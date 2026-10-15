"""LPAC variants for the ablation of the neighbour map (phase 2 of the project).

Three models, same CNN, same action MLP, same training recipe:

* ``V0`` the original LPAC: 4-channel CNN input (local map, two neighbour-map channels,
  obstacle map) and five ``TAGConv`` layers with K = 3 (receptive field 15 hops);
* ``V1`` LPAC without the neighbour map: 2-channel CNN input, ``TAGConv`` unchanged.
  Control: what does removing the fragile map alone do?
* ``V2`` V1 plus the neighbour geometry on the graph edges: the GNN is five
  ``GATv2Conv`` layers (4 heads of 64, concatenated to 256, ``edge_dim = 2``) whose
  edge attribute is the relative position ``p_j - p_i`` of the sender, divided by the
  world size (the same constant that normalizes ``data.pos``), never by the range.
  Receptive field 5 hops.
* ``V3`` V1 plus one ``PointNetConv`` layer in front of the unchanged ``TAGConv`` stack.
  The layer reads, for every edge, the relative position of the neighbour divided by a
  fixed constant of 256 m (never by the range, never by the world size), and adds its
  output to the TAGConv input (residual connection). Receptive field 16 hops.

The library's ``LPAC`` class is subclassed, not modified; the edge attributes are
computed in ``forward`` from ``data.pos`` and ``data.edge_index``.
"""
from __future__ import annotations

import copy

import torch
import torch_geometric
from coverage_control.nn import LPAC
from torch_geometric.nn import GATv2Conv, PointNetConv

VARIANTS = ("V0", "V1", "V2", "V3")
EDGE_DIM = 2
HEADS = 4
# V3 constants. ``data.pos`` is (p + W/2) / W with W the world size (1024 m in every dataset of
# this project): multiplying by WORLD_SIZE_M only undoes that normalization and gives metres
# back. POINTNET_SCALE_M is the one scale the layer ever sees: a neighbour 256 m away reads as
# 1.0 whatever the communication range at training or at test time.
WORLD_SIZE_M = 1024.0
POINTNET_SCALE_M = 256.0
POINTNET_HIDDEN = 256


def uses_comm_map(variant: str) -> bool:
    """Whether the CNN input carries the two neighbour-map channels."""
    return variant == "V0"


class LPACPointNet(LPAC):
    """V3: LPAC without neighbour maps, with one PointNetConv layer ahead of the TAGConv stack.

    Why this layer. In V1 the GNN only knows *that* two robots are linked (binary adjacency)
    plus each robot's absolute position; the relative geometry that the neighbour map used to
    rasterize is gone. ``PointNetConv`` (Qi et al., 2017, as implemented in PyG) is the simplest
    layer that reads geometry on edges: for every edge j -> i it feeds ``[x_j, p_j - p_i]`` to a
    small MLP (``local_nn``), takes the element-wise max over the neighbours of i, and maps the
    result through a second MLP (``global_nn``). Unlike V2, which replaced the whole GNN by GATv2
    attention, V3 keeps the five TAGConv layers exactly as in V1/V0 and only *adds* geometry in
    front of them.

    Design choices, each one deliberate:
    * positions are given to the layer in units of ``POINTNET_SCALE_M`` (256 m), a constant:
      the layer sees the same numbers for the same geometry at every communication range, so
      nothing in its input scales with the range (the flaw of the rasterized map);
    * ``add_self_loops=True`` (PyG default): a robot with no neighbour in range (31 % of the
      robots at 128 m) still gets one message, from itself with relative position 0, instead
      of an undefined max over an empty set;
    * max aggregation (PyG default): invariant to the number of neighbours, which varies from
      ~2 at 128 m to ~30 at 1024 m;
    * residual connection ``h = x + relu(pointnet(x))``: the V1 pathway is left intact, so the
      model can at worst ignore the layer, and the training-loss control against V1 is fair;
    * output width equal to the TAGConv input width (CNN latent 32 + position 2 = 34) so that
      the five TAGConv layers are unchanged, hyperparameters included.
    """

    def __init__(self, config: dict):
        super().__init__(variant_config(config, "V3"))
        width = self.cnn_backbone.latent_size + 2  # 34: what the first TAGConv layer expects
        self.pointnet = PointNetConv(
            local_nn=torch.nn.Sequential(torch.nn.Linear(width + 2, POINTNET_HIDDEN), torch.nn.ReLU(),
                                         torch.nn.Linear(POINTNET_HIDDEN, POINTNET_HIDDEN)),
            global_nn=torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Linear(POINTNET_HIDDEN, width)),
            add_self_loops=True)

    @staticmethod
    def scaled_positions(pos: torch.Tensor) -> torch.Tensor:
        """Positions in units of 256 m: ``PointNetConv`` subtracts them along each edge."""
        return pos * (WORLD_SIZE_M / POINTNET_SCALE_M)

    def forward(self, data: torch_geometric.data.Data) -> torch.Tensor:
        x, edge_index, pos = data.x, data.edge_index, data.pos
        cnn_output = self.cnn_backbone(x.view(-1, x.shape[-3], x.shape[-2], x.shape[-1]))
        gnn_input = torch.cat([cnn_output, pos], dim=-1)  # exactly V1's GNN input
        geometry = torch.relu(self.pointnet(gnn_input, self.scaled_positions(pos), edge_index))
        gnn_output = self.gnn_backbone(gnn_input + geometry, edge_index)  # the unchanged TAGConv stack
        return self.output_linear(self.gnn_mlp(gnn_output))


def variant_config(config: dict, variant: str) -> dict:
    """Copy of the release ``learning_params`` adapted to the variant (2 CNN channels for V1/V2)."""
    config = copy.deepcopy(config)
    if not uses_comm_map(variant):
        config["CNNBackBone"]["InputDim"] = 2
        config["ModelConfig"]["UseCommMaps"] = False
    return config


class EdgeGeometryGNN(torch.nn.Module):
    """Five GATv2 layers reading the relative position of the sender on every edge."""

    def __init__(self, input_dim: int, latent_size: int, num_layers: int, heads: int = HEADS):
        super().__init__()
        assert latent_size % heads == 0, "latent size must split evenly across heads"
        per_head = latent_size // heads
        self.num_layers = num_layers  # same attribute as the library's GNNBackBone
        self.layers = torch.nn.ModuleList()
        in_dim = input_dim
        for _ in range(num_layers):
            self.layers.append(GATv2Conv(in_dim, per_head, heads=heads, edge_dim=EDGE_DIM, concat=True))
            in_dim = latent_size

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = torch.relu(layer(x, edge_index, edge_attr))
        return x


class LPACEdgeGeometry(LPAC):
    """LPAC without neighbour maps, with ``p_j - p_i`` (normalized) as edge attribute of a GATv2 GNN."""

    def __init__(self, config: dict):
        super().__init__(variant_config(config, "V2"))
        self.gnn_backbone = EdgeGeometryGNN(self.cnn_backbone.latent_size + 2, self.latent_size, self.num_layers)

    @staticmethod
    def edge_attributes(pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Relative position of the sender ``j`` seen from the receiver ``i`` for every edge ``(j, i)``.

        ``data.pos`` is already ``(p + W/2) / W`` (world size ``W``), so the difference is
        ``(p_j - p_i) / W``: a fixed constant, independent of the communication range.
        """
        sender, receiver = edge_index[0], edge_index[1]
        return pos[sender] - pos[receiver]

    def forward(self, data: torch_geometric.data.Data) -> torch.Tensor:
        x, edge_index, pos = data.x, data.edge_index, data.pos
        cnn_output = self.cnn_backbone(x.view(-1, x.shape[-3], x.shape[-2], x.shape[-1]))
        gnn_input = torch.cat([cnn_output, pos], dim=-1)
        gnn_output = self.gnn_backbone(gnn_input, edge_index, self.edge_attributes(pos, edge_index))
        return self.output_linear(self.gnn_mlp(gnn_output))


def build_model(variant: str, config: dict) -> LPAC:
    """The model of a variant from the release ``learning_params`` dictionary."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}, expected one of {VARIANTS}")
    if variant == "V2":
        return LPACEdgeGeometry(config)
    if variant == "V3":
        return LPACPointNet(config)
    return LPAC(variant_config(config, variant))


def parameter_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def receptive_field_hops(variant: str, config: dict) -> int:
    """Graph hops seen by a node: layers x K for TAGConv, layers for GATv2, plus one for V3's front layer."""
    layers = config["GNNBackBone"]["NumLayers"]
    if variant == "V2":
        return layers
    return layers * config["GNNBackBone"]["NumHops"] + (1 if variant == "V3" else 0)
