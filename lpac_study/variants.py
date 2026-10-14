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

The library's ``LPAC`` class is subclassed, not modified; the edge attributes are
computed in ``forward`` from ``data.pos`` and ``data.edge_index``.
"""
from __future__ import annotations

import copy

import torch
import torch_geometric
from coverage_control.nn import LPAC
from torch_geometric.nn import GATv2Conv

VARIANTS = ("V0", "V1", "V2")
EDGE_DIM = 2
HEADS = 4


def uses_comm_map(variant: str) -> bool:
    """Whether the CNN input carries the two neighbour-map channels."""
    return variant == "V0"


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
    return LPAC(variant_config(config, variant))


def parameter_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def receptive_field_hops(variant: str, config: dict) -> int:
    """Graph hops seen by a node: layers x K for TAGConv, layers for GATv2."""
    layers = config["GNNBackBone"]["NumLayers"]
    return layers if variant == "V2" else layers * config["GNNBackBone"]["NumHops"]
