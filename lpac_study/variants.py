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
* ``V0R`` the original LPAC with one more GNN input per robot: its own communication range
  divided by the world size (experiment H2, trained on teams with a range drawn per robot).
* ``V3`` V1 plus one ``PointNetConv`` layer in front of the unchanged ``TAGConv`` stack.
* ``V4`` V3 with bounded, range-invariant edge features: the unit vector towards the neighbour
  and ``tanh(d / 256 m)`` instead of the raw relative position.
* ``V5`` the original LPAC with three pairs of neighbour-map channels drawn at fixed scales
  (256, 512, 1024 m) instead of one pair drawn at the communication range: 8 CNN channels.
* ``V6`` V5 plus V4's bounded-edge layer: the multi-scale maps feed the CNN and the bounded
  edge features feed a residual front layer ahead of the unchanged TAGConv stack.
* ``V2R`` / ``V2RL`` the GATv2 of V2 with the usual hygiene (pre-norm residual blocks and V4's
  bounded edge features), without and with the log-degree of each robot as a node feature:
  the diagnosis of why V2 underfits (question Q3 of the teaching assistant).
  The layer reads, for every edge, the relative position of the neighbour divided by a
  fixed constant of 256 m (never by the range, never by the world size), and adds its
  output to the TAGConv input (residual connection). Receptive field 16 hops.

The library's ``LPAC`` class is subclassed, not modified; the edge attributes are
computed in ``forward`` from ``data.pos`` and ``data.edge_index``.
"""
from __future__ import annotations

import copy
from pathlib import Path

import torch
import torch_geometric
from coverage_control.nn import LPAC
from coverage_control.nn.data_loaders.loaders import CNNGNNDataset
from torch_geometric.nn import GATv2Conv, PointNetConv

VARIANTS = ("V0", "V1", "V2", "V3", "V0R", "V4", "V5", "V6", "V2R", "V2RL", "V2B")
MULTISCALE_SCALES_M = (256.0, 512.0, 1024.0)  # V5: one pair of map channels per scale
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
    """Whether the CNN input carries neighbour-map channels (two for V0/V0R, six for V5/V6)."""
    return variant in ("V0", "V0R", "V5", "V6")


def uses_log_degree(variant: str) -> bool:
    """Whether the GNN input carries log(1 + in-degree) as an extra node feature (V2RL)."""
    return variant == "V2RL"


def uses_multiscale_maps(variant: str) -> bool:
    """Whether the neighbour maps are the three fixed-scale pairs of V5 (also V6)."""
    return variant in ("V5", "V6")


def uses_range_input(variant: str) -> bool:
    """Whether the GNN input carries each robot's own communication range (``data.ranges``)."""
    return variant == "V0R"


class LPACRangeInput(LPAC):
    """V0R: LPAC whose GNN also reads each robot's own communication range.

    The maps and the graph are those of the original model; the only change is one extra
    input per node, ``r_i / WORLD_SIZE_M`` (0.125 for 128 m, 1 for 1024 m), concatenated to the
    CNN latent and the position before the first TAGConv layer (35 inputs instead of 34).
    Trained on teams where every robot draws its range in {128, 256, 512, 1024} m, the model
    is told at which scale its own map is drawn, which the original model has to guess.
    ``data.ranges`` (one value per node, in metres) is set by the dataset class and by the
    policies; the TAGConv stack, the action MLP and the training recipe are unchanged.
    """

    def __init__(self, config: dict):
        super().__init__(variant_config(config, "V0R"))
        from coverage_control.nn.models.gnn_backbone import GNNBackBone  # the library's own stack, one input wider
        self.gnn_backbone = GNNBackBone(self.config, self.cnn_backbone.latent_size + 3)

    def forward(self, data: torch_geometric.data.Data) -> torch.Tensor:
        x, edge_index, pos = data.x, data.edge_index, data.pos
        ranges = (data.ranges / WORLD_SIZE_M).to(x.dtype).view(-1, 1)
        cnn_output = self.cnn_backbone(x.view(-1, x.shape[-3], x.shape[-2], x.shape[-1]))
        gnn_input = torch.cat([cnn_output, pos, ranges], dim=-1)
        return self.output_linear(self.gnn_mlp(self.gnn_backbone(gnn_input, edge_index)))


class RangeDataset(CNNGNNDataset):
    """``CNNGNNDataset`` that also attaches ``data.ranges`` from ``robot_ranges.pt`` (metres per robot).

    A dataset without that file (generated at a single range) gets the constant ``default_range``.
    """

    def __init__(self, data_dir, stage, use_comm_map, world_size, default_range: float | None = None):
        super().__init__(data_dir, stage, use_comm_map, world_size)
        path = Path(f"{data_dir}/{stage}/robot_ranges.pt")
        if path.is_file():
            self.robot_ranges = torch.load(path, weights_only=True).to(torch.float32)
        else:
            if default_range is None:
                raise FileNotFoundError(f"{path} is missing and no default range was given")
            self.robot_ranges = torch.full((len(self), self.robot_positions.shape[1]), float(default_range))

    def get(self, idx):
        data, targets = super().get(idx)
        data.ranges = self.robot_ranges[idx]
        return data, targets


class MultiScaleDataset(CNNGNNDataset):
    """Dataset for V5 that keeps the six neighbour-map channels sparse until a sample is requested.

    The library's loader densifies ``comm_maps.pt`` at load time; with six channels that is
    100,000 x 32 x 6 x 32 x 32 floats, 78 GB, which does not fit next to the other tensors.
    The maps hold at most a few tens of nonzero cells per robot, so they are kept as the
    sparse COO tensor written by ``redraw_ranges.py --multiscale`` and densified per sample in
    ``get``; the resulting input is identical to what ``DataLoaderUtils.load_maps`` would build
    (local map, six neighbour channels, obstacle map).
    """

    def __init__(self, data_dir, stage, use_comm_map=True, world_size=1024):
        super().__init__(data_dir, stage, False, world_size)  # local and obstacle maps, dense
        comm = torch.load(f"{data_dir}/{stage}/comm_maps.pt", weights_only=True)
        comm = comm.coalesce() if comm.is_sparse else comm.to_sparse_coo().coalesce()
        self.num_comm_channels = comm.shape[2]
        self.comm_shape = tuple(comm.shape[1:])
        # Indexing a coalesced COO tensor by sample scans all of its indices (190 M entries for the
        # 100k mixed set: about 0.3 s per sample, 20 min per epoch on 27 workers). The entries are
        # sorted by sample, so they are split once into per-sample index/value slices.
        indices, values = comm.indices(), comm.values()
        counts = torch.bincount(indices[0], minlength=comm.shape[0]).tolist()
        self.comm_indices = list(torch.split(indices[1:], counts, dim=1))
        self.comm_values = list(torch.split(values, counts))

    def comm_map(self, idx: int) -> torch.Tensor:
        """Dense ``(robots, channels, S, S)`` neighbour maps of one sample."""
        return torch.sparse_coo_tensor(self.comm_indices[idx], self.comm_values[idx], self.comm_shape).to_dense()

    def get(self, idx):
        from coverage_control.nn.data_loaders.data_loader_utils import DataLoaderUtils
        base = self.maps[idx]  # (robots, 2, S, S): local map, obstacle map
        comm = self.comm_map(idx)  # (robots, 6, S, S)
        maps = torch.cat([base[:, :1], comm, base[:, 1:2]], dim=1)
        data = DataLoaderUtils.to_torch_geometric_data(maps, self.edge_weights[idx], self.robot_positions[idx])
        targets = self.targets[idx]
        if targets.dim == 3:
            targets = targets.view(-1, targets.shape[-1])
        return data, targets


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
    if uses_multiscale_maps(variant):
        config["CNNBackBone"]["InputDim"] = 2 + 2 * len(MULTISCALE_SCALES_M)
    return config


class EdgeGeometryGNN(torch.nn.Module):
    """Five GATv2 layers reading the relative position of the sender on every edge."""

    def __init__(self, input_dim: int, latent_size: int, num_layers: int, heads: int = HEADS,
                 edge_dim: int = EDGE_DIM):
        super().__init__()
        assert latent_size % heads == 0, "latent size must split evenly across heads"
        per_head = latent_size // heads
        self.num_layers = num_layers  # same attribute as the library's GNNBackBone
        self.layers = torch.nn.ModuleList()
        in_dim = input_dim
        for _ in range(num_layers):
            self.layers.append(GATv2Conv(in_dim, per_head, heads=heads, edge_dim=edge_dim, concat=True))
            in_dim = latent_size

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = torch.relu(layer(x, edge_index, edge_attr))
        return x


class ResidualEdgeGeometryGNN(torch.nn.Module):
    """GATv2 layers in pre-norm residual blocks: ``x <- x + relu(GATv2(LayerNorm(x)))``.

    The stack of V2 applies ``relu(GATv2(x))`` with no residual and no normalization, so five
    attention layers have to carry the signal on their own; with 15 layers (to match the 15 hops
    of five K = 3 TAGConv layers) that stack does not train at all. The blocks here are the
    standard remedy and leave the attention itself untouched.
    """

    def __init__(self, input_dim: int, latent_size: int, num_layers: int, heads: int = HEADS, edge_dim: int = 3):
        super().__init__()
        assert latent_size % heads == 0, "latent size must split evenly across heads"
        per_head = latent_size // heads
        self.num_layers = num_layers  # same attribute as the library's GNNBackBone
        self.input_proj = torch.nn.Linear(input_dim, latent_size)
        self.layers = torch.nn.ModuleList(
            GATv2Conv(latent_size, per_head, heads=heads, edge_dim=edge_dim, concat=True) for _ in range(num_layers))
        self.norms = torch.nn.ModuleList(torch.nn.LayerNorm(latent_size) for _ in range(num_layers))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for layer, norm in zip(self.layers, self.norms):
            x = x + torch.relu(layer(norm(x), edge_index, edge_attr))
        return x


class LPACResidualEdgeGeometry(LPAC):
    """V2R / V2RL: the GATv2 model of V2 with residual blocks, LayerNorm and bounded edge features.

    Three changes to V2, each one a candidate explanation of its underfitting (train loss 0.62
    after 100 epochs against 0.21 for V0):
    * the edge input is V4's ``[(p_j - p_i) / d, tanh(d / 256 m)]`` instead of ``(p_j - p_i) / W``,
      whose values only span +-0.125 at the 128 m training range;
    * every layer is a pre-norm residual block, so depth does not have to be crossed in one pass;
    * ``V2RL`` adds ``log(1 + in-degree) / log(33)`` to the node input: softmax attention is a
      weighted *mean* and loses the number of neighbours, which the normalized sum of TAGConv
      partly keeps; if that is what V2 misses, the feature restores it.
    The number of layers is read from the config, so the 15-layer run (the receptive field of
    five K = 3 TAGConv layers) is the same class with ``NumLayers = 15``.
    """

    def __init__(self, config: dict, variant: str = "V2R"):
        super().__init__(variant_config(config, variant))
        self.log_degree = uses_log_degree(variant)
        width = self.cnn_backbone.latent_size + 2 + (1 if self.log_degree else 0)
        self.gnn_backbone = ResidualEdgeGeometryGNN(width, self.latent_size, self.num_layers)

    def forward(self, data: torch_geometric.data.Data) -> torch.Tensor:
        x, edge_index, pos = data.x, data.edge_index, data.pos
        cnn_output = self.cnn_backbone(x.view(-1, x.shape[-3], x.shape[-2], x.shape[-1]))
        parts = [cnn_output, pos]
        if self.log_degree:
            degree = torch.zeros(x.shape[0], device=x.device, dtype=cnn_output.dtype)
            degree.index_add_(0, edge_index[1], torch.ones(edge_index.shape[1], device=x.device, dtype=cnn_output.dtype))
            parts.append((torch.log1p(degree) / float(torch.log(torch.tensor(33.0)))).view(-1, 1))
        gnn_input = torch.cat(parts, dim=-1)
        edge_attr = LPACBoundedGeometry.edge_features(pos, edge_index)
        return self.output_linear(self.gnn_mlp(self.gnn_backbone(gnn_input, edge_index, edge_attr)))


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


class LPACBoundedEdgeGATv2(LPACEdgeGeometry):
    """V2B: exactly V2, with the bounded edge features of V4 and nothing else changed.

    The single-cause ablation of question Q3. V2res changes three things at once (pre-norm
    residual blocks, LayerNorm, and the edge input), so it identifies a bundle. This variant
    changes only the edge input, from V2's ``(p_j - p_i) / W``, which spans +-0.125 at the 128 m
    training range, to V4's ``[(p_j - p_i) / d, tanh(d / 256 m)]``. The attention stack stays
    plain: ``relu(GATv2(x))``, no residual, no normalization. If the training loss drops most of
    the way from 0.676 towards 0.432, the edge encoding carries the bundle; if it does not, the
    residual blocks do.
    """

    def __init__(self, config: dict):
        LPAC.__init__(self, variant_config(config, "V2"))
        self.gnn_backbone = EdgeGeometryGNN(self.cnn_backbone.latent_size + 2, self.latent_size,
                                            self.num_layers, edge_dim=3)

    def forward(self, data: torch_geometric.data.Data) -> torch.Tensor:
        x, edge_index, pos = data.x, data.edge_index, data.pos
        cnn_output = self.cnn_backbone(x.view(-1, x.shape[-3], x.shape[-2], x.shape[-1]))
        gnn_input = torch.cat([cnn_output, pos], dim=-1)
        edge_attr = LPACBoundedGeometry.edge_features(pos, edge_index)
        return self.output_linear(self.gnn_mlp(self.gnn_backbone(gnn_input, edge_index, edge_attr)))


class BoundedGeometryConv(torch_geometric.nn.MessagePassing):
    """PointNet-style layer whose edge input is bounded and does not scale with the range (V4).

    For every edge j -> i the message is ``local_nn([x_j, e_ij])`` with
    ``e_ij = [(p_j - p_i) / d, tanh(d / scale)]`` (unit direction and a squashed distance, three
    numbers in [-1, 1]); messages are max-aggregated over the neighbours of i (self-loop with
    ``e = 0`` included, so an isolated robot still gets one message) and mapped through
    ``global_nn``. The same structure as ``PointNetConv`` (Qi et al., 2017), with the raw
    relative position replaced by its bounded encoding.
    """

    def __init__(self, local_nn: torch.nn.Module, global_nn: torch.nn.Module):
        super().__init__(aggr="max")
        self.local_nn = local_nn
        self.global_nn = global_nn

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        loops = torch.arange(x.shape[0], device=x.device)
        edge_index = torch.cat([edge_index, torch.stack([loops, loops])], dim=1)
        edge_attr = torch.cat([edge_attr, torch.zeros(x.shape[0], edge_attr.shape[1], device=x.device, dtype=edge_attr.dtype)])
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return self.global_nn(out)

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        return self.local_nn(torch.cat([x_j, edge_attr], dim=-1))


def bounded_geometry_layer(width: int) -> "BoundedGeometryConv":
    """The front layer of V4 (and V6): ``local_nn`` on ``[x_j, e_ij]`` (width + 3 inputs), ``global_nn`` back to ``width``."""
    return BoundedGeometryConv(
        local_nn=torch.nn.Sequential(torch.nn.Linear(width + 3, POINTNET_HIDDEN), torch.nn.ReLU(),
                                     torch.nn.Linear(POINTNET_HIDDEN, POINTNET_HIDDEN)),
        global_nn=torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Linear(POINTNET_HIDDEN, width)))


class LPACBoundedGeometry(LPAC):
    """V4: V3 with bounded, range-invariant edge features.

    V3 reads ``(p_j - p_i) / 256 m`` on every edge: at 1024 m those inputs are four times larger
    than at training, and section 13 shows that one recipe for both regimes fits neither. V4
    keeps everything of V3 (no neighbour map, the same five TAGConv layers, the same residual
    front layer, the same hidden width) and only changes what the front layer reads:
    ``e_ij = [(p_j - p_i) / d, tanh(d / 256 m)]``, the unit direction towards the neighbour and
    a squashed distance. Every component lies in [-1, 1] whatever the range; scaling all
    positions by a factor leaves the direction unchanged and only moves the tanh, so a long
    range changes the *number* of edges the layer sees, not the range of its inputs. V3 against
    V4 therefore isolates the edge encoding.
    """

    def __init__(self, config: dict, variant: str = "V4"):
        super().__init__(variant_config(config, variant))
        width = self.cnn_backbone.latent_size + 2  # 34, what the first TAGConv layer expects
        self.geometry = bounded_geometry_layer(width)

    @staticmethod
    def edge_features(pos: torch.Tensor, edge_index: torch.Tensor, scale_m: float = POINTNET_SCALE_M) -> torch.Tensor:
        """``[(p_j - p_i) / d, tanh(d / scale)]`` per edge j -> i, from the world-normalized ``data.pos``."""
        sender, receiver = edge_index[0], edge_index[1]
        rel = (pos[sender] - pos[receiver]) * WORLD_SIZE_M  # metres
        dist = rel.norm(dim=-1, keepdim=True)
        direction = rel / dist.clamp_min(1e-6)
        return torch.cat([direction, torch.tanh(dist / scale_m)], dim=-1)

    def forward(self, data: torch_geometric.data.Data) -> torch.Tensor:
        x, edge_index, pos = data.x, data.edge_index, data.pos
        cnn_output = self.cnn_backbone(x.view(-1, x.shape[-3], x.shape[-2], x.shape[-1]))
        gnn_input = torch.cat([cnn_output, pos], dim=-1)
        geometry = torch.relu(self.geometry(gnn_input, edge_index, self.edge_features(pos, edge_index)))
        gnn_output = self.gnn_backbone(gnn_input + geometry, edge_index)
        return self.output_linear(self.gnn_mlp(gnn_output))


class LPACMultiScaleGeometry(LPACBoundedGeometry):
    """V6: V5's multi-scale maps and V4's bounded edge layer in one model.

    Motivation (section 14 of the journal): V5, whose CNN reads three pairs of neighbour maps at
    fixed scales, holds the 256 m level across ranges but does not improve beyond it; V4, whose
    front layer reads bounded edge features, is the one model whose cost keeps falling with the
    range but it pays 0.14 at 256 m, where it has no map. V6 gives the CNN the V5 input (8
    channels, built exactly as for V5) and the GNN the V4 front layer (same class, same width,
    residual, self-loops), with the TAGConv stack unchanged, to test whether the two gains add.
    Parameters: V5 plus the V4 layer.
    """

    def __init__(self, config: dict):
        super().__init__(config, variant="V6")


class LPACMultiScale(LPAC):
    """V5: LPAC whose neighbour maps are three pairs of channels at fixed scales (256, 512, 1024 m).

    The original model draws one pair of channels at the communication range, so the drawing
    scale moves with the range and the CNN reads an image it was not trained on (section 4).
    The inference-time fix draws one pair at a fixed 256 m scale (section 9) and loses the
    neighbours beyond 256 m. V5 gives the CNN all three: pair k holds the robots within
    min(range, scale_k) drawn at scale_k with the library's value convention (p_j - p_i) /
    scale_k, so at 128 m the three pairs show the same (few) neighbours at three zooms, and at
    1024 m the 256 m pair is the fixed-scale map of section 9 while the 1024 m pair is the
    original map. CNN input 8 channels (local map, six neighbour channels, obstacle map);
    nothing else changes. The datasets are redrawn from the stored positions
    (``training/redraw_ranges.py --multiscale``).
    """

    def __init__(self, config: dict):
        super().__init__(variant_config(config, "V5"))


def build_model(variant: str, config: dict) -> LPAC:
    """The model of a variant from the release ``learning_params`` dictionary."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}, expected one of {VARIANTS}")
    if variant == "V2":
        return LPACEdgeGeometry(config)
    if variant == "V3":
        return LPACPointNet(config)
    if variant == "V0R":
        return LPACRangeInput(config)
    if variant == "V4":
        return LPACBoundedGeometry(config)
    if variant == "V5":
        return LPACMultiScale(config)
    if variant == "V6":
        return LPACMultiScaleGeometry(config)
    if variant in ("V2R", "V2RL"):
        return LPACResidualEdgeGeometry(config, variant)
    if variant == "V2B":
        return LPACBoundedEdgeGATv2(config)
    return LPAC(variant_config(config, variant))


def parameter_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def receptive_field_hops(variant: str, config: dict) -> int:
    """Graph hops seen by a node: layers x K for TAGConv, layers for GATv2, plus one for V3's front layer."""
    layers = config["GNNBackBone"]["NumLayers"]
    if variant in ("V2", "V2R", "V2RL", "V2B"):
        return layers
    return layers * config["GNNBackBone"]["NumHops"] + (1 if variant in ("V3", "V4", "V6") else 0)
