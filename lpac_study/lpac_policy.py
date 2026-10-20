"""The published LPAC-K3 policy, instrumented to expose its communication graph.

The wrapper mirrors ``coverage_control.algorithms.ControllerNN`` (v1.3.1) but
keeps the PyG ``Data`` object fed to the network and captures the output of
each of the five ``TAGConv`` layers through forward hooks.  ``torch.compile``
is not used so that hooks and CPU execution stay simple and deterministic.
"""
from __future__ import annotations

from pathlib import Path

import torch
import coverage_control as cc
import coverage_control.nn as cc_nn
from coverage_control import CoverageEnvUtils, CoverageSystem, IOUtils, PointVector
from torch_geometric.data import Data

from .paths import DataPaths
from .neighbourhood import comm_maps_from_positions, in_range_mask, robot_positions_double
from .variants import MULTISCALE_SCALES_M, build_model, uses_comm_map, uses_multiscale_maps, uses_range_input

# Controller options of the authors' eval.toml ([[Controllers]] lpac entry).
USE_CNN = True
USE_COMM_MAP = True
CNN_MAP_SIZE = 32


def multiscale_channels(x: torch.Tensor, positions: torch.Tensor, comm_range: float,
                        scales=MULTISCALE_SCALES_M) -> torch.Tensor:
    """V5 input: local map, one pair of neighbour channels per fixed scale, obstacle map.

    ``x`` is the library's 4-channel input (local map, two neighbour channels at the range,
    obstacle map); pair k holds the robots within min(comm_range, scale_k) at scale_k.
    """
    pairs = [comm_maps_from_positions(positions, in_range_mask(positions, min(comm_range, s)), s, CNN_MAP_SIZE).to(x.device)
             for s in scales]
    return torch.cat([x[:, :1], *pairs, x[:, 3:4]], dim=1)


class LPACPolicy:
    """Pre-trained LPAC model with access to its graph and layer activations."""

    def __init__(self, paths: DataPaths, device: torch.device, state_dict: str | Path | None = None,
                 variant: str = "V0"):
        """``state_dict`` overrides the published checkpoint; ``variant`` selects the architecture (V0 = LPAC)."""
        learning_params = IOUtils.load_toml(str(paths.learning_params))
        self.device = device
        self.variant = variant
        self.use_comm_map = USE_COMM_MAP and uses_comm_map(variant)
        self.model = build_model(variant, learning_params).to(device)
        if state_dict is None:
            self.model.load_model(str(paths.model_state_dict))
        else:  # a checkpoint saved on a GPU must be mapped to the evaluation device
            weights = torch.load(str(state_dict), map_location=device, weights_only=True)
            self.model.load_state_dict(weights, strict=True)
        self.model.eval()
        self.actions_mean = self.model.actions_mean.to(device)
        self.actions_std = self.model.actions_std.to(device)
        self.num_layers = self.model.gnn_backbone.num_layers
        self._layer_outputs: list[torch.Tensor] = []
        self._register_layer_hooks()

    def _register_layer_hooks(self) -> None:
        """Capture the post-ReLU output of every TAGConv layer of the GNN backbone."""
        backbone = self.model.gnn_backbone
        layers = (list(backbone.layers) if hasattr(backbone, "layers")
                  else [backbone._modules[f"graph_conv_{i}"] for i in range(self.num_layers)])
        for layer in layers:
            layer.register_forward_hook(self._store_layer_output)

    def _store_layer_output(self, module, inputs, output) -> None:
        # GNNBackBone.forward applies torch.relu after each TAGConv call.
        self._layer_outputs.append(torch.relu(output).detach())

    def build_graph(self, env: CoverageSystem, params: cc.Parameters) -> Data:
        """PyG graph exactly as the library builds it for the controller.

        Node features are the four CNN input channels, ``pos`` the normalized
        positions, ``edge_index`` the symmetric radius graph at the current
        communication range.  ``edge_weight`` is present in the object but the
        LPAC forward pass does not pass it to the GNN, so the adjacency is binary.
        """
        data = CoverageEnvUtils.get_torch_geometric_data(
            env, params, USE_CNN, self.use_comm_map, CNN_MAP_SIZE
        ).to(self.device)
        if uses_range_input(self.variant):  # homogeneous team: every robot has the test range
            data.ranges = torch.full((data.x.shape[0],), float(params.pCommunicationRange), device=self.device)
        if uses_multiscale_maps(self.variant):
            data.x = multiscale_channels(data.x, robot_positions_double(env), float(params.pCommunicationRange)).to(self.device)
        return data

    @torch.no_grad()
    def act(self, data: Data) -> tuple[PointVector, bool, list[torch.Tensor]]:
        """Velocity actions for the current graph, plus convergence flag and layer outputs."""
        self._layer_outputs = []
        actions = self.model(data) * self.actions_std + self.actions_mean
        converged = bool(torch.allclose(actions, torch.zeros_like(actions), atol=1e-5))
        layer_outputs = list(self._layer_outputs)
        return PointVector(actions.cpu().numpy()), converged, layer_outputs
