"""Statistics of the communication graph actually fed to the GNN."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch_geometric.utils import degree


@dataclass(frozen=True)
class DegreeStats:
    """Per-step degree statistics of one communication graph."""

    mean: float
    std: float
    isolated_fraction: float


def degree_stats(edge_index: torch.Tensor, num_nodes: int) -> DegreeStats:
    """Mean and standard deviation of node degree from a PyG ``edge_index``.

    The communication graph is symmetric, so in- and out-degree coincide.
    """
    deg = degree(edge_index[1], num_nodes=num_nodes, dtype=torch.float32)
    return DegreeStats(
        mean=float(deg.mean()),
        std=float(deg.std(unbiased=False)),
        isolated_fraction=float((deg == 0).float().mean()),
    )
