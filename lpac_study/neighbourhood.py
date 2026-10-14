"""Neighbourhood construction for the inference-time factorial experiment.

The neighbourhood enters LPAC through two inputs that this module rebuilds
from raw robot positions:

* the communication graph given to the five TAGConv layers (``edge_index``);
* the two "communication map" channels given to the CNN, which encode the
  relative position of neighbours on a ``map_size x map_size`` grid whose
  resolution scales with the communication range.

The map reconstruction mirrors ``CoverageEnvUtils.get_communication_maps`` of
coverage_control 1.3.1 (a robot ``j`` is a neighbour of ``i`` when
``|p_j - p_i| < range``; cell index ``round(rel * map_size / (2 range) +
map_size / 2 - 1/2)``; cell value ``rel / range`` summed over the robots that
fall in the same cell).  ``checks.py`` verifies the reconstruction against the
library before any neighbour restriction is applied.
"""
from __future__ import annotations

import torch
from torch_geometric.utils import dense_to_sparse, to_undirected


def robot_positions_double(env) -> torch.Tensor:
    """Raw robot positions in float64, exactly as the C++ simulator holds them.

    The library converts positions to float32 before use; computing relative
    positions in double and casting afterwards reproduces the C++ maps bit for bit.
    """
    return torch.tensor([[p[0], p[1]] for p in env.GetRobotPositions()], dtype=torch.float64)


def relative_positions(positions: torch.Tensor) -> torch.Tensor:
    """``rel[i, j] = p_j - p_i`` for raw positions of shape ``(N, 2)``."""
    return positions.unsqueeze(0) - positions.unsqueeze(1)


def in_range_mask(positions: torch.Tensor, comm_range: float) -> torch.Tensor:
    """Boolean ``(N, N)`` mask of robots strictly within communication range (no self-loops)."""
    distances = torch.cdist(positions, positions)
    mask = distances < comm_range
    mask.fill_diagonal_(False)
    return mask


def knn_within_range_mask(positions: torch.Tensor, comm_range: float, k: int) -> torch.Tensor:
    """Directed mask keeping, for every robot, its ``k`` nearest robots among those in range.

    Robots with fewer than ``k`` neighbours in range keep all of them.  This is
    the k-nearest-neighbour rule of Muthusamy et al. (arXiv:2409.19829)
    restricted to physically reachable robots.
    """
    distances = torch.cdist(positions, positions)
    reachable = in_range_mask(positions, comm_range)
    distances = distances.masked_fill(~reachable, float("inf"))
    k = min(k, positions.shape[0] - 1)
    nearest = distances.topk(k, dim=1, largest=False).indices
    mask = torch.zeros_like(reachable)
    mask.scatter_(1, nearest, True)
    return mask & reachable


def symmetrize(mask: torch.Tensor) -> torch.Tensor:
    """Undirected version of a neighbour mask: keep an edge if either robot selected the other."""
    return mask | mask.T


def mask_to_edge_index(mask: torch.Tensor) -> torch.Tensor:
    """PyG ``edge_index`` (undirected, coalesced) from a symmetric boolean mask."""
    edge_index, _ = dense_to_sparse(mask.to(torch.float32))
    return to_undirected(edge_index, num_nodes=mask.shape[0])


def comm_maps_from_positions(
    positions: torch.Tensor, neighbour_mask: torch.Tensor, comm_range: float, map_size: int
) -> torch.Tensor:
    """Communication maps ``(N, 2, map_size, map_size)`` for the neighbours of ``neighbour_mask``.

    Args:
        positions: raw robot positions, shape ``(N, 2)``, in metres.
        neighbour_mask: ``(N, N)`` boolean, row ``i`` = robots drawn on robot ``i``'s map.
        comm_range: range used for the grid resolution and the value normalization.
        map_size: side of the square map (32 for the published model).
    """
    num_robots = positions.shape[0]
    rel = relative_positions(positions).to(torch.float32)  # the library works on float32 relative positions
    scaled = torch.round(rel * map_size / (comm_range * 2.0) + (map_size / 2.0 - 0.5)).long()
    values = rel / comm_range
    maps = torch.zeros((num_robots, 2, map_size, map_size), dtype=torch.float32)
    robot, neighbour = neighbour_mask.nonzero(as_tuple=True)
    cell_x, cell_y = scaled[robot, neighbour, 0], scaled[robot, neighbour, 1]
    if len(robot) and (cell_x.min() < 0 or cell_y.min() < 0 or cell_x.max() >= map_size or cell_y.max() >= map_size):
        raise ValueError("a neighbour falls outside the communication map")
    for channel in (0, 1):
        maps[:, channel].index_put_((robot, cell_x, cell_y), values[robot, neighbour, channel], accumulate=True)
    return maps
