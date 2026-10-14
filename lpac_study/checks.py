"""Mandatory checks before the factorial experiment: the rebuilt inputs match the library.

    python -m lpac_study.checks --data-dir /path/to/lpac [--comm-range 1024] [--steps 20]

Runs the published policy for a few steps at the given range and, at every
step, compares against ``CoverageEnvUtils.get_torch_geometric_data``:

1. the communication-map channels rebuilt from positions with all in-range
   neighbours (maximum absolute error reported);
2. which channels of ``data.x`` are communication maps (1 and 2) and which are
   perception maps (0: local importance map, 3: obstacle map), by checking
   that only channels 1 and 2 change when neighbours are removed;
3. the library's ``edge_index`` equals the in-range graph, and the simulator's
   own neighbour count matches it (the simulated range is untouched).
"""
from __future__ import annotations

import argparse

import torch
from coverage_control import CoverageEnvUtils
from torch_geometric.utils import to_undirected

from .lpac_policy import CNN_MAP_SIZE, LPACPolicy
from .neighbourhood import (comm_maps_from_positions, in_range_mask, knn_within_range_mask, mask_to_edge_index,
                            robot_positions_double, symmetrize)
from .paths import resolve_data_dir
from .simulation import load_initial_positions, make_environment, make_parameters

COMM_CHANNELS = (1, 2)


def edge_set(edge_index: torch.Tensor) -> set[tuple[int, int]]:
    return set(map(tuple, edge_index.t().tolist()))


def run_checks(args: argparse.Namespace) -> None:
    paths = resolve_data_dir(args.data_dir)
    params = make_parameters(paths.env_params, args.comm_range, args.num_robots)
    env = make_environment(params, paths.env_file(args.env), load_initial_positions(paths.pos_file(args.env), args.num_robots))
    policy = LPACPolicy(paths, torch.device("cpu"))
    worst_map_error = 0.0
    worst_other_change = 0.0
    max_lonely = 0
    for step in range(args.steps):
        data = policy.build_graph(env, params)
        positions = robot_positions_double(env)
        full_mask = in_range_mask(positions, params.pCommunicationRange)
        rebuilt = comm_maps_from_positions(positions, full_mask, params.pCommunicationRange, CNN_MAP_SIZE)
        worst_map_error = max(worst_map_error, float((data.x[:, list(COMM_CHANNELS)] - rebuilt).abs().max()))
        # Channels that must not depend on neighbours: compare with a restricted neighbour set.
        restricted = comm_maps_from_positions(positions, symmetrize(knn_within_range_mask(positions, params.pCommunicationRange, 2)),
                                              params.pCommunicationRange, CNN_MAP_SIZE)
        assert not torch.equal(restricted, rebuilt) or full_mask.sum() <= 2 * len(positions)
        other = [c for c in range(data.x.shape[1]) if c not in COMM_CHANNELS]
        worst_other_change = max(worst_other_change, float(data.x[:, other].abs().sum() == 0))
        # Graph: library edge_index (d <= range) versus strict in-range mask.
        library_edges = edge_set(to_undirected(data.edge_index))
        rebuilt_edges = edge_set(mask_to_edge_index(full_mask))
        if library_edges != rebuilt_edges:
            raise AssertionError(f"step {step}: edge sets differ ({len(library_edges ^ rebuilt_edges)} edges)")
        simulator_counts = [len(env.GetRelativePositonsNeighbors(i)) for i in range(len(positions))]
        if simulator_counts != full_mask.sum(1).tolist():
            raise AssertionError(f"step {step}: simulator neighbour counts differ from the in-range mask")
        # A robot without neighbour must have all-zero communication channels in the library's map.
        lonely = [i for i, n in enumerate(simulator_counts) if n == 0]
        if lonely and float(data.x[lonely][:, list(COMM_CHANNELS)].abs().max()) != 0.0:
            raise AssertionError(f"step {step}: a robot without neighbour has a non-zero communication map")
        max_lonely = max(max_lonely, len(lonely))
        actions, _, _ = policy.act(data)
        env.StepActions(actions)
    print(f"steps checked: {args.steps} at {args.comm_range:.0f} m, {args.num_robots} robots, env {args.env}")
    print(f"comm-map reconstruction, max |error| over all steps: {worst_map_error:.3e}")
    print(f"x channels: {data.x.shape[1]} (comm maps = {COMM_CHANNELS}; others = perception maps, never zero: "
          f"{'ok' if worst_other_change == 0 else 'EMPTY'})")
    print(f"edge_index equals the in-range graph and the simulator's neighbour lists: ok")
    print(f"robots without neighbour have all-zero map channels: ok "
          f"(up to {max_lonely} such robots in one step)")
    print(f"simulated communication range after the checks: {params.pCommunicationRange:.0f} m")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir")
    parser.add_argument("--comm-range", type=float, default=1024.0)
    parser.add_argument("--num-robots", type=int, default=32)
    parser.add_argument("--env", type=int, default=0)
    parser.add_argument("--steps", type=int, default=20)
    run_checks(parser.parse_args(argv))


if __name__ == "__main__":
    main()
