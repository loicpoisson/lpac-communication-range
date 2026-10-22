"""Heterogeneous communication ranges at inference: some robots hear farther than others.

    python -m lpac_study.heterogeneous --baseline results/results.csv --out-dir results_hetero \
        --fractions 0 0.25 0.5 0.75 1 --draws 0 1 --maps asis fixed256 --num-envs 11

A fraction ``f`` of the robots (drawn once per environment and draw index, fixed seed) gets
the long range (1024 m), the others the short range (128 m). Reception semantics: robot
``i`` knows the robots within *its own* range ``r_i``, so the graph has an edge ``j -> i``
whenever ``|p_j - p_i| < r_i`` (directed when ranges differ) and the neighbour map of robot
``i`` is drawn with those robots at the scale ``r_i``.

MAP modes:
  asis      the map of robot ``i`` is drawn at the scale ``r_i`` (what the published model does);
  fixed256  the map of robot ``i`` holds the robots within ``min(r_i, 256 m)`` at the 256 m scale
            (the inference-time fix of sections 9-11 of the journal), the graph is unchanged.

Frozen-state probe (``--probe-every``): every ``k`` steps of the mixed-team episode the actions
of the current state are also computed with every robot at the short range (same map mode).
For the short-range robots the two inputs are identical except for the messages received from
the long-range robots: the relative difference of their actions measures the contamination
through the graph alone. The same quantity for the long-range robots mixes their own input
change with the graph effect.

The simulator always runs with the long range as its parameter (only the model inputs are
rebuilt here); ``f = 0`` must reproduce the as-is run at the short range and ``f = 1`` the
as-is run at the long range (checked in the journal).
"""
from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

from .factorial import COMM_CHANNELS, NUM_ROBOTS, load_clairvoyant
from .graph_stats import degree_stats
from .lpac_policy import CNN_MAP_SIZE, LPACPolicy
from .neighbourhood import relative_positions, robot_positions_double
from .paths import resolve_data_dir
from .simulation import load_initial_positions, make_environment, make_parameters
from .variants import MULTISCALE_SCALES_M, uses_comm_map, uses_multiscale_maps, uses_range_input

FIELDS = ["env", "fraction", "draw", "map", "model", "num_long", "long_range", "short_range",
          "cost_mean", "cost_final", "ratio_mean", "ratio_final", "clairvoyant_cost_mean",
          "degree_mean", "degree_std", "probe_short", "probe_long", "probe_short_abs", "probe_count",
          "long_center_dist", "long_degree128", "long_idf_share", "long_spread",
          "converged_step", "wall_s"]
FIXED_SCALE = 256.0


def comm_maps_per_robot(positions: torch.Tensor, neighbour_mask: torch.Tensor, scales: torch.Tensor,
                        map_size: int) -> torch.Tensor:
    """Communication maps with a drawing scale per robot (row ``i`` of the mask drawn at ``scales[i]``).

    Same arithmetic as ``neighbourhood.comm_maps_from_positions`` (cell = round(rel * size / (2 scale)
    + size / 2 - 1/2), value = rel / scale summed per cell), with ``scale`` depending on the robot.
    """
    num_robots = positions.shape[0]
    rel = relative_positions(positions).to(torch.float32)
    scale = scales.to(torch.float32).view(num_robots, 1, 1)
    cells = torch.round(rel * map_size / (scale * 2.0) + (map_size / 2.0 - 0.5)).long()
    values = rel / scale
    maps = torch.zeros((num_robots, 2, map_size, map_size), dtype=torch.float32)
    robot, neighbour = neighbour_mask.nonzero(as_tuple=True)
    cell_x, cell_y = cells[robot, neighbour, 0], cells[robot, neighbour, 1]
    if len(robot) and (cell_x.min() < 0 or cell_y.min() < 0 or cell_x.max() >= map_size or cell_y.max() >= map_size):
        raise ValueError("a neighbour falls outside its robot's communication map")
    for channel in (0, 1):
        maps[:, channel].index_put_((robot, cell_x, cell_y), values[robot, neighbour, channel], accumulate=True)
    return maps


def reception_mask(positions: torch.Tensor, ranges: torch.Tensor) -> torch.Tensor:
    """``mask[i, j]`` true when robot ``i`` hears robot ``j``: ``|p_j - p_i| < r_i`` (no self-loops)."""
    distances = torch.cdist(positions, positions)
    mask = distances < ranges.to(positions.dtype).view(-1, 1)
    mask.fill_diagonal_(False)
    return mask


def reception_edge_index(mask: torch.Tensor) -> torch.Tensor:
    """Directed PyG ``edge_index`` with one edge ``j -> i`` (source ``j``, target ``i``) per true ``mask[i, j]``."""
    receiver, sender = mask.nonzero(as_tuple=True)
    return torch.stack([sender, receiver])


class HeterogeneousPolicy(LPACPolicy):
    """The published policy with a communication range per robot (reception semantics)."""

    def __init__(self, paths, device, ranges: torch.Tensor, map_mode: str = "asis", state_dict=None, variant: str = "V0"):
        super().__init__(paths, device, state_dict, variant)
        if map_mode not in ("asis", "fixed256"):
            raise ValueError("map_mode must be 'asis' or 'fixed256'")
        self.ranges = ranges.to(torch.float64)
        self.map_mode = map_mode

    def build_graph(self, env, params) -> Data:
        data = super().build_graph(env, params)  # local and obstacle maps; the rest is rebuilt below
        positions = robot_positions_double(env)
        heard = reception_mask(positions, self.ranges)
        data.edge_index = reception_edge_index(heard).to(self.device)
        if self.map_mode == "asis":
            map_mask, scales = heard, self.ranges
        else:
            scales = torch.full_like(self.ranges, FIXED_SCALE)
            map_mask = reception_mask(positions, torch.minimum(self.ranges, scales))
        if uses_multiscale_maps(self.variant):  # pair k: robots within min(r_i, scale_k) at scale_k, per robot
            pairs = [comm_maps_per_robot(positions, reception_mask(positions, torch.minimum(self.ranges, torch.full_like(self.ranges, s))),
                                         torch.full_like(self.ranges, s), CNN_MAP_SIZE).to(self.device) for s in MULTISCALE_SCALES_M]
            data.x = torch.cat([data.x[:, :1], *pairs, data.x[:, 3:4]], dim=1)
        elif uses_comm_map(self.variant):  # map-less variants (V1-V4) only see the graph
            data.x[:, COMM_CHANNELS] = comm_maps_per_robot(positions, map_mask, scales, CNN_MAP_SIZE).to(self.device)
        if uses_range_input(self.variant):
            data.ranges = self.ranges.to(torch.float32).to(self.device)
        return data


def subset_features(env, positions: np.ndarray, is_long: np.ndarray, world_size: float) -> dict:
    """Where the long-range robots stand at the start of the episode.

    * ``long_center_dist``: mean distance of the long-range robots to the world centre, in units of
      half the world size (0 = centre, 1 = mid-edge);
    * ``long_degree128``: mean number of other robots within 128 m of a long-range robot (how
      clustered they start);
    * ``long_idf_share``: share of the importance field (``env.GetWorldMap()``) whose nearest robot
      is a long-range one, i.e. the Voronoi share of the subset at t = 0;
    * ``long_spread``: mean pairwise distance between long-range robots over the same for all robots.
    """
    if not is_long.any() or is_long.all():
        return {"long_center_dist": float("nan"), "long_degree128": float("nan"), "long_idf_share": float("nan"),
                "long_spread": float("nan")}
    p = np.asarray(positions, dtype=np.float64)
    centre = np.array([world_size / 2.0, world_size / 2.0])
    dist = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
    np.fill_diagonal(dist, np.inf)
    world = np.asarray(env.GetWorldMap(), dtype=np.float64)
    ys, xs = np.mgrid[0:world.shape[0], 0:world.shape[1]]
    cells = np.stack([xs.ravel() + 0.5, ys.ravel() + 0.5], axis=1)  # cell centres in metres (resolution 1 m)
    nearest = np.argmin(np.linalg.norm(cells[:, None, :] - p[None, :, :], axis=-1), axis=1)
    mass = world.ravel()
    share = float(mass[is_long[nearest]].sum() / max(mass.sum(), 1e-9))
    pl = p[is_long]
    spread_long = np.mean(np.linalg.norm(pl[:, None] - pl[None, :], axis=-1)) if len(pl) > 1 else float("nan")
    spread_all = np.mean(np.linalg.norm(p[:, None] - p[None, :], axis=-1))
    return {"long_center_dist": float(np.linalg.norm(pl - centre, axis=1).mean() / (world_size / 2.0)),
            "long_degree128": float((dist[is_long] < 128.0).sum(1).mean()),
            "long_idf_share": share,
            "long_spread": float(spread_long / spread_all)}


def long_range_robots(env_index: int, draw: int, fraction: float, num_robots: int = NUM_ROBOTS) -> np.ndarray:
    """Indices of the robots given the long range: ``round(fraction * N)`` of them, fixed seed per (env, draw)."""
    num_long = int(round(fraction * num_robots))
    if num_long in (0, num_robots):
        return np.arange(num_long)
    rng = np.random.default_rng(100 * env_index + draw)
    return np.sort(rng.choice(num_robots, size=num_long, replace=False))


def run_episode(policy: HeterogeneousPolicy, reference: HeterogeneousPolicy | None, params, env, num_steps: int,
                probe_every: int, is_long: np.ndarray) -> dict:
    """Mixed-team episode with the frozen-state probe against ``reference`` (every robot at the short range)."""
    initial_cost = env.GetObjectiveValue()
    num_robots = env.GetNumRobots()
    costs = np.ones(num_steps)
    degree_sum = degree_sq_sum = 0.0
    probe_short, probe_long, probe_short_abs, probe_count = 0.0, 0.0, 0.0, 0
    converged_step = None
    steps_run = 0
    for step in range(1, num_steps):
        data = policy.build_graph(env, params)
        stats = degree_stats(data.edge_index, num_robots)
        degree_sum += stats.mean
        degree_sq_sum += stats.std**2 + stats.mean**2
        actions, converged, _ = policy.act(data)
        if reference is not None and probe_every and step % probe_every == 0 and is_long.any() and (~is_long).any():
            ref_actions, _, _ = reference.act(reference.build_graph(env, params))
            a, b = np.asarray(actions), np.asarray(ref_actions)
            diff = np.linalg.norm(a - b, axis=1)
            norm = np.linalg.norm(b, axis=1)
            probe_short += float(diff[~is_long].mean() / max(norm[~is_long].mean(), 1e-9))
            probe_short_abs += float(diff[~is_long].mean())
            probe_long += float(diff[is_long].mean() / max(norm[is_long].mean(), 1e-9))
            probe_count += 1
        env.StepActions(actions)
        costs[step] = env.GetObjectiveValue() / initial_cost
        steps_run += 1
        if converged:
            costs[step:] = costs[step]
            converged_step = step
            break
    degree_mean = degree_sum / steps_run
    return {
        "cost_mean": float(costs.mean()), "cost_final": float(costs[-1]),
        "degree_mean": degree_mean, "degree_std": math.sqrt(max(degree_sq_sum / steps_run - degree_mean**2, 0.0)),
        "probe_short": probe_short / probe_count if probe_count else float("nan"),
        "probe_long": probe_long / probe_count if probe_count else float("nan"),
        "probe_short_abs": probe_short_abs / probe_count if probe_count else float("nan"),
        "probe_count": probe_count, "converged_step": converged_step,
    }


def done_keys(path: Path) -> set[tuple]:
    if not path.is_file():
        return set()
    with path.open(newline="") as handle:
        return {(int(r["env"]), float(r["fraction"]), int(r["draw"]), r["map"]) for r in csv.DictReader(handle)}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir")
    parser.add_argument("--baseline", required=True, help="results.csv holding the clairvoyant rows of the environments")
    parser.add_argument("--out-dir", default="results_hetero")
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--draws", type=int, nargs="+", default=[0], help="draw indices of the long-range subset")
    parser.add_argument("--maps", nargs="+", default=["asis", "fixed256"], choices=["asis", "fixed256"])
    parser.add_argument("--long-range", type=float, default=1024.0)
    parser.add_argument("--short-range", type=float, default=128.0)
    parser.add_argument("--model", default=None, help="state dict of another LPAC-K3 checkpoint (default: published)")
    parser.add_argument("--model-tag", default="published")
    parser.add_argument("--variant", default="V0", choices=("V0", "V0R", "V4", "V5", "V6"), help="architecture of --model")
    parser.add_argument("--num-envs", type=int, default=11)
    parser.add_argument("--env-start", type=int, default=0)
    parser.add_argument("--env-end", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=600)
    parser.add_argument("--probe-every", type=int, default=50, help="steps between frozen-state probes (0 = none)")
    args = parser.parse_args(argv)

    paths = resolve_data_dir(args.data_dir)
    device = torch.device("cpu")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = out / "results_hetero.csv"
    done = done_keys(results)
    clairvoyant = load_clairvoyant(Path(args.baseline), NUM_ROBOTS)
    env_end = args.num_envs if args.env_end is None else min(args.env_end, args.num_envs)
    started = time.time()
    for env_index in range(args.env_start, env_end):
        for fraction in args.fractions:
            for draw in args.draws:
                if fraction in (0.0, 1.0) and draw != args.draws[0]:
                    continue  # no subset to draw
                for map_mode in args.maps:
                    key = (env_index, float(fraction), draw, map_mode)
                    if key in done:
                        continue
                    long_idx = long_range_robots(env_index, draw, fraction)
                    is_long = np.zeros(NUM_ROBOTS, dtype=bool)
                    is_long[long_idx] = True
                    ranges = torch.full((NUM_ROBOTS,), args.short_range, dtype=torch.float64)
                    ranges[is_long] = args.long_range
                    policy = HeterogeneousPolicy(paths, device, ranges, map_mode, args.model, args.variant)
                    reference = HeterogeneousPolicy(paths, device, torch.full((NUM_ROBOTS,), args.short_range, dtype=torch.float64),
                                                    map_mode, args.model, args.variant) if args.probe_every else None
                    params = make_parameters(paths.env_params, args.long_range, NUM_ROBOTS)
                    positions = load_initial_positions(paths.pos_file(env_index), NUM_ROBOTS)
                    env = make_environment(params, paths.env_file(env_index), positions)
                    clair_mean, clair_final = clairvoyant[env_index]
                    tic = time.time()
                    features = subset_features(env, positions, is_long, params.pWorldMapSize)
                    rec = run_episode(policy, reference, params, env, args.num_steps, args.probe_every, is_long)
                    rec.update(features)
                    row = {"env": env_index, "fraction": fraction, "draw": draw, "map": map_mode, "model": args.model_tag,
                           "num_long": int(is_long.sum()), "long_range": args.long_range, "short_range": args.short_range,
                           "ratio_mean": rec["cost_mean"] / clair_mean, "ratio_final": rec["cost_final"] / clair_final,
                           "clairvoyant_cost_mean": clair_mean, "wall_s": round(time.time() - tic, 1), **rec}
                    new = not results.is_file()
                    with results.open("a", newline="") as handle:
                        writer = csv.DictWriter(handle, fieldnames=FIELDS)
                        if new:
                            writer.writeheader()
                        writer.writerow({k: row.get(k) for k in FIELDS})
                    done.add(key)
                    print(f"[{time.time() - started:6.0f}s] env={env_index:2d} f={fraction:.2f} draw={draw} {map_mode:8s} "
                          f"n_long={row['num_long']:2d} ratio={row['ratio_mean']:.3f} deg={row['degree_mean']:5.2f} "
                          f"probe_short={row['probe_short']:.3f} probe_long={row['probe_long']:.3f} "
                          f"idf_share={row['long_idf_share']:.2f} centre={row['long_center_dist']:.2f} ({row['wall_s']:.0f}s)",
                          flush=True)


if __name__ == "__main__":
    main()
