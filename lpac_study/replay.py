"""Episode replays for the project site: where the robots go, step by step, under each controller.

    python -m lpac_study.replay --v4-model ~/lpac/train/farmshare_models/V4mixed_s0/model_best_state_dict.pt \
        --out site/data/replays.json

One released environment is replayed under ten controllers, with the loops of the evaluation
scripts reproduced step for step (``episode.py`` for the homogeneous runs, ``heterogeneous.py`` for
the mixed teams), so that the mean cost of every replay equals the value already published in the
result CSVs. That equality is checked before anything is written: a replay whose mean differs from
its CSV by more than ``--tolerance`` aborts the run.

The environment is the released one whose published-model cost at 1024 m is closest to the median
over the 11 (a typical collapse, not the worst one). Every fourth step is kept: robot positions
rounded to 0.1 m, the cost normalized by the clairvoyant controller's cost at the same step, and,
for robot 0 at three steps, its 2 x 32 x 32 neighbour map drawn three ways (C0 at the run's range,
C3 and K256, see ``map_stats.py``). The importance field is stored once, downsampled to 128 x 128.

Layout conventions, which the site relies on: ``positions[frame]`` is ``[x0, y0, x1, y1, ...]`` in
metres; ``idf.data[y * 128 + x]``; a map is flattened as ``[channel * 1024 + y * 32 + x]``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .episode import CVT_ALGORITHMS
from .factorial import FactorialPolicy
from .heterogeneous import HeterogeneousPolicy, long_range_robots
from .lpac_policy import LPACPolicy
from .neighbourhood import comm_maps_from_positions, in_range_mask, robot_positions_double
from .paths import resolve_data_dir
from .simulation import load_initial_positions, make_environment, make_parameters

NUM_ROBOTS, NUM_STEPS, STRIDE = 32, 600, 4
MAP_STEPS = (20, 200, 596)
MAP_ROBOT, MAP_SIZE, FIXED_SCALE = 0, 32, 256.0
IDF_SIZE = 128
LONG, SHORT, FRACTION, DRAW = 1024.0, 128.0, 0.5, 0
ROOT = Path(__file__).resolve().parents[1]


def choose_environment() -> tuple[int, float, float]:
    """The released environment whose published cost ratio at 1024 m is closest to the median."""
    frame = pd.read_csv(ROOT / "results/results.csv")
    frame = frame[(frame.controller == "lpac") & (frame.num_robots == NUM_ROBOTS) & (frame.sweep == "range") & (frame.comm_range == LONG)]
    median = float(frame.ratio_mean.median())
    row = frame.iloc[(frame.ratio_mean - median).abs().argsort().iloc[0]]
    return int(row.env), float(row.ratio_mean), median


def record(env, params, num_steps: int, act) -> tuple[np.ndarray, np.ndarray]:
    """Run ``act`` and keep the cost and positions of every step.

    ``act`` returns the actions and a callable giving the convergence flag, read after the step
    exactly where the evaluation loops read it.

    Same loop as ``episode.run_lpac_episode``: ``costs[0] = 1``, the cost is frozen once the
    controller reports convergence. Positions are frozen with it, the robots having stopped.
    """
    initial = env.GetObjectiveValue()
    costs = np.ones(num_steps)
    positions = np.zeros((num_steps, NUM_ROBOTS, 2))
    positions[0] = robot_positions_double(env).numpy()
    for step in range(1, num_steps):
        actions, converged = act()
        if env.StepActions(actions):
            raise RuntimeError("the simulator reported an error while stepping")
        costs[step] = env.GetObjectiveValue() / initial
        positions[step] = robot_positions_double(env).numpy()
        if converged():
            costs[step:] = costs[step]
            positions[step:] = positions[step]
            break
    return costs, positions


def fresh_env(paths, env_index: int, comm_range: float):
    params = make_parameters(paths.env_params, comm_range, NUM_ROBOTS)
    env = make_environment(params, paths.env_file(env_index), load_initial_positions(paths.pos_file(env_index), NUM_ROBOTS))
    return params, env


def run_cvt(paths, env_index, name, comm_range=LONG):
    params, env = fresh_env(paths, env_index, comm_range)
    algorithm = CVT_ALGORITHMS[name](params, env)

    def act():
        algorithm.ComputeActions()
        return algorithm.GetActions(), algorithm.IsConverged  # read after the step, as run_cvt_episode does

    return record(env, params, NUM_STEPS, act), env


def run_policy(paths, env_index, policy, comm_range):
    params, env = fresh_env(paths, env_index, comm_range)

    def act():
        actions, converged, _ = policy.act(policy.build_graph(env, params))
        return actions, lambda: converged

    return record(env, params, NUM_STEPS, act), env


def robot_maps(positions: np.ndarray, comm_range: float) -> dict:
    """Robot 0's neighbour map at the chosen steps, drawn three ways; flattened [channel, y, x]."""
    out = {"robot": MAP_ROBOT, "steps": list(MAP_STEPS), "C0": [], "C3": [], "K256": []}
    for step in MAP_STEPS:
        p = torch.tensor(positions[step])
        in_range = in_range_mask(p, comm_range)
        levels = {"C0": comm_maps_from_positions(p, in_range, comm_range, MAP_SIZE),
                  "C3": comm_maps_from_positions(p, in_range_mask(p, min(comm_range, FIXED_SCALE)), FIXED_SCALE, MAP_SIZE),
                  "K256": comm_maps_from_positions(p, in_range, FIXED_SCALE, MAP_SIZE, clamp=True)}
        for level, maps in levels.items():
            flat = maps[MAP_ROBOT].permute(0, 2, 1).reshape(-1)  # [channel, x, y] -> [channel, y, x]
            out[level].append([round(float(v), 4) for v in flat])
    return out


def importance_field(env, clairvoyant_positions: np.ndarray) -> dict:
    """The importance density downsampled to 128 x 128, stored as data[y * 128 + x].

    The orientation of the library's array is settled by the data rather than assumed: the
    clairvoyant controller ends on the density, so the reading under which its final positions sit
    on more density is the right one.
    """
    world = np.asarray(env.GetWorldMap(), dtype=np.float64)
    size = world.shape[0]
    final = np.clip(clairvoyant_positions[-1].astype(int), 0, size - 1)
    as_xy = world[final[:, 0], final[:, 1]].mean()  # world[x, y]
    as_yx = world[final[:, 1], final[:, 0]].mean()  # world[y, x]
    by_xy = as_xy >= as_yx
    block = size // IDF_SIZE
    small = world.reshape(IDF_SIZE, block, IDF_SIZE, block).mean(axis=(1, 3))
    grid_yx = small.T if by_xy else small  # index [y, x]
    return {"size": IDF_SIZE, "data": [round(float(v), 4) for v in grid_yx.reshape(-1)],
            "orientation_check": {"density_under_final_clairvoyant_positions": {"world[x,y]": float(as_xy), "world[y,x]": float(as_yx)},
                                  "library_array_indexed_as": "world[x, y]" if by_xy else "world[y, x]"}}


def csv_value(path: str, **where) -> float:
    frame = pd.read_csv(ROOT / path)
    for column, value in where.items():
        frame = frame[frame[column] == value]
    if len(frame) != 1:
        raise LookupError(f"{path} {where}: {len(frame)} rows")
    return float(frame.ratio_mean.iloc[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--v4-model", required=True, help="state dict of V4 mixed, seed 0 (best validation)")
    parser.add_argument("--out", default="site/data/replays.json")
    parser.add_argument("--tolerance", type=float, default=1e-6, help="largest accepted |replay mean - CSV value|")
    parser.add_argument("--only", nargs="*", default=None, help="run ids to record (default: all)")
    args = parser.parse_args()
    torch.manual_seed(0)
    paths = resolve_data_dir(args.data_dir)
    device = torch.device("cpu")
    env_index, env_ratio, median = choose_environment()
    print(f"environment {env_index}: published cost ratio at 1024 m {env_ratio:.4f}, median of the 11 {median:.4f}", flush=True)

    is_long = np.zeros(NUM_ROBOTS, dtype=bool)
    is_long[long_range_robots(env_index, DRAW, FRACTION)] = True
    ranges = torch.where(torch.tensor(is_long), torch.tensor(LONG, dtype=torch.float64), torch.tensor(SHORT, dtype=torch.float64))
    sweep = dict(env=env_index, num_robots=NUM_ROBOTS, sweep="range")
    het = dict(env=env_index, fraction=FRACTION, draw=DRAW, map="asis")

    def hetero(model, variant, semantics):
        return lambda: run_policy(paths, env_index, HeterogeneousPolicy(paths, device, ranges, "asis", model, variant, semantics), LONG)

    # id, label, model, range, map level, semantics, f, runner, (csv, filters)
    plan = [
        ("clairvoyant", "Clairvoyant CVT (knows the whole field)", "clairvoyant", LONG, "none", None, None,
         lambda: run_cvt(paths, env_index, "ClairvoyantCVT"), ("results/results.csv", dict(sweep, controller="ClairvoyantCVT", comm_range=LONG))),
        ("centralized_cvt", "Centralized CVT", "centralized_cvt", LONG, "none", None, None,
         lambda: run_cvt(paths, env_index, "CentralizedCVT"), ("results/results.csv", dict(sweep, controller="CentralizedCVT", comm_range=LONG))),
        ("published_256", "LPAC as published, 256 m", "published", 256.0, "C0", None, None,
         lambda: run_policy(paths, env_index, LPACPolicy(paths, device), 256.0), ("results/results.csv", dict(sweep, controller="lpac", comm_range=256.0))),
        ("published_1024", "LPAC as published, 1024 m", "published", LONG, "C0", None, None,
         lambda: run_policy(paths, env_index, LPACPolicy(paths, device), LONG), ("results/results.csv", dict(sweep, controller="lpac", comm_range=LONG))),
        ("published_1024_C3", "LPAC as published, 1024 m, fixed 256 m map", "published", LONG, "C3", None, None,
         lambda: run_policy(paths, env_index, FactorialPolicy(paths, device, "G0", "C3", 4), LONG),
         ("results_clamp/results_11.csv", dict(env=env_index, comm_range=LONG, map="C3"))),
        ("V4mixed_1024", "V4 (bounded edges, mixed ranges), 1024 m", "V4mixed", LONG, "none", None, None,
         lambda: run_policy(paths, env_index, LPACPolicy(paths, device, args.v4_model, "V4"), LONG),
         ("results_V45/V4mixed_s0_11.csv", dict(sweep, controller="lpac", comm_range=LONG))),
        ("published_hetero_reception", "LPAC as published, half the robots at 1024 m, reception", "published", LONG, "C0", "reception", FRACTION,
         hetero(None, "V0", "reception"), ("results_hetero/results_hetero8.csv", het)),
        ("published_hetero_transmission", "LPAC as published, half the robots at 1024 m, transmission", "published", LONG, "C0", "transmission", FRACTION,
         hetero(None, "V0", "transmission"), ("results_sem/published.csv", dict(het, semantics="transmission"))),
        ("V4mixed_hetero_reception", "V4 (mixed ranges), half the robots at 1024 m, reception", "V4mixed", LONG, "none", "reception", FRACTION,
         hetero(args.v4_model, "V4", "reception"), ("results_V45/results_hetero_V4mixed.csv", het)),
        ("V4mixed_hetero_transmission", "V4 (mixed ranges), half the robots at 1024 m, transmission", "V4mixed", LONG, "none", "transmission", FRACTION,
         hetero(args.v4_model, "V4", "transmission"), ("results_sem/V4mixed_s0.csv", dict(het, semantics="transmission"))),
    ]
    if args.only:
        plan = [p for p in plan if p[0] in args.only or p[0] == "clairvoyant"]

    frames = list(range(0, NUM_STEPS, STRIDE))
    runs, clair_costs, idf, failures = [], None, None, []
    for run_id, label, model, comm_range, map_level, semantics, fraction, runner, (csv_path, where) in plan:
        (costs, positions), env = runner()
        if run_id == "clairvoyant":
            clair_costs = costs
            idf = importance_field(env, positions)
        mean_ratio = float(costs.mean() / clair_costs.mean())
        expected = csv_value(csv_path, **where)
        gap = abs(mean_ratio - expected)
        status = "ok" if gap <= args.tolerance else "MISMATCH"
        print(f"{run_id:32s} mean ratio {mean_ratio:.6f}  csv {expected:.6f}  gap {gap:.2e}  {status}", flush=True)
        if gap > args.tolerance:
            failures.append(run_id)
        map_range = float(ranges[MAP_ROBOT]) if fraction else comm_range
        runs.append({
            "id": run_id, "label": label, "model": model, "range_m": int(comm_range), "map_level": map_level,
            "semantics": semantics, "f": fraction, "draw": DRAW if fraction else None,
            "robot_ranges": [int(r) for r in ranges.tolist()] if fraction else None,
            "positions": [[round(float(v), 1) for v in positions[s].reshape(-1)] for s in frames],
            "cost": [round(float(costs[s]), 5) for s in frames],
            "cost_ratio": [round(float(costs[s] / clair_costs[s]), 4) for s in frames],
            "mean_cost_ratio": mean_ratio, "sweep_cost_ratio": expected, "sweep_source": csv_path,
            "maps_range_m": int(map_range), "maps": robot_maps(positions, map_range),
        })
    if failures:
        raise SystemExit(f"replays not written: {failures} differ from their CSV by more than {args.tolerance}")
    payload = {
        "schema": "replays/1",
        "about": "one released environment under ten controllers; every 4th of 600 steps; positions in metres rounded to 0.1; "
                 "cost = coverage cost / initial cost; cost_ratio = cost / the clairvoyant controller's cost at the same step; "
                 "mean_cost_ratio = episode mean of the cost / the clairvoyant's, over all 600 steps, equal to sweep_cost_ratio",
        "layout": {"positions": "positions[frame] = [x0, y0, x1, y1, ...]", "idf": "idf.data[y * size + x]",
                   "maps": "maps[level][k][channel * 1024 + y * 32 + x], k indexes maps.steps; channel 0 = x offset, 1 = y offset, each / scale"},
        "env": {"env_set": "released11", "env": env_index, "world_size_m": 1024, "num_robots": NUM_ROBOTS, "num_steps": NUM_STEPS,
                "selection": {"rule": "released environment whose published cost ratio at 1024 m is closest to the median of the 11",
                              "value": env_ratio, "median": median, "source": "results/results.csv"},
                "idf": idf},
        "frame_steps": frames, "runs": runs,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB), {len(runs)} runs, {len(frames)} frames each")


if __name__ == "__main__":
    main()
