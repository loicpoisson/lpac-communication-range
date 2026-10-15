"""Communication-range sweep of the published LPAC model and the CVT baselines.

Example (full study, as reported):

    python -m lpac_study.sweep --data-dir /path/to/lpac --out-dir results

The script is resumable: rows already present in ``results.csv`` are skipped.
Outputs (all in ``--out-dir``):

* ``results.csv``   one row per (environment, range, robots, controller);
* ``rank.csv``      effective rank per GNN layer for the LPAC rows;
* ``curves.csv``    sub-sampled cost curves, for time-series figures;
* ``run_info.json`` versions, device and timing of the run.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch_geometric
import coverage_control as cc

from .episode import CONTROLLERS, LPAC_NAME, EpisodeRecord, run_cvt_episode, run_lpac_episode
from .lpac_policy import LPACPolicy
from .paths import DataPaths, resolve_data_dir
from .simulation import load_initial_positions, make_environment, make_parameters

DEFAULT_RANGES = (128, 256, 512, 768, 1024)
# (communication range, number of robots) pairs keeping the expected degree
# roughly constant: the number of neighbours within range scales like N * r^2.
DEFAULT_CONSTANT_DEGREE = ("256:32", "512:11", "768:7", "1024:6")
# These two baselines never read the communication range (checked in the C++ source).
RANGE_INDEPENDENT = ("CentralizedCVT", "ClairvoyantCVT")

RESULT_FIELDS = [
    "env", "comm_range", "num_robots", "controller", "sweep",
    "cost_mean", "cost_final", "ratio_mean", "ratio_final",
    "degree_mean", "degree_std", "isolated_fraction", "converged_step",
    "wall_s", "computed",
]
RANK_FIELDS = ["env", "comm_range", "num_robots", "layer", "erank_mean", "erank_std"]
CURVE_FIELDS = ["env", "comm_range", "num_robots", "controller", "step", "cost"]


@dataclass(frozen=True)
class Configuration:
    """One point of a sweep."""

    comm_range: float
    num_robots: int
    sweep: str  # "range" or "constant_degree"


def parse_pair(value: str) -> tuple[int, int]:
    """Parse one ``range:robots`` command-line pair."""
    comm_range, num_robots = value.split(":")
    return int(comm_range), int(num_robots)


def build_configurations(args: argparse.Namespace) -> list[Configuration]:
    """Range sweep at fixed team size, then the constant-degree sweep."""
    configs = [Configuration(float(r), args.num_robots, "range") for r in args.ranges]
    if not args.no_constant_degree:
        pairs = [parse_pair(p) for p in args.constant_degree]
        configs += [Configuration(float(r), n, "constant_degree") for r, n in pairs]
    return configs


def select_device(name: str) -> torch.device:
    """``auto`` picks CUDA when available, else CPU."""
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


@dataclass(frozen=True)
class CostSummary:
    """The two numbers of a clairvoyant run needed to normalize the other controllers."""

    cost_mean: float
    cost_final: float


class ResultWriter:
    """Append-only CSV writers that make the sweep resumable."""

    def __init__(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        self.results = out_dir / "results.csv"
        self.rank = out_dir / "rank.csv"
        self.curves = out_dir / "curves.csv"
        self.done = self._read_done_keys()

    def _read_done_keys(self) -> set[tuple]:
        """Keys of rows already in results.csv; clairvoyant costs are kept for normalization."""
        self.clairvoyant: dict[tuple, CostSummary] = {}
        if not self.results.is_file():
            return set()
        done = set()
        with self.results.open(newline="") as handle:
            for r in csv.DictReader(handle):
                key = (int(r["env"]), float(r["comm_range"]), int(r["num_robots"]), r["controller"])
                done.add(key)
                if r["controller"] == "ClairvoyantCVT":
                    self.clairvoyant[(key[0], key[2])] = CostSummary(float(r["cost_mean"]), float(r["cost_final"]))
        return done

    @staticmethod
    def _append(path: Path, fields: list[str], rows: list[dict]) -> None:
        new_file = not path.is_file()
        with path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if new_file:
                writer.writeheader()
            writer.writerows(rows)

    def write_result(self, row: dict) -> None:
        self._append(self.results, RESULT_FIELDS, [row])
        self.done.add((row["env"], row["comm_range"], row["num_robots"], row["controller"]))
        if row["controller"] == "ClairvoyantCVT":
            self.clairvoyant[(row["env"], row["num_robots"])] = CostSummary(row["cost_mean"], row["cost_final"])

    def write_rank(self, rows: list[dict]) -> None:
        if rows:
            self._append(self.rank, RANK_FIELDS, rows)

    def write_curve(self, rows: list[dict]) -> None:
        self._append(self.curves, CURVE_FIELDS, rows)


def evaluate_controller(
    controller: str,
    config: Configuration,
    env_index: int,
    paths: DataPaths,
    policy: LPACPolicy | None,
    num_steps: int,
    record_rank: bool,
) -> EpisodeRecord:
    """Build a fresh environment and run one controller on it."""
    params = make_parameters(paths.env_params, config.comm_range, config.num_robots)
    positions = load_initial_positions(paths.pos_file(env_index), config.num_robots)
    env = make_environment(params, paths.env_file(env_index), positions)
    if controller == LPAC_NAME:
        return run_lpac_episode(policy, params, env, num_steps, record_rank)
    return run_cvt_episode(controller, params, env, num_steps)


def result_row(controller, config, env_index, record, clairvoyant, wall_s, computed) -> dict:
    """One line of ``results.csv``; ratios are relative to the clairvoyant run."""
    return {
        "env": env_index, "comm_range": config.comm_range, "num_robots": config.num_robots,
        "controller": controller, "sweep": config.sweep,
        "cost_mean": record.cost_mean, "cost_final": record.cost_final,
        "ratio_mean": record.cost_mean / clairvoyant.cost_mean,
        "ratio_final": record.cost_final / clairvoyant.cost_final,
        "degree_mean": record.degree_mean, "degree_std": record.degree_std,
        "isolated_fraction": record.isolated_fraction,
        "converged_step": "" if record.converged_step is None else record.converged_step,
        "wall_s": round(wall_s, 2), "computed": computed,
    }


def rank_rows(config, env_index, record) -> list[dict]:
    """Effective rank per GNN layer, averaged over the steps of the episode."""
    if record.layer_eranks is None:
        return []
    return [
        {"env": env_index, "comm_range": config.comm_range, "num_robots": config.num_robots,
         "layer": layer, "erank_mean": float(mean), "erank_std": float(std)}
        for layer, (mean, std) in enumerate(zip(record.layer_eranks, record.layer_eranks_std))
    ]


def curve_rows(controller, config, env_index, record, stride) -> list[dict]:
    """Cost curve sub-sampled every ``stride`` steps (the last step is always kept)."""
    steps = list(range(0, len(record.costs), stride))
    if steps[-1] != len(record.costs) - 1:
        steps.append(len(record.costs) - 1)
    return [
        {"env": env_index, "comm_range": config.comm_range, "num_robots": config.num_robots,
         "controller": controller, "step": s, "cost": float(record.costs[s])}
        for s in steps
    ]


def run_sweep(args: argparse.Namespace) -> None:
    """Main loop: configurations x environments x controllers, written as they finish."""
    paths = resolve_data_dir(args.data_dir)
    device = select_device(args.device)
    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    writer = ResultWriter(out_dir)
    policy = LPACPolicy(paths, device, args.model, args.variant) if LPAC_NAME in args.controllers else None
    num_envs = min(args.num_envs, paths.count_environments())
    env_indices = range(args.env_start, num_envs if args.env_end is None else min(args.env_end, num_envs))
    cache: dict[tuple, EpisodeRecord] = {}  # (env, robots, controller) of range-independent CVTs
    started = time.time()
    print(f"device={device} envs={num_envs} steps={args.num_steps} out={out_dir}", flush=True)

    # Clairvoyant first: every other controller is normalized by it.
    order = sorted(args.controllers, key=lambda c: c != "ClairvoyantCVT")
    for config in build_configurations(args):
        for env_index in env_indices:
            for controller in order:
                key = (env_index, config.comm_range, config.num_robots, controller)
                if key in writer.done:
                    continue
                cache_key = (env_index, config.num_robots, controller)
                reuse = args.reuse_range_independent and controller in RANGE_INDEPENDENT
                tic = time.time()
                if reuse and cache_key in cache:
                    record, computed = cache[cache_key], False
                else:
                    record = evaluate_controller(
                        controller, config, env_index, paths, policy, args.num_steps, not args.no_rank
                    )
                    cache[cache_key] = record
                    computed = True
                wall = time.time() - tic
                clairvoyant = cache.get((env_index, config.num_robots, "ClairvoyantCVT"))                     or writer.clairvoyant.get((env_index, config.num_robots))
                if clairvoyant is None:
                    raise RuntimeError("ClairvoyantCVT must be evaluated first to normalize costs")
                writer.write_result(result_row(controller, config, env_index, record, clairvoyant, wall, computed))
                writer.write_curve(curve_rows(controller, config, env_index, record, args.curve_stride))
                if controller == LPAC_NAME:
                    writer.write_rank(rank_rows(config, env_index, record))
                print(
                    f"[{time.time() - started:7.0f}s] r={config.comm_range:6.0f} N={config.num_robots:2d} "
                    f"env={env_index:2d} {controller:16s} cost_mean={record.cost_mean:.4f} "
                    f"ratio={record.cost_mean / clairvoyant.cost_mean:.3f} deg={record.degree_mean:5.2f} "
                    f"({wall:5.1f}s{'' if computed else ', reused'})",
                    flush=True,
                )
    write_run_info(out_dir, args, device, time.time() - started)


def write_run_info(out_dir: Path, args: argparse.Namespace, device: torch.device, elapsed: float) -> None:
    """Record software versions and timing next to the results."""
    info = {
        "args": vars(args), "device": str(device), "elapsed_s": round(elapsed, 1), "model": args.model,
        "variant": args.variant,
        "python": sys.version.split()[0], "platform": platform.platform(),
        "torch": torch.__version__, "torch_geometric": torch_geometric.__version__,
        "coverage_control": getattr(cc, "__version__", "unknown"),
    }
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", help="extracted 'lpac' folder (or set $LPAC_DATA_DIR)")
    parser.add_argument("--out-dir", default="results", help="where CSV outputs are written")
    parser.add_argument("--ranges", type=float, nargs="*", default=DEFAULT_RANGES,
                        help="communication ranges in metres for the range sweep (empty list skips it)")
    parser.add_argument("--num-robots", type=int, default=32, help="team size of the range sweep")
    parser.add_argument("--constant-degree", nargs="+", default=DEFAULT_CONSTANT_DEGREE,
                        metavar="RANGE:ROBOTS", help="(range, robots) pairs of the constant-degree sweep")
    parser.add_argument("--no-constant-degree", action="store_true", help="skip the constant-degree sweep")
    parser.add_argument("--num-envs", type=int, default=11, help="number of release environments to use")
    parser.add_argument("--num-steps", type=int, default=600, help="episode length (NumSteps of eval.toml)")
    parser.add_argument("--env-start", type=int, default=0, help="first environment index (to split work)")
    parser.add_argument("--env-end", type=int, default=None, help="one past the last environment index")
    parser.add_argument("--controllers", nargs="+", default=list(CONTROLLERS), choices=CONTROLLERS)
    parser.add_argument("--reuse-range-independent", action="store_true",
                        help="run Centralized/Clairvoyant CVT once per (env, robots) instead of once per range")
    parser.add_argument("--no-rank", action="store_true", help="skip the effective-rank measurement")
    parser.add_argument("--curve-stride", type=int, default=10, help="keep every k-th step of the cost curves")
    parser.add_argument("--device", default="auto", help="auto, cpu or cuda")
    parser.add_argument("--seed", type=int, default=0, help="torch seed (the pipeline is deterministic anyway)")
    parser.add_argument("--model", default=None, help="state dict of another LPAC-K3 checkpoint (default: the published one)")
    parser.add_argument("--variant", default="V0", choices=("V0", "V1", "V2", "V3"), help="architecture of --model (lpac_study.variants)")
    return parser


def main(argv: list[str] | None = None) -> None:
    run_sweep(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
