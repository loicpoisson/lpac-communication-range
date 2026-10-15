"""Inference-time factorial experiment: which input carries the neighbourhood that breaks LPAC?

    python -m lpac_study.factorial --data-dir /path/to/lpac --baseline results/results.csv \
        --out-dir results_2x2 --ranges 1024 --cells G0C0 G0C1 G0C2 G1C0 G1C1 G1C2 --k 4

Factor GRAPH (what the GNN sees through ``edge_index``):
  G0  every edge within range (the published behaviour);
  G1  each robot keeps its k nearest robots among those in range, made undirected;
  R<m> edges only between robots within min(<m>, test range) metres: the graph analogue of
       the fixed-scale map (R256 = edges capped at 256 m), for the map-less variants.
Factor MAP (what the CNN sees through the two communication-map channels):
  C0  the library's map (all robots in range, resolution of the test range);
  C1  only the k nearest neighbours of G1 (same undirected set), test-range resolution;
  C2  only robots within min(128 m, test range), drawn with the 128 m normalization and resolution;
  C3  only robots within min(256 m, test range), drawn with the 256 m normalization and resolution;
  C4  empty map: both neighbour channels set to zero (the library's value for "no neighbour");
  S<m> map of the robots within min(<m>, test range) metres, drawn with the <m> m normalization
       and resolution (C2 = S128, C3 = S256); only robots within radio range are ever drawn.
``--model`` evaluates another LPAC-K3 checkpoint (same architecture); ``--model-tag`` labels it.
The simulator itself always runs at the test range; only the model inputs change.
Costs are normalized by the clairvoyant rows of ``--baseline`` (range-independent).

Sensitivity repeats (``--repeats N --noise-sigma 1``): the initial positions are shifted
by Gaussian noise with a fixed seed per (environment, repeat), and the clairvoyant is
re-run on the same shifted start to normalize.  ``repeat = 0`` is the nominal episode.

Outputs ``results_2x2.csv`` (one row per environment, range, cell, k, repeat) in
``--out-dir``; the script is resumable.
"""
from __future__ import annotations

import argparse
import csv
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from .episode import run_cvt_episode, run_lpac_episode
from .lpac_policy import CNN_MAP_SIZE, LPACPolicy
from .neighbourhood import (comm_maps_from_positions, in_range_mask, knn_within_range_mask, mask_to_edge_index,
                            robot_positions_double, symmetrize)
from .paths import DataPaths, resolve_data_dir
from .simulation import load_initial_positions, make_environment, make_parameters

NUM_ROBOTS = 32
COMM_CHANNELS = [1, 2]
MAP_SCALE_LEVELS = {"C2": 128.0, "C3": 256.0}  # map drawn as if the range were this value
CELLS = tuple(f"G{g}C{c}" for g in (0, 1) for c in range(5))
FIELDS = ["env", "comm_range", "model", "graph", "map", "cell", "k", "repeat", "noise_sigma",
          "cost_mean", "cost_final", "ratio_mean", "ratio_final", "clairvoyant_cost_mean",
          "degree_mean", "degree_std", "map_neighbours_mean", "converged_step", "wall_s"]
WORLD_MARGIN = 0.5  # metres kept from the world boundary when perturbing initial positions


class FactorialPolicy(LPACPolicy):
    """The published policy whose graph and communication maps are rebuilt per factor level."""

    def __init__(self, paths, device, graph: str, comm_map: str, k: int, state_dict=None, variant: str = "V0"):
        super().__init__(paths, device, state_dict, variant)
        self.graph, self.comm_map, self.k = graph, comm_map, k
        self.map_neighbour_counts: list[float] = []

    def build_graph(self, env, params) -> Data:
        data = super().build_graph(env, params)
        positions = robot_positions_double(env)
        test_range = params.pCommunicationRange
        capped = symmetrize(knn_within_range_mask(positions, test_range, self.k))
        if self.graph == "G1":
            data.edge_index = mask_to_edge_index(capped)
        elif graph_scale(self.graph) is not None:
            data.edge_index = mask_to_edge_index(in_range_mask(positions, min(graph_scale(self.graph), test_range)))
        map_mask = self._map_mask(positions, test_range, capped)
        if self.comm_map == "C4":
            data.x[:, COMM_CHANNELS] = 0.0
        elif self.comm_map != "C0":
            map_range = map_scale(self.comm_map) or test_range
            data.x[:, COMM_CHANNELS] = comm_maps_from_positions(positions, map_mask, map_range, CNN_MAP_SIZE)
        self.map_neighbour_counts.append(float(map_mask.sum(1).float().mean()))
        return data

    def _map_mask(self, positions, test_range, capped):
        """Robots drawn on each robot's communication map for the current MAP level."""
        if self.comm_map == "C1":
            return capped
        scale = map_scale(self.comm_map)
        if scale is not None:
            # A robot can only draw neighbours it can hear: within the test range, and within
            # the drawing scale so that they fit on the map (corrected on 14 October; before,
            # a scale larger than the test range drew robots out of radio range).
            return in_range_mask(positions, min(scale, test_range))
        if self.comm_map == "C4":
            return torch.zeros_like(capped)
        return in_range_mask(positions, test_range)


def map_scale(level: str):
    """Fixed drawing scale (metres) of a MAP level: C2 = 128, C3 = 256, S<metres> = any value, else None."""
    if level in MAP_SCALE_LEVELS:
        return MAP_SCALE_LEVELS[level]
    if level.startswith("S") and level[1:].isdigit():
        return float(level[1:])
    return None


def graph_scale(level: str):
    """Edge cap (metres) of a GRAPH level R<metres>, else None."""
    if level.startswith("R") and level[1:].isdigit():
        return float(level[1:])
    return None


def parse_cell(cell: str):
    """Split a cell name such as G0C3, G1S192 or R256C0 into its GRAPH and MAP levels."""
    match = re.fullmatch(r"(G[01]|R\d+)([CS]\d+)", cell)
    if match is None:
        raise ValueError("bad cell name %r: expected G<0|1> or R<metres>, followed by C<0-4> or S<metres>" % cell)
    return match.group(1), match.group(2)


def load_clairvoyant(baseline: Path, num_robots: int) -> dict[int, tuple[float, float]]:
    """Per-environment clairvoyant (cost_mean, cost_final) from an existing results.csv."""
    rows = pd.read_csv(baseline)
    rows = rows[(rows["controller"] == "ClairvoyantCVT") & (rows["num_robots"] == num_robots)]
    rows = rows.drop_duplicates("env")
    return {int(r.env): (float(r.cost_mean), float(r.cost_final)) for r in rows.itertuples()}


def done_keys(path: Path) -> set[tuple]:
    """Keys (env, range, cell, k, repeat) already present in a results file."""
    if not path.is_file():
        return set()
    with path.open(newline="") as handle:
        return {(int(r["env"]), float(r["comm_range"]), r.get("model") or "published", r["cell"], int(r["k"]),
                 int(r.get("repeat") or 0)) for r in csv.DictReader(handle)}


def append_row(path: Path, row: dict) -> None:
    new = not path.is_file()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(row)


def perturbed_positions(positions: np.ndarray, env_index: int, repeat: int, sigma: float, world_size: float) -> np.ndarray:
    """Initial positions shifted by N(0, sigma) metres with a fixed seed; repeat 0 is unchanged."""
    if repeat == 0 or sigma <= 0:
        return positions
    rng = np.random.default_rng(1000 * env_index + repeat)
    shifted = positions + rng.normal(0.0, sigma, size=positions.shape)
    return np.clip(shifted, WORLD_MARGIN, world_size - WORLD_MARGIN)


def run_cell(paths: DataPaths, env_index: int, comm_range: float, cell: str, k: int, repeat: int,
             sigma: float, num_steps: int, clairvoyant: dict, device, model=None, model_tag="published",
             variant="V0") -> dict:
    """One LPAC episode for one factorial cell on one (possibly perturbed) environment."""
    graph, comm_map = parse_cell(cell)
    policy = FactorialPolicy(paths, device, graph, comm_map, k, model, variant)
    params = make_parameters(paths.env_params, comm_range, NUM_ROBOTS)
    positions = load_initial_positions(paths.pos_file(env_index), NUM_ROBOTS)
    positions = perturbed_positions(positions, env_index, repeat, sigma, params.pWorldMapSize)
    if repeat == 0:
        clair_mean, clair_final = clairvoyant[env_index]
    else:  # the shifted start changes the reference too: re-run the clairvoyant on it
        reference = run_cvt_episode("ClairvoyantCVT", params, make_environment(params, paths.env_file(env_index), positions), num_steps)
        clair_mean, clair_final = reference.cost_mean, reference.cost_final
    env = make_environment(params, paths.env_file(env_index), positions)
    tic = time.time()
    record = run_lpac_episode(policy, params, env, num_steps, record_rank=False)
    return {
        "env": env_index, "comm_range": comm_range, "model": model_tag, "graph": graph, "map": comm_map,
        "cell": cell, "k": k,
        "repeat": repeat, "noise_sigma": sigma if repeat else 0.0,
        "cost_mean": record.cost_mean, "cost_final": record.cost_final,
        "ratio_mean": record.cost_mean / clair_mean, "ratio_final": record.cost_final / clair_final,
        "clairvoyant_cost_mean": clair_mean,
        "degree_mean": record.degree_mean, "degree_std": record.degree_std,
        "map_neighbours_mean": sum(policy.map_neighbour_counts) / len(policy.map_neighbour_counts),
        "converged_step": "" if record.converged_step is None else record.converged_step,
        "wall_s": round(time.time() - tic, 1),
    }


def run_factorial(args: argparse.Namespace) -> None:
    paths = resolve_data_dir(args.data_dir)
    device = torch.device("cpu")
    torch.manual_seed(0)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = out / "results_2x2.csv"
    done = done_keys(results)
    clairvoyant = load_clairvoyant(Path(args.baseline), NUM_ROBOTS)
    env_end = args.num_envs if args.env_end is None else min(args.env_end, args.num_envs)
    repeats = range(args.repeat_start, args.repeats + 1)
    started = time.time()
    for comm_range in args.ranges:
        for env_index in range(args.env_start, env_end):
            for cell in args.cells:
                for k in args.k:
                    graph, comm_map = parse_cell(cell)
                    if comm_map != "C1" and graph != "G1" and k != args.k[0]:
                        continue  # these cells do not depend on k: computed once
                    for repeat in repeats:
                        key = (env_index, float(comm_range), args.model_tag, cell, int(k), repeat)
                        if key in done:
                            continue
                        row = run_cell(paths, env_index, float(comm_range), cell, int(k), repeat,
                                       args.noise_sigma, args.num_steps, clairvoyant, device,
                                       args.model, args.model_tag, args.variant)
                        append_row(results, row)
                        done.add(key)
                        print(f"[{time.time() - started:6.0f}s] r={comm_range:5.0f} env={env_index:2d} "
                              f"{args.model_tag} {cell} k={k:2d} "
                              f"rep={repeat} ratio={row['ratio_mean']:.3f} deg={row['degree_mean']:5.2f} "
                              f"map_nb={row['map_neighbours_mean']:5.2f} ({row['wall_s']:.0f}s)", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir")
    parser.add_argument("--baseline", required=True, help="results.csv of the range sweep (clairvoyant normalization)")
    parser.add_argument("--out-dir", default="results_2x2")
    parser.add_argument("--ranges", type=float, nargs="+", default=[1024.0])
    parser.add_argument("--cells", nargs="+", default=list(CELLS[:3]) + list(CELLS[5:8]),
                        help="cells G<0|1><level>, level in C0..C4 or S<metres> (map drawn at that fixed scale)")
    parser.add_argument("--model", default=None, help="state dict of another LPAC-K3 checkpoint (default: published)")
    parser.add_argument("--model-tag", default="published", help="label of the checkpoint written in the CSV")
    parser.add_argument("--variant", default="V0", choices=("V0", "V1", "V2", "V3"), help="architecture of --model")
    parser.add_argument("--k", type=int, nargs="+", default=[4], help="neighbour caps for G1/C1")
    parser.add_argument("--num-envs", type=int, default=11)
    parser.add_argument("--env-start", type=int, default=0)
    parser.add_argument("--env-end", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=600)
    parser.add_argument("--repeats", type=int, default=0, help="number of perturbed repeats per episode (0 = nominal only)")
    parser.add_argument("--repeat-start", type=int, default=0, help="first repeat index to run (1 skips the nominal episode)")
    parser.add_argument("--noise-sigma", type=float, default=1.0, help="std (m) of the initial-position noise of repeats")
    return parser


def main(argv: list[str] | None = None) -> None:
    run_factorial(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
