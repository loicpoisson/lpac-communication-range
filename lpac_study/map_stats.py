"""Magnitude of the neighbour map under three ways of drawing it (question Q1, journal 16.1).

    python -m lpac_study.map_stats --out results_clamp/map_stats.csv

For every released environment, at its initial robot positions, and for every communication
range, the 2 x 32 x 32 neighbour maps of the 32 robots are built at three levels:

* ``C0``   as published: robots within the range, drawn at the scale of the range;
* ``C3``   fixed 256 m scale, robots beyond min(range, 256 m) dropped;
* ``K256`` fixed 256 m scale, robots beyond 256 m projected onto the 256 m circle.

One row per (level, range, environment): the largest absolute cell value over the team, the
mass (sum of absolute values) per robot, the number of nonzero cells per robot and the number of
robots drawn per map. These are the numbers quoted in the journal (at 1024 m: largest cell 1.10 /
4.29 / 1.72 for C3 / K256 / C0 as environment means, mass 4.0 / 36.1, nonzero cells 9.9 / 49.4).
No model and no simulation are involved, only the positions.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from .neighbourhood import comm_maps_from_positions, in_range_mask
from .paths import resolve_data_dir
from .simulation import load_initial_positions

RANGES = (128.0, 256.0, 512.0, 768.0, 1024.0)
SCALE, SIZE, NUM_ROBOTS = 256.0, 32, 32
FIELDS = ["level", "range_m", "env", "max_abs_cell", "mass_per_robot", "nonzero_cells_per_robot", "robots_drawn_per_map"]


def stats(level: str, positions: torch.Tensor, comm_range: float) -> dict:
    in_range = in_range_mask(positions, comm_range)
    if level == "C0":
        drawn, maps = in_range, comm_maps_from_positions(positions, in_range, comm_range, SIZE)
    elif level == "C3":
        drawn = in_range_mask(positions, min(comm_range, SCALE))
        maps = comm_maps_from_positions(positions, drawn, SCALE, SIZE)
    else:  # K256
        drawn, maps = in_range, comm_maps_from_positions(positions, in_range, SCALE, SIZE, clamp=True)
    return {"max_abs_cell": float(maps.abs().max()), "mass_per_robot": float(maps.abs().sum()) / NUM_ROBOTS,
            "nonzero_cells_per_robot": float((maps != 0).sum()) / NUM_ROBOTS,
            "robots_drawn_per_map": float(drawn.sum()) / NUM_ROBOTS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--num-envs", type=int, default=11)
    parser.add_argument("--out", default="results_clamp/map_stats.csv")
    args = parser.parse_args()
    paths = resolve_data_dir(args.data_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        for env in range(args.num_envs):
            positions = torch.tensor(load_initial_positions(paths.pos_file(env), NUM_ROBOTS))
            for comm_range in RANGES:
                for level in ("C0", "C3", "K256"):
                    writer.writerow({"level": level, "range_m": comm_range, "env": env, **stats(level, positions, comm_range)})
    print("wrote", out)


if __name__ == "__main__":
    main()
