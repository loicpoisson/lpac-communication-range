"""Rebuild the range-dependent tensors of a generated dataset at another communication range.

    python training/redraw_ranges.py --src /path/data100k --dst /path/r512/data --range 512
    python training/redraw_ranges.py --check /path/part_0 --range 512      # compare with stored tensors

Only two tensors of the authors' datasets depend on the communication range: ``comm_maps.pt``
(the two neighbour-map channels, ``CoverageEnvUtils.get_communication_maps``) and
``edge_weights.pt`` (``CoverageEnvUtils.get_weights``: exp(-d^2 / range^2), zeroed below 1/e,
i.e. beyond the range). The clairvoyant trajectories, local maps, obstacle maps and target
actions do not. So a dataset "generated at 512 m" is the 128 m dataset with those two
tensors redrawn from ``robot_positions.pt``. ``--check`` rebuilds them at the range a dataset
was generated with and reports how exactly the stored tensors are reproduced.

The rebuilt files are written to ``dst/<split>/``; every other file of the source dataset is
linked (hard link, or copied when linking fails), including the root normalization tensors.
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
from pathlib import Path

import torch

MAP_SIZE = 32
SPLITS = ("train", "val", "test")
RANGE_FILES = ("comm_maps.pt", "edge_weights.pt")


def edge_weights(positions: torch.Tensor, comm_range: float) -> torch.Tensor:
    """``CoverageEnvUtils.get_weights`` on a batch of float32 positions ``(B, N, 2)``."""
    distances = torch.cdist(positions, positions, 2)
    weights = torch.exp(-distances.square() / (comm_range * comm_range))
    weights.masked_fill_(weights < 1.0 / math.exp(1.0), 0.0)
    idx = torch.arange(positions.shape[1])
    weights[:, idx, idx] = 0.0
    return weights


def comm_maps(positions: torch.Tensor, comm_range: float, map_size: int = MAP_SIZE) -> torch.Tensor:
    """``CoverageEnvUtils.get_communication_maps`` on a batch ``(B, N, 2)``: dense ``(B, N, 2, S, S)``.

    Neighbours are the robots strictly within ``comm_range``; cell = round(rel * S / (2 range) + S/2 - 1/2),
    value = rel / range, summed per cell (sparse_coo_tensor semantics of the library).
    """
    batch, num_robots, _ = positions.shape
    rel = positions.unsqueeze(1) - positions.unsqueeze(2)  # rel[b, i, j] = p_j - p_i
    dist = torch.cdist(positions, positions, 2)
    mask = dist < comm_range
    idx = torch.arange(num_robots)
    mask[:, idx, idx] = False
    cells = torch.round(rel * map_size / (comm_range * 2.0) + (map_size / 2.0 - 0.5)).long()
    values = rel / comm_range
    maps = torch.zeros((batch, num_robots, 2, map_size, map_size), dtype=torch.float32)
    b, i, j = mask.nonzero(as_tuple=True)
    cx, cy = cells[b, i, j, 0].clamp(0, map_size - 1), cells[b, i, j, 1].clamp(0, map_size - 1)
    for channel in (0, 1):
        maps[:, :, channel].index_put_((b, i, cx, cy), values[b, i, j, channel], accumulate=True)
    return maps


def rebuild(positions: torch.Tensor, comm_range: float, chunk: int = 1000):
    """Sparse ``comm_maps`` and ``edge_weights`` for all samples, built chunk by chunk."""
    maps, weights = [], []
    for start in range(0, positions.shape[0], chunk):
        p = positions[start:start + chunk]
        maps.append(comm_maps(p, comm_range).to_sparse_coo())
        weights.append(edge_weights(p, comm_range).to_sparse_coo())
    return torch.cat(maps).coalesce(), torch.cat(weights).coalesce()


def load(path: Path) -> torch.Tensor:
    t = torch.load(path, weights_only=True)
    return t.to_dense() if t.is_sparse else t


def check(folder: Path, comm_range: float) -> None:
    positions = load(folder / "robot_positions.pt")
    maps, weights = rebuild(positions, comm_range)
    for name, mine in (("comm_maps.pt", maps), ("edge_weights.pt", weights)):
        stored = load(folder / name)
        mine = mine.to_dense()
        diff = (stored - mine).abs()
        per_sample = diff.flatten(1).max(dim=1).values
        print(f"{name}: {positions.shape[0]} samples, max |diff| {diff.max():.2e}, "
              f"samples exact {(per_sample == 0).sum().item()}, within 1e-6 {(per_sample <= 1e-6).sum().item()}, "
              f"nonzeros stored {int((stored != 0).sum())} rebuilt {int((mine != 0).sum())}")


def write(src: Path, dst: Path, comm_range: float) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.is_file():
            link(item, dst / item.name)
    for split in SPLITS:
        if not (src / split).is_dir():
            continue
        (dst / split).mkdir(exist_ok=True)
        positions = load(src / split / "robot_positions.pt")
        maps, weights = rebuild(positions, comm_range)
        torch.save(maps, dst / split / "comm_maps.pt")
        torch.save(weights, dst / split / "edge_weights.pt")
        for item in (src / split).iterdir():
            if item.name not in RANGE_FILES:
                link(item, dst / split / item.name)
        print(f"{split}: {positions.shape[0]} samples redrawn at {comm_range:.0f} m, "
              f"{weights.values().numel() / positions.shape[0] / positions.shape[1]:.2f} neighbours per robot")


def link(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", help="source dataset root (train/val/test inside)")
    parser.add_argument("--dst", help="destination dataset root")
    parser.add_argument("--check", help="folder holding robot_positions.pt, comm_maps.pt, edge_weights.pt to compare against")
    parser.add_argument("--range", type=float, required=True, help="communication range in metres")
    args = parser.parse_args()
    if args.check:
        check(Path(args.check), args.range)
    else:
        if not (args.src and args.dst):
            parser.error("--src and --dst are required unless --check is given")
        write(Path(args.src), Path(args.dst), args.range)


if __name__ == "__main__":
    main()
