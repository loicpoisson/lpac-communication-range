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


def edge_weights(positions: torch.Tensor, comm_range) -> torch.Tensor:
    """``CoverageEnvUtils.get_weights`` on a batch of float32 positions ``(B, N, 2)`` (range: scalar or ``(B, 1, 1)``)."""
    distances = torch.cdist(positions, positions, 2)
    weights = torch.exp(-distances.square() / (comm_range * comm_range))
    weights.masked_fill_(weights < 1.0 / math.exp(1.0), 0.0)
    idx = torch.arange(positions.shape[1])
    weights[:, idx, idx] = 0.0
    return weights


def comm_maps(positions: torch.Tensor, comm_range: float, map_size: int = MAP_SIZE,
              scale: float | None = None) -> torch.Tensor:
    """``CoverageEnvUtils.get_communication_maps`` on a batch ``(B, N, 2)``: dense ``(B, N, 2, S, S)``.

    Neighbours are the robots strictly within ``comm_range``; cell = round(rel * S / (2 scale) + S/2 - 1/2),
    value = rel / scale, summed per cell (sparse_coo_tensor semantics of the library). ``scale`` is the
    drawing scale (default: the range itself). With a fixed ``scale`` the neighbours drawn are those
    within ``min(comm_range, scale)``: the fixed-scale map of the journal (level C3 when scale = 256 m).
    """
    batch, num_robots, _ = positions.shape
    comm_range = torch.as_tensor(comm_range, dtype=torch.float32).reshape(-1)  # one value, or one per sample
    draw_range = comm_range if scale is None else torch.clamp(comm_range, max=scale)
    if scale is None:
        if comm_range.numel() != 1:
            raise ValueError("a per-sample range needs a fixed drawing scale (--fixed-scale or --multiscale)")
        scale = float(comm_range)
    rel = positions.unsqueeze(1) - positions.unsqueeze(2)  # rel[b, i, j] = p_j - p_i
    dist = torch.cdist(positions, positions, 2)
    mask = dist < draw_range.view(-1, 1, 1)
    idx = torch.arange(num_robots)
    mask[:, idx, idx] = False
    cells = torch.round(rel * map_size / (scale * 2.0) + (map_size / 2.0 - 0.5)).long()
    values = rel / scale
    maps = torch.zeros((batch, num_robots, 2, map_size, map_size), dtype=torch.float32)
    b, i, j = mask.nonzero(as_tuple=True)
    cx, cy = cells[b, i, j, 0].clamp(0, map_size - 1), cells[b, i, j, 1].clamp(0, map_size - 1)
    for channel in (0, 1):
        maps[:, :, channel].index_put_((b, i, cx, cy), values[b, i, j, channel], accumulate=True)
    return maps


HETERO_RANGES = (128.0, 256.0, 512.0, 1024.0)


def hetero_tensors(positions: torch.Tensor, ranges: torch.Tensor, map_size: int = MAP_SIZE):
    """Per-robot ranges (reception semantics): dense comm maps ``(B, N, 2, S, S)`` and edge weights ``(B, N, N)``.

    Robot ``i`` hears the robots within its own range ``r_i``; its map is drawn at the scale ``r_i``; the
    weight matrix holds ``W[j, i] = exp(-d^2 / r_i^2)`` for ``d < r_i`` (source ``j``, target ``i``, the
    orientation of ``edge_index = W.to_sparse().indices()`` in the library), zero otherwise.
    """
    batch, num_robots, _ = positions.shape
    rel = positions.unsqueeze(1) - positions.unsqueeze(2)  # rel[b, i, j] = p_j - p_i
    dist = torch.cdist(positions, positions, 2)
    r_i = ranges.view(batch, num_robots, 1)
    mask = dist < r_i
    idx = torch.arange(num_robots)
    mask[:, idx, idx] = False
    cells = torch.round(rel * map_size / (r_i.unsqueeze(-1) * 2.0) + (map_size / 2.0 - 0.5)).long()
    values = rel / r_i.unsqueeze(-1)
    maps = torch.zeros((batch, num_robots, 2, map_size, map_size), dtype=torch.float32)
    b, i, j = mask.nonzero(as_tuple=True)
    cx, cy = cells[b, i, j, 0].clamp(0, map_size - 1), cells[b, i, j, 1].clamp(0, map_size - 1)
    for channel in (0, 1):
        maps[:, :, channel].index_put_((b, i, cx, cy), values[b, i, j, channel], accumulate=True)
    weights_receiver = torch.exp(-dist.square() / r_i.square())  # indexed [b, i(receiver), j(sender)]
    weights_receiver = weights_receiver.masked_fill(~mask, 0.0)
    weights = weights_receiver.transpose(1, 2).contiguous()  # [b, j(source), i(target)]
    return maps, weights


def draw_ranges(num_samples: int, num_robots: int, seed: int = 0) -> torch.Tensor:
    """One range per robot and sample, uniform over HETERO_RANGES, reproducible."""
    gen = torch.Generator().manual_seed(seed)
    choice = torch.randint(len(HETERO_RANGES), (num_samples, num_robots), generator=gen)
    return torch.tensor(HETERO_RANGES)[choice]


def rebuild_hetero(positions: torch.Tensor, ranges: torch.Tensor, chunk: int = 1000):
    maps, weights = [], []
    for start in range(0, positions.shape[0], chunk):
        m, w = hetero_tensors(positions[start:start + chunk], ranges[start:start + chunk])
        maps.append(m.to_sparse_coo())
        weights.append(w.to_sparse_coo())
    return torch.cat(maps).coalesce(), torch.cat(weights).coalesce()


MULTISCALE = (256.0, 512.0, 1024.0)


def infer_ranges(positions: torch.Tensor, weights: torch.Tensor, candidates=(128.0, 256.0, 512.0, 1024.0),
                 max_mismatch: int = 4) -> torch.Tensor:
    """The range each sample was generated at, recovered from its stored edge weights.

    An edge weight is nonzero exactly when the distance is below the range, so the candidate
    whose in-range mask equals the nonzero pattern is the sample's range (a merged dataset
    mixing several ranges keeps no other record of it).
    """
    dist = torch.cdist(positions, positions, 2)
    nonzero = weights > 0
    idx = torch.arange(positions.shape[1])
    mismatches = []
    for r in candidates:
        mask = dist < r
        mask[:, idx, idx] = False
        mismatches.append((mask != nonzero).flatten(1).sum(1))
    mismatches = torch.stack(mismatches, dim=1)  # (samples, candidates)
    best = mismatches.argmin(dim=1)
    worst = int(mismatches.gather(1, best.view(-1, 1)).max())
    if worst > max_mismatch:
        raise ValueError(f"a sample differs from its closest candidate range in {worst} pairs (> {max_mismatch}); candidates {candidates}")
    # a pair at d ~ r can land on either side of the float32 threshold in the stored weights: a few are tolerated
    return torch.tensor(candidates)[best]


def rebuild(positions: torch.Tensor, comm_range, chunk: int = 1000, scale: float | None = None,
            multiscale: bool = False, stored_weights: torch.Tensor | None = None):
    """Sparse ``comm_maps`` and ``edge_weights`` for all samples, built chunk by chunk.

    ``comm_range`` is one value or one per sample; ``multiscale``: six channels, one pair per scale
    of MULTISCALE (robots within min(range, scale) drawn at that scale), the V5 input. The edge
    weights keep the range (``stored_weights`` are reused unchanged when given).
    """
    maps, weights = [], []
    ranges = torch.as_tensor(comm_range, dtype=torch.float32).reshape(-1)
    for start in range(0, positions.shape[0], chunk):
        p = positions[start:start + chunk]
        r = ranges if ranges.numel() == 1 else ranges[start:start + chunk]
        if multiscale:
            maps.append(torch.cat([comm_maps(p, r, scale=s) for s in MULTISCALE], dim=2).to_sparse_coo())
        else:
            maps.append(comm_maps(p, r, scale=scale).to_sparse_coo())
        if stored_weights is None:
            weights.append(edge_weights(p, float(r) if r.numel() == 1 else r.view(-1, 1, 1)).to_sparse_coo())
    if stored_weights is not None:
        return torch.cat(maps).coalesce(), stored_weights
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


def write_hetero(src: Path, dst: Path, seed: int = 0) -> None:
    """Dataset with a range per robot drawn in HETERO_RANGES; adds ``robot_ranges.pt`` to each split."""
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.is_file():
            link(item, dst / item.name)
    for split_index, split in enumerate(s for s in SPLITS if (src / s).is_dir()):
        (dst / split).mkdir(exist_ok=True)
        positions = load(src / split / "robot_positions.pt")
        ranges = draw_ranges(positions.shape[0], positions.shape[1], seed=1000 * seed + split_index)
        maps, weights = rebuild_hetero(positions, ranges)
        torch.save(maps, dst / split / "comm_maps.pt")
        torch.save(weights, dst / split / "edge_weights.pt")
        torch.save(ranges, dst / split / "robot_ranges.pt")
        for item in (src / split).iterdir():
            if item.name not in RANGE_FILES:
                link(item, dst / split / item.name)
        print(f"{split}: {positions.shape[0]} samples with per-robot ranges, "
              f"{weights.values().numel() / positions.shape[0] / positions.shape[1]:.2f} incoming edges per robot")


def write(src: Path, dst: Path, comm_range, scale: float | None = None, multiscale: bool = False,
          infer_range: bool = False) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    splits = [s for s in SPLITS if (src / s).is_dir()]
    if not splits:  # an unsplit part written by data_generation.py --split False
        splits = ["."]
    for item in src.iterdir():
        if item.is_file() and (splits != ["."] or item.name not in RANGE_FILES):
            link(item, dst / item.name)
    for split in splits:
        (dst / split).mkdir(exist_ok=True)
        positions = load(src / split / "robot_positions.pt")
        if infer_range:
            stored = torch.load(src / split / "edge_weights.pt", weights_only=True)
            comm_range = infer_ranges(positions, stored.to_dense() if stored.is_sparse else stored)
            counts = {float(r): int((comm_range == r).sum()) for r in comm_range.unique()}
            print(f"{split}: ranges inferred from the edge weights: {counts}")
            maps, weights = rebuild(positions, comm_range, scale=scale, multiscale=multiscale, stored_weights=stored)
        else:
            maps, weights = rebuild(positions, comm_range, scale=scale, multiscale=multiscale)
        torch.save(maps, dst / split / "comm_maps.pt")
        torch.save(weights, dst / split / "edge_weights.pt")
        if split != ".":
            for item in (src / split).iterdir():
                if item.name not in RANGE_FILES:
                    link(item, dst / split / item.name)
        shown = f"{float(comm_range):.0f} m" if torch.as_tensor(comm_range).numel() == 1 else "their own ranges"
        print(f"{split}: {positions.shape[0]} samples redrawn at {shown}"
              f"{'' if scale is None else f' (map scale {scale:.0f} m)'}, "
              f"{(weights.values().numel() if weights.is_sparse else int((weights != 0).sum())) / positions.shape[0] / positions.shape[1]:.2f} neighbours per robot, "
              f"{maps.values().numel() / maps.shape[2] / positions.shape[0] / positions.shape[1]:.2f} robots drawn per map"
              f"{' (6 channels)' if multiscale else ''}")


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
    parser.add_argument("--range", type=float, default=None, help="communication range in metres")
    parser.add_argument("--hetero", action="store_true", help="draw a range per robot in {128, 256, 512, 1024} m instead")
    parser.add_argument("--multiscale", action="store_true", help="V5 maps: three pairs at fixed scales 256/512/1024 m")
    parser.add_argument("--infer-range", action="store_true",
                        help="recover each sample's range from its stored edge weights (merged mixed-range sets) and keep them")
    parser.add_argument("--seed", type=int, default=0, help="seed of the per-robot range draw (--hetero)")
    parser.add_argument("--fixed-scale", type=float, default=None,
                        help="draw the maps at this fixed scale with the robots within min(range, scale); the graph keeps the range")
    args = parser.parse_args()
    if args.hetero:
        if not (args.src and args.dst):
            parser.error("--src and --dst are required with --hetero")
        write_hetero(Path(args.src), Path(args.dst), args.seed)
    elif args.check:
        check(Path(args.check), args.range)
    else:
        if not (args.src and args.dst):
            parser.error("--src and --dst are required unless --check is given")
        if args.range is None and not args.infer_range:
            parser.error("--range or --infer-range is required")
        write(Path(args.src), Path(args.dst), args.range, args.fixed_scale, args.multiscale, args.infer_range)


if __name__ == "__main__":
    main()
