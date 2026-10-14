"""Merge several unsplit datasets written by the authors' data_generation.py into one split dataset.

    python merge_datasets.py --parts /path/part_a /path/part_b --out /path/data \
        [--train-ratio 0.7 --val-ratio 0.2]

Each part holds the tensors saved by ``DatasetGenerator.save_dataset`` with
``--split False`` (sparse maps, raw and normalized actions, positions, edge
weights, coverage features, objectives).  The merged dataset is written in the
layout ``CNNGNNDataset`` expects (``train/``, ``val/``, ``test/`` plus
``actions_mean.pt`` / ``actions_std.pt`` at the root), with the actions
re-normalized over the merged set exactly as the generator does
(``normalize_tensor(..., is_symmetric=True, zero_mean=True)``).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

TENSORS = ["robot_positions.pt", "local_maps.pt", "obstacle_maps.pt", "edge_weights.pt",
           "comm_maps.pt", "actions.pt", "coverage_features.pt", "objectives.pt"]
SPARSE = {"local_maps.pt", "obstacle_maps.pt", "edge_weights.pt", "comm_maps.pt"}


def load_concat(parts: list[Path], name: str) -> torch.Tensor:
    """Concatenate one tensor across parts along the sample axis (densified if sparse)."""
    chunks = []
    for part in parts:
        tensor = torch.load(part / name, weights_only=True)
        chunks.append(tensor.to_dense() if tensor.is_sparse else tensor)
    return torch.cat(chunks, dim=0)


def normalize_actions(actions: torch.Tensor, epsilon: float = 1e-6):
    """The generator's normalization: zero mean, one shared std over both action components."""
    dims = list(range(actions.dim() - 1))
    std = actions.std(dim=dims)
    std = torch.ones_like(std) * std.mean()
    std = torch.where(std < epsilon, torch.full_like(std, epsilon), std)
    mean = torch.zeros_like(std)
    return (actions - mean) / std, mean, std


def save_split(tensor: torch.Tensor, name: str, out: Path, sizes: tuple[int, int, int], sparse: bool) -> None:
    start = 0
    for stage, size in zip(("train", "val", "test"), sizes):
        chunk = tensor[start:start + size].clone()
        torch.save(chunk.to_sparse() if sparse else chunk, out / stage / name)
        start += size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parts", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0, help="seed of the sample permutation before splitting")
    args = parser.parse_args()
    parts = [Path(p) for p in args.parts]
    out = Path(args.out)
    for stage in ("train", "val", "test"):
        (out / stage).mkdir(parents=True, exist_ok=True)

    actions = load_concat(parts, "actions.pt")
    total = actions.shape[0]
    order = torch.randperm(total, generator=torch.Generator().manual_seed(args.seed))
    n_train = int(total * args.train_ratio)
    n_val = int(total * args.val_ratio)
    sizes = (n_train, n_val, total - n_train - n_val)
    print(f"samples: {total} from {len(parts)} parts -> train {sizes[0]}, val {sizes[1]}, test {sizes[2]}")

    for name in TENSORS:
        if not all((p / name).is_file() for p in parts):
            continue
        tensor = load_concat(parts, name)[order]
        save_split(tensor, name, out, sizes, name in SPARSE)
    normalized, mean, std = normalize_actions(actions[order])
    save_split(normalized, "normalized_actions.pt", out, sizes, False)
    torch.save(mean, out / "actions_mean.pt")
    torch.save(std, out / "actions_std.pt")
    (out / "metrics.txt").write_text(f"merged from {[str(p) for p in parts]}\nsamples {total}\nsplit {sizes}\nseed {args.seed}\n")
    print("written to", out)


if __name__ == "__main__":
    main()
