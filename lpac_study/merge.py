"""Merge the CSV outputs of several sweep (or factorial) processes and print a summary table.

    python -m lpac_study.merge --parts results/part_a results/part_b --out-dir results
    python -m lpac_study.merge --kind factorial --parts results_2x2/f1024a results_2x2/r768_env_0 --out-dir results_2x2

Rows are de-duplicated on (env, comm_range, num_robots, controller); the
first occurrence wins.  The summary printed at the end is the mean and the
standard deviation across environments of the clairvoyant-normalized cost.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

KEY = ["env", "comm_range", "num_robots", "controller"]
FILES = {"results.csv": KEY, "rank.csv": ["env", "comm_range", "num_robots", "layer"],
         "curves.csv": KEY + ["step"]}
FACTORIAL_KEY = ["env", "comm_range", "model", "cell", "k", "repeat"]
FACTORIAL_FILES = {"results_2x2.csv": FACTORIAL_KEY}


def merge_file(parts: list[Path], name: str, key: list[str]) -> pd.DataFrame:
    """Concatenate one CSV across parts, dropping duplicate keys."""
    frames = [pd.read_csv(p / name) for p in parts if (p / name).is_file()]
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    if "repeat" in key:  # files written before the sensitivity repeats existed have no such column
        merged["repeat"] = merged.get("repeat", 0).fillna(0).astype(int)
        merged["noise_sigma"] = merged.get("noise_sigma", 0.0).fillna(0.0)
    if "model" in key:  # files written before --model-tag existed hold the published checkpoint
        merged["model"] = merged.get("model", "published").fillna("published")
    return merged.drop_duplicates(subset=key).sort_values(key)


def factorial_summary(results: pd.DataFrame, metric: str = "ratio_mean") -> pd.DataFrame:
    """Mean ± std across environments per (range, cell, k) of the factorial experiment."""
    nominal = results[results["repeat"] == 0] if "repeat" in results else results
    grouped = nominal.groupby(["comm_range", "cell", "k"])
    table = grouped[metric].agg(mean="mean", std="std", n_env="count").reset_index()
    extras = grouped[["degree_mean", "map_neighbours_mean"]].mean().reset_index()
    return table.merge(extras).round(3)


def summary(results: pd.DataFrame, metric: str = "ratio_mean") -> pd.DataFrame:
    """Mean ± std across environments, one row per (sweep, range, robots, controller)."""
    grouped = results.groupby(["sweep", "comm_range", "num_robots", "controller"])[metric]
    table = grouped.agg(mean="mean", std="std", n_env="count").reset_index()
    degrees = results[results["controller"] == "lpac"].groupby(
        ["sweep", "comm_range", "num_robots"])["degree_mean"].mean().rename("degree_lpac")
    return table.merge(degrees.reset_index(), how="left").round(4)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts", nargs="+", required=True, help="output folders of sweep processes")
    parser.add_argument("--out-dir", required=True, help="folder receiving the merged CSVs")
    parser.add_argument("--kind", choices=["sweep", "factorial"], default="sweep",
                        help="sweep: results/rank/curves of lpac_study.sweep; factorial: results_2x2.csv")
    args = parser.parse_args(argv)
    parts = [Path(p) for p in args.parts]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = FILES if args.kind == "sweep" else FACTORIAL_FILES
    for name, key in files.items():
        merged = merge_file(parts, name, key)
        if not merged.empty:
            merged.to_csv(out_dir / name, index=False)
            print(f"{name}: {len(merged)} rows")
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        if args.kind == "sweep":
            print(summary(pd.read_csv(out_dir / "results.csv")).to_string(index=False))
        else:
            print(factorial_summary(pd.read_csv(out_dir / "results_2x2.csv")).to_string(index=False))


if __name__ == "__main__":
    main()
