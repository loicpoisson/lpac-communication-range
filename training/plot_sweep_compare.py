"""Range sweep of retrained LPAC checkpoints next to the published model and the CVT baselines.

    python training/plot_sweep_compare.py --baseline results/results.csv \
        --sweeps "retrained, 100k samples=results_train/results_sweep100k_gpu_epoch13.csv" --out fig_range_retrained.png
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

COLORS = {"lpac": "#2a78d6", "DecentralizedCVT": "#eb6834", "CentralizedCVT": "#1baf7a", "ClairvoyantCVT": "#eda100"}
SWEEP_COLORS = ["#184f95", "#6da7ec", "#e87ba4"]
STYLE = dict(marker="o", markersize=6, linewidth=2, capsize=4)


def curve(frame: pd.DataFrame, controller: str) -> pd.DataFrame:
    rows = frame[frame["controller"] == controller]
    return rows.groupby("comm_range")["ratio_mean"].agg(["mean", "std"]).sort_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--sweeps", nargs="+", required=True, help="label=path pairs of sweep results.csv files")
    parser.add_argument("--out", default="fig_range_retrained.png")
    args = parser.parse_args()
    base = pd.read_csv(args.baseline)
    base = base[(base["sweep"] == "range") & (base["num_robots"] == 32)]
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=150)
    for ctrl, label, style in (("lpac", "LPAC-K3 as published (trained at 128 m)", "--"),
                               ("DecentralizedCVT", "Decentralized CVT", "-"), ("CentralizedCVT", "Centralized CVT", "-"),
                               ("ClairvoyantCVT", "Clairvoyant CVT (= 1)", "-")):
        c = curve(base, ctrl)
        ax.errorbar(c.index, c["mean"], yerr=c["std"].fillna(0), color=COLORS[ctrl], linestyle=style,
                    alpha=0.6 if ctrl == "lpac" else 1.0, label=label, **STYLE)
    for color, item in zip(SWEEP_COLORS, args.sweeps):
        label, path = item.split("=", 1)
        c = curve(pd.read_csv(path), "lpac")
        ax.errorbar(c.index, c["mean"], yerr=c["std"].fillna(0), color=color, label=f"LPAC-K3 {label}", **STYLE)
    ax.set_xlabel("Communication range at test time (m)")
    ax.set_ylabel("Coverage cost / clairvoyant cost\n(episode average)")
    ax.set_title("Retrained LPAC-K3 under the same range sweep — 32 robots, 11 environments\n"
                 "bars = std across environments; retrained on our own clairvoyant data, release hyperparameters", fontsize=10)
    ax.set_xticks(sorted(base["comm_range"].unique()))
    ax.set_ylim(bottom=0)
    ax.grid(True, color="#dddddd")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(args.out)
    print("written", args.out)


if __name__ == "__main__":
    main()
