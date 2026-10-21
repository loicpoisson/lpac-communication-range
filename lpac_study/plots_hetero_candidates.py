"""Heterogeneous 128/1024 m teams: the candidate models side by side (cost against the fraction of
robots with the long range).

    python -m lpac_study.plots_hetero_candidates --published results_hetero/results_hetero8.csv \
        --series "V4 trained on mixed ranges (seed 0)=results_V45/results_hetero_V4mixed.csv" \
                 "V0R, own range as input (seed 0)=results_H2/results_hetero_V0R.csv" \
        --mixture results_h1/results_h1.csv --cvt 2.52 --out fig_hetero_candidates.png
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .plots import COLORS, ERRORBAR_STYLE, TEXT, style_axis

STYLE = {k: v for k, v in ERRORBAR_STYLE.items() if k != "marker"}


def hetero_means(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby(["fraction", "env"])["ratio_mean"].mean().groupby("fraction").agg(["mean", "std"]).reset_index().sort_values("fraction")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--published", required=True, help="results_hetero*.csv of the published model (as-is and fixed maps)")
    parser.add_argument("--series", nargs="*", default=[], help="label=results_hetero.csv of other models (as-is map rows)")
    parser.add_argument("--mixture", default=None, help="results_h1.csv of the mixture of specialists")
    parser.add_argument("--cvt", type=float, default=None)
    parser.add_argument("--out", default="fig_hetero_candidates.png")
    args = parser.parse_args()
    pub = pd.read_csv(args.published)
    fig, ax = plt.subplots(figsize=(8.5, 5.4), dpi=150)
    for frame, color, style, marker, label in (
            (pub[pub["map"] == "asis"], "#9ec5f4", "--", "o", "published model, map at each robot's own range"),
            (pub[pub["map"] == "fixed256"], "#184f95", "--", "o", "published model, map at a fixed 256 m scale")):
        r = hetero_means(frame)
        ax.errorbar(r["fraction"], r["mean"], yerr=r["std"], color=color, linestyle=style, marker=marker, label=label, **STYLE)
    if args.mixture:
        h1 = pd.read_csv(args.mixture)
        frame = pd.concat([h1[h1["mode"] == "mixture"], h1[(h1["mode"] == "short") & (h1["fraction"] == 0)], h1[(h1["mode"] == "long") & (h1["fraction"] == 1)]])
        r = hetero_means(frame)
        ax.errorbar(r["fraction"], r["mean"], yerr=r["std"], color="#7b3fb3", linestyle="--", marker="^", label="mixture of the 128 m and 1024 m specialists", **STYLE)
    for color, marker, item in zip(("#d95926", "#2a78d6", "#1baf7a"), ("s", "D", "v"), args.series):
        label, path = item.split("=", 1)
        frame = pd.read_csv(path)
        frame = frame[frame["map"] == "asis"] if "map" in frame else frame
        r = hetero_means(frame)
        ax.errorbar(r["fraction"], r["mean"], yerr=r["std"], color=color, linestyle="-", marker=marker, label=label, **STYLE)
    if args.cvt:
        ax.axhline(args.cvt, color=COLORS["CentralizedCVT"], linestyle=":", linewidth=1.5, label="centralized CVT")
    style_axis(ax, "Fraction of robots with the 1024 m range (others 128 m)", "Coverage cost / clairvoyant cost\n(episode average)",
               "Heterogeneous teams: one model for everyone against the inference-time options\n11 release environments, bars = std across environments")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_ylim(0, 4.5)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out)
    plt.close(fig)
    print("figure written to", args.out)


if __name__ == "__main__":
    main()
