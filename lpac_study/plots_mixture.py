"""Figure of the mixture-of-specialists experiment (H1): cost against the fraction of long-range
robots for the mixture, each specialist used by the whole team, the published model and the
fixed-scale map.

    python -m lpac_study.plots_mixture --results results_h1/results_h1.csv \
        --hetero results_hetero/results_hetero8.csv --cvt 2.52 --out fig_mixture.png
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .plots import COLORS, ERRORBAR_STYLE, TEXT, style_axis

SERIES = {"short": ("128 m specialist on every robot", "#eb6834", "-", "o"),
          "long": ("1024 m specialist on every robot", "#1baf7a", "-", "^"),
          "mixture": ("mixture: each robot runs its own range's specialist", "#2a78d6", "-", "s")}


def env_mean(frame: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Mean over draws per environment, then mean/std across environments."""
    return frame.groupby(by + ["env"])["ratio_mean"].mean().groupby(by).agg(["mean", "std"]).reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", required=True, help="results_h1.csv")
    parser.add_argument("--hetero", required=True, help="results_hetero*.csv of the published model (as-is and fixed maps)")
    parser.add_argument("--cvt", type=float, default=None)
    parser.add_argument("--extra", nargs="*", default=[], help="label=results_h1.csv of other mixtures (dashed, mixture mode only)")
    parser.add_argument("--out", default="fig_mixture.png")
    args = parser.parse_args()
    h1 = pd.read_csv(args.results)
    het = pd.read_csv(args.hetero)
    fig, ax = plt.subplots(figsize=(8.5, 5.4), dpi=150)
    for mode, (label, color, style, marker) in SERIES.items():
        r = env_mean(h1[h1["mode"] == mode], ["fraction"]).sort_values("fraction")
        if mode == "mixture":  # the endpoints of the mixture are the specialists at their own range
            ends = pd.concat([env_mean(h1[(h1["mode"] == "short") & (h1["fraction"] == 0)], ["fraction"]),
                              env_mean(h1[(h1["mode"] == "long") & (h1["fraction"] == 1)], ["fraction"])])
            r = pd.concat([r, ends]).sort_values("fraction")
        ax.errorbar(r["fraction"], r["mean"], yerr=r["std"], color=color, linestyle=style, marker=marker, label=label,
                    **{k: v for k, v in ERRORBAR_STYLE.items() if k != "marker"})
    for color, item in zip(("#6da7ec", "#184f95", "#7b3fb3"), args.extra):
        label, path = item.split("=", 1)
        other = pd.read_csv(path)
        r = env_mean(other[other["mode"] == "mixture"], ["fraction"]).sort_values("fraction")
        ax.errorbar(r["fraction"], r["mean"], yerr=r["std"], color=color, linestyle=":", marker="s", markersize=5,
                    linewidth=1.6, capsize=3, label=label)
    for map_mode, label, color, style in (("asis", "published model, map at each robot's own range", "#9ec5f4", "--"),
                                          ("fixed256", "published model, map at a fixed 256 m scale", "#184f95", "--")):
        r = env_mean(het[het["map"] == map_mode], ["fraction"]).sort_values("fraction")
        ax.errorbar(r["fraction"], r["mean"], yerr=r["std"], color=color, linestyle=style, marker="D", markersize=5,
                    linewidth=1.6, capsize=3, label=label)
    if args.cvt:
        ax.axhline(args.cvt, color=COLORS["CentralizedCVT"], linestyle=":", linewidth=1.5, label="centralized CVT")
    style_axis(ax, "Fraction of robots with the 1024 m range (others 128 m)", "Coverage cost / clairvoyant cost\n(episode average)",
               "Mixture of range specialists in a heterogeneous team — 11 release environments\n"
               "specialists trained at 128 m and 1024 m (seed 0); bars = std across environments, 4 draws of the subset")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out)
    plt.close(fig)
    print("figure written to", args.out)


if __name__ == "__main__":
    main()
