"""Figure of experiment H2: V0R (the robot's own range as a GNN input, trained on teams with a range
drawn per robot) on homogeneous teams across ranges and on the heterogeneous 128/1024 m grid.

    python -m lpac_study.plots_range_input --baseline results/results.csv \
        --v0r results_H2/V0R_s0_11.csv --v0 results_train/results_sweep100k_gpu_epoch13.csv \
        --v0-fixed results_m1/results_2x2.csv --v0f results_A/A_mixed_11.csv --mixed results_night2/mixed_s0_11.csv \
        --v0r-hetero results_H2/results_hetero_V0R.csv --hetero results_hetero/results_hetero8.csv --h1 results_h1/results_h1.csv \
        --out fig_range_input.png
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .plots import COLORS, ERRORBAR_STYLE, TEXT, style_axis

Y_LABEL = "Coverage cost / clairvoyant cost\n(episode average)"
STYLE = {k: v for k, v in ERRORBAR_STYLE.items() if k != "marker"}


def sweep_means(path: str) -> pd.DataFrame:
    f = pd.read_csv(path)
    f = f[(f["controller"] == "lpac") & (f["num_robots"] == 32)]
    if "sweep" in f:
        f = f[f["sweep"] == "range"]
    return f.groupby("comm_range")["ratio_mean"].agg(["mean", "std"]).reset_index().sort_values("comm_range")


def factorial_means(path: str, model: str, level: str = "C3") -> pd.DataFrame:
    f = pd.read_csv(path)
    f = f[(f["map"] == level) & (f["graph"] == "G0") & (f["repeat"] == 0) & (f["model"] == model)]
    return f.groupby("comm_range")["ratio_mean"].agg(["mean", "std"]).reset_index().sort_values("comm_range")


def hetero_means(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby(["fraction", "env"])["ratio_mean"].mean().groupby("fraction").agg(["mean", "std"]).reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--v0r", required=True)
    parser.add_argument("--v0", required=True)
    parser.add_argument("--v0-fixed", required=True)
    parser.add_argument("--v0-fixed-model", default="r128_s0")
    parser.add_argument("--v0f", required=True)
    parser.add_argument("--mixed", required=True)
    parser.add_argument("--v0r-hetero", required=True)
    parser.add_argument("--hetero", required=True)
    parser.add_argument("--h1", required=True)
    parser.add_argument("--out", default="fig_range_input.png")
    args = parser.parse_args()
    base = pd.read_csv(args.baseline)
    base = base[(base["sweep"] == "range") & (base["num_robots"] == 32)]
    cvt = base[base["controller"] == "CentralizedCVT"]["ratio_mean"].mean()

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=150, sharey=True)
    for r, color, style, marker, label in (
            (sweep_means(args.v0r), "#2a78d6", "-", "s", "V0R: own range as GNN input, trained on per-robot ranges (seed 0)"),
            (sweep_means(args.v0), "#9ec5f4", "--", "o", "V0 trained at 128 m, as is"),
            (factorial_means(args.v0_fixed, args.v0_fixed_model), "#184f95", "--", "o", "V0 trained at 128 m, fixed-scale map at inference"),
            (factorial_means(args.v0f, "A_mixed"), "#1baf7a", "--", "D", "V0F: fixed-scale map at training, mixed ranges"),
            (sweep_means(args.mixed), "#eb6834", "--", "^", "V0 trained on mixed ranges, as-is map")):
        ax_a.errorbar(r["comm_range"], r["mean"], yerr=r["std"], color=color, linestyle=style, marker=marker, label=label, **STYLE)
    ax_a.axhline(cvt, color=COLORS["CentralizedCVT"], linestyle=":", linewidth=1.5, label="centralized CVT")
    style_axis(ax_a, "Communication range at test time (m), homogeneous team", Y_LABEL, "Homogeneous teams")
    ax_a.set_xticks([128, 256, 512, 768, 1024])
    ax_a.set_ylim(0, 5)
    ax_a.legend(frameon=False, fontsize=7.5)

    v0r = pd.read_csv(args.v0r_hetero)
    het = pd.read_csv(args.hetero)
    h1 = pd.read_csv(args.h1)
    mix = h1[h1["mode"] == "mixture"]
    ends = pd.concat([h1[(h1["mode"] == "short") & (h1["fraction"] == 0)], h1[(h1["mode"] == "long") & (h1["fraction"] == 1)]])
    for frame, color, style, marker, label in (
            (v0r[v0r["map"] == "asis"], "#2a78d6", "-", "s", "V0R, one model for the whole team (its own convention)"),
            (het[het["map"] == "asis"], "#9ec5f4", "--", "o", "published model, map at each robot's own range"),
            (het[het["map"] == "fixed256"], "#184f95", "--", "o", "published model, map at a fixed 256 m scale"),
            (pd.concat([mix, ends]), "#7b3fb3", "--", "^", "mixture of the 128 m and 1024 m specialists")):
        r = hetero_means(frame).sort_values("fraction")
        ax_b.errorbar(r["fraction"], r["mean"], yerr=r["std"], color=color, linestyle=style, marker=marker, label=label, **STYLE)
    ax_b.axhline(cvt, color=COLORS["CentralizedCVT"], linestyle=":", linewidth=1.5)
    style_axis(ax_b, "Fraction of robots with the 1024 m range (others 128 m)", "", "Heterogeneous teams")
    ax_b.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax_b.legend(frameon=False, fontsize=7.5)
    fig.suptitle("One model with the robot's own range as input (V0R) — 11 release environments, bars = std across environments",
                 fontsize=10, color=TEXT)
    fig.tight_layout()
    fig.savefig(args.out)
    plt.close(fig)
    print("figure written to", args.out)


if __name__ == "__main__":
    main()
