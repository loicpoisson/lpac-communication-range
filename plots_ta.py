"""Figure for the two CPU questions of the teaching assistant (sections 16.1 and 16.2).

    python plots_ta.py --out fig_ta_q1q2.png

Left panel, question Q1: the published model at long range with the three neighbour maps, on the
40 new environments. Right panel, question Q2: V4 trained on mixed ranges in a heterogeneous team,
for the four definitions of a communication link, averaged over three seeds.
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RANGES = [512.0, 768.0, 1024.0]
FRACTIONS = [0.0, 0.25, 0.5, 0.75, 1.0]
DRAWS = [0, 1, 2, 3]
# one hue and one marker per series, so the identity never rests on colour alone
MAPS = [("C0", "map as published", "#eb6834", "s", "-"),
        ("K256", "fixed 256 m scale, far robots\nprojected on the circle", "#6da7ec", "^", "-"),
        ("C3", "fixed 256 m scale, far robots\ndropped", "#184f95", "o", "-")]
RULES = [("reception", "reception: j -> i if d < r_i (current)", "#184f95", "o", "-"),
         ("transmission", "transmission: j -> i if d < r_j", "#eb6834", "s", "-"),
         ("sym_min", "both must reach (AND)", "#1baf7a", "D", "--"),
         ("sym_max", "either reaches (OR)", "#8d4bbb", "^", "-")]


def panel_q1(ax: plt.Axes, clamp_path: str, sweep_path: str) -> None:
    c = pd.read_csv(clamp_path)
    c = c[(c.graph == "G0") & (c.repeat == 0)]
    sweep = pd.read_csv(sweep_path)
    sweep = sweep[(sweep.controller == "lpac") & (sweep.num_robots == 32)]
    if "sweep" in sweep:
        sweep = sweep[sweep.sweep == "range"]
    tables = {level: c[c["map"] == level].pivot_table(index="env", columns="comm_range", values="ratio_mean")
              for level in ("C3", "K256")}
    tables["C0"] = sweep.pivot_table(index="env", columns="comm_range", values="ratio_mean")
    optimum = tables["C0"][256.0]
    ax.axhline(optimum.mean(), color="#7a7a7a", linewidth=1.2, linestyle=":", zorder=1)
    ax.text(1024.0, optimum.mean() - 0.12, f"optimum at 256 m, {optimum.mean():.2f}",
            ha="right", va="top", fontsize=8.5, color="#5a5a5a")
    for level, label, colour, marker, style in MAPS:
        t = tables[level]
        xs = [r for r in RANGES if r in t]
        mean = [t[r].mean() for r in xs]
        se = [t[r].std() / np.sqrt(t[r].notna().sum()) for r in xs]
        ax.errorbar(xs, mean, yerr=se, color=colour, marker=marker, linestyle=style,
                    markersize=6, linewidth=2, capsize=4, label=label, zorder=3)
    ax.set_xticks(RANGES)
    ax.set_xlabel("communication range at inference (m)")
    ax.set_ylabel("coverage cost / clairvoyant cost")
    ax.set_title("Q1. Keeping the far robots as border pixels\nhurts (published model, 40 new envs)", fontsize=10.5)
    ax.legend(fontsize=8.5, loc="upper left", frameon=False)
    ax.grid(alpha=0.25, linewidth=0.7)


def panel_q2(ax: plt.Axes, reception_paths: list[str], semantics_paths: list[str]) -> None:
    def grid(frame: pd.DataFrame, **sel: object) -> pd.DataFrame:
        g = frame
        for key, value in sel.items():
            g = g[g[key] == value]
        g = g[g.draw.isin(DRAWS)]
        return g.groupby(["fraction", "env"]).ratio_mean.mean().unstack("fraction")

    per_rule = {"reception": [grid(pd.read_csv(p), **{"map": "asis"}) for p in reception_paths]}
    for rule, _, _, _, _ in RULES[1:]:
        per_rule[rule] = [grid(pd.read_csv(p), **{"map": "asis", "semantics": rule}) for p in semantics_paths]
    for rule, label, colour, marker, style in RULES:
        runs = per_rule[rule]
        means = np.array([[g[f].mean() for f in FRACTIONS] for g in runs])
        ax.errorbar(FRACTIONS, means.mean(axis=0), yerr=means.std(axis=0, ddof=1), color=colour,
                    marker=marker, linestyle=style, markersize=6, linewidth=2, capsize=4, label=label, zorder=3)
    ax.set_xticks(FRACTIONS)
    ax.set_xlabel("fraction of robots that hear at 1024 m (the others hear at 128 m)")
    ax.set_ylabel("coverage cost / clairvoyant cost")
    ax.set_title("Q2. With a geometry-reading model the link rule matters\n(V4 mixed, 3 seeds, 11 envs, 4 draws)", fontsize=10.5)
    ax.legend(fontsize=8.5, loc="lower left", frameon=False)
    ax.grid(alpha=0.25, linewidth=0.7)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clamp", default="results_clamp/results_40.csv")
    parser.add_argument("--sweep", default="results_new40/results_sweep_published.csv")
    parser.add_argument("--reception", nargs="+", default=["results_V45/results_hetero_V4mixed.csv",
                                                           "results_V45/results_hetero_V4mixed_s1.csv",
                                                           "results_V45/results_hetero_V4mixed_s2.csv"])
    parser.add_argument("--semantics", nargs="+", default=["results_sem/V4mixed_s0.csv",
                                                            "results_sem/V4mixed_s1.csv",
                                                            "results_sem/V4mixed_s2.csv"])
    parser.add_argument("--out", default="fig_ta_q1q2.png")
    args = parser.parse_args()
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.2), dpi=150)
    panel_q1(axes[0], args.clamp, args.sweep)
    panel_q2(axes[1], args.reception, args.semantics)
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
