"""Figure and table of the heterogeneous-range experiment (lpac_study.heterogeneous).

    python -m lpac_study.plots_hetero --results results_hetero/results_hetero.csv --out fig_hetero.png

Left: cost ratio versus the fraction of long-range robots, for the as-is map and the map at a
fixed 256 m scale (mean over environments and draws, bars = std across environments of the
per-environment mean over draws). Right: frozen-state probe, the relative change of the actions
of the short-range robots (graph contamination alone) and of the long-range robots.
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .plots import COLORS, ERRORBAR_STYLE, TEXT, style_axis

MAP_LABELS = {"asis": "map at each robot's own range (as published)", "fixed256": "map at a fixed 256 m scale"}
MAP_COLORS = {"asis": "#2a78d6", "fixed256": "#184f95"}


def per_env(frame: pd.DataFrame, col: str) -> pd.DataFrame:
    """Mean over draws per (map, fraction, env), then mean/std across environments."""
    env_mean = frame.groupby(["map", "fraction", "env"])[col].mean().reset_index()
    return env_mean.groupby(["map", "fraction"])[col].agg(["mean", "std", "count"]).reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", required=True)
    parser.add_argument("--cvt", type=float, default=None, help="centralized CVT cost ratio (dotted line)")
    parser.add_argument("--out", default="fig_hetero.png")
    args = parser.parse_args()
    frame = pd.read_csv(args.results)
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=150)
    cost = per_env(frame, "ratio_mean")
    for map_mode in ("asis", "fixed256"):
        r = cost[cost["map"] == map_mode].sort_values("fraction")
        if r.empty:
            continue
        ax_a.errorbar(r["fraction"], r["mean"], yerr=r["std"], color=MAP_COLORS[map_mode], label=MAP_LABELS[map_mode],
                      **ERRORBAR_STYLE)
    if args.cvt:
        ax_a.axhline(args.cvt, color=COLORS["CentralizedCVT"], linestyle=":", linewidth=1.5, label="centralized CVT")
    style_axis(ax_a, "Fraction of robots with the 1024 m range (others 128 m)", "Coverage cost / clairvoyant cost\n(episode average)",
               "Cost of a mixed-range team, published model")
    ax_a.set_ylim(bottom=0)
    ax_a.legend(frameon=False, fontsize=8)

    probe = frame[(frame["fraction"] > 0) & (frame["fraction"] < 1)]
    for map_mode, style in (("asis", "-"), ("fixed256", "--")):
        for col, marker, who in (("probe_short", "o", "short-range robots (graph effect only)"),
                                 ("probe_long", "^", "long-range robots (own input + graph)")):
            r = per_env(probe[probe["map"] == map_mode], col).sort_values("fraction")
            if r.empty:
                continue
            ax_b.errorbar(r["fraction"], r["mean"], yerr=r["std"], color=MAP_COLORS[map_mode], linestyle=style, marker=marker,
                          markersize=6, linewidth=2, capsize=4, label=f"{who}, {'as-is map' if map_mode == 'asis' else '256 m map'}")
    style_axis(ax_b, "Fraction of robots with the 1024 m range", "|action(mixed team) − action(all at 128 m)|\n/ |action(all at 128 m)|",
               "Frozen-state probe: how much the mixed team changes each robot's action")
    ax_b.set_ylim(bottom=0)
    ax_b.legend(frameon=False, fontsize=7.5)
    for ax in (ax_a, ax_b):
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    fig.suptitle("Heterogeneous communication ranges at inference — 32 robots, 11 release environments, 2 draws of the long-range subset\n"
                 "bars = std across environments", fontsize=10, color=TEXT)
    fig.tight_layout()
    fig.savefig(args.out)
    plt.close(fig)
    print(cost.round(3).to_string(index=False))
    print(per_env(probe, "probe_short").round(3).to_string(index=False))
    print(per_env(probe, "probe_long").round(3).to_string(index=False))
    print("figure written to", args.out)


if __name__ == "__main__":
    main()
