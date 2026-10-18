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


DESCRIPTORS = {"long_idf_share": "Voronoi share of the importance field held by the long-range robots at t = 0",
               "long_center_dist": "mean distance of the long-range robots to the world centre (half-world units)",
               "long_degree128": "mean number of robots within 128 m of a long-range robot at t = 0",
               "long_spread": "spread of the long-range robots / spread of the whole team"}
FRACTION_COLORS = {0.25: "#6da7ec", 0.5: "#2a78d6", 0.75: "#184f95"}


def plot_position(frame: pd.DataFrame, path: str) -> None:
    """Cost minus its (environment, fraction) mean, against each start descriptor minus the same mean, across draws."""
    mid = frame[(frame["fraction"] > 0) & (frame["fraction"] < 1) & (frame["map"] == "asis")].copy()
    keys = ["fraction", "env"]
    mid["dcost"] = mid["ratio_mean"] - mid.groupby(keys)["ratio_mean"].transform("mean")
    fig, axes = plt.subplots(1, len(DESCRIPTORS), figsize=(4.2 * len(DESCRIPTORS), 4.3), dpi=150, sharey=True)
    for ax, (col, label) in zip(axes, DESCRIPTORS.items()):
        mid["dx"] = mid[col] - mid.groupby(keys)[col].transform("mean")
        for fr, color in FRACTION_COLORS.items():
            g = mid[mid["fraction"] == fr]
            ax.scatter(g["dx"], g["dcost"], s=14, color=color, alpha=0.75, label=f"f = {fr:g}", linewidths=0)
        rho = mid[["dx", "dcost"]].corr(method="spearman").iloc[0, 1]
        style_axis(ax, "descriptor - its (env, f) mean", "cost - its (env, f) mean" if ax is axes[0] else "",
                   f"{label}\nSpearman rho = {rho:+.2f} (within env and f, across draws)")
        ax.axhline(0, color="#999999", linewidth=0.8)
        ax.axvline(0, color="#999999", linewidth=0.8)
    axes[0].legend(frameon=False, fontsize=8, title="fraction at 1024 m")
    fig.suptitle("Which long-range subset hurts: cost variation across draws against where those robots start (as-is map, 11 envs x 3 fractions x 8 draws)",
                 fontsize=9.5, color=TEXT)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", required=True)
    parser.add_argument("--cvt", type=float, default=None, help="centralized CVT cost ratio (dotted line)")
    parser.add_argument("--out", default="fig_hetero.png")
    parser.add_argument("--out-position", default=None,
                        help="second figure: cost against the start position of the long-range subset, across draws")
    parser.add_argument("--draws-label", default="2 draws of the long-range subset")
    args = parser.parse_args()
    frame = pd.read_csv(args.results)
    if args.out_position:
        plot_position(frame, args.out_position)
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
    fig.suptitle(f"Heterogeneous communication ranges at inference — 32 robots, 11 release environments, {args.draws_label}\n"
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
