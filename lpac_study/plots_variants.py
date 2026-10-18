"""Figure and table of the model variants (V0 as published, V1 without neighbour map, V2 with
edge geometry) across communication ranges, with the seed spread and the fixed-map references.

    python -m lpac_study.plots_variants --baseline results/results.csv \
        --v0 results_train/results_sweep100k_gpu_epoch13.csv results_train/results_sweep100k_seed1_best.csv ... \
        --v1 results_variants/V1_s0_11.csv ... --v2 results_variants/V2_s0_11.csv ... \
        --fixed-map results_m1/results_2x2.csv --r256 results_train/results_sweep256_best.csv \
        --title "11 release environments" --out fig_variants.png

Each ``--v*`` list holds one merged sweep ``results.csv`` per training seed. Curves show the
mean over seeds of the environment mean; the band is the min-max across seeds. The table
printed on stdout gives mean ± std across seeds of the environment mean at every range,
and the paired differences used in the journal.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .plots import COLORS, ERRORBAR_STYLE, TEXT, style_axis

Y_LABEL = "Coverage cost / clairvoyant cost\n(episode average)"
VARIANT_COLORS = {"V0": "#2a78d6", "V1": "#eb6834", "V2": "#1baf7a", "V3": "#eda100", "mixed": "#7b3fb3"}
VARIANT_LABELS = {"V0": "V0: as published (neighbour map + TAGConv)",
                  "V1": "V1: no neighbour map (TAGConv)",
                  "V2": "V2: no map, relative positions on edges (GATv2)",
                  "V3": "V3: no map, PointNetConv on relative positions / 256 m, then TAGConv",
                  "mixed": "V0 trained at mixed ranges 128/256/512/1024 m"}


def env_means(path: str, controller: str = "lpac") -> pd.Series:
    """Mean over environments of the cost ratio, per range, for one sweep file."""
    frame = pd.read_csv(path)
    frame = frame[(frame["controller"] == controller) & (frame["num_robots"] == 32)]
    if "sweep" in frame:
        frame = frame[frame["sweep"] == "range"]
    return frame.groupby("comm_range")["ratio_mean"].mean()


def seed_table(paths: list[str]) -> pd.DataFrame:
    """Rows = seeds, columns = ranges: environment mean of the cost ratio."""
    return pd.DataFrame([env_means(p) for p in paths]).reset_index(drop=True)


def draw_variant(ax, table: pd.DataFrame, color: str, label: str) -> None:
    ranges = table.columns.to_numpy()
    mean = table.mean(axis=0).to_numpy()
    ax.plot(ranges, mean, color=color, label=f"{label} ({len(table)} seed{'s' if len(table) > 1 else ''})",
            **{k: v for k, v in ERRORBAR_STYLE.items() if k != "capsize"})
    if len(table) > 1:
        ax.fill_between(ranges, table.min(axis=0).to_numpy(), table.max(axis=0).to_numpy(), color=color, alpha=0.18,
                        linewidth=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", required=True, help="results.csv of the published model (CVT references)")
    parser.add_argument("--v0", nargs="*", default=[], help="sweep files of the V0 seeds")
    parser.add_argument("--v1", nargs="*", default=[], help="sweep files of the V1 seeds")
    parser.add_argument("--v2", nargs="*", default=[], help="sweep files of the V2 seeds")
    parser.add_argument("--v3", nargs="*", default=[], help="sweep files of the V3 seeds")
    parser.add_argument("--mixed", nargs="*", default=[], help="sweep files of the V0 seeds trained at mixed ranges")
    parser.add_argument("--fixed-map", help="results_2x2.csv holding C3 rows of the V0 seeds (map at 256 m scale)")
    parser.add_argument("--r256", help="sweep file of the model trained at 256 m")
    parser.add_argument("--extra", nargs="*", default=[], help="label=path sweeps drawn as extra dashed curves (1 seed)")
    parser.add_argument("--suptitle", default="Model variants trained at 128 m (100,000 samples), tested across ranges")
    parser.add_argument("--title", default="11 release environments")
    parser.add_argument("--out", default="fig_variants.png")
    args = parser.parse_args()

    base = pd.read_csv(args.baseline)
    base = base[(base["sweep"] == "range") & (base["num_robots"] == 32)]
    fig, ax = plt.subplots(figsize=(8.5, 5.4), dpi=150)
    tables = {}
    for variant, paths in (("V0", args.v0), ("V1", args.v1), ("V2", args.v2), ("V3", args.v3), ("mixed", args.mixed)):
        if paths:
            tables[variant] = seed_table(paths)
            draw_variant(ax, tables[variant], VARIANT_COLORS[variant], VARIANT_LABELS[variant])
    if args.fixed_map:
        fixed = pd.read_csv(args.fixed_map)
        fixed = fixed[(fixed["map"] == "C3") & (fixed["graph"] == "G0") & (fixed.get("repeat", 0) == 0)
                      & fixed["model"].str.startswith("r128")]
        if not fixed.empty:
            t = fixed.groupby(["model", "comm_range"])["ratio_mean"].mean().unstack("comm_range").sort_index()
            t = t.dropna(axis=1)  # keep the ranges measured for every seed
            if "V0" in tables and 256.0 not in t.columns and 256.0 in tables["V0"].columns:
                # at 256 m the fixed-scale map is the as-is map (bit-identical, M1): take the V0 seeds
                # in the same order (r128_s0, r128_s1, r128_s2 = --v0 files in seed order)
                t[256.0] = tables["V0"][256.0].to_numpy()[: len(t)]
                t = t.sort_index(axis=1)
            tables["V0 + map at 256 m scale"] = t.reset_index(drop=True)
            r = t.mean(axis=0)
            ax.plot(r.index, r.to_numpy(), color="#184f95", linestyle="--", marker="s", markersize=5, linewidth=1.8,
                    label=f"V0 with the map drawn at a 256 m scale ({len(t)} seeds)")
    if args.r256:
        r = env_means(args.r256)
        tables["trained at 256 m"] = pd.DataFrame([r]).reset_index(drop=True)
        ax.plot(r.index, r.to_numpy(), color="#e87ba4", linestyle="--", marker="^", markersize=6, linewidth=1.8,
                label="V0 trained at 256 m (1 seed)")
    for color, marker, item in zip(("#eda100", "#7b3fb3", "#008300"), ("D", "v", "P"), args.extra):
        label_x, path = item.split("=", 1)
        r = env_means(path)
        tables[label_x] = pd.DataFrame([r]).reset_index(drop=True)
        ax.plot(r.index, r.to_numpy(), color=color, linestyle="--", marker=marker, markersize=6, linewidth=1.8, label=label_x)
    for controller, style in (("CentralizedCVT", ":"), ("DecentralizedCVT", "-.")):
        value = base[base["controller"] == controller]["ratio_mean"].mean()
        ax.axhline(value, color=COLORS[controller], linestyle=style, linewidth=1.5,
                   label=f"{'centralized' if controller.startswith('C') else 'decentralized'} CVT")
    style_axis(ax, "Communication range at test time (m)", Y_LABEL,
               f"{args.suptitle} — {args.title}\n"
               "line = mean over seeds of the environment mean, band = min-max across seeds")
    ax.set_xticks([128, 256, 512, 768, 1024])
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(args.out)
    plt.close(fig)

    print(f"# {args.title}: environment mean, mean ± std across seeds (n seeds)")
    for name, t in tables.items():
        cells = [f"{t[c].mean():.2f} ± {t[c].std():.2f}" if len(t) > 1 else f"{t[c].mean():.2f}" for c in t.columns]
        print(f"| {name} | " + " | ".join(cells) + f" | n={len(t)} |")
    print("figure written to", args.out)


if __name__ == "__main__":
    main()
