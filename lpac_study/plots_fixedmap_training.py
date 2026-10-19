"""Figure of experiment A: V0 trained with the fixed-scale neighbour map ("V0F"), against the
inference-time fix and the as-is recipes.

    python -m lpac_study.plots_fixedmap_training --baseline results/results.csv \
        --v0f-mixed results_A/A_mixed_11.csv --v0f-128 results_A/A_128_11.csv \
        --v0 results_train/results_sweep100k_gpu_epoch13.csv --v0-fixed results_m1/results_2x2.csv --v0-fixed-model r128_s0 \
        --mixed results_night2/mixed_s0_11.csv results_night2/mixed_s1_11.csv results_night2/mixed_s2_11.csv \
        --r256 results_train/results_sweep256_best.csv --title "11 release environments" --out fig_fixedmap_training.png
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .plots import COLORS, ERRORBAR_STYLE, TEXT, style_axis

Y_LABEL = "Coverage cost / clairvoyant cost\n(episode average)"


def sweep_means(path: str) -> pd.DataFrame:
    f = pd.read_csv(path)
    f = f[(f["controller"] == "lpac") & (f["num_robots"] == 32)]
    if "sweep" in f:
        f = f[f["sweep"] == "range"]
    return f.groupby("comm_range")["ratio_mean"].agg(["mean", "std"]).reset_index()


def factorial_means(path: str, model: str | None = None, level: str = "C3") -> pd.DataFrame:
    f = pd.read_csv(path)
    f = f[(f["map"] == level) & (f["graph"] == "G0") & (f["repeat"] == 0)]
    if model:
        f = f[f["model"] == model]
    return f.groupby("comm_range")["ratio_mean"].agg(["mean", "std"]).reset_index()


def seed_means(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Mean over seeds of the environment mean; the std column becomes the std across seeds."""
    stacked = pd.concat(frames)
    return stacked.groupby("comm_range")["mean"].agg(["mean", "std"]).fillna(0.0).reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--v0f-mixed", nargs="+", required=True, help="factorial CSVs of V0F trained on the mixed-range set (one per seed)")
    parser.add_argument("--v0f-mixed-models", nargs="*", default=["A_mixed", "A_mixed_s1", "A_mixed_s2"], help="model tags of those files")
    parser.add_argument("--v0f-128", required=True, help="factorial CSV of V0F trained on the 128 m set")
    parser.add_argument("--v0", required=True, help="sweep of V0 r128 seed 0 as is")
    parser.add_argument("--v0-fixed", required=True, help="factorial CSV with the C3 rows of V0 r128 seed 0")
    parser.add_argument("--v0-fixed-model", default="r128_s0")
    parser.add_argument("--mixed", nargs="*", default=[], help="sweeps of the V0 seeds trained at mixed ranges (as-is map)")
    parser.add_argument("--r256", default=None)
    parser.add_argument("--title", default="11 release environments")
    parser.add_argument("--out", default="fig_fixedmap_training.png")
    args = parser.parse_args()
    base = pd.read_csv(args.baseline)
    base = base[(base["sweep"] == "range") & (base["num_robots"] == 32)]
    fig, ax = plt.subplots(figsize=(8.5, 5.4), dpi=150)
    series = [
        (seed_means([factorial_means(f, m) for f, m in zip(args.v0f_mixed, args.v0f_mixed_models)]), "#2a78d6", "-", "s",
         f"V0F: trained on mixed ranges with the fixed-scale map ({len(args.v0f_mixed)} seed{'s' if len(args.v0f_mixed) > 1 else ''})"),
        (factorial_means(args.v0f_128, "A_128"), "#7b3fb3", "-", "s", "V0F: trained at 128 m with the fixed-scale map (seed 0)"),
        (sweep_means(args.v0), "#9ec5f4", "--", "o", "V0 trained at 128 m, as is"),
        (factorial_means(args.v0_fixed, args.v0_fixed_model), "#184f95", "--", "o", "V0 trained at 128 m, fixed-scale map at inference only"),
    ]
    if args.mixed:
        frames = [sweep_means(p) for p in args.mixed]
        m = pd.concat(frames).groupby("comm_range")["mean"].agg(["mean", "std"]).reset_index()
        series.append((m, "#eb6834", "--", "^", f"V0 trained on mixed ranges, as-is map ({len(frames)} seeds)"))
    if args.r256:
        series.append((sweep_means(args.r256), "#e87ba4", "--", "v", "V0 trained at 256 m, as is"))
    for r, color, style, marker, label in series:
        r = r.sort_values("comm_range")
        ax.errorbar(r["comm_range"], r["mean"], yerr=r["std"], color=color, linestyle=style, marker=marker, label=label,
                    **{k: v for k, v in ERRORBAR_STYLE.items() if k != "marker"})
    ax.axhline(base[base["controller"] == "CentralizedCVT"]["ratio_mean"].mean(), color=COLORS["CentralizedCVT"], linestyle=":",
               linewidth=1.5, label="centralized CVT")
    style_axis(ax, "Communication range at test time (m)", Y_LABEL,
               f"Training with the fixed-scale neighbour map (256 m) — {args.title}\nbars = std across environments")
    ax.set_xticks([128, 256, 512, 768, 1024])
    ax.set_ylim(0, 5)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out)
    plt.close(fig)
    print("figure written to", args.out)


if __name__ == "__main__":
    main()
