"""Figures and summary tables of the two validation measures (14 October).

    python -m lpac_study.plots_measures --m1 results_m1/results_2x2.csv \
        --sweeps results_train/results_sweep100k_gpu_epoch13.csv ... \
        --new40-sweeps published=results_new40/results_sweep_published.csv ... \
        --new40-factorial results_new40/results_2x2.csv --baseline results/results.csv --fig-dir .

* ``fig_map_levels_retrained.png``  measure 1: map levels C2/C3/C4 (full graph) on the retrained
  models at 512/768/1024 m, next to each model's own full map (C0, from its sweep);
* ``fig_new_envs.png``  measure 2: published and retrained models, as is and with C3, on the
  11 release environments and on 40 new ones;
* ``fig_map_scale.png``  measure 2b: cost at 1024 m versus the drawing scale of the map.
Labels are in English for the blog post.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .plots import COLORS, ERRORBAR_STYLE, GRID, TEXT, style_axis

MODEL_LABELS = {"published": "published (trained at 128 m)", "r128_s0": "retrained at 128 m, seed 0",
                "r128_s1": "retrained at 128 m, seed 1", "r128_s2": "retrained at 128 m, seed 2",
                "r256": "retrained at 256 m"}
LEVEL_LABELS = {"C0": "full map (as is)", "C2": "map at 128 m scale", "C3": "map at 256 m scale", "C4": "empty map"}
LEVEL_COLORS = {"C0": "#9ec5f4", "C2": "#184f95", "C3": "#2a78d6", "C4": "#eb6834"}
Y_LABEL = "Coverage cost / clairvoyant cost\n(episode average)"


def mean_std(frame: pd.DataFrame, by: list[str], col: str = "ratio_mean") -> pd.DataFrame:
    return frame.groupby(by)[col].agg(["mean", "std", "count"]).reset_index()


def plot_map_levels(m1: pd.DataFrame, sweeps: dict[str, pd.DataFrame], path: Path) -> None:
    """One panel per retrained model: cost vs range for C0 (own sweep), C2, C3, C4."""
    models = [m for m in ("r128_s0", "r128_s1", "r128_s2", "r256") if m in set(m1["model"])]
    if not models:
        return
    ncols = 2 if len(models) > 1 else 1
    nrows = (len(models) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.2 * nrows), dpi=150, sharey=True)
    axes = list(axes.flat) if len(models) > 1 else [axes]
    for ax, model in zip(axes, models):
        if model in sweeps:
            own = sweeps[model]
            own = mean_std(own[(own["controller"] == "lpac") & (own["comm_range"] >= 512)], ["comm_range"])
            ax.errorbar(own["comm_range"], own["mean"], yerr=own["std"], color=LEVEL_COLORS["C0"],
                        label=LEVEL_LABELS["C0"], **ERRORBAR_STYLE)
        rows = m1[(m1["model"] == model) & (m1["repeat"] == 0) & (m1["comm_range"] >= 512)]
        for level in ("C2", "C3", "C4"):
            r = mean_std(rows[rows["map"] == level], ["comm_range"]).sort_values("comm_range")
            if not r.empty:
                ax.errorbar(r["comm_range"], r["mean"], yerr=r["std"], color=LEVEL_COLORS[level],
                            label=LEVEL_LABELS[level], **ERRORBAR_STYLE)
        ax.axhline(2.52, color=COLORS["CentralizedCVT"], linestyle=":", linewidth=1.5)
        style_axis(ax, "Range at test time (m)", Y_LABEL if axes.index(ax) % ncols == 0 else "", MODEL_LABELS[model])
        ax.set_xticks([512, 768, 1024])
        ax.set_ylim(bottom=0)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Neighbour-map levels on retrained models (full graph) — 32 robots, 11 environments\n"
                 "bars = std across environments; dotted = centralized CVT", fontsize=10, color=TEXT)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_new_envs(release: dict[str, pd.DataFrame], new40: dict[str, pd.DataFrame], path: Path) -> None:
    """Release (11) versus new (40) environments: cost vs range for each (model, map) condition."""
    conditions = [k for k in release if k in new40]
    if not conditions:
        return
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 4.8), dpi=150, sharey=True)
    palette = ["#2a78d6", "#1baf7a", "#eb6834", "#e87ba4", "#184f95", "#d95926", "#eda100", "#008300"]
    for color, cond in zip(palette, conditions):
        for ax, source in ((ax_a, release), (ax_b, new40)):
            r = mean_std(source[cond], ["comm_range"]).sort_values("comm_range")
            ax.errorbar(r["comm_range"], r["mean"], yerr=r["std"], color=color, label=cond, **ERRORBAR_STYLE)
    n_new = int(max(new40[c]["env"].nunique() for c in conditions))
    style_axis(ax_a, "Range at test time (m)", Y_LABEL, "11 release environments")
    style_axis(ax_b, "Range at test time (m)", "", f"{n_new} new environments (generated with the library, saved)")
    for ax in (ax_a, ax_b):
        ax.set_xticks([128, 256, 512, 1024])
        ax.set_ylim(bottom=0)
    ax_a.legend(frameon=False, fontsize=8)
    fig.suptitle("Does the diagnosis hold on environments never seen before? bars = std across environments",
                 fontsize=10, color=TEXT)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_map_scale(scale11: pd.DataFrame, scale40: pd.DataFrame | None, as_is: dict[str, float], path: Path) -> None:
    """Cost at 1024 m versus the drawing scale of the neighbour map (published model)."""
    if scale11.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    for frame, color, label in ((scale11, COLORS["lpac"], "11 release environments"),
                                (scale40, "#184f95", "40 new environments")):
        if frame is None or frame.empty:
            continue
        frame = frame.copy()
        frame["scale"] = frame["map"].str[1:].astype(float)
        r = mean_std(frame, ["scale"]).sort_values("scale")
        ax.errorbar(r["scale"], r["mean"], yerr=r["std"], color=color, label=f"map drawn at a fixed scale, {label}",
                    **ERRORBAR_STYLE)
    for key, (value, color, style) in as_is.items():
        ax.axhline(value, color=color, linestyle=style, linewidth=1.5, label=key)
    style_axis(ax, "Drawing scale of the neighbour map (m)", Y_LABEL,
               "Published LPAC-K3 evaluated at 1024 m: cost versus the map scale\nbars = std across environments")
    ax.set_xticks([128, 192, 256, 384, 512])
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def load_pairs(items: list[str]) -> dict[str, pd.DataFrame]:
    """``label=path`` pairs to DataFrames."""
    out = {}
    for item in items or []:
        label, path = item.split("=", 1)
        out[label] = pd.read_csv(path)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--m1", help="merged results_2x2.csv of measure 1 (and the 11-env scale sweep)")
    parser.add_argument("--sweeps", nargs="*", default=[], help="model=path pairs of release-environment sweeps")
    parser.add_argument("--baseline", required=True, help="results.csv of the published model on the release envs")
    parser.add_argument("--new40-sweeps", nargs="*", default=[], help="model=path pairs of 40-env sweeps")
    parser.add_argument("--new40-factorial", help="merged results_2x2.csv of the 40-env factorial runs")
    parser.add_argument("--factorial-release", help="results_2x2.csv of the published model on the release envs (C3 rows)")
    parser.add_argument("--fig-dir", default=".")
    args = parser.parse_args()
    fig_dir = Path(args.fig_dir)
    base = pd.read_csv(args.baseline)
    base = base[(base["sweep"] == "range") & (base["num_robots"] == 32)]
    sweeps = load_pairs(args.sweeps)
    m1 = pd.read_csv(args.m1) if args.m1 else pd.DataFrame()
    if not m1.empty:
        plot_map_levels(m1[~m1["map"].str.startswith("S")], sweeps, fig_dir / "fig_map_levels_retrained.png")

    # Measure 2: conditions on both environment sets.
    def label(model: str, level: str) -> str:
        return f"{MODEL_LABELS.get(model, model)}, {'full map' if level == 'C0' else 'map at 256 m scale'}"

    release = {label("published", "C0"): base[base["controller"] == "lpac"],
               "centralized CVT": base[base["controller"] == "CentralizedCVT"]}
    for model, frame in sweeps.items():
        release[label(model, "C0")] = frame[frame["controller"] == "lpac"]
    if args.factorial_release:
        rel = pd.read_csv(args.factorial_release)
        rel = rel[(rel["cell"] == "G0C3") & (rel.get("repeat", 0) == 0)]
        if not rel.empty:
            release[label("published", "C3")] = rel
    if not m1.empty:
        for model in sorted(set(m1["model"])):
            rows = m1[(m1["model"] == model) & (m1["map"] == "C3")]
            if not rows.empty:
                key = label(model, "C3")
                if key in release:  # published C3 rows come from two files: keep one row per (env, range)
                    rows = pd.concat([release[key], rows]).drop_duplicates(subset=["env", "comm_range"], keep="last")
                release[key] = rows
    new40 = {}
    for model, frame in load_pairs(args.new40_sweeps).items():
        new40[label(model, "C0")] = frame[frame["controller"] == "lpac"]
        if model == "published":
            new40["centralized CVT"] = frame[frame["controller"] == "CentralizedCVT"]
    fact40 = pd.read_csv(args.new40_factorial) if args.new40_factorial else pd.DataFrame()
    scale40 = None
    if not fact40.empty:
        for model in sorted(set(fact40["model"])):
            rows = fact40[(fact40["model"] == model) & (fact40["map"] == "C3")]
            if not rows.empty:
                new40[label(model, "C3")] = rows
        scale40 = fact40[(fact40["model"] == "published") & fact40["map"].str.startswith("S")]
    plot_new_envs(release, new40, fig_dir / "fig_new_envs.png")

    scale11 = m1[(m1["model"] == "published") & m1["map"].str.startswith("S")] if not m1.empty else pd.DataFrame()
    as_is = {"published, full map, 11 envs": (float(base[(base["controller"] == "lpac") & (base["comm_range"] == 1024)]["ratio_mean"].mean()),
                                              COLORS["lpac"], "--"),
             "centralized CVT (11 envs)": (float(base[(base["controller"] == "CentralizedCVT")]["ratio_mean"].mean()),
                                           COLORS["CentralizedCVT"], ":")}
    plot_map_scale(scale11, scale40, as_is, fig_dir / "fig_map_scale.png")
    print("figures written to", fig_dir)


if __name__ == "__main__":
    main()
