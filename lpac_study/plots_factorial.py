"""Figures of the inference-time factorial experiment (English labels, for the blog post).

    python -m lpac_study.plots_factorial --results results_2x2/results_2x2.csv \
        --baseline results/results.csv --fig-dir .

* ``fig_2x2.png``  the six cells at the main range (mean ± std across environments);
* ``fig_dose.png`` cost versus neighbour cap k for the best cell (dose-response);
* ``fig_fix.png``  the best cell across communication ranges, against the
  published behaviour and the CVT baselines of the range sweep;
* ``fig_scale256.png`` the published model, G1C2, G0C3 and the CVTs across ranges.
Each figure is drawn only when the corresponding rows exist.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plots import COLORS, ERRORBAR_STYLE, GRID, TEXT, style_axis

CELL_ORDER = ["G0C0", "G0C1", "G0C2", "G1C0", "G1C1", "G1C2"]
CELL_LABELS = {
    "G0C0": "G0 + C0\nfull graph,\nfull map",
    "G0C1": "G0 + C1\nfull graph,\nmap: k nearest",
    "G0C2": "G0 + C2\nfull graph,\nmap: 128 m",
    "G1C0": "G1 + C0\ngraph: k nearest,\nfull map",
    "G1C1": "G1 + C1\ngraph & map:\nk nearest",
    "G1C2": "G1 + C2\ngraph: k nearest,\nmap: 128 m",
}
CELL_COLORS = {"G0": COLORS["DecentralizedCVT"], "G1": COLORS["lpac"]}
BASELINE_LABELS = {"DecentralizedCVT": "Decentralized CVT", "CentralizedCVT": "Centralized CVT"}
Y_LABEL = "Coverage cost / clairvoyant cost\n(episode average)"


def cell_stats(rows: pd.DataFrame, metric: str = "ratio_mean") -> pd.DataFrame:
    grouped = rows.groupby("cell")[metric].agg(["mean", "std", "count"]).reindex(CELL_ORDER).dropna()
    return grouped


def plot_cells(rows: pd.DataFrame, baseline: pd.DataFrame, path: Path, comm_range: float, k: int) -> None:
    """Dot-and-error-bar chart of the six factorial cells at one range."""
    sel = rows[(rows["comm_range"] == comm_range) & ((rows["k"] == k) | (rows["cell"] == "G0C0"))]
    if sel.empty:
        return
    stats = cell_stats(sel)
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=150)
    x = np.arange(len(stats))
    for i, (cell, row) in enumerate(stats.iterrows()):
        ax.errorbar(i, row["mean"], yerr=row["std"], color=CELL_COLORS[cell[:2]], **ERRORBAR_STYLE)
        ax.annotate(f"{row['mean']:.2f}", (i, row["mean"]), textcoords="offset points", xytext=(10, 4), fontsize=9, color=TEXT)
    base = baseline[(baseline["comm_range"] == comm_range) & (baseline["num_robots"] == 32)]
    for ctrl, style in (("DecentralizedCVT", "--"), ("CentralizedCVT", ":")):
        value = base[base["controller"] == ctrl]["ratio_mean"].mean()
        if not np.isnan(value):
            ax.axhline(value, color=COLORS[ctrl], linestyle=style, linewidth=1.5, label=BASELINE_LABELS[ctrl])
    ax.axhline(1.0, color=COLORS["ClairvoyantCVT"], linewidth=1.5, label="Clairvoyant CVT (= 1)")
    ax.set_xticks(x)
    ax.set_xticklabels([CELL_LABELS[c] for c in stats.index], fontsize=8.5)
    n_env = int(sel.groupby("cell")["env"].nunique().min())
    style_axis(ax, "", Y_LABEL,
               f"Where does the neighbourhood break LPAC? Inference-time factorial at {comm_range:.0f} m, "
               f"32 robots, k = {k}\n{n_env} environments; bars = std across environments; "
               "blue = capped graph (G1), orange = full graph (G0)")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_dose(rows: pd.DataFrame, path: Path, comm_range: float, cell: str) -> None:
    """Cost versus the neighbour cap k for one cell; the uncapped model is the last point."""
    capped = rows[(rows["comm_range"] == comm_range) & (rows["cell"] == cell)]
    full = rows[(rows["comm_range"] == comm_range) & (rows["cell"] == "G0C0")]
    if capped["k"].nunique() < 2:
        return
    stats = capped.groupby("k")["ratio_mean"].agg(["mean", "std"]).sort_index()
    degrees = capped.groupby("k")["degree_mean"].mean()
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    positions = list(range(len(stats)))
    ax.errorbar(positions, stats["mean"], yerr=stats["std"], color=COLORS["lpac"], label=f"{cell}: k nearest neighbours", **ERRORBAR_STYLE)
    labels = [f"k = {int(k)}\n({degrees[k]:.1f} nb.)" for k in stats.index]
    if not full.empty:
        positions.append(len(positions))
        ax.errorbar(positions[-1], full["ratio_mean"].mean(), yerr=full["ratio_mean"].std(), color=COLORS["DecentralizedCVT"],
                    label="G0C0: all neighbours in range", **ERRORBAR_STYLE)
        labels.append(f"all\n({full['degree_mean'].mean():.1f} nb.)")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=9)
    n_env = int(capped.groupby("k")["env"].nunique().min())
    style_axis(ax, "Neighbour cap k (mean degree actually seen by the GNN)", Y_LABEL,
               f"Dose-response of the neighbour cap at {comm_range:.0f} m, 32 robots, {n_env} environments\n"
               "bars = std across environments")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_fix(rows: pd.DataFrame, baseline: pd.DataFrame, path: Path, cell: str, k: int) -> None:
    """The chosen cell across ranges, next to the published model and the CVT baselines."""
    fixed = rows[(rows["cell"] == cell) & (rows["k"] == k)]
    if fixed["comm_range"].nunique() < 2:
        return
    sweep = baseline[(baseline["sweep"] == "range") & (baseline["num_robots"] == 32)]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    for ctrl, label in (("lpac", "LPAC-K3 as published (all neighbours)"),
                        ("DecentralizedCVT", "Decentralized CVT"), ("CentralizedCVT", "Centralized CVT"),
                        ("ClairvoyantCVT", "Clairvoyant CVT (= 1)")):
        s = sweep[sweep["controller"] == ctrl].groupby("comm_range")["ratio_mean"].agg(["mean", "std"]).sort_index()
        ax.errorbar(s.index, s["mean"], yerr=s["std"].fillna(0), color=COLORS[ctrl], label=label, alpha=0.55 if ctrl == "lpac" else 1.0,
                    linestyle="--" if ctrl == "lpac" else "-", **ERRORBAR_STYLE)
    f = fixed.groupby("comm_range")["ratio_mean"].agg(["mean", "std"]).sort_index()
    ax.errorbar(f.index, f["mean"], yerr=f["std"].fillna(0), color=COLORS["lpac"], label=f"LPAC-K3 + {cell}, k = {k}", **ERRORBAR_STYLE)
    n_env = int(fixed.groupby("comm_range")["env"].nunique().min())
    style_axis(ax, "Communication range at test time (m)", Y_LABEL,
               f"Inference-time fix across ranges — 32 robots, {n_env} environments, no retraining\n"
               "bars = std across environments")
    ax.set_xticks(sorted(sweep["comm_range"].unique()))
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


SCALE_SERIES = (  # (cell or controller, label, colour, linestyle)
    ("lpac", "LPAC-K3 as published (all neighbours in range)", COLORS["lpac"], "--"),
    ("G1C2", "LPAC-K3 + G1C2: map at the 128 m scale, graph capped (k = 4)", "#184f95", "-"),
    ("G0C3", "LPAC-K3 + G0C3: map at the 256 m scale, full graph", "#3987e5", "-"),
    ("DecentralizedCVT", "Decentralized CVT", COLORS["DecentralizedCVT"], "-"),
    ("CentralizedCVT", "Centralized CVT", COLORS["CentralizedCVT"], "-"),
    ("ClairvoyantCVT", "Clairvoyant CVT (= 1)", COLORS["ClairvoyantCVT"], "-"),
)


def plot_scale(rows: pd.DataFrame, baseline: pd.DataFrame, path: Path, k: int = 4) -> None:
    """Published model, two inference-time map fixes and the CVT baselines across ranges."""
    nominal = rows[(rows.get("repeat", 0) == 0) & ((rows["k"] == k) | rows["cell"].str.startswith("G0"))]
    sweep = baseline[(baseline["sweep"] == "range") & (baseline["num_robots"] == 32)]
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=150)
    drawn = 0
    for name, label, color, style in SCALE_SERIES:
        source = sweep[sweep["controller"] == name] if name in COLORS else nominal[nominal["cell"] == name]
        if source.empty:
            continue
        s = source.groupby("comm_range")["ratio_mean"].agg(["mean", "std"]).sort_index()
        ax.errorbar(s.index, s["mean"], yerr=s["std"].fillna(0), color=color, label=label, linestyle=style,
                    alpha=0.6 if name == "lpac" else 1.0, **ERRORBAR_STYLE)
        drawn += 1
    if drawn == 0:
        plt.close(fig)
        return
    n_env = int(nominal.groupby(["comm_range", "cell"])["env"].nunique().min())
    style_axis(ax, "Communication range at test time (m)", Y_LABEL,
               f"Neighbour map drawn at a fixed scale — 32 robots, {n_env} environments, no retraining\n"
               "bars = std across environments")
    ax.set_xticks(sorted(sweep["comm_range"].unique()))
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", required=True, help="results_2x2.csv")
    parser.add_argument("--baseline", required=True, help="results.csv of the range sweep")
    parser.add_argument("--fig-dir", default=".")
    parser.add_argument("--comm-range", type=float, default=1024.0)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--best-cell", default="G1C2", help="cell drawn across ranges in fig_fix.png")
    parser.add_argument("--dose-cell", default="G1C1", help="cell whose k is varied in fig_dose.png")
    args = parser.parse_args(argv)
    rows = pd.read_csv(args.results)
    baseline = pd.read_csv(args.baseline)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_cells(rows, baseline, fig_dir / "fig_2x2.png", args.comm_range, args.k)
    plot_dose(rows, fig_dir / "fig_dose.png", args.comm_range, args.dose_cell)
    plot_fix(rows, baseline, fig_dir / "fig_fix.png", args.best_cell, args.k)
    plot_scale(rows, baseline, fig_dir / "fig_scale256.png", args.k)
    print("figures written to", fig_dir)


if __name__ == "__main__":
    main()
