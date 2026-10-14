"""Figures of the study, built only from measured rows of ``results.csv`` / ``rank.csv``.

    python -m lpac_study.plots --results-dir results --fig-dir .

Produces ``fig_portee.png`` (cost vs range, 4 controllers), ``fig_degre.png``
(measured degree vs range), ``fig_degre_constant.png`` (constant-degree sweep)
and ``fig_rang.png`` (effective rank per GNN layer) when the data exist.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .sweep import DEFAULT_CONSTANT_DEGREE

# Categorical palette, fixed order (validated colour-blind-safe slots).
COLORS = {
    "lpac": "#2a78d6",
    "DecentralizedCVT": "#eb6834",
    "CentralizedCVT": "#1baf7a",
    "ClairvoyantCVT": "#eda100",
}
LABELS = {
    "lpac": "LPAC-K3 (modèle publié, entraîné à 128 m)",
    "DecentralizedCVT": "CVT décentralisé",
    "CentralizedCVT": "CVT centralisé",
    "ClairvoyantCVT": "CVT clairvoyant (référence = 1)",
}
LABELS_EN = {
    "lpac": "LPAC-K3 (published model, trained at 128 m)",
    "DecentralizedCVT": "Decentralized CVT",
    "CentralizedCVT": "Centralized CVT",
    "ClairvoyantCVT": "Clairvoyant CVT (reference = 1)",
}
# Sequential blue ramp (light to dark) for ordered communication ranges.
RANGE_RAMP = ["#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]
TEXT = "#333333"
GRID = "#dddddd"
ERRORBAR_STYLE = dict(marker="o", markersize=6, linewidth=2, capsize=4)


def style_axis(ax, xlabel: str, ylabel: str, title: str) -> None:
    """Recessive grid and spines, explicit axis labels with units."""
    ax.set_xlabel(xlabel, color=TEXT)
    ax.set_ylabel(ylabel, color=TEXT)
    ax.set_title(title, fontsize=10, color=TEXT)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT)


def aggregate(frame: pd.DataFrame, value: str) -> pd.DataFrame:
    """Mean and standard deviation across environments for one metric."""
    grouped = frame.groupby(["comm_range", "num_robots", "controller"])[value]
    out = grouped.agg(["mean", "std", "count"]).reset_index()
    out["std"] = out["std"].fillna(0.0)
    return out


def count_environments(frame: pd.DataFrame) -> int:
    """Smallest number of environments behind any plotted point."""
    return int(frame.groupby(["comm_range", "num_robots", "controller"])["env"].nunique().min())


def draw_controllers(ax, stats: pd.DataFrame, labels: dict = LABELS) -> None:
    """One error-bar curve per controller, in the fixed palette order."""
    for controller, color in COLORS.items():
        rows = stats[stats["controller"] == controller].sort_values("comm_range")
        if not rows.empty:
            ax.errorbar(rows["comm_range"], rows["mean"], yerr=rows["std"], color=color,
                        label=labels[controller], **ERRORBAR_STYLE)


def plot_cost_vs_range(results: pd.DataFrame, path: Path, metric: str = "ratio_mean", lang: str = "fr") -> None:
    """Normalized cost versus communication range; error bars = std across environments."""
    sweep = results[results["sweep"] == "range"]
    if sweep.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    n_robots = int(sweep["num_robots"].iloc[0])
    n_env = count_environments(sweep)
    if lang == "en":
        draw_controllers(ax, aggregate(sweep, metric), LABELS_EN)
        style_axis(ax, "Communication range at test time (m)",
                   "Coverage cost / clairvoyant cost\n(episode average)",
                   f"Normalized cost versus communication range — {n_robots} robots, {n_env} environments\n"
                   "bars = std across environments; the model was trained at 128 m only")
    else:
        draw_controllers(ax, aggregate(sweep, metric))
        style_axis(
            ax, "Portée de communication (m)",
            "Coût de couverture / coût du clairvoyant\n(moyenne sur l'épisode)",
            f"Coût normalisé selon la portée — {n_robots} robots, {n_env} environnements\n"
            "barres = écart-type entre environnements",
        )
    ax.set_xticks(sorted(sweep["comm_range"].unique()))
    ax.legend(frameon=False, fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_degree_vs_range(results: pd.DataFrame, path: Path) -> None:
    """Mean degree of the LPAC communication graph along the trajectories."""
    rows = results[(results["sweep"] == "range") & (results["controller"] == "lpac")]
    if rows.empty:
        return
    stats = aggregate(rows, "degree_mean").sort_values("comm_range")
    n_robots = int(rows["num_robots"].iloc[0])
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    ax.errorbar(stats["comm_range"], stats["mean"], yerr=stats["std"], color=COLORS["lpac"],
                label="degré moyen mesuré sur l'edge_index fourni au GNN", **ERRORBAR_STYLE)
    ax.axhline(n_robots - 1, color=GRID, linestyle="--", linewidth=1.5)
    ax.annotate(f"graphe complet ({n_robots - 1} voisins)", (stats["comm_range"].min(), n_robots - 1),
                textcoords="offset points", xytext=(0, 4), fontsize=8, color=TEXT, ha="left")
    for x, y in zip(stats["comm_range"], stats["mean"]):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(-6, 8), fontsize=8,
                    color=TEXT, ha="right")
    style_axis(
        ax, "Portée de communication (m)", "Degré moyen par robot (voisins)",
        f"Degré réel du graphe de communication le long des trajectoires LPAC ({n_robots} robots)\n"
        "moyenne sur robots, pas et environnements ; barres = écart-type entre environnements",
    )
    ax.set_xticks(sorted(rows["comm_range"].unique()))
    ax.set_ylim(0, n_robots + 1)
    ax.legend(frameon=False, fontsize=9, loc="center left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def select_constant_degree(results: pd.DataFrame) -> pd.DataFrame:
    """Rows of the constant-degree sweep.

    The (range, robots) pairs are those labelled ``constant_degree`` plus any
    pair of the default list that the range sweep already covers (for example
    (256 m, 32 robots) is shared by both sweeps and is stored once).
    """
    labelled = results[results["sweep"] == "constant_degree"]
    pairs = {(float(r), int(n)) for r, n in zip(labelled["comm_range"], labelled["num_robots"])}
    pairs |= {(float(p.split(":")[0]), int(p.split(":")[1])) for p in DEFAULT_CONSTANT_DEGREE}
    mask = [(float(r), int(n)) in pairs for r, n in zip(results["comm_range"], results["num_robots"])]
    return results[mask]


def plot_constant_degree(results: pd.DataFrame, path: Path, metric: str = "ratio_mean") -> None:
    """Range sweep at (approximately) constant degree, team size varying with range."""
    sweep = select_constant_degree(results)
    if sweep.empty:
        return
    degrees = aggregate(sweep[sweep["controller"] == "lpac"], "degree_mean").sort_values("comm_range")
    fig, (ax, ax_deg) = plt.subplots(2, 1, figsize=(8, 7), dpi=150, sharex=True,
                                     gridspec_kw={"height_ratios": [3, 1.3]})
    draw_controllers(ax, aggregate(sweep, metric))
    ax_deg.errorbar(degrees["comm_range"], degrees["mean"], yerr=degrees["std"], color=COLORS["lpac"],
                    **ERRORBAR_STYLE)
    for _, row in degrees.iterrows():
        ax_deg.annotate(f"{row['mean']:.1f}", (row["comm_range"], row["mean"]),
                        textcoords="offset points", xytext=(8, 6), fontsize=8, color=TEXT)
    pairs = sweep[["comm_range", "num_robots"]].drop_duplicates().sort_values("comm_range")
    ax_deg.set_xticks(list(pairs["comm_range"]))
    ax_deg.set_xticklabels([f"{int(r)} m\n{int(n)} robots" for r, n in zip(pairs["comm_range"], pairs["num_robots"])])
    style_axis(
        ax, "", "Coût / coût du clairvoyant\n(moyenne sur l'épisode)",
        f"Balayage à degré approximativement constant — {count_environments(sweep)} environnements\n"
        "barres = écart-type entre environnements",
    )
    style_axis(ax_deg, "Portée de communication et taille de l'équipe", "Degré moyen\n(LPAC)", "")
    ax.legend(frameon=False, fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_effective_rank(rank: pd.DataFrame, results: pd.DataFrame, path: Path) -> None:
    """Effective rank of each TAGConv layer output, one curve per communication range."""
    main = results[(results["sweep"] == "range") & (results["controller"] == "lpac")]
    if rank.empty or main.empty:
        return
    n_robots = int(main["num_robots"].iloc[0])
    rank = rank[rank["num_robots"] == n_robots]
    stats = rank.groupby(["comm_range", "layer"])["erank_mean"].agg(["mean", "std"]).reset_index()
    ranges = sorted(stats["comm_range"].unique())
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    for color, comm_range in zip(RANGE_RAMP, ranges):
        rows = stats[stats["comm_range"] == comm_range].sort_values("layer")
        ax.errorbar(rows["layer"] + 1, rows["mean"], yerr=rows["std"].fillna(0.0), color=color,
                    label=f"{int(comm_range)} m", **ERRORBAR_STYLE)
    n_env = int(rank.groupby("comm_range")["env"].nunique().min())
    style_axis(
        ax, "Couche TAGConv (sortie, après ReLU)",
        "Rang effectif des représentations des robots\n(exp. de l'entropie des valeurs singulières)",
        f"Rang effectif par couche selon la portée — LPAC, {n_robots} robots, {n_env} environnements\n"
        f"moyenne sur les pas de temps ; barres = écart-type entre environnements ; maximum = {n_robots}",
    )
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_ylim(bottom=0)
    ax.legend(title="Portée", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def load_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results", help="folder holding results.csv and rank.csv")
    parser.add_argument("--fig-dir", default=".", help="where PNG figures are written")
    args = parser.parse_args(argv)
    results_dir, fig_dir = Path(args.results_dir), Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    results = load_table(results_dir / "results.csv")
    rank = load_table(results_dir / "rank.csv")
    if results.empty:
        raise SystemExit(f"No results.csv in {results_dir}")
    plot_cost_vs_range(results, fig_dir / "fig_portee.png")
    plot_cost_vs_range(results, fig_dir / "fig_range_en.png", lang="en")
    plot_degree_vs_range(results, fig_dir / "fig_degre.png")
    plot_constant_degree(results, fig_dir / "fig_degre_constant.png")
    plot_effective_rank(rank, results, fig_dir / "fig_rang.png")
    print("figures written to", fig_dir)


if __name__ == "__main__":
    main()
