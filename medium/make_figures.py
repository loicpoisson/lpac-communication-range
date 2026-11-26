"""Figures of the Medium post, drawn from the site's data bundle (site/data/*.json).

    python medium/make_figures.py            # writes medium/figures/*.png and the hook GIF

Nothing is measured here: every number comes from the bundle, itself built from the tracked CSVs.
Cost ratio = episode-mean coverage cost / clairvoyant controller's, lower is better.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "site" / "data"
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)

# Okabe-Ito, colour-blind safe; one colour per entity, fixed across figures.
C = {"published": "#D55E00", "fixed": "#0072B2", "v4": "#009E73", "cvt": "#7f7f7f",
     "lpac256": "#CC79A7", "v4_128": "#56B4E9", "other": "#E69F00", "v1": "#000000"}
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 200, "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False})
RANGES = [128, 256, 512, 768, 1024]


def load(name):
    return json.loads((DATA / f"{name}.json").read_text(encoding="utf8"))


def table(d):
    return pd.DataFrame(d["rows"], columns=d["columns"])


costs = load("costs")
cost = table(costs)
cost["model"] = cost.run_id.map(lambda r: costs["runs"][r]["model"])
cost["primary"] = cost.run_id.map(lambda r: costs["runs"][r].get("primary", True))
cost = cost[cost.primary]


def curve(model, env_set="new40", map_level=None, graph="G0"):
    """Mean over environments within each run, then mean and sd over runs, per range."""
    d = cost[(cost.model == model) & (cost.env_set == env_set) & (cost.graph_level == graph)]
    if map_level is not None:
        d = d[d.map_level.isin([map_level] if isinstance(map_level, str) else map_level)]
    per_run = d.groupby(["run_id", "range_m"]).cost_ratio.mean().reset_index()
    return per_run.groupby("range_m").cost_ratio.agg(["mean", "std", "count"]).reindex(RANGES)


def range_axis(ax):
    ax.set_xscale("log", base=2)
    ax.set_xticks(RANGES)
    ax.set_xticklabels([str(r) for r in RANGES])
    ax.set_xlabel("communication range (m)")
    ax.set_ylabel("cost / clairvoyant optimum (lower is better)")


def plot_curve(ax, s, label, color, **kw):
    s = s.dropna(subset=["mean"])
    ax.plot(s.index, s["mean"], marker="o", color=color, label=label, lw=2, ms=5, **kw)
    if (s["count"] > 1).any():
        ax.fill_between(s.index, s["mean"] - s["std"].fillna(0), s["mean"] + s["std"].fillna(0), color=color, alpha=0.15)


# 1. The anomaly -------------------------------------------------------------------------------
def fig_anomaly():
    fig, ax = plt.subplots(figsize=(7, 4.3))
    plot_curve(ax, curve("published", map_level="C0"), "LPAC as published (40 new environments)", C["published"])
    plot_curve(ax, curve("published", "released11", "C0"), "LPAC as published (11 released environments)",
               C["published"], ls="--", alpha=0.6)
    cvt = curve("centralized_cvt", map_level=None)["mean"].dropna().mean()
    ax.axhline(cvt, color=C["cvt"], ls=":", lw=1.5)
    ax.text(1024, cvt + 0.08, "centralized CVT (sees the whole team)", ha="right", color=C["cvt"], fontsize=9)
    range_axis(ax)
    ax.set_title("More communication, worse coverage", loc="left", fontweight="bold")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_anomaly.png")
    plt.close(fig)


# 2. Factorial at 1024 m -----------------------------------------------------------------------
def fig_factorial():
    f = load("factorial")
    d = table(f["cells"])
    d = d[(d.range_m == 1024) & (d.repeat == 0) & (d.env_set == "released11")]
    m = d.groupby(["graph_level", "map_level"]).cost_ratio.agg(["mean", "sem"]).reset_index()
    maps = [("C0", "map as published"), ("C4", "empty map"), ("C2", "map at 128 m scale"),
            ("C3", "map at 256 m scale")]
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    w = 0.38
    for gi, (g, glabel, col) in enumerate([("G0", "full graph", C["published"]), ("G1", "graph capped to 4 nearest", C["other"])]):
        vals = [m[(m.graph_level == g) & (m.map_level == ml)] for ml, _ in maps]
        y = [v["mean"].iloc[0] if len(v) else np.nan for v in vals]
        e = [v["sem"].iloc[0] if len(v) else 0 for v in vals]
        x = np.arange(len(maps)) + (gi - 0.5) * w
        ax.bar(x, y, w * 0.95, yerr=e, color=col, label=glabel, capsize=3, edgecolor="white")
        for xi, yi in zip(x, y):
            if not np.isnan(yi):
                ax.text(xi, yi + 0.15, f"{yi:.2f}", ha="center", fontsize=8.5)
    ax.set_xticks(np.arange(len(maps)))
    ax.set_xticklabels([l for _, l in maps])
    ax.set_ylabel("cost / optimum at 1024 m")
    ax.set_title("Same weights, same simulator: change one input at a time", loc="left", fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_factorial.png")
    plt.close(fig)


# 3. What the CNN sees ---------------------------------------------------------------------------
def fig_maps():
    r = load("replays")
    run = next(x for x in r["runs"] if x["id"] == "published_1024")
    maps = run["maps"]
    k = 0  # first stored step (step 20)
    step = maps["steps"][k] if "steps" in maps else 20
    levels = [("C0", "as published\n1024 m scale, 64 m per cell"), ("C3", "fixed 256 m scale\nfar robots dropped"),
              ("K256", "fixed 256 m scale\nfar robots pushed to the border")]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.9), gridspec_kw={"wspace": 0.08})
    vmax = 0
    imgs = []
    for lvl, _ in levels:
        a = np.array(maps[lvl][k], dtype=float).reshape(2, 32, 32)
        mag = np.hypot(a[0], a[1])
        imgs.append(mag)
        vmax = max(vmax, mag.max())
    for ax, (lvl, title), img in zip(axes, levels, imgs):
        im = ax.imshow(img, origin="lower", cmap="magma", vmin=0, vmax=vmax)
        ax.set_title(title, fontsize=9.5)
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(1, 29, f"max {img.max():.2f}\n{int((img > 0).sum())} cells", color="white", fontsize=8.5, va="top")
    fig.colorbar(im, ax=axes, shrink=0.8, label="|value| of the neighbour cell")
    fig.suptitle(f"Robot 0's neighbour map at 1024 m (step {step}), three ways", x=0.02, ha="left", fontweight="bold")
    fig.savefig(OUT / "fig3_maps.png", bbox_inches="tight")
    plt.close(fig)


# 4. Fixes and variants ---------------------------------------------------------------------------
def fig_variants():
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    plot_curve(ax, curve("published", map_level="C0"), "LPAC as published", C["published"])
    fixed = curve("published", map_level="C3")
    fixed.loc[256] = curve("published", map_level="C0").loc[256]  # at 256 m the fixed map is the published map
    plot_curve(ax, fixed.sort_index(), "same weights, fixed 256 m map (no retraining)", C["fixed"])
    plot_curve(ax, curve("lpac256", map_level="C0"), "LPAC retrained at 256 m", C["lpac256"])
    plot_curve(ax, curve("V4_128", map_level=["none", "C0"]), "V4 bounded edges, trained at 128 m", C["v4_128"])
    plot_curve(ax, curve("V4mixed", map_level=["none", "C0"]), "V4 bounded edges, trained on mixed ranges", C["v4"])
    range_axis(ax)
    ax.set_ylim(1.5, 3.2)
    ax.text(1024, 3.12, "published at 1024 m: 6.15, off the chart ↑", ha="right", fontsize=8.5, color=C["published"])
    ax.set_title("Fixing the input, then removing it", loc="left", fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_variants.png")
    plt.close(fig)


# 5. Heterogeneous teams ---------------------------------------------------------------------------
def fig_hetero():
    h = load("hetero")
    d = table(h)
    d["model"] = d.run_id.str.replace(r"_s\d+$", "", regex=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    spec = [("V4mixed", None, "transmission", "V4 mixed, transmission rule", C["v4"], "-"),
            ("V4mixed", None, "reception", "V4 mixed, reception rule", C["v4"], "--"),
            ("published", "asis", "reception", "LPAC as published, reception rule", C["published"], "--"),
            ("published", "asis", "transmission", "LPAC as published, transmission rule", C["published"], "-")]
    for model, mp, sem, label, col, ls in spec:
        s = d[(d.model == model) & (d.semantics == sem)]
        if mp:
            s = s[s["map"] == mp]
        s = s[s["mode"].isna()]  # mode is only set for the mixture-of-specialists runs
        per_run = s.groupby(["run_id", "f"]).cost_ratio.mean().reset_index()
        g = per_run.groupby("f").cost_ratio.mean()
        if len(g):
            ax.plot(g.index, g.values, marker="o", ls=ls, color=col, label=label, lw=2)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels(["0", "1/4", "1/2", "3/4", "1"])
    ax.set_xlabel("fraction of the team that hears at 1024 m (the rest at 128 m)")
    ax.set_ylabel("cost / clairvoyant optimum")
    ax.set_title("Who hears whom? Two link rules, two models", loc="left", fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_hetero.png")
    plt.close(fig)


# 6. GATv2 training curves ---------------------------------------------------------------------------
def fig_curves():
    cu = load("curves")["runs"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    spec = [("V2", "GATv2, plain stack (V2)", C["published"], "-"), ("V2B", "GATv2 + bounded edge features (V2B)", C["other"], "--"),
            ("V2res", "GATv2 in residual blocks + LayerNorm (V2res)", C["v4"], "-"),
            ("V1", "TAGConv, no map (V1)", C["v1"], ":")]
    for key, label, col, ls in spec:
        r = cu[key]
        e = np.array(r["epoch"]); t = np.array(r["train"], dtype=float)
        sel = e <= 30
        ax.plot(e[sel], t[sel], ls=ls, color=col, lw=2, label=label)
    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss (MSE on the normalized action)")
    ax.set_title("GATv2 was not too weak: it could not optimize", loc="left", fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig6_gatv2.png")
    plt.close(fig)


# 0. Hook animation --------------------------------------------------------------------------------
def gif_hook(every=2):
    from matplotlib.animation import FuncAnimation, PillowWriter
    r = load("replays")
    size = r["env"]["idf"]["size"]
    idf = np.array(r["env"]["idf"]["data"], dtype=float).reshape(size, size)
    W = r["env"]["world_size_m"]
    runs = {x["id"]: x for x in r["runs"]}
    panels = [("published_1024", "LPAC as published, 1024 m range", C["published"]),
              ("published_1024_C3", "same weights, fixed 256 m map", C["fixed"])]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.4))
    dots, texts = [], []
    for ax, (rid, title, col) in zip(axes, panels):
        ax.imshow(idf, origin="lower", extent=[0, W, 0, W], cmap="Greys", alpha=0.85)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.set_title(title, fontsize=10, color=col, fontweight="bold")
        p = np.array(runs[rid]["positions"][0]).reshape(-1, 2)
        dots.append(ax.scatter(p[:, 0], p[:, 1], s=22, color=col, edgecolor="white", linewidth=0.6, zorder=3))
        texts.append(ax.text(10, W - 10, "", va="top", fontsize=9, bbox=dict(fc="white", ec="none", alpha=0.8)))
    steps = r["frame_steps"]
    frames = list(range(0, len(steps), every))

    def update(i):
        for (rid, _, _), dot, txt in zip(panels, dots, texts):
            p = np.array(runs[rid]["positions"][i]).reshape(-1, 2)
            dot.set_offsets(p)
            txt.set_text(f"step {steps[i]}   cost / optimum {runs[rid]['cost_ratio'][i]:.2f}")
        return dots + texts

    fig.tight_layout()
    FuncAnimation(fig, update, frames=frames, blit=False).save(OUT / "fig0_hook.gif", writer=PillowWriter(fps=12))
    plt.close(fig)


if __name__ == "__main__":
    fig_anomaly(); fig_factorial(); fig_maps(); fig_variants(); fig_hetero(); fig_curves(); gif_hook()
    for p in sorted(OUT.iterdir()):
        print(p.name, f"{p.stat().st_size / 1e3:.0f} kB")
