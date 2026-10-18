"""Reproduction of Fig. 20 of Agarwal et al. (T-RO 2025): one model per training range, each
evaluated at its own range (the "diagonal"), against the CVT baselines.

    python -m lpac_study.plots_fig20 --baseline results/results.csv \
        --models 128=results_train/results_sweep100k_gpu_epoch13.csv 256=results_train/results_sweep256_best.csv \
                 512=results_fig20/r512_s0_11.csv 1024=results_fig20/r1024_s0_11.csv \
        --title "11 release environments" --out fig_fig20.png

Each ``--models`` entry is ``training range=merged sweep file`` of the model trained at that range
and evaluated at every range. Left: the diagonal with the CVTs (what the paper plots). Right: the
full matrix, one curve per model across test ranges, the diagonal points marked.
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .plots import COLORS, ERRORBAR_STYLE, TEXT, style_axis

Y_LABEL = "Coverage cost / clairvoyant cost\n(episode average)"
MODEL_COLORS = {128: "#2a78d6", 256: "#1baf7a", 512: "#eda100", 1024: "#eb6834"}


def env_table(path: str, controller: str = "lpac") -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[(frame["controller"] == controller) & (frame["num_robots"] == 32)]
    if "sweep" in frame:
        frame = frame[frame["sweep"] == "range"]
    return frame.pivot_table(index="env", columns="comm_range", values="ratio_mean")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", required=True, help="sweep holding the CVT rows of the environments")
    parser.add_argument("--models", nargs="+", required=True, help="range=path pairs")
    parser.add_argument("--title", default="11 release environments")
    parser.add_argument("--out", default="fig_fig20.png")
    args = parser.parse_args()
    models = {}
    for item in args.models:
        r, path = item.split("=", 1)
        models[float(r)] = env_table(path)
    base = pd.read_csv(args.baseline)
    base = base[(base["sweep"] == "range") & (base["num_robots"] == 32)]
    cvt = {c: base[base["controller"] == c].pivot_table(index="env", columns="comm_range", values="ratio_mean")
           for c in ("CentralizedCVT", "DecentralizedCVT")}

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 4.9), dpi=150, sharey=True)
    ranges = sorted(models)
    diag = [(r, models[r][r].mean(), models[r][r].std()) for r in ranges if r in models[r]]
    ax_a.errorbar([d[0] for d in diag], [d[1] for d in diag], yerr=[d[2] for d in diag], color=COLORS["lpac"],
                  label="LPAC trained at the test range (one model per range, seed 0)", **ERRORBAR_STYLE)
    for c, label, style in (("CentralizedCVT", "centralized CVT", ":"), ("DecentralizedCVT", "decentralized CVT", "-.")):
        t = cvt[c]
        cols = [r for r in ranges if r in t.columns]
        ax_a.errorbar(cols, [t[r].mean() for r in cols], yerr=[t[r].std() for r in cols], color=COLORS[c], linestyle=style,
                      label=label, **ERRORBAR_STYLE)
    style_axis(ax_a, "Communication range (training = test), m", Y_LABEL, "The diagonal: each model at its own range (as in Fig. 20)")
    ax_a.set_xticks([128, 256, 512, 1024])
    ax_a.set_ylim(bottom=0)
    ax_a.legend(frameon=False, fontsize=8)
    for r in ranges:
        t = models[r]
        cols = sorted(t.columns)
        ax_b.plot(cols, [t[c].mean() for c in cols], color=MODEL_COLORS.get(int(r), "#444444"), marker="o", markersize=5,
                  linewidth=1.8, label=f"trained at {r:.0f} m")
        if r in t:
            ax_b.plot([r], [t[r].mean()], marker="s", markersize=11, markerfacecolor="none", markeredgewidth=2,
                      color=MODEL_COLORS.get(int(r), "#444444"))
    ax_b.axhline(cvt["CentralizedCVT"].mean().mean(), color=COLORS["CentralizedCVT"], linestyle=":", linewidth=1.5)
    style_axis(ax_b, "Communication range at test time (m)", "", "Every model at every range (squares = diagonal)")
    ax_b.set_xticks([128, 256, 512, 768, 1024])
    ax_b.legend(frameon=False, fontsize=8)
    fig.suptitle(f"LPAC-K3, authors' recipe, 100,000 samples per training range — {args.title}, bars = std across environments",
                 fontsize=10, color=TEXT)
    fig.tight_layout()
    fig.savefig(args.out)
    plt.close(fig)
    print("diagonal (range: mean ± std across envs):", ", ".join(f"{r:.0f} m: {m:.2f} ± {s:.2f}" for r, m, s in diag))
    for c in cvt:
        print(c, ", ".join(f"{r:.0f} m: {cvt[c][r].mean():.2f}" for r in ranges if r in cvt[c].columns))
    print("figure written to", args.out)


if __name__ == "__main__":
    main()
