"""Builds the milestone figures (ranges.pdf, hetero.pdf) and prints the tables, from the result CSVs of the repository."""
import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RANGES = [128, 256, 512, 768, 1024]


def load(path, **match):
    """{(env, range): ratio_mean} for the rows matching `match` (default: controller lpac when present)."""
    out = {}
    with open(ROOT / path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if "controller" in r and "controller" not in match and r["controller"] != "lpac":
                continue
            if all(r.get(k) == v for k, v in match.items()):
                out[(int(r["env"]), int(float(r["comm_range"])))] = float(r["ratio_mean"])
    return out


def recipe(paths, **match):
    """Per-range mean over environments, averaged over runs (one dict per run)."""
    runs = [load(p, **match) for p in paths]
    return {r: st.mean(st.mean(v for (e, rr), v in d.items() if rr == r) for d in runs)
            for r in RANGES if all(any(rr == r for (_, rr) in d) for d in runs)}


def seeds(fmt, n):
    return [fmt.format(s=s, n=n) for s in ("_s0", "_s1", "_s2")]


def table(n):
    t = {}
    t["LPAC published (trained 128 m)"] = recipe(["results.csv" if n == 11 else "results_new40/results_sweep_published.csv"], sweep="range")
    t["Centralized CVT"] = recipe(["results.csv" if n == 11 else "results_new40/results_sweep_published.csv"], controller="CentralizedCVT", sweep="range")
    t["Published + fixed 256 m map (inference)"] = recipe(["results_2x2.csv" if n == 11 else "results_new40/results_2x2.csv"], cell="G0C3", k="4", repeat="0", **({} if n == 11 else {"model": "published"}))
    if n == 40:  # the 40-env C3 run at 128 m predates the min(R, scale) mask fix: drop it
        t["Published + fixed 256 m map (inference)"].pop(128, None)
    if n == 11:
        t["LPAC retrained 256 m"] = recipe(["results_train/results_sweep256_best.csv"])
    t["A: fixed map in training, mixed"] = recipe([f"results_A/A_mixed{s}_{n}.csv" for s in ("", "_s1", "_s2")], repeat="0")
    t["V3: PointNet edges (trained 128 m)"] = recipe([f"results_variants/V3{s}_{n}.csv" for s in ("_s0", "_s1", "_s2")])
    t["V4: bounded edges (trained 128 m)"] = recipe([f"results_V45/V4_128{s}_{n}.csv" for s in ("_s0", "_s1", "_s2")])
    t["V4: bounded edges, mixed"] = recipe([f"results_V45/V4mixed{s}_{n}.csv" for s in ("_s0", "_s1", "_s2")])
    t["V5: multi-scale maps, mixed (6 runs)"] = recipe([f"results_V45/V5mixed{s}{v}_{n}.csv" for s in ("_s0", "_s1", "_s2") for v in ("", "_v2")])
    t["V6: V5 + V4, mixed"] = recipe([f"results_V45/V6mixed{s}_{n}.csv" for s in ("_s0", "_s1", "_s2")])
    return t


T = {n: table(n) for n in (11, 40)}
for n in (11, 40):
    print(f"--- {n} envs")
    for k, v in T[n].items():
        print(f"{k:42s}", " ".join(f"{v.get(r, float('nan')):.2f}" for r in RANGES))

plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.linewidth": 0.6})
STYLE = {
    "Centralized CVT": dict(color="#1baf7a", lw=1.1),
    "LPAC published (trained 128 m)": dict(color="#2a78d6", lw=1.3, marker="o", ms=3),
    "Published + fixed 256 m map (inference)": dict(color="#eda100", lw=1.8, marker="D", ms=3),
    "A: fixed map in training, mixed": dict(color="#d55181", lw=1.2, ls="--", marker="s", ms=2.5),
    "V5: multi-scale maps, mixed (6 runs)": dict(color="#6250d6", lw=1.2, ls="-.", marker="^", ms=3),
    "V4: bounded edges (trained 128 m)": dict(color="#0b6065", lw=1.0, ls=":", marker="v", ms=3),
    "V4: bounded edges, mixed": dict(color="#0b6065", lw=1.8, marker="o", ms=3),
}
fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.5), gridspec_kw={"width_ratios": [1, 1.25]})
for ax, ymax, keys in ((axes[0], 5.6, list(STYLE)), (axes[1], 2.6, [k for k in STYLE if k not in ("LPAC published (trained 128 m)",)])):
    ax.axhline(1.0, color="0.55", lw=0.8, ls=(0, (4, 3)))
    for k in keys:
        v = T[40][k]
        xs = [r for r in RANGES if r in v]
        ax.plot(xs, [v[r] for r in xs], label=k, **STYLE[k])
    ax.set_xticks(RANGES)
    ax.set_xlim(100, 1050)
    ax.set_xlabel("Communication range at test time (m)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="0.9", lw=0.5)
axes[0].set_ylim(0, 6.4)
axes[1].set_ylim(1.6, 2.6)
axes[0].set_ylabel("Cost / clairvoyant cost")
axes[0].set_title("Full scale", fontsize=8)
axes[1].set_title("Zoom on the trained remedies", fontsize=8)
fig.tight_layout(pad=0.3, rect=(0, 0.2, 1, 1))
fig.legend(*axes[0].get_legend_handles_labels(), fontsize=6.5, frameon=False, loc="lower center", ncol=3, columnspacing=1.2)
fig.savefig(OUT / "ranges.pdf")

# Heterogeneous teams: a fraction f of the robots hears at 1024 m, the rest at 128 m.
def hetero(path, **match):
    per = defaultdict(list)
    with open(ROOT / path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if all(r.get(k) == v for k, v in match.items()):
                per[float(r["fraction"])].append(float(r["ratio_mean"]))
    return {f: st.mean(v) for f, v in sorted(per.items())}

H = {
    "LPAC published": (hetero("results_hetero/results_hetero8.csv", map="asis", model="published"), dict(color="#2a78d6", marker="o")),
    "Published + fixed 256 m map": (hetero("results_hetero/results_hetero8.csv", map="fixed256", model="published"), dict(color="#eda100", marker="D")),
    "V4 bounded edges, mixed (3 seeds)": (hetero("results_V45/results_hetero_V4mixed_3seeds.csv"), dict(color="#0b6065", marker="o", lw=1.8)),
}
print("--- hetero")
for k, (v, _) in H.items():
    print(f"{k:36s}", " ".join(f"{f:.3f}:{x:.2f}" for f, x in v.items()))
fig, ax = plt.subplots(figsize=(3.35, 2.3))
for k, (v, sty) in H.items():
    ax.plot(list(v), list(v.values()), label=k, ms=3, **{"lw": 1.3, **sty})
ax.set_xlabel("Fraction of robots at 1024 m (others at 128 m)")
ax.set_ylabel("Cost / clairvoyant cost")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="y", color="0.9", lw=0.5)
ax.legend(fontsize=6.5, frameon=False)
fig.tight_layout(pad=0.6)
fig.savefig(OUT / "hetero.pdf", bbox_inches="tight", pad_inches=0.02)
