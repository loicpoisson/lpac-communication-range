"""Builds figures/results.pdf for the proposal from the result CSVs of the repository."""
import csv
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RANGES = [128, 256, 512, 768, 1024]


def rows(path):
    with open(ROOT / path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def mean_ratio(table, rng, **match):
    v = [float(r["ratio_mean"]) for r in table if float(r["comm_range"]) == rng
         and all(r.get(k, "") == val for k, val in match.items())]
    assert len(v) == 11, (match, rng, len(v))
    return st.mean(v)


base = rows("results.csv")
published = [mean_ratio(base, r, controller="lpac", sweep="range") for r in RANGES]
central = [mean_ratio(base, r, controller="CentralizedCVT", sweep="range") for r in RANGES]
fact = rows("results_2x2.csv")
fixed_ranges = [256, 512, 768, 1024]
fixed = [mean_ratio(fact, r, cell="G0C3", k="4", repeat="0") for r in fixed_ranges]
seeds = [rows(p) for p in ("results_train/results_sweep100k_gpu_epoch13.csv",
                           "results_train/results_sweep100k_seed1_best.csv",
                           "results_train/results_sweep100k_seed2_best.csv")]
r128 = [[mean_ratio(s, r, controller="lpac") for r in RANGES] for s in seeds]
r256 = [mean_ratio(rows("results_train/results_sweep256_best.csv"), r, controller="lpac") for r in RANGES]
v1 = [[mean_ratio(rows(f"results_variants/V1_s{i}_11.csv"), r, controller="lpac") for r in RANGES] for i in range(3)]

plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.linewidth": 0.6})
fig, ax = plt.subplots(figsize=(3.35, 2.75))
ax.axhline(1.0, color="0.55", lw=0.8, ls=(0, (4, 3)))
ax.plot(RANGES, central, color="#1baf7a", lw=1.2, label="Centralized CVT")
r128_mean = [st.mean(c) for c in zip(*r128)]
ax.fill_between(RANGES, [min(c) for c in zip(*r128)], [max(c) for c in zip(*r128)], color="#008300", alpha=0.18, lw=0)
ax.plot(RANGES, published, color="#2a78d6", lw=1.4, marker="o", ms=3, label="LPAC published (trained 128 m)")
ax.plot(RANGES, r128_mean, color="#008300", lw=1.2, marker="o", ms=2.5, label="LPAC retrained 128 m (3 seeds)")
ax.plot(RANGES, r256, color="#6250d6", lw=1.2, marker="^", ms=3, label="LPAC retrained 256 m")
ax.plot(RANGES, [st.mean(c) for c in zip(*v1)], color="#d55181", lw=1.2, ls="--", marker="s", ms=2.5, label="V1: no neighbor map (3 seeds)")
ax.plot(fixed_ranges, fixed, color="#eda100", lw=2.0, marker="D", ms=3, label="Published + fixed 256 m map")
ax.set_xticks(RANGES)
ax.set_xlim(100, 1050)
ax.set_ylim(0, 5.6)
ax.set_xlabel("Communication range at test time (m)")
ax.set_ylabel("Cost / clairvoyant cost")
ax.text(1035, 1.08, "clairvoyant", fontsize=6.5, color="0.4", ha="right")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="y", color="0.9", lw=0.5)
fig.tight_layout(pad=0.3, rect=(0, 0.2, 1, 1))
fig.legend(*ax.get_legend_handles_labels(), fontsize=6, frameon=False, loc="lower center", ncol=2, handlelength=2.0, columnspacing=1.0, borderaxespad=0.1)
fig.savefig(Path(__file__).with_name("results.pdf"))
print("published", [round(x, 2) for x in published])
print("retrained128", [round(x, 2) for x in r128_mean], "| r256", [round(x, 2) for x in r256])
print("fixed", [round(x, 2) for x in fixed], "| V1", [round(st.mean(c), 2) for c in zip(*v1)], "| central", [round(x, 2) for x in central])
