"""Analysis of the final-report runs (journal section 21).

    python -m lpac_study.final_runs > results_final/analysis.txt

Reads the merged CSVs of results_final/ and the earlier runs they extend. Cost = ratio_mean, the
episode-mean coverage cost divided by the clairvoyant controller's (lower is better). "Paired"
means the per-environment difference, seed-averaged first when several seeds exist; reported as
mean, its standard error over environments, and the number of environments where the first model
is better (lower).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
F = ROOT / "results_final"
RANGES = [128, 256, 512, 768, 1024]
FRACTIONS = [0.0, 0.25, 0.5, 0.75, 1.0]


def sweep(path, controller="lpac"):
    d = pd.read_csv(ROOT / path)
    if "sweep" in d:
        d = d[d.sweep == "range"]
    d = d[(d.controller == controller) & (d.num_robots == 32)]
    return d.set_index(["env", "comm_range"]).ratio_mean.unstack()  # env x range


def runs(paths):
    return [sweep(p) for p in paths]


V4MIX = {s: {n: (f"results_V45/V4mixed_s{s}_{n}.csv" if s < 3 else f"results_final/V4mixed_s{s}_{n}.csv")
             for n in (11, 40)} for s in range(6)}
LPAC256 = {0: {11: "results_train/results_sweep256_best.csv", 40: "results_new40/results_sweep_r256.csv"},
           **{s: {n: f"results_final/LPAC256_s{s}_{n}.csv" for n in (11, 40)} for s in (1, 2)}}
LPAC128_40 = "results_new40/results_sweep_r128_s0.csv"
PUBLISHED = {11: "results/results.csv", 40: "results_new40/results_sweep_published.csv"}


def curve_line(label, frames):
    per_run = pd.DataFrame({i: f.mean() for i, f in enumerate(frames)})  # range x run
    cells = []
    for r in RANGES:
        if r in per_run.index and per_run.loc[r].notna().any():
            v = per_run.loc[r].dropna()
            cells.append(f"{v.mean():.2f}" + (f" ± {v.std():.2f}" if len(v) > 1 else ""))
        else:
            cells.append("n/a")
    return f"| {label} | " + " | ".join(cells) + " |"


def per_run_line(label, frames):
    cells = []
    for r in RANGES:
        vals = [f"{f[r].mean():.2f}" for f in frames if r in f]
        cells.append(" / ".join(vals) if vals else "n/a")
    return f"| {label} | " + " | ".join(cells) + " |"


def avg(frames):
    return sum(frames) / len(frames)


def paired(a: pd.Series, b: pd.Series) -> str:
    d = (a - b).dropna()
    return f"{d.mean():+.3f} (SE {d.std(ddof=1) / np.sqrt(len(d)):.3f}, better in {(d < 0).sum()}/{len(d)})"


def header(title):
    print(f"\n## {title}\n")
    print("| | " + " | ".join(f"{r} m" for r in RANGES) + " |")
    print("|---" * (len(RANGES) + 1) + "|")


# A. V4 mixed with six seeds ----------------------------------------------------------------------
def section_a():
    for n in (40, 11):
        header(f"A. V4 mixed ranges, {n} environments")
        old = runs([V4MIX[s][n] for s in range(3)])
        new = runs([V4MIX[s][n] for s in range(3, 6)])
        print(per_run_line("seeds 0 / 1 / 2", old))
        print(per_run_line("seeds 3 / 4 / 5 (new)", new))
        print(curve_line("seeds 0-2, mean ± sd", old))
        print(curve_line("seeds 3-5, mean ± sd", new))
        print(curve_line("all six seeds, mean ± sd", old + new))
        allv = avg(old + new)
        print(f"\n(b) 1024 m − 256 m, paired, six-seed average: {paired(allv[1024], allv[256])}")
        print("(b) per seed: " + ", ".join(f"{(f[1024] - f[256]).mean():+.3f}" for f in old + new))
        print(f"(b) seeds 3-5 only: {paired(avg(new)[1024], avg(new)[256])}")


# B. LPAC retrained at 256 m with three seeds -----------------------------------------------------
def section_b():
    for n in (40, 11):
        header(f"B. LPAC retrained at 256 m, {n} environments")
        l256 = runs([LPAC256[s][n] for s in range(3)])
        print(per_run_line("seeds 0 / 1 / 2", l256))
        print(curve_line("mean ± sd", l256))
        v4 = avg(runs([V4MIX[s][n] for s in range(6)]))
        l = avg(l256)
        print()
        for r in (256, 512, 1024):
            if r in l and r in v4:
                print(f"V4 mixed (6 seeds) − LPAC 256 m (3 seeds) at {r} m: {paired(v4[r], l[r])}")
        print(f"LPAC 256 m, 1024 m − 256 m: {paired(l[1024], l[256])}")
        pub = sweep(PUBLISHED[n])
        print(f"V4 mixed − published at 256 m: {paired(v4[256], pub[256])}")
        if n == 40:
            l128 = sweep(LPAC128_40)
            print(f"V4 mixed − LPAC retrained at 128 m (seed 0) at 256 m: {paired(v4[256], l128[256])}")


# C. V4 on the 50/50 mixture -------------------------------------------------------------------------
def section_c():
    for n in (40, 11):
        header(f"C. V4 trained on half 128 m / half 1024 m, {n} environments")
        half = sweep(f"results_final/V4half_s0_{n}.csv")
        v4 = runs([V4MIX[s][n] for s in range(6)])
        print(curve_line("V4 half (seed 0)", [half]))
        print(curve_line("V4 mixed, six seeds", v4))
        print()
        for r in RANGES:
            print(f"V4 half − V4 mixed (6-seed avg) at {r} m: {paired(half[r], avg(v4)[r])}; "
                  f"seed spread of V4 mixed {pd.Series([f[r].mean() for f in v4]).std():.3f}")
        print(f"V4 half, 1024 m − 256 m: {paired(half[1024], half[256])}")


# D. heterogeneous teams ------------------------------------------------------------------------------
def hetero(path, semantics=None):
    d = pd.read_csv(ROOT / path)
    if "semantics" not in d:
        d["semantics"] = "reception"
    if "map" in d:
        d = d[d["map"] == "asis"]
    if semantics:
        d = d[d.semantics == semantics]
    return d.groupby(["semantics", "env", "fraction"]).ratio_mean.mean()  # mean over draws


def hetero_runs_v4mixed():
    """Per seed, a Series indexed (semantics, env, fraction) with the four link rules."""
    out = {}
    recv = {0: "results_V45/results_hetero_V4mixed.csv", 1: "results_V45/results_hetero_V4mixed_s1.csv",
            2: "results_V45/results_hetero_V4mixed_s2.csv"}
    for s in range(6):
        if s < 3:
            parts = [hetero(recv[s]), hetero(f"results_sem/V4mixed_s{s}.csv")]
        else:
            parts = [hetero(f"results_final/hetero_V4mixed_s{s}_reception.csv"),
                     hetero(f"results_final/hetero_V4mixed_s{s}_transmission.csv")]
        out[s] = pd.concat(parts)
    return out


def hetero_table(label, series_by_seed, semantics):
    rows = [s.loc[semantics].groupby("fraction").mean() for s in series_by_seed.values()]
    m = pd.DataFrame(rows)
    cells = [f"{m[f].mean():.2f}" + (f" ± {m[f].std():.2f}" if len(m) > 1 else "") for f in FRACTIONS]
    print(f"| {label} | " + " | ".join(cells) + " |")


def section_d():
    print("\n## D. Heterogeneous teams (128 m and 1024 m robots), 11 environments, 4 draws\n")
    print("| model, rule | f = 0 | 1/4 | 1/2 | 3/4 | 1 |\n|---|---|---|---|---|---|")
    v4 = hetero_runs_v4mixed()
    old = {s: v4[s] for s in range(3)}
    new = {s: v4[s] for s in range(3, 6)}
    for sem in ("reception", "transmission", "sym_min", "sym_max"):
        hetero_table(f"V4 mixed, {sem}, seeds 0-2", old, sem)
        hetero_table(f"V4 mixed, {sem}, seeds 3-5", new, sem)
        hetero_table(f"V4 mixed, {sem}, six seeds", v4, sem)
    v4t = {s: pd.concat([hetero(f"results_final/hetero_V4T_s{s}_transmission.csv"),
                         hetero(f"results_final/hetero_V4T_s{s}_reception.csv")]) for s in (0, 1, 2)}
    for sem in ("transmission", "reception", "sym_min", "sym_max"):
        hetero_table(f"V4T (trained on transmission teams), {sem}, seeds 0-2", v4t, sem)
    half = {0: hetero("results_final/hetero_V4half_s0_reception.csv")}
    hetero_table("V4 half 128/1024 m, reception, seed 0", half, "reception")

    def seedavg(group, sem):
        return sum(s.loc[sem] for s in group.values()) / len(group)

    print()
    for f in (0.25, 0.5, 0.75):
        for sem in ("transmission", "sym_max", "sym_min"):
            a = seedavg(v4, sem).xs(f, level="fraction")
            b = seedavg(v4, "reception").xs(f, level="fraction")
            print(f"V4 mixed (6 seeds), {sem} − reception at f = {f}: {paired(a, b)}")
        a = seedavg(new, "transmission").xs(f, level="fraction")
        b = seedavg(new, "reception").xs(f, level="fraction")
        print(f"V4 mixed seeds 3-5 only, transmission − reception at f = {f}: {paired(a, b)}")
        a = seedavg(v4t, "transmission").xs(f, level="fraction")
        b = seedavg(v4, "transmission").xs(f, level="fraction")
        print(f"V4T − V4 mixed, both under transmission, at f = {f}: {paired(a, b)}")
        a = seedavg(v4t, "transmission").xs(f, level="fraction")
        b = seedavg(v4t, "reception").xs(f, level="fraction")
        print(f"V4T, transmission − reception at f = {f}: {paired(a, b)}")
        a = half[0].loc["reception"].xs(f, level="fraction")
        b = seedavg(v4, "reception").xs(f, level="fraction")
        print(f"V4 half − V4 mixed, reception, at f = {f}: {paired(a, b)}")
    a = seedavg(v4, "reception")
    print(f"\nV4 mixed reception, f = 1 − f = 0: {paired(a.xs(1.0, level='fraction'), a.xs(0.0, level='fraction'))}")
    a = seedavg(v4, "transmission")
    print(f"V4 mixed transmission, f = 1 − f = 0: {paired(a.xs(1.0, level='fraction'), a.xs(0.0, level='fraction'))}")


# E. V4T in homogeneous teams ------------------------------------------------------------------------
def section_e():
    header("E. V4T (trained on transmission teams) in homogeneous teams, 11 environments")
    v4t = runs([f"results_final/V4T_s{s}_11.csv" for s in (0, 1, 2)])
    v4 = runs([V4MIX[s][11] for s in range(6)])
    print(per_run_line("V4T seeds 0 / 1 / 2", v4t))
    print(curve_line("V4 mixed, six seeds", v4))
    print()
    for r in RANGES:
        print(f"V4T (3 seeds) − V4 mixed (6 seeds) at {r} m: {paired(avg(v4t)[r], avg(v4)[r])}")


# F. V2M ---------------------------------------------------------------------------------------------
def section_f():
    for n in (11, 40):
        header(f"F. GATv2 variants, {n} environments")
        v2m = sweep(f"results_final/V2M_{n}.csv")
        v1 = runs([f"results_variants/V1_s{s}_{n}.csv" for s in range(3)])
        v2 = runs([f"results_variants/V2_s{s}_{n}.csv" for s in range(3)])
        print(curve_line("V2M (V2res with max aggregation), seed 0", [v2m]))
        frames = {"V1": v1, "V2": v2}
        if n == 11:
            v2res = sweep("results_Q3/V2res_eval_11.csv")
            print(curve_line("V2res (softmax attention), seed 0", [v2res]))
            frames["V2res"] = [v2res]
        print(curve_line("V1 (TAGConv, no map), 3 seeds", v1))
        print(curve_line("V2 (plain GATv2), 3 seeds", v2))
        print(curve_line("published", [sweep(PUBLISHED[n])]))
        print()
        for name, fr in frames.items():
            for r in (128, 256, 1024):
                if r in v2m and r in avg(fr):
                    print(f"V2M − {name} at {r} m: {paired(v2m[r], avg(fr)[r])}")
        print(f"V2M, 1024 m − 256 m: {paired(v2m[1024], v2m[256])}")


if __name__ == "__main__":
    section_a(); section_b(); section_c(); section_d(); section_e(); section_f()
