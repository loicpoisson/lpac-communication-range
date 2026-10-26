"""Checkpoint-selection test (section 14): V5/V6 mixed evaluated at the best-val epoch (~5), epoch 12 and epoch 25.

Run on FarmShare: python3 lpac_study/checkpoint_epochs.py (reads results_V45/<run>/env_*/results.csv)."""
import csv, glob, os, statistics as st
B = os.path.expanduser("~/lpac_project/results_V45")
R = [128, 256, 512, 768, 1024]
def load(d):
    out = {}
    for f in glob.glob(f"{B}/{d}/env_*/results.csv"):
        for r in csv.DictReader(open(f)):
            if r["controller"] == "lpac" and r["sweep"] == "range":
                out[(int(r["env"]), int(float(r["comm_range"])))] = float(r["ratio_mean"])
    return out
recipes = {
 "V4mixed": ["V4mixed_s%d"], "V5best": ["V5mixed_s%d_v2"], "V5e25": ["V5mixed_s%d_e25"],
 "V6best": ["V6mixed_s%d"], "V6e12": ["V6mixed_s%d_e12"], "V6e25": ["V6mixed_s%d_e25"]}
for n in (40, 11):
    print(f"===== {n} envs")
    data = {}
    for k, pats in recipes.items():
        seeds = [load(f"{p % s}_{n}") for p in pats for s in range(3)]
        envs = sorted({e for d in seeds for (e, r) in d})
        # per-env mean over seeds
        m = {(e, r): st.mean(d[(e, r)] for d in seeds) for e in envs for r in R if all((e, r) in d for d in seeds)}
        data[k] = m
        per_seed = [[st.mean(v for (e, rr), v in d.items() if rr == r) for r in R] for d in seeds]
        means = [st.mean(v for (e, rr), v in m.items() if rr == r) for r in R]
        dB = [m[(e, 1024)] - m[(e, 256)] for e in envs if (e, 1024) in m]
        sd = st.stdev(dB) / len(dB) ** .5
        seedb = [ps[4] - ps[1] for ps in per_seed]
        print(f"{k:8s} n={len(envs)} " + " ".join(f"{x:.2f}" for x in means) +
              f" | b={st.mean(dB):+.3f} SE {sd:.3f} better {sum(x<0 for x in dB)}/{len(dB)} | seeds b " + " ".join(f"{x:+.2f}" for x in seedb))
    for a, b in [("V6e25", "V4mixed"), ("V6e25", "V5e25"), ("V6e25", "V6best"), ("V5e25", "V5best"), ("V6e12", "V6best")]:
        row = []
        for r in R:
            d = [data[a][k] - data[b][k] for k in data[a] if k[1] == r and k in data[b]]
            row.append(f"{st.mean(d):+.2f}(SE{st.stdev(d)/len(d)**.5:.2f},{sum(x<0 for x in d)}/{len(d)})")
        print(f"{a}-{b}: " + " ".join(row))
