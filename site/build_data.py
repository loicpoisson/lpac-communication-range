"""Build the data bundle of the project site from the result files of the repository.

    python site/build_data.py            # writes site/data/*.json and checks milestone Table 1

Nothing is measured here. Every number is copied, unrounded, from a CSV or JSON file tracked in
this repository, and every row keeps the path of the file it came from, so a figure on the site
can be traced back to a file and from there to the journal (docs/journal.md).

Table files share one shape, chosen to stay small (one object per record would cost about
300 bytes for each of 14 000 evaluations):

    {"schema": "<name>/1", "commit": "<git sha>", "runs": {run_id: {...}},
     "sources": [paths], "columns": [...], "rows": [[...], ...]}

The ``source`` column is an index into ``sources``; run-level facts (model, seed, checkpoint) live
once in ``runs``. To get one object per record in JavaScript::

    const recs = d.rows.map(r => { const o = Object.fromEntries(d.columns.map((c, i) => [c, r[i]]));
                                   return {...d.runs[o.run_id], ...o, source: d.sources[o.source]}; });

Files written: costs.json, hetero.json, factorial.json, curves.json, maps.json, models.json.
replays.json is written by lpac_study/replay.py (it needs the simulator); this script only
checks that the whole folder stays under the size budget.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "site" / "data"
SIZE_BUDGET_BYTES = 5_000_000
RANGES = (128.0, 256.0, 512.0, 768.0, 1024.0)

# --------------------------------------------------------------------------------------------
# Runs. One entry per trained model (or baseline) and seed. ``primary`` is false for the runs the
# journal's tables do not pool: CPU reruns of a GPU run, other checkpoints of the same run, and
# the early training trials. ``files`` maps an environment set to a sweep CSV (one row per
# environment, range and controller); factorial-format files are attached further down.
# --------------------------------------------------------------------------------------------
MIXED = "mixed ranges, 25k samples each at 128, 256, 512 and 1024 m"
R128 = "128 m, 100k samples generated with the authors' recipe"


def run(model, label, trained_on, variant, seed, checkpoint="best validation", primary=True, note="", map_level="C0", **files):
    return dict(model=model, model_label=label, trained_on=trained_on, variant=variant, seed=seed,
                checkpoint=checkpoint, primary=primary, note=note, map_level=map_level, files=files)


RUNS: dict[str, dict] = {
    "published": run("published", "LPAC as published", "128 m, the authors' dataset and checkpoint", "V0", None,
                     checkpoint="authors' release", released11="results/results.csv",
                     new40="results_new40/results_sweep_published.csv"),
    "lpac128_s0": run("lpac128", "LPAC retrained at 128 m", R128, "V0", 0, checkpoint="best validation (epoch 13)",
                      released11="results_train/results_sweep100k_gpu_epoch13.csv",
                      new40="results_new40/results_sweep_r128_s0.csv"),
    "lpac128_s1": run("lpac128", "LPAC retrained at 128 m", R128, "V0", 1,
                      released11="results_train/results_sweep100k_seed1_best.csv"),
    "lpac128_s2": run("lpac128", "LPAC retrained at 128 m", R128, "V0", 2,
                      released11="results_train/results_sweep100k_seed2_best.csv"),
    "lpac128_every20_s0": run("lpac128_every20", "LPAC retrained at 128 m, one sample every 20 steps",
                              "128 m, 100k decorrelated samples (one every 20 steps instead of 5)", "V0", 0,
                              released11="results_train/results_sweep_e20_best.csv"),
    "lpac256_s0": run("lpac256", "LPAC retrained at 256 m", "256 m, 100k samples", "V0", 0,
                      released11="results_train/results_sweep256_best.csv", new40="results_new40/results_sweep_r256.csv"),
    "lpac512_s0": run("lpac512", "LPAC retrained at 512 m", "512 m, 100k samples", "V0", 0,
                      released11="results_fig20/r512_s0_11.csv", new40="results_fig20/r512_s0_40.csv"),
    "lpac1024_s0": run("lpac1024", "LPAC retrained at 1024 m", "1024 m, 100k samples", "V0", 0,
                       released11="results_fig20/r1024_s0_11.csv", new40="results_fig20/r1024_s0_40.csv"),
    "V3mixed_s0": run("V3mixed", "V3: PointNet edges, mixed ranges", MIXED, "V3", 0, map_level="none",
                      released11="results_B/V3mixed_s0_11.csv", new40="results_B/V3mixed_s0_40.csv"),
    "V0R_s0": run("V0R", "V0R: own range as GNN input", "per-robot ranges drawn in 128, 256, 512, 1024 m", "V0R", 0,
                  released11="results_H2/V0R_s0_11.csv", new40="results_H2/V0R_s0_40.csv"),
    "V5_128_s0": run("V5_128", "V5: multi-scale maps, trained at 128 m", R128, "V5", 0, map_level="multiscale",
                     released11="results_V45/V5_128_s0_11.csv", new40="results_V45/V5_128_s0_40.csv"),
    "V2res_s0": run("V2res", "V2res: GATv2 with residual blocks, LayerNorm and bounded edges", R128, "V2R", 0,
                    map_level="none", note="30 epochs", released11="results_Q3/V2res_eval_11.csv"),
}
for s in range(3):
    RUNS[f"v0mixed_s{s}"] = run("v0mixed", "LPAC trained on mixed ranges", MIXED, "V0", s,
                                released11=f"results_night2/mixed_s{s}_11.csv", new40=f"results_night2/mixed_s{s}_40.csv")
    for v, label in (("V1", "V1: no neighbour map"), ("V2", "V2: GATv2 with edge geometry"), ("V3", "V3: PointNet edges, trained at 128 m")):
        RUNS[f"{v}_s{s}"] = run(v, label, R128, v, s, map_level="none",
                                released11=f"results_variants/{v}_s{s}_11.csv", new40=f"results_variants/{v}_s{s}_40.csv")
    RUNS[f"V4_128_s{s}"] = run("V4_128", "V4: bounded edges, trained at 128 m", R128, "V4", s, map_level="none",
                               released11=f"results_V45/V4_128_s{s}_11.csv", new40=f"results_V45/V4_128_s{s}_40.csv")
    RUNS[f"V4mixed_s{s}"] = run("V4mixed", "V4: bounded edges, mixed ranges", MIXED, "V4", s, map_level="none",
                                released11=f"results_V45/V4mixed_s{s}_11.csv", new40=f"results_V45/V4mixed_s{s}_40.csv")
    RUNS[f"V6mixed_s{s}"] = run("V6mixed", "V6: V5 maps plus the V4 layer, mixed ranges", MIXED, "V6", s, map_level="multiscale",
                                released11=f"results_V45/V6mixed_s{s}_11.csv", new40=f"results_V45/V6mixed_s{s}_40.csv")
    # V5 mixed was launched twice (the first launch was stopped by the slow loader and restarted);
    # milestone Table 1 pools the six runs, so both launches are primary.
    RUNS[f"V5mixed_s{s}"] = run("V5mixed", "V5: multi-scale maps, mixed ranges", MIXED, "V5", s, map_level="multiscale",
                                note="first launch", released11=f"results_V45/V5mixed_s{s}_11.csv",
                                new40=f"results_V45/V5mixed_s{s}_40.csv")
    RUNS[f"V5mixed_s{s}_v2"] = run("V5mixed", "V5: multi-scale maps, mixed ranges", MIXED, "V5", s, map_level="multiscale",
                                   note="second launch, same seed (GPU training is not deterministic)",
                                   released11=f"results_V45/V5mixed_s{s}_v2_11.csv", new40=f"results_V45/V5mixed_s{s}_v2_40.csv")
# final-report runs (journal 20-21), evaluated by training/submit_final_evals.sh into results_final/
HALF = "two ranges only, half the samples at 128 m and half at 1024 m"
HETERO_T = "teams mixing 128 m and 1024 m robots, links by the transmission rule"
for s in (3, 4, 5):
    RUNS[f"V4mixed_s{s}"] = run("V4mixed", "V4: bounded edges, mixed ranges", MIXED, "V4", s, map_level="none",
                                released11=f"results_final/V4mixed_s{s}_11.csv", new40=f"results_final/V4mixed_s{s}_40.csv")
for s in (1, 2):
    RUNS[f"lpac256_s{s}"] = run("lpac256", "LPAC retrained at 256 m", "256 m, 100k samples", "V0", s,
                                released11=f"results_final/LPAC256_s{s}_11.csv", new40=f"results_final/LPAC256_s{s}_40.csv")
RUNS["V4half_s0"] = run("V4half", "V4: bounded edges, trained on 128 m and 1024 m only", HALF, "V4", 0, map_level="none",
                        released11="results_final/V4half_s0_11.csv", new40="results_final/V4half_s0_40.csv")
for s in (0, 1, 2):
    RUNS[f"V4T_s{s}"] = run("V4T", "V4 trained on heterogeneous teams (transmission)", HETERO_T, "V4", s, map_level="none",
                            released11=f"results_final/V4T_s{s}_11.csv")
RUNS["V2M_s0"] = run("V2M", "V2M: V2res with max aggregation", R128, "V2M", 0, map_level="none", note="30 epochs",
                     released11="results_final/V2M_11.csv", new40="results_final/V2M_40.csv")
for v in ("V1", "V2", "V3"):
    RUNS[f"{v}_s0_cpu"] = run(v, RUNS[f"{v}_s0"]["model_label"], R128, v, 0, primary=False, map_level="none",
                              note="12-epoch CPU run of seed 0", released11=f"results_variants/{v}_s0_cpu_11.csv",
                              new40=f"results_variants/{v}_s0_cpu_40.csv")
RUNS["v0mixed_s0_cpu"] = run("v0mixed", "LPAC trained on mixed ranges", MIXED, "V0", 0, primary=False, note="CPU run of seed 0",
                             released11="results_night2/mixed_s0_cpu_11.csv", new40="results_night2/mixed_s0_cpu_40.csv")
for name, note in (("eval100k_cpu_best", "100k samples, CPU, best validation"), ("eval100k_cpu_epoch5", "100k samples, CPU, epoch 5"),
                   ("eval100k_gpu_best", "100k samples, GPU, best validation"), ("eval100k_gpu_epoch13", "100k samples, GPU, epoch 13"),
                   ("eval40k_best", "40k samples, best validation"), ("eval40k_last", "40k samples, last epoch"),
                   ("evalcpu20_best", "40k samples, 20 CPU epochs, best validation")):
    RUNS[f"trial_{name}"] = run("lpac128_trial", "LPAC retraining trials at 128 m", "128 m, early training trials", "V0", 0,
                                checkpoint=note, primary=False, note="training trial of 13 October, evaluated at 128 and 256 m only",
                                released11=f"results_train/results_{name}.csv")
# models that only appear in factorial-format files (their own `model` column)
RUNS["A_128_s0"] = run("A_128", "A: fixed-scale map at training, 128 m", R128 + ", neighbour map drawn at the fixed 256 m scale", "V0", 0, map_level="C3")
for s, name in enumerate(("A_mixed", "A_mixed_s1", "A_mixed_s2")):
    RUNS[f"A_mixed_s{s}"] = run("A_mixed", "A: fixed-scale map at training, mixed ranges", MIXED + ", neighbour map drawn at the fixed 256 m scale", "V0", s, map_level="C3")
BASELINES = {"ClairvoyantCVT": ("clairvoyant", "Clairvoyant CVT (the normalization, cost ratio 1)"),
             "CentralizedCVT": ("centralized_cvt", "Centralized CVT"),
             "DecentralizedCVT": ("decentralized_cvt", "Decentralized CVT")}
for controller, (model, label) in BASELINES.items():
    RUNS[model] = run(model, label, "not learned", None, None, checkpoint=None, map_level="none")
# These two controllers do not use the communication range: their cost is the same at every range
# (checked below, to the last digit). The 40 new environments were not swept at 768 m, so a table
# that prints them at 768 m repeats the value of the measured ranges; no row is invented for it.
RANGE_INDEPENDENT = ("clairvoyant", "centralized_cvt")
for model in RANGE_INDEPENDENT:
    RUNS[model]["range_independent"] = True

# factorial-format files: (path, env_set, {value of the `model` column -> run_id}); a file with no
# `model` column holds the published model. Listed by priority: when two files hold the same
# measurement the first one wins and the pair is checked for equality.
FACTORIAL_FILES = [
    ("results_clamp/results_40.csv", "new40", {"published": "published"}),
    ("results_clamp/results_11.csv", "released11", {"published": "published"}),
    ("results_fix_c3_128/results_40envs.csv", "new40", {"published": "published", "r128_s0": "lpac128_s0"}),
    ("results_fix_c3_128/results_11envs.csv", "released11", {"published": "published", "r128_s0": "lpac128_s0"}),
    ("results_2x2/results_2x2.csv", "released11", {None: "published"}),
    ("results_new40/results_2x2.csv", "new40", {"published": "published", "r128_s0": "lpac128_s0"}),
    ("results_m1/results_2x2.csv", "released11", {"published": "published", "r128_s0": "lpac128_s0", "r128_s1": "lpac128_s1",
                                                  "r128_s2": "lpac128_s2", "r256": "lpac256_s0"}),
    ("results_night2/cap_V1_s0_11.csv", "released11", {"V1_s0": "V1_s0"}),
    ("results_night2/cap_V2_s0_11.csv", "released11", {"V2_s0": "V2_s0"}),
    ("results_A/A_128_11.csv", "released11", {"A_128": "A_128_s0"}), ("results_A/A_128_40.csv", "new40", {"A_128": "A_128_s0"}),
    ("results_A/A_mixed_11.csv", "released11", {"A_mixed": "A_mixed_s0"}), ("results_A/A_mixed_40.csv", "new40", {"A_mixed": "A_mixed_s0"}),
    ("results_A/A_mixed_s1_11.csv", "released11", {"A_mixed_s1": "A_mixed_s1"}), ("results_A/A_mixed_s1_40.csv", "new40", {"A_mixed_s1": "A_mixed_s1"}),
    ("results_A/A_mixed_s2_11.csv", "released11", {"A_mixed_s2": "A_mixed_s2"}), ("results_A/A_mixed_s2_40.csv", "new40", {"A_mixed_s2": "A_mixed_s2"}),
]

# Rows of milestone Table 1 (40 new environments). Each row: the run_ids pooled, the map level
# read, and the values printed in milestone/main.tex (None where the table shows "--").
TABLE1 = [
    ("LPAC published (trained 128 m)", ["published"], "C0", [1.96, 1.70, 2.87, None, 6.15]),
    ("+ fixed 256 m map at inference", ["published"], "C3", [None, 1.70, 1.71, 1.73, 1.77]),
    ("Centralized CVT", ["centralized_cvt"], "none", [2.33, 2.33, 2.33, 2.33, 2.33]),
    ("A: fixed map, mixed", ["A_mixed_s0", "A_mixed_s1", "A_mixed_s2"], "C3", [2.09, 1.91, 1.91, 1.91, 1.92]),
    ("V3: PointNet edges, trained 128 m", ["V3_s0", "V3_s1", "V3_s2"], "none", [2.13, 2.03, 2.25, 2.39, 2.44]),
    ("V4: bounded edges, trained 128 m", ["V4_128_s0", "V4_128_s1", "V4_128_s2"], "none", [2.05, 1.87, 1.90, 1.90, 1.90]),
    ("V4: bounded edges, mixed", ["V4mixed_s0", "V4mixed_s1", "V4mixed_s2"], "none", [2.14, 1.87, 1.81, 1.81, 1.82]),
    ("V5: multi-scale maps, mixed", [f"V5mixed_s{s}{v}" for s in range(3) for v in ("", "_v2")], "multiscale", [2.09, 1.89, 1.89, 1.90, 1.90]),
    ("V6: V5 + V4, mixed", ["V6mixed_s0", "V6mixed_s1", "V6mixed_s2"], "multiscale", [2.12, 1.91, 1.87, 1.88, 1.89]),
]


class Sources:
    """Index of the source files, so a row stores a small integer instead of a path."""

    def __init__(self) -> None:
        self.paths: list[str] = []

    def index(self, path: str) -> int:
        if not (ROOT / path).is_file():
            raise FileNotFoundError(path)
        if path not in self.paths:
            self.paths.append(path)
        return self.paths.index(path)


def number(value):
    """A JSON-safe copy of a CSV value: full precision, NaN as null, integral floats as int."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (int, float)) or hasattr(value, "item"):
        value = value.item() if hasattr(value, "item") else value
        return int(value) if isinstance(value, float) and value.is_integer() and abs(value) < 1e15 else value
    return value


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def write(name: str, payload: dict) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    with open(OUT / name, "w", encoding="utf-8", newline="\n") as handle:  # same bytes on Windows and Linux
        handle.write(text + "\n")
    return len(text) + 1


# --------------------------------------------------------------------------------------------
def build_costs(commit: str) -> dict:
    sources = Sources()
    columns = ["run_id", "env_set", "env", "range_m", "map_level", "graph_level", "cost_ratio", "source"]
    rows: dict[tuple, list] = {}
    conflicts: list[dict] = []

    def add(run_id, env_set, env, comm_range, map_level, graph_level, ratio, path):
        key = (run_id, env_set, int(env), float(comm_range), map_level, graph_level)
        if key in rows:  # the same measurement in two files: they must agree, the first file listed wins
            kept = rows[key]
            if abs(kept[6] - float(ratio)) > 1e-9:
                conflicts.append({"run_id": run_id, "env_set": env_set, "env": int(env), "range_m": number(float(comm_range)),
                                  "map_level": map_level, "graph_level": graph_level, "kept": kept[6], "kept_source": sources.paths[kept[7]],
                                  "dropped": float(ratio), "dropped_source": path})
            return
        rows[key] = [run_id, env_set, int(env), number(float(comm_range)), map_level, graph_level, float(ratio), sources.index(path)]

    # factorial-format files first: they hold the inference-time map and graph levels
    for path, env_set, mapping in FACTORIAL_FILES:
        frame = pd.read_csv(ROOT / path)
        frame = frame[(frame.repeat == 0) & (frame.noise_sigma == 0.0) & (frame.k == 4)]
        frame = frame[frame["map"] != "C1"]  # C1 varies k: it is the dose-response of factorial.json
        for record in frame.itertuples(index=False):
            model = getattr(record, "model", None)
            run_id = mapping[model if "model" in frame.columns else None]
            add(run_id, env_set, record.env, record.comm_range, record.map, record.graph, record.ratio_mean, path)
    # sweep files: the model as trained, its own map level, the unmodified graph
    for run_id, meta in RUNS.items():
        for env_set, path in meta["files"].items():
            frame = pd.read_csv(ROOT / path)
            frame = frame[(frame.num_robots == 32) & (frame.sweep == "range")]
            for record in frame[frame.controller == "lpac"].itertuples(index=False):
                add(run_id, env_set, record.env, record.comm_range, meta["map_level"], "G0", record.ratio_mean, path)
            if run_id == "published":
                for controller, (model, _) in BASELINES.items():
                    for record in frame[frame.controller == controller].itertuples(index=False):
                        add(model, env_set, record.env, record.comm_range, "none", "G0", record.ratio_mean, path)
    runs = {rid: {k: v for k, v in meta.items() if k not in ("files", "map_level")} for rid, meta in RUNS.items()}
    used = {r[0] for r in rows.values()}
    return {"schema": "costs/1", "commit": commit,
            "about": "cost_ratio = episode mean of the coverage cost divided by the clairvoyant controller's, 600 steps, 32 robots",
            "runs": {rid: meta for rid, meta in runs.items() if rid in used}, "sources": sources.paths, "columns": columns,
            "rows": sorted(rows.values(), key=lambda r: (r[0], r[1], r[4], r[5], r[3], r[2])),
            "conflicts": conflicts,
            "headline": {"table": "milestone Table 1", "env_set": "new40", "ranges_m": [number(r) for r in RANGES],
                         "rows": [{"label": label, "run_ids": ids, "map_level": level, "printed": printed,
                                   "range_independent": all(r in RANGE_INDEPENDENT for r in ids)}
                                  for label, ids, level, printed in TABLE1],
                         "notes": ["mean over environments within a run, then mean over the runs listed",
                                   "map_level C3 at 256 m is read from C0: at 256 m the fixed 256 m map is the published map",
                                   "a range_independent row has no data at 768 m on new40; its value there is the value at any measured range"]}}


def check_table1(costs: dict) -> list[str]:
    """Recompute milestone Table 1 from the bundle; every printed value must match to 2 decimals."""
    frame = pd.DataFrame(costs["rows"], columns=costs["columns"])
    frame = frame[(frame.env_set == "new40") & (frame.graph_level == "G0")]
    lines, failures = [], []
    for label, run_ids, level, printed in TABLE1:
        cells = []
        for comm_range, expected in zip(RANGES, printed):
            sel = frame[frame.run_id.isin(run_ids) & (frame.map_level == level) & (frame.range_m == comm_range)]
            if level == "C3" and comm_range == 256.0 and sel.empty:
                # at 256 m the fixed 256 m map is the published map itself (min(R, 256) = 256, same scale)
                sel = frame[frame.run_id.isin(run_ids) & (frame.map_level == "C0") & (frame.range_m == comm_range)]
            if expected is None:
                cells.append("  -- ")
                continue
            if sel.empty and all(r in RANGE_INDEPENDENT for r in run_ids):
                every = frame[frame.run_id.isin(run_ids)].pivot_table(index=["run_id", "env"], columns="range_m", values="cost_ratio")
                spread = float((every.max(axis=1) - every.min(axis=1)).max())
                if spread != 0.0:
                    failures.append(f"{label}: declared range independent but varies by {spread} across ranges")
                sel = frame[frame.run_id.isin(run_ids) & (frame.range_m == every.columns[0])]
            per_run = sel.groupby("run_id").cost_ratio.mean()
            if len(per_run) != len(run_ids):
                failures.append(f"{label} at {comm_range:.0f} m: {len(per_run)} of {len(run_ids)} runs found")
                cells.append("  ?? ")
                continue
            value = per_run.mean()
            cells.append(f"{value:5.2f}")
            if abs(round(value, 2) - expected) > 1e-9:
                failures.append(f"{label} at {comm_range:.0f} m: bundle gives {value:.4f}, the table prints {expected:.2f}")
        lines.append(f"  {label:36s} " + "  ".join(cells))
    print("milestone Table 1 recomputed from costs.json (40 new environments):")
    print("\n".join(lines))
    return failures


# --------------------------------------------------------------------------------------------
HETERO_FILES = [
    ("results_hetero/results_hetero8.csv", {"published": "published"}),
    ("results_sem/published.csv", {"published": "published"}),
    ("results_V45/results_hetero_V4mixed.csv", {"V4mixed_s0": "V4mixed_s0"}),
    ("results_V45/results_hetero_V4mixed_s1.csv", {"V4mixed_s1": "V4mixed_s1"}),
    ("results_V45/results_hetero_V4mixed_s2.csv", {"V4mixed_s2": "V4mixed_s2"}),
    ("results_sem/V4mixed_s0.csv", {"V4mixed_s0": "V4mixed_s0"}), ("results_sem/V4mixed_s1.csv", {"V4mixed_s1": "V4mixed_s1"}),
    ("results_sem/V4mixed_s2.csv", {"V4mixed_s2": "V4mixed_s2"}),
    ("results_H2/results_hetero_V0R.csv", {"V0R": "V0R_s0"}),
    ("results_h1/results_h1.csv", {"128+1024": "mixture_s0"}), ("results_h1/results_h1_s1.csv", {"128+1024": "mixture_s1"}),
    ("results_h1/results_h1_s2.csv", {"128+1024": "mixture_s2"}),
    *[(f"results_final/hetero_V4mixed_s{s}_{sem}.csv", {f"V4mixed_s{s}": f"V4mixed_s{s}"})
      for s in (3, 4, 5) for sem in ("reception", "transmission")],
    *[(f"results_final/hetero_V4T_s{s}_{sem}.csv", {f"V4T_s{s}": f"V4T_s{s}"})
      for s in (0, 1, 2) for sem in ("transmission", "reception")],
    ("results_final/hetero_V4half_s0_reception.csv", {"V4half_s0": "V4half_s0"}),
]


def build_hetero(commit: str) -> dict:
    sources = Sources()
    columns = ["run_id", "map", "semantics", "f", "env", "draw", "mode", "cost_ratio", "source"]
    rows, runs = [], {}
    for path, mapping in HETERO_FILES:
        frame = pd.read_csv(ROOT / path)
        for record in frame.itertuples(index=False):
            run_id = mapping[record.model]
            if run_id.startswith("mixture"):
                seed = int(run_id[-1])
                runs[run_id] = {"model": "mixture128+1024", "model_label": "Mixture of two specialists (LPAC at 128 m and at 1024 m)",
                                "seed": seed, "note": "mode 'mixture': each robot runs the specialist of its own range; mode 'short': every robot runs the 128 m model"}
            else:
                meta = RUNS[run_id]
                runs[run_id] = {"model": meta["model"], "model_label": meta["model_label"], "seed": meta["seed"]}
            rows.append([run_id, record.map, getattr(record, "semantics", "reception"), number(float(record.fraction)), int(record.env),
                         int(record.draw), getattr(record, "mode", None), float(record.ratio_mean), sources.index(path)])
    return {"schema": "hetero/1", "commit": commit,
            "about": "teams mixing robots that hear at 128 m and at 1024 m; f is the fraction at 1024 m, draw the random choice of which robots; "
                     "map 'asis' draws each robot's map at its own range, 'fixed256' at the fixed 256 m scale",
            "long_range_m": 1024, "short_range_m": 128, "runs": runs, "sources": sources.paths, "columns": columns,
            "rows": sorted(rows, key=lambda r: (r[0], r[1], r[2], r[3], r[4], r[5], str(r[6])))}


def build_factorial(commit: str) -> dict:
    path = "results_2x2/results_2x2.csv"
    frame = pd.read_csv(ROOT / path)
    cell_columns = ["env_set", "model", "cell", "graph_level", "map_level", "range_m", "env", "repeat", "noise_sigma", "k",
                    "cost_ratio", "map_neighbours_mean", "degree_mean", "source"]
    cells = [["released11", "published", r.cell, r.graph, r.map, number(float(r.comm_range)), int(r.env), int(r.repeat),
              number(float(r.noise_sigma)), int(r.k), float(r.ratio_mean), number(float(r.map_neighbours_mean)), number(float(r.degree_mean)), 0]
             for r in frame.itertuples(index=False)]
    dose = frame[(frame["map"] == "C1") & (frame.repeat == 0) & (frame.noise_sigma == 0.0)]
    dose_rows = [[r.graph, int(r.k), number(float(r.comm_range)), int(r.env), float(r.ratio_mean), number(float(r.map_neighbours_mean)), 0]
                 for r in dose.itertuples(index=False)]
    return {"schema": "factorial/1", "commit": commit,
            "about": "published model, 11 released environments. Graph: G0 all robots in range, G1 the k nearest. Map: C0 as published, "
                     "C1 only the k nearest drawn (at the range's scale), C2 drawn as at training (128 m), C3 fixed 256 m scale, C4 empty. "
                     "repeat > 0 with noise_sigma > 0 are the noise-floor repetitions (journal, M3).",
            "sources": [path],
            "cells": {"columns": cell_columns, "rows": cells},
            "dose_response": {"about": "map level C1: the k nearest robots are drawn in the map; k against cost",
                              "columns": ["graph_level", "k", "range_m", "env", "cost_ratio", "map_neighbours_mean", "source"], "rows": dose_rows}}


CURVES = [  # id, label, variant, dataset, seed, source
    ("V1", "V1: no neighbour map (TAGConv K=3)", "V1", R128, 0, "results_variants/train_V1_s0/loss_history.json"),
    ("V2", "V2: GATv2 with edge geometry, as first trained", "V2", R128, 0, "results_variants/train_V2_s0/loss_history.json"),
    ("V2lr", "V2 with ten times the learning rate", "V2", R128, 0, "results_Q3/V2lr_loss_history.json"),
    ("V2B", "V2 with bounded edge features only", "V2B", R128, 0, "results_Q3/V2B_loss_history.json"),
    ("V2res", "V2 with residual blocks, LayerNorm and bounded edges", "V2R", R128, 0, "results_Q3/V2res_loss_history.json"),
    ("V2deep", "the V2res block, 15 layers", "V2R", R128, 0, "results_Q3/V2deep_loss_history.json"),
    ("V2logdeg", "V2res plus a log-degree node feature", "V2RL", R128, 0, "results_Q3/V2logdeg_loss_history.json"),
    ("V1K1", "V1 with TAGConv K=1 (five hops, like V2)", "V1", R128, 0, "results_Q3/V1K1_loss_history.json"),
    ("V2M", "V2res with max aggregation (V2M)", "V2M", R128, 0, "results_final/train_V2M/loss_history.json"),
    ("V4mixed_s0", "V4: bounded edges, mixed ranges", "V4", MIXED, 0, "results_V45/train_V4mixed_s0/loss_history.json"),
    ("V5mixed_s0", "V5: multi-scale maps, mixed ranges", "V5", MIXED, 0, "results_V45/train_V5mixed_s0/loss_history.json"),
    ("lpac256_s0", "LPAC retrained at 256 m", "V0", "256 m, 100k samples", 0, "training/logs/loss_256m_gpu.json"),
]


def build_curves(commit: str) -> dict:
    runs = {}
    for run_id, label, variant, dataset, seed, path in CURVES:
        history = json.loads((ROOT / path).read_text())
        assert len(history["train"]) == len(history["val"]), path
        runs[run_id] = {"label": label, "variant": variant, "dataset": dataset, "seed": seed, "source": path,
                        "epoch": list(range(1, len(history["train"]) + 1)), "train": history["train"], "val": history["val"]}
    return {"schema": "curves/1", "commit": commit,
            "about": "mean squared error on the normalized control, per epoch; 70 000 training and 20 000 validation samples", "runs": runs}


def build_maps(commit: str) -> dict:
    path = "results_clamp/map_stats.csv"
    frame = pd.read_csv(ROOT / path)
    columns = ["level", "range_m", "env", "max_abs_cell", "mass_per_robot", "nonzero_cells_per_robot", "robots_drawn_per_map", "source"]
    rows = [[r.level, number(float(r.range_m)), int(r.env), float(r.max_abs_cell), float(r.mass_per_robot),
             float(r.nonzero_cells_per_robot), float(r.robots_drawn_per_map), 0] for r in frame.itertuples(index=False)]
    return {"schema": "maps/1", "commit": commit,
            "about": "neighbour maps (2 x 32 x 32 per robot) at the initial positions of the 11 released environments. C0 as published, "
                     "C3 fixed 256 m scale with far robots dropped, K256 fixed 256 m scale with far robots projected onto the 256 m circle. "
                     "Generated by lpac_study/map_stats.py.",
            "env_set": "released11", "sources": [path], "columns": columns, "rows": rows}


def introduced_in(symbol: str) -> str | None:
    """Short hash of the first commit in which ``symbol`` appears in lpac_study/variants.py."""
    out = subprocess.run(["git", "log", "--reverse", "--format=%h", f"-S{symbol}", "--", "lpac_study/variants.py"],
                         cwd=ROOT, capture_output=True, text=True).stdout.split()
    return out[0] if out else None


def params_of(path: str) -> int:
    return json.loads((ROOT / path).read_text())["params"]


def build_models(commit: str) -> dict:
    tag = "TAGConv, K = 3: normalized sum over neighbours, 5 layers"
    models = {
        "V0": dict(label="LPAC (the authors' architecture)", description="CNN on four 32 x 32 maps, five TAGConv layers, MLP; the two neighbour-map channels are drawn at the scale of the communication range.",
                   parameters=params_of("results_A/train_A_128/run_info.json"), map_input="2 neighbour-map channels at the scale of the range (or at a fixed 256 m scale for experiment A)",
                   edge_features="none (binary adjacency)", aggregation=tag, receptive_field_hops=15,
                   training_sets=["authors' release (published)", R128, "256 m", "512 m", "1024 m", MIXED], epochs="35 to 100, best validation checkpoint",
                   **{"class": "coverage_control.nn.LPAC"}, introduced_in_commit=None, source="results_A/train_A_128/run_info.json"),
        "V1": dict(label="V1: no neighbour map", description="LPAC with the two neighbour-map channels removed; the robots only learn about each other through the graph.",
                   parameters=params_of("results_variants/train_V1_s0/run_info.json"), map_input="none", edge_features="none (binary adjacency)", aggregation=tag,
                   receptive_field_hops=15, training_sets=[R128], epochs="100 (30 for the K = 1 control of Q3)", **{"class": "LPAC with variant_config"},
                   introduced_in_commit=introduced_in('"V1"'), source="results_variants/train_V1_s0/run_info.json"),
        "V2": dict(label="V2: GATv2 with edge geometry", description="No neighbour map; five GATv2 layers read (p_j - p_i) / W on every edge. Underfits as first trained (journal 10 and 16.3).",
                   parameters=params_of("results_variants/train_V2_s0/run_info.json"), map_input="none", edge_features="(p_j - p_i) / W, 2 numbers, W the world size",
                   aggregation="GATv2 softmax attention, 4 heads, 5 layers, no residual, no normalization", receptive_field_hops=5, training_sets=[R128], epochs="100",
                   **{"class": "LPACEdgeGeometry"}, introduced_in_commit=introduced_in("LPACEdgeGeometry"), source="results_variants/train_V2_s0/run_info.json"),
        "V2B": dict(label="V2B: V2 with bounded edge features only", description="V2 with only the edge input replaced by V4's bounded features; the single-cause ablation of Q3. Trains exactly like V2.",
                    parameters=params_of("results_Q3/V2B_run_info.json"), map_input="none", edge_features="[(p_j - p_i) / d, tanh(d / 256 m)], 3 numbers",
                    aggregation="GATv2 softmax attention, 4 heads, 5 layers, no residual, no normalization", receptive_field_hops=5, training_sets=[R128], epochs="30",
                    **{"class": "LPACBoundedEdgeGATv2"}, introduced_in_commit=introduced_in("LPACBoundedEdgeGATv2"), source="results_Q3/V2B_run_info.json"),
        "V2R": dict(label="V2res: GATv2 in residual blocks", description="V2 with pre-norm residual blocks, LayerNorm and bounded edge features; removes V2's underfitting (journal 16.3, 16.4).",
                    parameters=params_of("results_Q3/V2res_run_info.json"), map_input="none", edge_features="[(p_j - p_i) / d, tanh(d / 256 m)], 3 numbers",
                    aggregation="GATv2 softmax attention, 4 heads, in x + relu(GATv2(LayerNorm(x))) blocks, 5 layers (15 in V2deep)", receptive_field_hops=5,
                    training_sets=[R128], epochs="30", **{"class": "LPACResidualEdgeGeometry"}, introduced_in_commit=introduced_in("LPACResidualEdgeGeometry"),
                    source="results_Q3/V2res_run_info.json"),
        "V2RL": dict(label="V2logdeg: V2res plus log-degree", description="V2res with log(1 + in-degree) added to the node input, to restore the neighbour count that attention averages away. Slightly worse than V2res.",
                     parameters=params_of("results_Q3/V2logdeg_run_info.json"), map_input="none", edge_features="[(p_j - p_i) / d, tanh(d / 256 m)], 3 numbers",
                     aggregation="as V2res", receptive_field_hops=5, training_sets=[R128], epochs="30", **{"class": "LPACResidualEdgeGeometry"},
                     introduced_in_commit=introduced_in("uses_log_degree"), source="results_Q3/V2logdeg_run_info.json"),
        "V2M": dict(label="V2M: V2res with max aggregation", description="V2res with each GATv2 layer replaced by the same GATv2-style message, leaky_relu(W_l x_i + W_r x_j + W_e e_ij), max-aggregated instead of softmax-weighted. Fits better than TAGConv and halves V2res's range degradation (journal 21).",
                    parameters=params_of("results_final/train_V2M/run_info.json"), map_input="none", edge_features="[(p_j - p_i) / d, tanh(d / 256 m)], 3 numbers",
                    aggregation="max over neighbours of GATv2-style messages, in x + relu(conv(LayerNorm(x))) blocks, 5 layers", receptive_field_hops=5,
                    training_sets=[R128], epochs="30", **{"class": "MaxEdgeConv in ResidualEdgeGeometryGNN"}, introduced_in_commit=introduced_in("MaxEdgeConv"),
                    source="results_final/train_V2M/run_info.json"),
        "V3": dict(label="V3: PointNet edges", description="V1 plus one PointNet-style layer in front of the TAGConv stack, reading (p_j - p_i) / 256 m on every edge.",
                   parameters=params_of("results_variants/train_models100k_V3_s0/run_info.json"), map_input="none", edge_features="(p_j - p_i) / 256 m, 2 numbers, unbounded",
                   aggregation="max over neighbours in the front layer (residual), then " + tag, receptive_field_hops=16, training_sets=[R128, MIXED], epochs="100 at 128 m, 25 on mixed ranges",
                   **{"class": "LPACPointNet"}, introduced_in_commit=introduced_in("LPACPointNet"), source="results_variants/train_models100k_V3_s0/run_info.json"),
        "V4": dict(label="V4: bounded edge features", description="V3 with bounded, range-invariant edge features: the unit direction to the neighbour and a squashed distance. Trained on mixed ranges it is the only model whose cost falls with the range.",
                   parameters=params_of("results_V45/train_V4mixed_s0/run_info.json"), map_input="none", edge_features="[(p_j - p_i) / d, tanh(d / 256 m)], 3 numbers in [-1, 1]",
                   aggregation="max over neighbours in the front layer (residual), then " + tag, receptive_field_hops=16, training_sets=[R128, MIXED], epochs="100 at 128 m, 25 on mixed ranges",
                   **{"class": "LPACBoundedGeometry"}, introduced_in_commit=introduced_in("LPACBoundedGeometry"), source="results_V45/train_V4mixed_s0/run_info.json"),
        "V5": dict(label="V5: multi-scale fixed maps", description="LPAC with three pairs of neighbour-map channels at fixed scales of 256, 512 and 1024 m instead of one pair at the scale of the range.",
                   parameters=params_of("results_V45/train_V5mixed_s0/run_info.json"), map_input="6 neighbour-map channels at fixed scales 256 / 512 / 1024 m", edge_features="none (binary adjacency)",
                   aggregation=tag, receptive_field_hops=15, training_sets=[R128, MIXED], epochs="40 at 128 m, 25 on mixed ranges", **{"class": "LPACMultiScale"},
                   introduced_in_commit=introduced_in("LPACMultiScale"), source="results_V45/train_V5mixed_s0/run_info.json"),
        "V6": dict(label="V6: V5 maps plus the V4 layer", description="The multi-scale maps of V5 feeding the CNN and the bounded-edge front layer of V4. Equals V5 in cost: the two gains do not add up.",
                   parameters=params_of("results_V45/train_V6mixed_s0/run_info.json"), map_input="6 neighbour-map channels at fixed scales 256 / 512 / 1024 m",
                   edge_features="[(p_j - p_i) / d, tanh(d / 256 m)], 3 numbers in [-1, 1]", aggregation="max over neighbours in the front layer (residual), then " + tag,
                   receptive_field_hops=16, training_sets=[MIXED], epochs="25", **{"class": "LPACMultiScaleGeometry"},
                   introduced_in_commit=introduced_in("LPACMultiScaleGeometry"), source="results_V45/train_V6mixed_s0/run_info.json"),
        "V0R": dict(label="V0R: own range as GNN input", description="LPAC with each robot's own communication range as an extra GNN input, trained on teams with a range drawn per robot.",
                    parameters=params_of("results_H2/train_V0R_s0/run_info.json"), map_input="2 neighbour-map channels at the scale of the robot's own range", edge_features="none (binary adjacency)",
                    aggregation=tag, receptive_field_hops=15, training_sets=["per-robot ranges drawn in 128, 256, 512, 1024 m"], epochs="40", **{"class": "LPACRangeInput"},
                    introduced_in_commit=introduced_in("LPACRangeInput"), source="results_H2/train_V0R_s0/run_info.json"),
    }
    return {"schema": "models/1", "commit": commit,
            "about": "architectures of lpac_study/variants.py; parameter counts are read from the run_info.json of a training run of each", "models": models}


def main() -> None:
    commit = git_commit()
    costs = build_costs(commit)
    sizes = {"costs.json": write("costs.json", costs), "hetero.json": write("hetero.json", build_hetero(commit)),
             "factorial.json": write("factorial.json", build_factorial(commit)), "curves.json": write("curves.json", build_curves(commit)),
             "maps.json": write("maps.json", build_maps(commit)), "models.json": write("models.json", build_models(commit))}
    print(f"costs.json: {len(costs['rows'])} rows, {len(costs['runs'])} runs, {len(costs['sources'])} source files, "
          f"{len(costs['conflicts'])} conflicting duplicates")
    for conflict in costs["conflicts"][:10]:
        print("  conflict:", conflict)
    failures = check_table1(costs)
    total = sum(p.stat().st_size for p in OUT.glob("*.json"))
    for name in sorted(p.name for p in OUT.glob("*.json")):
        print(f"  {name:16s} {(OUT / name).stat().st_size / 1e6:6.3f} MB")
    print(f"  total            {total / 1e6:6.3f} MB (budget {SIZE_BUDGET_BYTES / 1e6:.0f} MB)")
    if total > SIZE_BUDGET_BYTES:
        failures.append(f"site/data is {total} bytes, over the {SIZE_BUDGET_BYTES} budget")
    if failures:
        print("FAILED:\n  " + "\n  ".join(failures))
        sys.exit(1)
    print("Table 1 matches to 2 decimals; bundle within budget.")


if __name__ == "__main__":
    main()
