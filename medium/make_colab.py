"""Generate ``medium/lpac_range_colab.ipynb``, the Colab notebook that goes with the Medium post.

    python medium/make_colab.py

The notebook embeds the ``lpac_study`` modules it needs as ``%%writefile`` cells, so it runs top
to bottom on a fresh Colab runtime with no other file. Re-run this script after editing the
package. Quick mode (default): 2 environments, full 600-step episodes, on the
free CPU runtime (about 45 minutes); full mode: the 11 released environments, several hours.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent / "lpac_study"
MODULES = ["__init__", "paths", "simulation", "graph_stats", "effective_rank", "neighbourhood",
           "variants", "lpac_policy", "episode", "sweep", "factorial", "merge", "checks"]
SITE = "https://loicpoisson.github.io/lpac-range-transfer/"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n")}


INTRO = f"""
# When more communication hurts: LPAC and the communication range

Companion notebook of the Medium post *When More Communication Hurts: Debugging a Graph Neural
Network for Robot Swarms* (Stanford CS224W, Autumn 2026). Full study and interactive figures:
[{SITE}]({SITE}).

**What it does, with no training at all:**

1. Runs the published LPAC checkpoint [Agarwal et al., IEEE T-RO 2025] at 256, 512 and 1024 m and
   shows that its coverage cost explodes as the communication range grows.
2. Runs an inference-time factorial at 1024 m: same weights and simulator, change only the GNN's
   graph or only the CNN's neighbour map. The map is the culprit, not the graph.
3. Applies the fix: draw the neighbour map at a fixed 256 m scale. The cost comes back to the
   model's own optimum.
4. Runs V4, our map-less variant with bounded edge features (its checkpoint is downloaded, 7.6 MB).

**Runtime.** `QUICK = True` (default) uses 2 environments with full 600-step episodes and runs them
in parallel on Colab's 2 CPUs: about 35 minutes for parts 1-3 and 15 more for part 4.
`QUICK = False` uses the 11 released environments: several hours. No GPU needed. Every cost is the coverage cost divided by that of a
clairvoyant controller that knows the whole field, averaged over the episode: lower is better,
1.0 is perfect.
"""

INSTALL = """
# The simulator ships as a manylinux wheel (Python 3.10-3.13); torch and torchvision are preinstalled on Colab.
import importlib.util, subprocess, sys
for module, spec in [("coverage_control", "coverage_control==1.3.1"), ("torch_geometric", "torch_geometric"),
                     ("torchvision", "torchvision")]:
    if importlib.util.find_spec(module) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", spec])
import torch, torch_geometric, coverage_control
print("torch", torch.__version__, "| PyG", torch_geometric.__version__, "| coverage_control", coverage_control.__version__)
"""

SETTINGS = """
import os
from pathlib import Path
QUICK = True                     # False: the 11 released environments (several hours)
NUM_ENVS = 2 if QUICK else 11
NUM_STEPS = 600                  # episode length used throughout the study
WORKERS = 2                      # parallel processes, one environment each (Colab CPU runtime has 2 cores)
V4_URL = ("https://raw.githubusercontent.com/loicpoisson/lpac-range-transfer/models/"
          "v4_mixed_ranges_seed0.pt")   # V4 checkpoint (7.6 MB); set to "" to skip part 4
ROOT = Path(os.environ.get("LPAC_NB_ROOT", "/content"))
OUT = ROOT / "lpac_range"
OUT.mkdir(parents=True, exist_ok=True)
os.environ.update(OMP_NUM_THREADS="1", OMP_THREAD_LIMIT="1", OMP_WAIT_POLICY="PASSIVE")
"""

DOWNLOAD = """
# Published checkpoint, the 11 evaluation environments and the authors' parameters (release v1.3.1, ~300 MB).
import tarfile, urllib.request
DATA_ROOT = ROOT / "lpac_data"
DATA_DIR = DATA_ROOT / "lpac"
if not (DATA_DIR / "models" / "model_k3_1024_state_dict.pt").is_file():
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    archive = DATA_ROOT / "lpac_CoverageControl.tar.gz"
    urllib.request.urlretrieve("https://github.com/KumarRobotics/CoverageControl/releases/download/v1.3.1/"
                               "lpac_CoverageControl.tar.gz", archive)
    with tarfile.open(archive) as tar:
        tar.extractall(DATA_ROOT)
    archive.unlink()
os.environ["LPAC_DATA_DIR"] = str(DATA_DIR)
print(sorted(p.name for p in DATA_DIR.iterdir()))
"""

PACKAGE_MD = """
## The code: package `lpac_study`

The next cells write our study package to the runtime's disk, unchanged from the project
repository. Everything graph-related goes through PyTorch Geometric: the library builds a radius
graph over the robots, the GNN is five `TAGConv` layers (K = 3), and our variants are
`MessagePassing` layers. `neighbourhood.py` rebuilds the neighbour map and the graph so that
either can be changed at inference; `factorial.py` runs those interventions.
"""

HELPERS = """
import subprocess, sys, time
import pandas as pd

def run_parallel(make_cmd, envs, label):
    \"\"\"Run one process per environment, WORKERS at a time; make_cmd(env) returns the argument list.\"\"\"
    t0, pending, running = time.time(), list(envs), []
    while pending or running:
        while pending and len(running) < WORKERS:
            env = pending.pop(0)
            log = open(OUT / f"{label}_env{env}.log", "w")
            running.append((env, subprocess.Popen(make_cmd(env), stdout=log, stderr=subprocess.STDOUT), log))
        time.sleep(5)
        for item in list(running):
            env, proc, log = item
            if proc.poll() is not None:
                log.close(); running.remove(item)
                if proc.returncode != 0:
                    print(open(OUT / f"{label}_env{env}.log").read()[-3000:])
                    raise RuntimeError(f"{label}: environment {env} failed")
                print(f"{label}: environment {env} done ({time.time() - t0:.0f} s)")

def env_args(env):
    return ["--num-envs", str(NUM_ENVS), "--env-start", str(env), "--env-end", str(env + 1), "--num-steps", str(NUM_STEPS)]

def merge(parts, out_dir, kind="sweep"):
    subprocess.run([sys.executable, "-m", "lpac_study.merge", "--kind", kind, "--parts", *map(str, parts),
                    "--out-dir", str(out_dir)], check=True, capture_output=True)

ENVS = list(range(NUM_ENVS))
"""

CHECK = """
# Sanity check: the neighbour map and graph rebuilt by lpac_study match the library's own, bit for bit.
subprocess.run([sys.executable, "-m", "lpac_study.checks", "--comm-range", "1024", "--steps", "5"], check=True)
"""

PART1_MD = """
## 1. The anomaly: more range, worse coverage

The published checkpoint was trained with a 128 m range. We run it, and the clairvoyant
controller used for normalization, at 256, 512 and 1024 m. On the 40-environment set of the post
the costs are 1.70, 2.87 and 6.15.
"""

PART1 = """
SWEEP = OUT / "sweep"
run_parallel(lambda env: [sys.executable, "-m", "lpac_study.sweep", "--out-dir", str(SWEEP / f"env_{env}"),
                          "--ranges", "256", "512", "1024", "--controllers", "lpac", "ClairvoyantCVT",
                          "--no-constant-degree", "--no-rank", *env_args(env)], ENVS, "sweep")
merge([SWEEP / f"env_{e}" for e in ENVS], SWEEP)
sweep = pd.read_csv(SWEEP / "results.csv")
published = sweep[sweep.controller == "lpac"].pivot_table(index="env", columns="comm_range", values="ratio_mean")
published.loc["mean"] = published.mean()
published.round(2)
"""

PART2_MD = """
## 2. Graph or picture of the graph? A factorial at 1024 m

The neighbourhood enters LPAC twice: as the `edge_index` of the GNN (**G0** every robot in range,
**G1** only the 4 nearest) and as the two neighbour-map channels of the CNN (**C0** as published,
drawn at the scale of the range; **C3** drawn at a fixed 256 m scale, robots beyond it left out;
**C4** empty). Same weights, same simulator; only the model's inputs change.
"""

PART2 = """
FACT = OUT / "factorial"
def factorial(tag, ranges, cells):
    run_parallel(lambda env: [sys.executable, "-m", "lpac_study.factorial", "--baseline", str(SWEEP / "results.csv"),
                              "--out-dir", str(FACT / f"{tag}_env_{env}"), "--ranges", *ranges, "--k", "4",
                              "--cells", *cells, *env_args(env)], ENVS, f"factorial_{tag}")
factorial("r1024", ["1024"], ["G1C0", "G0C3", "G0C4"])   # the factorial proper
factorial("r512", ["512"], ["G0C3"])                      # the fix at 512 m, for part 3
merge([FACT / f"{t}_env_{e}" for t in ("r1024", "r512") for e in ENVS], FACT, kind="factorial")
fact = pd.read_csv(FACT / "results_2x2.csv")
fact = fact[fact.repeat == 0]
asis = sweep[sweep.controller == "lpac"].assign(cell="G0C0")[["env", "comm_range", "cell", "ratio_mean"]]
cells = pd.concat([asis, fact[["env", "comm_range", "cell", "ratio_mean"]]])
NAMES = {"G0C0": "as published", "G1C0": "graph capped to 4 nearest", "G0C4": "empty map",
         "G0C3": "map at fixed 256 m scale"}
table_1024 = (cells[cells.comm_range == 1024].groupby("cell").ratio_mean.mean()
              .rename(index=NAMES).sort_values(ascending=False))
table_1024.round(2)
"""

PART3_MD = """
## 3. The fix: a fixed 256 m map, no retraining

Drawing the neighbour map at a fixed 256 m scale, whatever the radio range, brings the published
model back to its own 256 m optimum at every range (at 256 m the fixed map *is* the published map).
"""

PART3 = """
import matplotlib.pyplot as plt
mean_asis = published.loc["mean"]
fixed = cells[cells.cell == "G0C3"].groupby("comm_range").ratio_mean.mean()
fixed.loc[256] = mean_asis[256]
fixed = fixed.sort_index()
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
ax = axes[0]
ax.plot(mean_asis.index, mean_asis.values, "o-", color="#D55E00", lw=2, label="LPAC as published")
ax.plot(fixed.index, fixed.values, "o-", color="#0072B2", lw=2, label="same weights, fixed 256 m map")
ax.set_xscale("log", base=2); ax.set_xticks([256, 512, 1024]); ax.set_xticklabels(["256", "512", "1024"])
ax.set_xlabel("communication range (m)"); ax.set_ylabel("cost / clairvoyant (lower is better)")
ax.set_title("More communication, worse coverage, and the fix"); ax.legend(); ax.grid(alpha=0.3)
ax = axes[1]
COLOURS = {"as published": "#D55E00", "graph capped to 4 nearest": "#E69F00", "empty map": "#999999",
           "map at fixed 256 m scale": "#0072B2"}
ax.barh(table_1024.index, table_1024.values, color=[COLOURS[n] for n in table_1024.index])
ax.set_xlabel("cost / clairvoyant at 1024 m"); ax.set_title("Factorial at 1024 m: change one input")
ax.grid(alpha=0.3, axis="x")
plt.tight_layout(); plt.savefig(OUT / "summary.png", dpi=150); plt.show()
"""

PART4_MD = """
## 4. V4: let the graph carry the geometry

V4 removes the two neighbour-map channels and adds one PointNet-style `MessagePassing` layer in
front of the unchanged `TAGConv` stack. On every edge j → i at distance d it reads the unit
direction (p_j − p_i) / d and tanh(d / 256 m), three numbers in [−1, 1] whatever the range, and
aggregates with a max (`lpac_study/variants.py`, class `BoundedGeometryConv`). Trained on mixed
ranges, it is the only model of the study whose cost keeps falling beyond 256 m. The checkpoint
is seed 0 of that recipe; the clairvoyant rows of part 1 are reused.
"""

PART4 = """
if not V4_URL:
    print("V4_URL is empty: part 4 skipped.")
else:
    import urllib.request
    V4 = OUT / "v4"
    ckpt = OUT / "v4_state_dict.pt"
    if not ckpt.is_file():
        urllib.request.urlretrieve(V4_URL, ckpt)
    for env in ENVS:  # seed each folder with the clairvoyant rows so that they are not recomputed
        (V4 / f"env_{env}").mkdir(parents=True, exist_ok=True)
        part = pd.read_csv(SWEEP / f"env_{env}" / "results.csv")
        part[part.controller == "ClairvoyantCVT"].to_csv(V4 / f"env_{env}" / "results.csv", index=False)
    run_parallel(lambda env: [sys.executable, "-m", "lpac_study.sweep", "--out-dir", str(V4 / f"env_{env}"),
                              "--ranges", "256", "512", "1024", "--controllers", "lpac", "ClairvoyantCVT",
                              "--no-constant-degree", "--no-rank", "--model", str(ckpt), "--variant", "V4",
                              *env_args(env)], ENVS, "v4")
    merge([V4 / f"env_{e}" for e in ENVS], V4)
    v4 = pd.read_csv(V4 / "results.csv")
    v4 = v4[v4.controller == "lpac"].groupby("comm_range").ratio_mean.mean()
    display(pd.DataFrame({"LPAC as published": mean_asis, "fixed 256 m map": fixed, "V4 (mixed ranges)": v4}).round(2))
"""

OUTRO = f"""
## What next

* The post explains each result and its limits; the 40-environment numbers there are more precise
  than the 2 environments of quick mode.
* The project site has every number of the study, interactive: [{SITE}]({SITE}).
* The last cell zips `{{OUT}}` for download.
"""

ZIP = """
import shutil
zip_path = shutil.make_archive(str(OUT.parent / "lpac_range_results"), "zip", root_dir=OUT)  # outside OUT
print("archive:", zip_path)
try:
    from google.colab import files
    files.download(zip_path)
except Exception as error:  # not on Colab
    print("download it manually:", error)
"""


def package_cells() -> list[dict]:
    cells = [code("import os\nos.makedirs('lpac_study', exist_ok=True)")]
    for name in MODULES:
        source = (PACKAGE / f"{name}.py").read_text(encoding="utf8")
        cells.append(code(f"%%writefile lpac_study/{name}.py\n{source}"))
    return cells


def build() -> dict:
    cells = [md(INTRO), code(INSTALL), code(SETTINGS), code(DOWNLOAD), md(PACKAGE_MD), *package_cells(),
             code(HELPERS), code(CHECK), md(PART1_MD), code(PART1), md(PART2_MD), code(PART2), md(PART3_MD),
             code(PART3), md(PART4_MD), code(PART4), md(OUTRO), code(ZIP)]
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                         "language_info": {"name": "python"}, "colab": {"provenance": []}},
            "nbformat": 4, "nbformat_minor": 5}


if __name__ == "__main__":
    out = HERE / "lpac_range_colab.ipynb"
    out.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n", encoding="utf8")
    print("wrote", out)
