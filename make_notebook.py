"""Generate ``lpac_eval.ipynb``, a self-contained Colab notebook of the study.

    python make_notebook.py

The notebook embeds the ``lpac_study`` package as ``%%writefile`` cells so
that it runs top to bottom on a fresh Colab runtime (GPU T4 or CPU) with no
other file than itself.  Re-run this script after editing the package.
"""
from __future__ import annotations

import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
PACKAGE = PROJECT / "lpac_study"
MODULES = ["__init__", "paths", "simulation", "graph_stats", "effective_rank",
           "lpac_policy", "episode", "sweep", "merge", "plots",
           "neighbourhood", "checks", "factorial", "plots_factorial"]
RELEASE_URL = ("https://github.com/KumarRobotics/CoverageControl/releases/download/"
               "v1.3.1/lpac_CoverageControl.tar.gz")


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n")}


INTRO = """
# LPAC et portée de communication : le modèle publié se dégrade-t-il au-delà de 512 m ?

Projet CS 224W (Stanford, automne 2026). Ce notebook reproduit, **sans aucun entraînement**,
l'expérience de la section VI-H de *LPAC* (Agarwal et al., IEEE T-RO 2025, arXiv:2401.04855)
avec le modèle pré-entraîné publié dans la release v1.3.1 de
[KumarRobotics/CoverageControl](https://github.com/KumarRobotics/CoverageControl).

Il s'exécute de haut en bas (`Runtime → Run all`) et écrit ses résultats dans `OUT_DIR`
(par défaut `/content/lpac_results`, ou sur Google Drive si `USE_DRIVE = True`) :

| fichier | contenu |
|---|---|
| `results.csv` | une ligne par (environnement, portée, nombre de robots, contrôleur) : coût normalisé, degré moyen |
| `rank.csv` | rang effectif en sortie de chaque couche TAGConv (LPAC) |
| `curves.csv` | courbes de coût sous-échantillonnées |
| `fig_portee.png`, `fig_degre.png`, `fig_degre_constant.png`, `fig_rang.png` | figures |
| `lpac_results.zip` | le tout, à télécharger |

**Durée.** Les baselines CVT tournent en C++ sur CPU (Colab n'en a que 2) et LPAC sur GPU.
Comptez 3 à 5 h pour le balayage complet. Le balayage est **reprenable** : relancer le
notebook saute les lignes déjà présentes dans `results.csv` (d'où l'intérêt de Drive).
Pour un test rapide, mettez `QUICK = True` (1 environnement, 60 pas, ~2 min).

**Mise en garde.** Un seul modèle est publié, entraîné à 128 m. L'évaluer à 1024 m mélange
l'effet de la densité du graphe et l'écart au régime d'entraînement ; le balayage à degré
constant sert à les séparer, au prix d'un autre défaut (le nombre de robots change la tâche).
"""

INSTALL = """
# Dépendances : le simulateur (roue manylinux, Python 3.10 à 3.13) et PyTorch Geometric.
# torch et torchvision sont déjà présents sur Colab.
import importlib, subprocess, sys
for pkg, spec in [("coverage_control", "coverage_control==1.3.1"), ("torch_geometric", "torch_geometric")]:
    if importlib.util.find_spec(pkg) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", spec])
import torch, torch_geometric, coverage_control
print("torch", torch.__version__, "| PyG", torch_geometric.__version__,
      "| coverage_control", coverage_control.__version__, "| cuda", torch.cuda.is_available())
"""

SETTINGS = """
# Réglages du run. Tout est public et déterministe (environnements et modèle de la release).
import os
from pathlib import Path
QUICK = os.environ.get("LPAC_QUICK", "0") == "1"   # True : 1 environnement, 60 pas, pour vérifier que tout tourne
USE_DRIVE = False      # True : résultats persistants dans Google Drive (reprise après déconnexion)
ROOT = Path(os.environ.get("LPAC_NB_ROOT", "/content"))  # racine de travail (Colab : /content)
NUM_ENVS = 11          # environnements d'évaluation disponibles dans la release
NUM_STEPS = 600        # NumSteps de eval.toml
RANGES = [128, 256, 512, 768, 1024]                      # portées (m), 32 robots
CONSTANT_DEGREE = ["256:32", "512:11", "768:7", "1024:6"]  # (portée, robots) à degré ~constant

if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    OUT_DIR = Path("/content/drive/MyDrive/lpac_results")
else:
    OUT_DIR = ROOT / "lpac_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
if QUICK:
    NUM_ENVS, NUM_STEPS, RANGES, CONSTANT_DEGREE = 1, 60, [128, 1024], ["1024:6"]
print("OUT_DIR =", OUT_DIR)
"""

DOWNLOAD = f"""
# Modèle pré-entraîné, 11 environnements d'évaluation et paramètres des auteurs.
import os, tarfile, urllib.request
DATA_ROOT = ROOT / "lpac_data"
DATA_DIR = DATA_ROOT / "lpac"
if not (DATA_DIR / "models" / "model_k3_1024_state_dict.pt").is_file():
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    archive = DATA_ROOT / "lpac_CoverageControl.tar.gz"
    urllib.request.urlretrieve("{RELEASE_URL}", archive)
    with tarfile.open(archive) as tar:
        tar.extractall(DATA_ROOT)
os.environ["LPAC_DATA_DIR"] = str(DATA_DIR)
print(sorted(p.name for p in DATA_DIR.iterdir()), "|", len(list((DATA_DIR / "envs").glob("*.env"))), "environnements")
"""

PACKAGE_INTRO = """
## Le code : package `lpac_study`

Les cellules suivantes écrivent le package sur le disque du runtime. Il est identique à celui
du dépôt du projet. Tout ce qui touche au graphe passe par PyTorch Geometric :
`CoverageEnvUtils.get_torch_geometric_data` construit l'objet `Data` (graphe à rayon,
symétrique), `torch_geometric.utils.degree` mesure le degré, et le GNN est une pile de
cinq `torch_geometric.nn.TAGConv` (K = 3). Note de lecture : `LPAC.forward` ne transmet
pas `edge_weight` au GNN, l'adjacence vue par TAGConv est donc binaire, normalisée
symétriquement, sans boucles propres.
"""

RUN = """
# Balayage complet, reprenable. Le GPU sert au forward de LPAC ; les CVT tournent en C++ (CPU).
import subprocess, sys
cmd = [sys.executable, "-m", "lpac_study.sweep", "--data-dir", str(DATA_DIR), "--out-dir", str(OUT_DIR / "raw"),
       "--num-envs", str(NUM_ENVS), "--num-steps", str(NUM_STEPS), "--ranges", *map(str, RANGES),
       "--constant-degree", *CONSTANT_DEGREE, "--device", "auto"]
print(" ".join(cmd))
subprocess.run(cmd, check=True)
"""

MERGE = """
# Fusion (un seul processus ici), tableau récapitulatif et figures.
import subprocess, sys
subprocess.run([sys.executable, "-m", "lpac_study.merge", "--parts", str(OUT_DIR / "raw"), "--out-dir", str(OUT_DIR)], check=True)
subprocess.run([sys.executable, "-m", "lpac_study.plots", "--results-dir", str(OUT_DIR), "--fig-dir", str(OUT_DIR)], check=True)
from IPython.display import Image, display
for name in ["fig_portee.png", "fig_degre.png", "fig_degre_constant.png", "fig_rang.png"]:
    if (OUT_DIR / name).is_file():
        display(Image(filename=str(OUT_DIR / name)))
"""

EXPORT = """
# Archive à télécharger et à rapporter.
import shutil
zip_path = shutil.make_archive(str(OUT_DIR / "lpac_results"), "zip", root_dir=OUT_DIR,
                               base_dir=".", verbose=False)
print("archive :", zip_path)
try:
    from google.colab import files
    files.download(zip_path)
except Exception as error:  # hors Colab
    print("téléchargement manuel :", error)
"""


FACTORIAL_INTRO = """
## Inference-time factorial: which input carries the neighbourhood that breaks the model?

The neighbourhood enters LPAC twice: as the `edge_index` of the five `TAGConv` layers (factor
**GRAPH**: G0 every edge in range, G1 the k nearest robots in range, made undirected with
`torch_geometric.utils.to_undirected`) and as the two communication-map channels of the CNN input
(factor **MAP**: C0 the library's map, C1 only the k nearest, C2 only robots within 128 m drawn at the
128 m resolution). The simulator always runs at the test range; only the model inputs change. The
checks cell first verifies that the rebuilt maps and graph match the library exactly.
"""

CHECKS = """
subprocess.run([sys.executable, "-m", "lpac_study.checks", "--data-dir", str(DATA_DIR), "--comm-range", "1024", "--steps", "10"], check=True)
"""

FACTORIAL = """
# Factorial at 1024 m (six cells, k = 4), normalized by the clairvoyant rows of the range sweep above.
FACT_RANGES, FACT_K, FACT_ENVS, FACT_STEPS = ["1024"], ["4"], NUM_ENVS, NUM_STEPS
if QUICK:
    FACT_ENVS, FACT_STEPS = 1, 20
cmd = [sys.executable, "-m", "lpac_study.factorial", "--data-dir", str(DATA_DIR), "--baseline", str(OUT_DIR / "results.csv"),
       "--out-dir", str(OUT_DIR / "factorial"), "--ranges", *FACT_RANGES, "--k", *FACT_K,
       "--num-envs", str(FACT_ENVS), "--num-steps", str(FACT_STEPS)]
subprocess.run(cmd, check=True)
subprocess.run([sys.executable, "-m", "lpac_study.plots_factorial", "--results", str(OUT_DIR / "factorial" / "results_2x2.csv"),
                "--baseline", str(OUT_DIR / "results.csv"), "--fig-dir", str(OUT_DIR)], check=True)
for name in ["fig_2x2.png", "fig_dose.png", "fig_fix.png"]:
    if (OUT_DIR / name).is_file():
        display(Image(filename=str(OUT_DIR / name)))
"""


def package_cells() -> list[dict]:
    cells = [markdown(PACKAGE_INTRO), code("import os\nos.makedirs('lpac_study', exist_ok=True)")]
    for name in MODULES:
        source = (PACKAGE / f"{name}.py").read_text(encoding="utf-8")
        cells.append(code(f"%%writefile lpac_study/{name}.py\n{source}"))
    return cells


def build_notebook() -> dict:
    cells = [markdown(INTRO), code(INSTALL), code(SETTINGS), code(DOWNLOAD)]
    cells += package_cells()
    cells += [markdown("## Balayages"), code(RUN), markdown("## Résultats"), code(MERGE),
              markdown(FACTORIAL_INTRO), code(CHECKS), code(FACTORIAL), code(EXPORT)]
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU", "colab": {"name": "lpac_eval.ipynb", "provenance": []},
        },
        "cells": cells,
    }


if __name__ == "__main__":
    target = PROJECT / "lpac_eval.ipynb"
    target.write_text(json.dumps(build_notebook(), indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", target)
