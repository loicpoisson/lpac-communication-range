#!/usr/bin/env bash
# Run the whole study from scratch on a Linux/WSL machine (CPU is enough, no training).
#
#   bash run_all.sh [WORK_DIR]
#
# WORK_DIR (default ~/lpac) receives the venv, the release archive and the raw
# results; the deliverables (results.csv, rank.csv, curves.csv, figures) are
# written next to this script.  Re-running resumes: finished rows are skipped.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${1:-$HOME/lpac}"
VENV="$WORK_DIR/venv"
DATA_DIR="$WORK_DIR/data/lpac"
RESULTS_DIR="$WORK_DIR/results"
RELEASE_URL="https://github.com/KumarRobotics/CoverageControl/releases/download/v1.3.1/lpac_CoverageControl.tar.gz"
NUM_PARALLEL="${NUM_PARALLEL:-4}"

mkdir -p "$WORK_DIR/data" "$RESULTS_DIR"

# 1. Python environment (uv avoids the python3-venv system package).
if [ ! -x "$VENV/bin/python" ]; then
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  uv venv --python 3.12 "$VENV"
  uv pip install --python "$VENV/bin/python" pip
  "$VENV/bin/python" -m pip install -q torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
  "$VENV/bin/python" -m pip install -q torch_geometric numpy pyyaml rich toml matplotlib pandas
  "$VENV/bin/python" -m pip install -q coverage_control==1.3.1
fi

# 2. Published model, environments and parameters.
if [ ! -f "$DATA_DIR/models/model_k3_1024_state_dict.pt" ]; then
  curl -sSL -o "$WORK_DIR/data/lpac_CoverageControl.tar.gz" "$RELEASE_URL"
  tar -xzf "$WORK_DIR/data/lpac_CoverageControl.tar.gz" -C "$WORK_DIR/data"
fi

# 3. Sweeps, split across independent processes (each one is resumable).
export LPAC_DATA_DIR="$DATA_DIR" OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 OMP_WAIT_POLICY=PASSIVE
cd "$PROJECT_DIR"
run() { "$VENV/bin/python" -m lpac_study.sweep --reuse-range-independent "$@"; }
if [ "$NUM_PARALLEL" -ge 4 ]; then
  run --out-dir "$RESULTS_DIR/part_a" --ranges 128 256 --no-constant-degree &
  run --out-dir "$RESULTS_DIR/part_b" --ranges 512 768 --no-constant-degree &
  run --out-dir "$RESULTS_DIR/part_c" --ranges 1024 --constant-degree 256:32 &
  run --out-dir "$RESULTS_DIR/part_d" --ranges --constant-degree 512:11 768:7 1024:6 &
  wait
else
  run --out-dir "$RESULTS_DIR/part_a"
fi

# 4. Merge, figures.
"$VENV/bin/python" -m lpac_study.merge --parts "$RESULTS_DIR"/part_* --out-dir "$PROJECT_DIR/results"
cp "$PROJECT_DIR/results/results.csv" "$PROJECT_DIR/results.csv"
"$VENV/bin/python" -m lpac_study.plots --results-dir "$PROJECT_DIR/results" --fig-dir "$PROJECT_DIR"
# 5. Inference-time factorial (graph x neighbour map) at 1024 m, then figures.
"$VENV/bin/python" -m lpac_study.checks --comm-range 1024 --steps 10
"$VENV/bin/python" -m lpac_study.factorial --baseline "$PROJECT_DIR/results/results.csv"   --out-dir "$PROJECT_DIR/results_2x2" --ranges 1024 --k 4
cp "$PROJECT_DIR/results_2x2/results_2x2.csv" "$PROJECT_DIR/results_2x2.csv"
"$VENV/bin/python" -m lpac_study.plots_factorial --results "$PROJECT_DIR/results_2x2/results_2x2.csv"   --baseline "$PROJECT_DIR/results/results.csv" --fig-dir "$PROJECT_DIR"
echo "Done: see $PROJECT_DIR/results.csv, results_2x2.csv and fig_*.png"
