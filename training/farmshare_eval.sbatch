#!/usr/bin/env bash
#SBATCH --job-name=lpac-eval
#SBATCH --array=0-10
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=02:00:00
#SBATCH --constraint=CPU_MNF:INTEL
#SBATCH --output=slurm-eval-%A_%a.out
# Evaluate a retrained LPAC-K3 checkpoint with lpac_study.sweep, one environment per task.
#
#   MODEL=/path/model_best_state_dict.pt OUT=~/lpac_project/results_train sbatch training/farmshare_eval.sbatch
#
# Variables: MODEL (state dict, required), OUT (output folder), RANGES (default "128 256").
set -euo pipefail
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export LPAC_DATA_DIR="$HOME/lpac/data/lpac" OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 OMP_WAIT_POLICY=PASSIVE
ENV_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
cd "$PROJECT_DIR"
# shellcheck disable=SC2086
"$HOME/lpac/venv/bin/python" -m lpac_study.sweep --out-dir "${OUT:-$PROJECT_DIR/results_train}/env_$ENV_INDEX" \
  --ranges ${RANGES:-128 256} --no-constant-degree --controllers lpac ClairvoyantCVT \
  --env-start "$ENV_INDEX" --env-end "$((ENV_INDEX + 1))" --model "$MODEL"
