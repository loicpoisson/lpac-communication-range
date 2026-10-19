#!/usr/bin/env bash
#SBATCH --job-name=lpac-mix
#SBATCH --array=0-10
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=02:00:00
#SBATCH --constraint=CPU_MNF:INTEL
#SBATCH --output=slurm-mix-%A_%a.out
# Mixture of range specialists in a heterogeneous team (lpac_study.mixture), one environment per task.
#
#   SHORT_MODEL=~/lpac/train/large/models100k/model_best_state_dict.pt LONG_MODEL=~/lpac/train/r1024/models/model_best_state_dict.pt \
#     OUT_ROOT=~/lpac_project/results_h1 sbatch farmshare_mixture.sbatch
#
# Variables: SHORT_MODEL, LONG_MODEL (state dicts), SHORT (128), LONG (1024), MODES ("mixture short long"),
# FRACTIONS, DRAWS, MAPS ("asis"), DATA_DIR, BASELINE, OUT_ROOT, NUM_ENVS. Intel nodes only.
set -euo pipefail
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
WORK_DIR="${WORK_DIR:-$HOME/lpac}"
export LPAC_DATA_DIR="${DATA_DIR:-$WORK_DIR/data/lpac}" OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 OMP_WAIT_POLICY=PASSIVE
ENV_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
cd "$PROJECT_DIR"
# shellcheck disable=SC2086
"$WORK_DIR/venv/bin/python" -m lpac_study.mixture \
  --baseline "${BASELINE:-$PROJECT_DIR/results/results.csv}" \
  --out-dir "${OUT_ROOT:-$PROJECT_DIR/results_h1}/env_$ENV_INDEX" \
  --specialists "${SHORT:-128}=$SHORT_MODEL" "${LONG:-1024}=$LONG_MODEL" \
  --modes ${MODES:-mixture short long} --fractions ${FRACTIONS:-0 0.25 0.5 0.75 1} --draws ${DRAWS:-0 1 2 3} \
  --maps ${MAPS:-asis} --env-start "$ENV_INDEX" --env-end "$((ENV_INDEX + 1))" --num-envs "${NUM_ENVS:-11}"
