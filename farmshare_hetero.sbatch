#!/usr/bin/env bash
#SBATCH --job-name=lpac-hetero
#SBATCH --array=0-10
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=02:00:00
#SBATCH --constraint=CPU_MNF:INTEL
#SBATCH --output=slurm-hetero-%A_%a.out
# Heterogeneous communication ranges at inference (lpac_study.heterogeneous), one environment per task.
#
#   OUT_ROOT=~/lpac_project/results_hetero sbatch farmshare_hetero.sbatch
#
# Variables: FRACTIONS (default "0 0.25 0.5 0.75 1"), DRAWS (default "0 1"), MAPS (default "asis fixed256"),
# LONG (default 1024), SHORT (default 128), PROBE_EVERY (default 50), EXTRA_ARGS (e.g. "--model ... --model-tag ..."),
# DATA_DIR, BASELINE, OUT_ROOT, NUM_ENVS. Intel nodes only (bit-reproducibility, see farmshare_factorial.sbatch).
set -euo pipefail
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
WORK_DIR="${WORK_DIR:-$HOME/lpac}"
export LPAC_DATA_DIR="${DATA_DIR:-$WORK_DIR/data/lpac}" OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 OMP_WAIT_POLICY=PASSIVE
ENV_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
cd "$PROJECT_DIR"
# shellcheck disable=SC2086
"$WORK_DIR/venv/bin/python" -m lpac_study.heterogeneous \
  --baseline "${BASELINE:-$PROJECT_DIR/results/results.csv}" \
  --out-dir "${OUT_ROOT:-$PROJECT_DIR/results_hetero}/env_$ENV_INDEX" \
  --fractions ${FRACTIONS:-0 0.25 0.5 0.75 1} --draws ${DRAWS:-0 1} --maps ${MAPS:-asis fixed256} \
  --semantics ${SEMANTICS:-reception} \
  --long-range "${LONG:-1024}" --short-range "${SHORT:-128}" --probe-every "${PROBE_EVERY:-50}" \
  --env-start "$ENV_INDEX" --env-end "$((ENV_INDEX + 1))" --num-envs "${NUM_ENVS:-11}" ${EXTRA_ARGS:-}
