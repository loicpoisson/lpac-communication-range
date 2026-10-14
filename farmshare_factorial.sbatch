#!/usr/bin/env bash
#SBATCH --job-name=lpac-2x2
#SBATCH --array=0-10
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=06:00:00
#SBATCH --constraint=CPU_MNF:INTEL
# Intel nodes only: on the AMD nodes (rye, barley) the CNN kernels round differently and the
# 600-step trajectories diverge (up to 5 % on one environment's mean cost); Intel AVX-512 nodes
# reproduce the laptop results bit for bit.
#SBATCH --output=slurm-%A_%a.out
# Slurm array job for Stanford FarmShare (CPU only, no training): one task per environment.
#
#   # once, on a login node, from the project folder copied to FarmShare:
#   bash run_all.sh ~/lpac            # or just steps 1-2 of it, to create ~/lpac/venv and ~/lpac/data
#   RANGES=768 sbatch farmshare_factorial.sbatch  # six cells, k = 4, env = array index, one out-dir per range
#   # then: python -m lpac_study.merge-like concatenation of results_2x2/env_*/results_2x2.csv
#
# Variables: RANGES (default "1024"), CELLS (default the six cells of the first factorial), KS (default "4"),
# EXTRA_ARGS (passed to lpac_study.factorial, e.g. "--repeats 5 --repeat-start 1"), TAG (suffix of the
# output folder, so that two arrays at the same range never append to the same CSV).
set -euo pipefail
# Under sbatch the script runs from a spool copy: the project is the submission directory.
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
WORK_DIR="${WORK_DIR:-$HOME/lpac}"
export LPAC_DATA_DIR="$WORK_DIR/data/lpac" OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 OMP_WAIT_POLICY=PASSIVE
ENV_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
cd "$PROJECT_DIR"
# shellcheck disable=SC2086
"$WORK_DIR/venv/bin/python" -m lpac_study.factorial \
  --baseline "$PROJECT_DIR/results/results.csv" \
  --out-dir "$PROJECT_DIR/results_2x2/r${RANGES:-1024}${TAG:-}_env_$ENV_INDEX" \
  --ranges ${RANGES:-1024} --cells ${CELLS:-G0C0 G0C1 G0C2 G1C0 G1C1 G1C2} --k ${KS:-4} \
  --env-start "$ENV_INDEX" --env-end "$((ENV_INDEX + 1))" ${EXTRA_ARGS:-}
