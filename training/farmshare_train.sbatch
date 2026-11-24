#!/usr/bin/env bash
#SBATCH --job-name=lpac-train
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-train-%j.out
# LPAC training on FarmShare with the authors' building blocks (train_launch.py).
#
#   sbatch -p gpu --gres=gpu:1 farmshare_train.sbatch   # GPU (oat nodes, L40S)
#   sbatch -p normal --constraint=CPU_MNF:INTEL farmshare_train.sbatch   # CPU fallback
#
# Variables: CONFIG (learning_params.toml, required), EPOCHS, VENV (default ~/lpac/venv_gpu if it
# exists else ~/lpac/venv), WORLD (default 1024), SEED (default 0), WORKERS (default cpus-per-task - 2).
set -euo pipefail
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
# train_launch.py lives in training/; submitting from the repository root used to put PROJECT_DIR
# one level above it and the job died in five seconds with "No such file or directory"
[ -f "$PROJECT_DIR/train_launch.py" ] || PROJECT_DIR="$PROJECT_DIR/training"
[ -f "$PROJECT_DIR/train_launch.py" ] || { echo "train_launch.py not found under $PROJECT_DIR" >&2; exit 2; }
VENV="${VENV:-$([ -x "$HOME/lpac/venv_gpu/bin/python" ] && echo "$HOME/lpac/venv_gpu" || echo "$HOME/lpac/venv")}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
cd "$PROJECT_DIR"
nvidia-smi -L 2>/dev/null || echo "no GPU visible"
# Two guards, both added after runs of 7-9 November were lost. (1) A hard cap just under the wall
# time: one job hung in the multiprocessing shutdown and held its GPU for 28 h after Slurm's own
# TIMEOUT. (2) A first-epoch check: with too few loader workers an epoch took 3000-4300 s instead
# of 364 s and six jobs hit the wall with a quarter of their epochs; a job whose first epoch
# exceeds FIRST_EPOCH_LIMIT seconds (default 600), or that has none after 25 minutes, is stopped.
MODEL_DIR="$(grep -m1 -E '^Dir *= *' "$CONFIG" | sed -E 's/^Dir *= *"(.*)".*/\1/')"
rm -f "$MODEL_DIR/timing.csv"
timeout --kill-after=60 "${HARD_CAP:-350m}" "$VENV/bin/python" train_launch.py "$CONFIG" "${WORLD:-1024}" ${EPOCHS:+--epochs "$EPOCHS"} \
  --seed "${SEED:-0}" --num-workers "${WORKERS:-$(( ${SLURM_CPUS_PER_TASK:-4} - 2 ))}" --variant "${VARIANT:-V0}" &
TRAIN_PID=$!
(
  for _ in $(seq 1 150); do
    sleep 10
    kill -0 "$TRAIN_PID" 2>/dev/null || exit 0
    if [ -f "$MODEL_DIR/timing.csv" ] && [ "$(wc -l < "$MODEL_DIR/timing.csv")" -ge 2 ]; then
      first=$(awk -F, 'NR==2{print int($4+$5)}' "$MODEL_DIR/timing.csv")
      if [ "$first" -gt "${FIRST_EPOCH_LIMIT:-600}" ]; then
        echo "GUARD: first epoch took ${first}s, limit ${FIRST_EPOCH_LIMIT:-600}s; stopping the job"; kill "$TRAIN_PID"
      else
        echo "GUARD: first epoch took ${first}s, within the limit"
      fi
      exit 0
    fi
  done
  echo "GUARD: no epoch finished within 25 minutes; stopping the job"; kill "$TRAIN_PID"
) &
wait "$TRAIN_PID"
