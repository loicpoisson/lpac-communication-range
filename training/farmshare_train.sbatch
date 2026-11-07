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
"$VENV/bin/python" train_launch.py "$CONFIG" "${WORLD:-1024}" ${EPOCHS:+--epochs "$EPOCHS"} \
  --seed "${SEED:-0}" --num-workers "${WORKERS:-$(( ${SLURM_CPUS_PER_TASK:-4} - 2 ))}" --variant "${VARIANT:-V0}"
