#!/usr/bin/env bash
#SBATCH --job-name=lpac-gen
#SBATCH --array=0-9
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --constraint=CPU_MNF:INTEL
#SBATCH --output=slurm-gen-%A_%a.out
# Data generation with the authors' data_generation.py (CoverageControl v1.3.1), one part per array task.
#
#   SAMPLES_PER_PART=2000 OUT=~/lpac/train/large sbatch training/farmshare_generate.sbatch
#
# Each task writes OUT/part_<index>/ (unsplit); training/merge_datasets.py merges the parts
# into OUT/data/ with the train/val/test split and the authors' action normalization.
# Note: the simulator seeds its environment generator from std::random_device, so the
# generated samples are not reproducible from a seed; the generated files are the reference.
set -euo pipefail
OUT="${OUT:-$HOME/lpac/train/large}"
SAMPLES="${SAMPLES_PER_PART:-2000}"
REPO="${REPO:-$HOME/lpac/src/CoverageControl}"
VENV="${VENV:-$HOME/lpac/venv}"
PART="$(( ${PART_OFFSET:-0} + ${SLURM_ARRAY_TASK_ID:-0} ))"  # PART_OFFSET lets a second array add parts
export OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 OMP_WAIT_POLICY=PASSIVE
PARAMS="${PARAMS:-$HOME/lpac/data/lpac/params/coverage_control_params.toml}"  # e.g. a copy with another CommunicationRange
mkdir -p "$OUT/params"
cp -n "$PARAMS" "$OUT/params/coverage_control_params.toml" 2>/dev/null || true
CONFIG="$OUT/params/data_params_part_$PART.toml"
sed -e "s|^DataDir = .*|DataDir = \"$OUT\"|" \
    -e "s|^EnvironmentConfig = .*|EnvironmentConfig = \"$OUT/params/coverage_control_params.toml\"|" \
    -e "s|^NumDataset = .*|NumDataset = $SAMPLES|" \
    -e "s|^EveryNumSteps = .*|EveryNumSteps = ${EVERY_NUM_STEPS:-5}|" \
    "$HOME/lpac/data/lpac/params/data_params.toml" > "$CONFIG"
cd "$REPO/python/data_generation"
/usr/bin/time -f "GEN_WALL %e s MAXRSS %M kB" "$VENV/bin/python" data_generation.py "$CONFIG" --append-dir "part_$PART" --split False
