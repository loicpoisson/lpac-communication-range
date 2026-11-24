#!/usr/bin/env bash
# Queue the evaluations of the final-report trainings, each one starting only when its training
# job ends successfully (--dependency=afterok), so no login session is needed in between.
# Same protocols as the earlier seeds (run_info.json of results_V45/V4mixed_s0_40 and results_sem):
#   sweep 11 release envs and 40 new envs at 128/256/512/768/1024 m, controllers lpac + ClairvoyantCVT;
#   heterogeneous teams: 11 envs, fractions 0-1, draws 0-3, map asis, long 1024 m, short 128 m.
#
#   cd ~/lpac_project && bash training/submit_final_evals.sh R-V4mix-s3=2018603 ...
# Each argument is TAG=TRAINJOBID; the table below maps each tag to its checkpoint and protocol.
set -euo pipefail
cd "$HOME/lpac_project"
T="$HOME/lpac/train"
OUT="$HOME/lpac_project/results_final"
mkdir -p "$OUT"
RANGES5="128 256 512 768 1024"

sweep() {  # dep tag model variant envs ranges
  local dep=$1 tag=$2 model=$3 variant=$4 envs=$5 ranges=$6 data array
  if [ "$envs" = 40 ]; then data="$HOME/lpac/data/lpac_new40"; array=0-39; else data="$HOME/lpac/data/lpac"; array=0-10; fi
  MODEL="$model" VARIANT="$variant" OUT="$OUT/${tag}_${envs}" RANGES="$ranges" NUM_ENVS="$envs" DATA_DIR="$data" \
    sbatch --parsable --dependency=afterok:"$dep" --kill-on-invalid-dep=yes --array="$array" --time=04:00:00 \
    --export=ALL --job-name="ev-${tag}-${envs}" training/farmshare_eval.sbatch
}
hetero() {  # dep tag model variant semantics...
  local dep=$1 tag=$2 model=$3 variant=$4; shift 4
  SEMANTICS="$*" DRAWS="0 1 2 3" MAPS="asis" OUT_ROOT="$OUT/hetero_${tag}_${1}" \
    EXTRA_ARGS="--model $model --model-tag $tag --variant $variant" \
    sbatch --parsable --dependency=afterok:"$dep" --kill-on-invalid-dep=yes --array=0-10 --time=08:00:00 \
    --export=ALL --job-name="hx-${tag}-${1}" farmshare_hetero.sbatch
}

for pair in "$@"; do
  name=${pair%%=*}; dep=${pair#*=}
  case "$name" in
    R-V4mix-s[0-9])
      s=${name##*-s}; tag="V4mixed_s$s"; m="$T/V45/models_V4mixed_s$s/model_best_state_dict.pt"
      sweep "$dep" "$tag" "$m" V4 11 "$RANGES5"; sweep "$dep" "$tag" "$m" V4 40 "$RANGES5"
      hetero "$dep" "$tag" "$m" V4 reception; hetero "$dep" "$tag" "$m" V4 transmission sym_max sym_min ;;
    R-LPAC256-s[0-9])
      s=${name##*-s}; tag="LPAC256_s$s"; m="$T/large256/models_s$s/model_best_state_dict.pt"
      sweep "$dep" "$tag" "$m" V0 11 "$RANGES5"; sweep "$dep" "$tag" "$m" V0 40 "$RANGES5" ;;
    H-V4T-s[0-9])
      s=${name##*-s}; tag="V4T_s$s"; m="$T/heteroT/models_V4T_s$s/model_best_state_dict.pt"
      sweep "$dep" "$tag" "$m" V4 11 "$RANGES5"
      hetero "$dep" "$tag" "$m" V4 transmission; hetero "$dep" "$tag" "$m" V4 reception sym_max sym_min ;;
    M-V4half-s[0-9])
      s=${name##*-s}; tag="V4half_s$s"; m="$T/half_models/models_V4half_s$s/model_best_state_dict.pt"
      sweep "$dep" "$tag" "$m" V4 11 "$RANGES5"; sweep "$dep" "$tag" "$m" V4 40 "$RANGES5"
      hetero "$dep" "$tag" "$m" V4 reception ;;
    Q3-V2M)
      m="$T/Q3/models_V2M/model_best_state_dict.pt"
      sweep "$dep" V2M "$m" V2M 11 "$RANGES5"; sweep "$dep" V2M "$m" V2M 40 "$RANGES5" ;;
    *) echo "unknown training tag $name" >&2; exit 2 ;;
  esac
done
