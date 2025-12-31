#!/bin/bash
#SBATCH -J plant03_long720
#SBATCH -p gpuh100
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-5
#SBATCH -o /shared/%u/miracle/logs/%x_%A_%a.out
#SBATCH -e /shared/%u/miracle/logs/%x_%A_%a.err

set -euo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

REPO="/shared/$USER/miracle/pv_forecast_30d"
IMG="/shared/$USER/miracle/containers/tft_env_v1.sif"
export SINGULARITY_BINDPATH="/shared:/shared,/home/$USER:/home/$USER,/tmp:/tmp,/dev/shm:/dev/shm"

# ---- data ----
PLANT_HOUR_DIR="$REPO/data/processed/plant_level/plant_03/hourly_longhead"
TRAIN="$PLANT_HOUR_DIR/train.parquet"
VAL="$PLANT_HOUR_DIR/val.parquet"

# ---- global longhead winner (warm start source) ----
GLOBAL_RUN="$REPO/experiments/tft/runs/germany/longhead/global_noleak_target03/hourly720/lr2e-3_do0.15_bs64_acc8_seed42/20251230_135616"
GLOBAL_SD="$GLOBAL_RUN/checkpoints/best_state_dict.pt"

# ---- long head lengths ----
ENC_LEN=168
PRED_LEN=720

# ---- training knobs ----
BS=64
ACCUM=8
MAX_EPOCHS=50
PATIENCE=5
DROP=0.15

# seeds mapping
SEEDS=(42 43 44 42 43 44)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

# regime mapping
if [[ "$SLURM_ARRAY_TASK_ID" -lt 3 ]]; then
  REGIME="cold"
  LR="2e-3"
  INIT=""
else
  REGIME="warm"
  # lower LR for finetune, same logic you used in short-head
  LR="8e-4"
  INIT="--init_state_dict $GLOBAL_SD"
fi

RUN_ROOT="experiments/tft/runs/germany/plant_03/longhead/hourly720/${REGIME}/lr${LR}_do${DROP}_bs${BS}_acc${ACCUM}_seed${SEED}"

cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

echo "REGIME=$REGIME SEED=$SEED LR=$LR DROP=$DROP"
echo "TRAIN=$TRAIN"
echo "VAL=$VAL"
echo "GLOBAL_SD=$GLOBAL_SD"

singularity exec --nv "$IMG" bash -lc "
  set -euo pipefail
  cd \"$REPO\"
  export PYTHONPATH=\"$REPO:\${PYTHONPATH:-}\"
  python3 -m src.training.train_tft_longhead_v1 \
    --train_parquet \"$TRAIN\" \
    --val_parquet   \"$VAL\" \
    --run_root      \"$RUN_ROOT\" \
    --enc_len       \"$ENC_LEN\" \
    --pred_len      \"$PRED_LEN\" \
    --lr            \"$LR\" \
    --dropout       \"$DROP\" \
    --batch_size    \"$BS\" \
    --grad_accum_steps \"$ACCUM\" \
    --max_epochs    \"$MAX_EPOCHS\" \
    --patience      \"$PATIENCE\" \
    --seed          \"$SEED\" \
    --precision     \"bf16-mixed\" \
    --num_workers   8 \
    --prefetch_factor 2 \
    $INIT
"
