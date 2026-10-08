#!/bin/bash
#SBATCH -J longhead_glob720
#SBATCH -p gpuh100
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-7
#SBATCH -o /shared/%u/miracle/logs/%x_%A_%a.out
#SBATCH -e /shared/%u/miracle/logs/%x_%A_%a.err

set -euo pipefail
export OMP_NUM_THREADS=1

REPO="/shared/$USER/miracle/pv_forecast_30d"
IMG="/shared/$USER/miracle/containers/tft_env_v1.sif"
export SINGULARITY_BINDPATH="/shared:/shared,/home/$USER:/home/$USER,/tmp:/tmp,/dev/shm:/dev/shm"

# Source: already no-leak 15min PVLib parquets (plant_03 excluded)
SRC_DIR="$REPO/data/processed/pretraining/germany/global/tft_inputs_pvlib_noleak_target03"
SRC_TRAIN="$SRC_DIR/train.parquet"
SRC_VAL="$SRC_DIR/val.parquet"

# Hourly output
HOUR_DIR="$REPO/data/processed/pretraining/germany/global/tft_inputs_pvlib_noleak_target03_hourly"
HOUR_TRAIN="$HOUR_DIR/train.parquet"
HOUR_VAL="$HOUR_DIR/val.parquet"

# Sweep grid (same as before)
LRS=(8e-4 1.2e-3 2e-3 3e-3 8e-4 1.2e-3 2e-3 3e-3)
DROPS=(0.05 0.05 0.05 0.05 0.15 0.15 0.15 0.15)

LR="${LRS[$SLURM_ARRAY_TASK_ID]}"
DROP="${DROPS[$SLURM_ARRAY_TASK_ID]}"

# Long head lengths
ENC_LEN=168
PRED_LEN=720

# Training knobs for long horizon
BS=64
ACCUM=8
MAX_EPOCHS=30
PATIENCE=3
SEED=42

cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

# Step 1: build hourly parquets once (cheap). Guard against race with a simple check.
if [[ ! -f "$HOUR_TRAIN" || ! -f "$HOUR_VAL" ]]; then
  echo "[INFO] Building hourly parquets at: $HOUR_DIR"
  singularity exec --nv "$IMG" python3 -m src.data.make_hourly_from_15min_parquets \
    --src_train "$SRC_TRAIN" \
    --src_val   "$SRC_VAL" \
    --out_dir   "$HOUR_DIR" \
    --freq "1H"
else
  echo "[INFO] Hourly parquets already exist. Skipping build."
fi

# Step 2: train long-head global model
RUN_ROOT="experiments/tft/runs/germany/longhead/global_noleak_target03/hourly720/lr${LR}_do${DROP}_bs${BS}_acc${ACCUM}_seed${SEED}"

singularity exec --nv "$IMG" python3 -m src.training.train_tft_longhead_v1 \
  --train_parquet "$HOUR_TRAIN" \
  --val_parquet   "$HOUR_VAL" \
  --run_root "$RUN_ROOT" \
  --enc_len "$ENC_LEN" \
  --pred_len "$PRED_LEN" \
  --lr "$LR" \
  --dropout "$DROP" \
  --batch_size "$BS" \
  --grad_accum_steps "$ACCUM" \
  --max_epochs "$MAX_EPOCHS" \
  --patience "$PATIENCE" \
  --seed "$SEED" \
  --precision "bf16-mixed" \
  --num_workers 8 \
  --prefetch_factor 2
