#!/bin/bash
#SBATCH --job-name=tft_manual
#SBATCH --partition=gpuh100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=250G
#SBATCH --time=04:00:00
#SBATCH --output=/shared/%u/miracle/logs/%x_%j.out
#SBATCH --error=/shared/%u/miracle/logs/%x_%j.err

set -euo pipefail

# --- CONFIG ---
REPO="/shared/$USER/miracle/pv_forecast_30d"
IMG="/shared/$USER/miracle/containers/tft_env_v1.sif"
LOCAL_DIR="/tmp/$USER/$SLURM_JOB_ID"

ALLOCATED_GPU="${SLURM_JOB_GPUS:-0}"

mkdir -p "$LOCAL_DIR"

echo "=== COPYING DATA ==="
# Adjust these paths if they differ
cp "$REPO/data/processed/pretraining/germany/global/tft_inputs/regional_train_tft_full.parquet" "$LOCAL_DIR/train.parquet"
cp "$REPO/data/processed/pretraining/germany/global/tft_inputs/regional_val_tft_full.parquet" "$LOCAL_DIR/val.parquet"

# --- RUNNING MANUAL LOOP ---
singularity exec -C --nv \
  --env SLURM_ID_PASS=$ALLOCATED_GPU \
  --bind "/shared/$USER:/shared/$USER,/home/$USER:/home/$USER,/tmp:/tmp,/dev/shm:/dev/shm" \
  --pwd "$REPO" \
  "$IMG" \
  bash -c "
    export CUDA_VISIBLE_DEVICES=\$SLURM_ID_PASS
    export PYTHONPATH=$REPO:\$PYTHONPATH
    
    echo 'Starting Manual Training Loop...'
    
    python3 src/training/train_manual.py \
      --train_parquet $LOCAL_DIR/train.parquet \
      --val_parquet   $LOCAL_DIR/val.parquet \
      --batch_size 2048 \
      --max_epochs 30
  "

rm -rf "$LOCAL_DIR"