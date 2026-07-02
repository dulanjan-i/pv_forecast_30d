#!/bin/bash
#SBATCH --job-name=tft_fast
#SBATCH --partition=gpuh100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/shared/%u/miracle/logs/%x_%j.out
#SBATCH --error=/shared/%u/miracle/logs/%x_%j.err

set -euo pipefail

# --- 1. DETECT ASSIGNED GPU (CRITICAL FIX) ---
# If SLURM_JOB_GPUS is set, use it. Otherwise default to 0.
ALLOCATED_GPU="${SLURM_JOB_GPUS:-0}"
echo "------------------------------------------------"
echo "SLURM Assigned GPU ID: $ALLOCATED_GPU"
echo "------------------------------------------------"

# --- 2. SETUP LOCAL SCRATCH ---
SOURCE_REPO="/shared/$USER/miracle/pv_forecast_30d"
CONTAINER="/shared/$USER/miracle/containers/tft_env_v1.sif"
LOCAL_DIR="/tmp/$USER/$SLURM_JOB_ID"

mkdir -p "$LOCAL_DIR"
echo "Created local scratch: $LOCAL_DIR"

SRC_TRAIN="$SOURCE_REPO/data/processed/pretraining/germany/global/tft_inputs/regional_train_tft_full.parquet"
SRC_VAL="$SOURCE_REPO/data/processed/pretraining/germany/global/tft_inputs/regional_val_tft_full.parquet"
DEST_TRAIN="$LOCAL_DIR/train.parquet"
DEST_VAL="$LOCAL_DIR/val.parquet"

# --- 3. COPY DATA ---
echo "Copying data to local SSD..."
cp "$SRC_TRAIN" "$DEST_TRAIN"
cp "$SRC_VAL" "$DEST_VAL"
echo "Data copy finished."

# --- 4. RUN TRAINING ---
echo "Starting training on H100..."

# We pass the ALLOCATED_GPU to the container explicitly.
# We also reduce num_workers to 4 to be safe.
singularity exec -C --nv \
  --env CUDA_VISIBLE_DEVICES=$ALLOCATED_GPU \
  --bind /shared/$USER:/shared/$USER,/home/$USER:/home/$USER,/tmp:/tmp \
  --pwd "$SOURCE_REPO" \
  "$CONTAINER" \
  bash -c "
    echo \"Container now sees ONLY GPU(s): \$CUDA_VISIBLE_DEVICES\"
    
    export PYTHONPATH=$SOURCE_REPO:\$PYTHONPATH
    python3 -m src.training.train_tft_v1 \
      --train_parquet $DEST_TRAIN \
      --val_parquet   $DEST_VAL \
      --use_lstm_encodings \
      --enc_lag 96 \
      --max_epochs 30 \
      --batch_size 512 \
      --num_workers 0 \
      --gpus 1 \
      --precision 32-true
"

# --- 5. CLEANUP ---
rm -rf "$LOCAL_DIR"
echo "Done."