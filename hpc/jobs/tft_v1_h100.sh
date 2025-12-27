#!/bin/bash
#SBATCH --job-name=tft_h100_fix
#SBATCH --partition=gpuh100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/shared/%u/miracle/logs/%x_%j.out
#SBATCH --error=/shared/%u/miracle/logs/%x_%j.err

set -euo pipefail

echo "HOST=$(hostname)"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-NA}"
date

REPO="/shared/$USER/miracle/pv_forecast_30d"
IMG="/shared/$USER/miracle/containers/tft_env_v1.sif"

SRC_TRAIN="$REPO/data/processed/pretraining/germany/global/tft_inputs_pca32/train_pca32.parquet"
SRC_VAL="$REPO/data/processed/pretraining/germany/global/tft_inputs_pca32/val_pca32.parquet"

LOCAL_DIR="/tmp/$USER/$SLURM_JOB_ID"

echo "=== SETUP ==="
echo "HOST: $(hostname)"
echo "REPO: $REPO"
echo "IMG:  $IMG"
mkdir -p "$LOCAL_DIR"

echo "=== COPY DATA TO LOCAL SSD ==="
cp "$SRC_TRAIN" "$LOCAL_DIR/train.parquet"
cp "$SRC_VAL"   "$LOCAL_DIR/val.parquet"
ls -lah "$LOCAL_DIR" | head

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Use env-based binding to avoid forbidden singularity flags
export SINGULARITY_BINDPATH="/shared/$USER:/shared/$USER,/home/$USER:/home/$USER,/tmp:/tmp,/dev/shm:/dev/shm"

echo "=== CONTAINER SMOKE TEST ==="
singularity exec --nv "$IMG" bash -lc "
  set -euo pipefail
  echo 'IN CONTAINER HOST:' \$(hostname)
  echo 'PWD:' \$(pwd)
  python3 - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda_available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu:', torch.cuda.get_device_name(0))
PY
  ls -lah $LOCAL_DIR | head
"

echo "=== TRAIN ==="
singularity exec --nv "$IMG" bash -lc "
  set -euo pipefail
  cd $REPO
  export PYTHONPATH="$REPO:${PYTHONPATH:-}"

  python3 -m src.training.train_tft_v1 \
    --train_parquet $LOCAL_DIR/train.parquet \
    --val_parquet   $LOCAL_DIR/val.parquet \
    --use_lstm_encodings \
    --enc_lag 96 \
    --max_epochs 30 \
    --batch_size 512 \
    --grad_accum_steps 8 \
    --num_workers 16 \
    --prefetch_factor 2 \
    --precision bf16-mixed \
    --enable_amp \
    --lr 2e-3
"

echo "=== CLEANUP ==="
rm -rf "$LOCAL_DIR"
echo "DONE"
