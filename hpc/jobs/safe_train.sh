#!/bin/bash
#SBATCH --job-name=tft_v1_manual
#SBATCH --partition=gpuh100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=250G
#SBATCH --time=04:00:00
#SBATCH --output=/shared/%u/miracle/logs/%x_%j.out
#SBATCH --error=/shared/%u/miracle/logs/%x_%j.err

# Version: v1.1.1
# Date: 2025-12-26
# Changes from v1.1:
#   - Fixed ENABLE_AMP bug (was not passing --enable_amp flag correctly)
#   - Hardcoded --enable_amp for reliability
#   - Added nvidia-smi power diagnostics

set -euo pipefail

echo "=== JOB START (UTC) $(date -u +'%Y-%m-%dT%H:%M:%SZ') ==="
echo "Host: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-}"

REPO="/shared/$USER/miracle/pv_forecast_30d"
IMG="/shared/$USER/miracle/containers/tft_env_v1.sif"
LOCAL_DIR="/tmp/$USER/${SLURM_JOB_ID:-manual}"
RUN_ROOT="/shared/$USER/miracle/experiments/tft/runs/germany/v1_0"

mkdir -p "$LOCAL_DIR"
trap 'echo "=== CLEANUP ==="; rm -rf "$LOCAL_DIR"' EXIT

# --- GPU selection ---
ALL_GPUS="${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-}}"
echo "Slurm assigned GPUs: ${ALL_GPUS}"

TARGET_GPU="$(echo "${ALL_GPUS}" | cut -d',' -f1)"
if [ -z "${TARGET_GPU}" ]; then
  echo "[WARN] GPU env var empty, defaulting TARGET_GPU=0"
  TARGET_GPU="0"
fi

echo "Using GPU: ${TARGET_GPU}"

# --- Copy parquet locally ---
echo "=== COPYING DATA ==="
SRC_TRAIN="$REPO/data/processed/pretraining/germany/global/tft_inputs/regional_train_tft_full.parquet"
SRC_VAL="$REPO/data/processed/pretraining/germany/global/tft_inputs/regional_val_tft_full.parquet"

cp "$SRC_TRAIN" "$LOCAL_DIR/train.parquet"
cp "$SRC_VAL"   "$LOCAL_DIR/val.parquet"

echo "Local train: $LOCAL_DIR/train.parquet"
echo "Local val:   $LOCAL_DIR/val.parquet"

echo "=== RUNNING TRAINING ==="

BIND_PATHS="/shared:/shared,/tmp:/tmp"

singularity exec -C --nv --bind "$BIND_PATHS" "$IMG" bash -lc "
  set -euo pipefail

  cd '$REPO'

  export CUDA_VISIBLE_DEVICES='${TARGET_GPU}'
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export PYTHONPATH='$REPO':\${PYTHONPATH:-}

  echo 'Inside container:'
  python3 -V
  nvidia-smi -L

  echo '=== GPU POWER/CLOCK DIAGNOSTICS ==='
  nvidia-smi -q -d POWER,CLOCK | grep -E 'Power Limit|Power Draw|Graphics|SM'
  echo '==================================='

  # Hardcode precision and enable_amp for reliability
  PREC='${PRECISION:-bf16-mixed}'
  
  echo \"Running with precision=\$PREC, enable_amp=TRUE\"

  python3 -u -m src.training.train_tft_v1 \\
    --train_parquet '$LOCAL_DIR/train.parquet' \\
    --val_parquet   '$LOCAL_DIR/val.parquet' \\
    --run_root      '$RUN_ROOT' \\
    --enable_amp \\
    --precision \"\$PREC\" \\
    --enc_lag 96 \\
    --max_epochs 30 \\
    --batch_size 2048 \\
    --num_workers 12 \\
    --prefetch_factor 4 \\
    --grad_accum_steps 1 \\
    --lr 3e-4 \\
    --weight_decay 1e-4 \\
    --patience 3 \\
    --min_delta 1e-5 \\
    --log_every_n_steps 0 \\
    --progress_every 25 \\
    --grad_clip 1.0
"

echo "=== JOB END (UTC) $(date -u +'%Y-%m-%dT%H:%M:%SZ') ==="
command -v sacct >/dev/null 2>&1 && sacct -j "${SLURM_JOB_ID}" --format=JobID,Elapsed,State,AllocTRES%50,MaxRSS