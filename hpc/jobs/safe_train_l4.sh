#!/bin/bash
# Version: v1.2-L4
# Date: 2025-12-26
# Optimized for NVIDIA L4 (24GB VRAM, 72W TDP)
# Changes from H100 version:
#   - Reduced batch_size from 2048 to 256 (VRAM limit)
#   - Reduced num_workers from 12 to 4 (fewer CPU cores)
#   - Reduced prefetch_factor from 4 to 2
#   - Added gradient accumulation to compensate for smaller batches
#   - Kept mixed precision (critical for L4 performance)

set -euo pipefail

echo "=== JOB START (UTC) $(date -u +'%Y-%m-%dT%H:%M:%SZ') ==="
echo "Host: $(hostname)"

REPO="$HOME/pv_forecast_30d"
RUN_ROOT="$HOME/experiments/tft/runs/germany/v1_0"
DATA_DIR="$HOME/data/processed/pretraining/germany/global/tft_inputs"

# Check if running in container or bare metal
if [ -f "/.dockerenv" ] || [ -f "/.singularity.d/Singularity" ]; then
    echo "Running inside container"
else
    echo "Running on bare metal"
fi

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

cd "$REPO"

echo "=== ENVIRONMENT ==="
python3 -V
nvidia-smi -L

echo "=== GPU DIAGNOSTICS ==="
nvidia-smi -q -d MEMORY,POWER,CLOCK | grep -E 'Memory|Power Limit|Power Draw|Graphics|SM' || true
echo "====================="

# L4-optimized parameters
PREC="bf16-mixed"
BATCH_SIZE=256        # L4 has 24GB VRAM vs H100's 80GB
GRAD_ACCUM=8          # Effective batch = 256 * 8 = 2048
NUM_WORKERS=4         # L4 typically on 8-16 core systems
PREFETCH=2            # Lower prefetch for limited RAM

echo "Running L4-optimized training:"
echo "  batch_size=$BATCH_SIZE (effective=$(($BATCH_SIZE * $GRAD_ACCUM)))"
echo "  num_workers=$NUM_WORKERS"
echo "  precision=$PREC"

python3 -u -m src.training.train_tft_v1 \
  --train_parquet "$DATA_DIR/regional_train_tft_full.parquet" \
  --val_parquet   "$DATA_DIR/regional_val_tft_full.parquet" \
  --run_root      "$RUN_ROOT" \
  --enable_amp \
  --precision "$PREC" \
  --enc_lag 96 \
  --max_epochs 30 \
  --batch_size $BATCH_SIZE \
  --num_workers $NUM_WORKERS \
  --prefetch_factor $PREFETCH \
  --grad_accum_steps $GRAD_ACCUM \
  --lr 3e-4 \
  --weight_decay 1e-4 \
  --patience 5 \
  --min_delta 1e-5 \
  --log_every_n_steps 0 \
  --progress_every 10 \
  --grad_clip 1.0

echo "=== JOB END (UTC) $(date -u +'%Y-%m-%dT%H:%M:%SZ') ==="