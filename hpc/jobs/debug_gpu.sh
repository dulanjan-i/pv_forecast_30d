#!/bin/bash
#SBATCH --job-name=h100_debug
#SBATCH --partition=gpuh100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=/shared/%u/miracle/logs/%x_%j.out
#SBATCH --error=/shared/%u/miracle/logs/%x_%j.err

set -euo pipefail

# --- CONFIG ---
REPO="/shared/$USER/miracle/pv_forecast_30d"
IMG="/shared/$USER/miracle/containers/tft_env_v1.sif"

# 1. DETECT ASSIGNED GPU
ALLOCATED_GPU="${SLURM_JOB_GPUS:-0}"
echo "------------------------------------------------"
echo "SLURM Assigned GPU ID: $ALLOCATED_GPU"
echo "------------------------------------------------"

# 2. RUN DEBUG TOOL
# We use the same 'Isolation Fix' (--env SLURM_ID_PASS) to ensure
# the test matches your real training environment exactly.

singularity exec -C --nv \
  --env SLURM_ID_PASS=$ALLOCATED_GPU \
  --bind "/shared/$USER:/shared/$USER,/home/$USER:/home/$USER,/tmp:/tmp,/dev/shm:/dev/shm" \
  --pwd "$REPO" \
  "$IMG" \
  bash -c "
    echo '--- CONTAINER START ---'
    
    # Force Python to see only 1 GPU
    export CUDA_VISIBLE_DEVICES=\$SLURM_ID_PASS
    export PYTHONPATH=$REPO:\$PYTHONPATH
    
    # Run the debug tool
    python3 src/utils/debug_gpu.py
  "