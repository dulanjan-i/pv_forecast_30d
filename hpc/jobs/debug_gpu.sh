***REMOVED***!/bin/bash
***REMOVED***SBATCH --job-name=h100_debug
***REMOVED***SBATCH --partition=gpuh100
***REMOVED***SBATCH --gres=gpu:1
***REMOVED***SBATCH --cpus-per-task=16
***REMOVED***SBATCH --mem=32G
***REMOVED***SBATCH --time=00:10:00
***REMOVED***SBATCH --output=/shared/%u/miracle/logs/%x_%j.out
***REMOVED***SBATCH --error=/shared/%u/miracle/logs/%x_%j.err

set -euo pipefail

***REMOVED*** --- CONFIG ---
REPO="/shared/$USER/miracle/pv_forecast_30d"
IMG="/shared/$USER/miracle/containers/tft_env_v1.sif"

***REMOVED*** 1. DETECT ASSIGNED GPU
ALLOCATED_GPU="${SLURM_JOB_GPUS:-0}"
echo "------------------------------------------------"
echo "SLURM Assigned GPU ID: $ALLOCATED_GPU"
echo "------------------------------------------------"

***REMOVED*** 2. RUN DEBUG TOOL
***REMOVED*** We use the same 'Isolation Fix' (--env SLURM_ID_PASS) to ensure
***REMOVED*** the test matches your real training environment exactly.

singularity exec -C --nv \
  --env SLURM_ID_PASS=$ALLOCATED_GPU \
  --bind "/shared/$USER:/shared/$USER,/home/$USER:/home/$USER,/tmp:/tmp,/dev/shm:/dev/shm" \
  --pwd "$REPO" \
  "$IMG" \
  bash -c "
    echo '--- CONTAINER START ---'
    
    ***REMOVED*** Force Python to see only 1 GPU
    export CUDA_VISIBLE_DEVICES=\$SLURM_ID_PASS
    export PYTHONPATH=$REPO:\$PYTHONPATH
    
    ***REMOVED*** Run the debug tool
    python3 src/utils/debug_gpu.py
  "