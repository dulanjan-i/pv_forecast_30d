***REMOVED***!/bin/bash
***REMOVED***SBATCH --job-name=tft_h100_fix
***REMOVED***SBATCH --partition=gpuh100
***REMOVED***SBATCH --gres=gpu:1
***REMOVED***SBATCH --cpus-per-task=16
***REMOVED***SBATCH --mem=64G
***REMOVED***SBATCH --time=04:00:00
***REMOVED***SBATCH --output=/shared/%u/miracle/logs/%x_%j.out
***REMOVED***SBATCH --error=/shared/%u/miracle/logs/%x_%j.err

set -euo pipefail

***REMOVED*** --- CONFIGURATION ---
REPO="/shared/$USER/miracle/pv_forecast_30d"
IMG="/shared/$USER/miracle/containers/tft_env_v1.sif"

***REMOVED*** Source Data
SRC_TRAIN="$REPO/data/processed/pretraining/germany/global/tft_inputs/regional_train_tft_full.parquet"
SRC_VAL="$REPO/data/processed/pretraining/germany/global/tft_inputs/regional_val_tft_full.parquet"

***REMOVED*** Local Scratch (Fast SSD)
LOCAL_DIR="/tmp/$USER/$SLURM_JOB_ID"

echo "=== 1. SETUP ==="
mkdir -p "$LOCAL_DIR"
echo "Created local scratch: $LOCAL_DIR"

echo "=== 2. DATA COPY (To Local SSD) ==="
cp "$SRC_TRAIN" "$LOCAL_DIR/train.parquet"
cp "$SRC_VAL"   "$LOCAL_DIR/val.parquet"
echo "Data copy finished."

***REMOVED*** --- CRITICAL PERFORMANCE SETTINGS ---
***REMOVED*** 1. Threads: Limit OMP threads so they don't fight the DataLoader workers
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

***REMOVED*** 2. Singularity Bind: MUST include /dev/shm for PyTorch workers to function!
BINDS="/shared/$USER:/shared/$USER,/home/$USER:/home/$USER,/tmp:/tmp,/dev/shm:/dev/shm"

echo "=== 3. RUNNING TRAINING ==="
***REMOVED*** Note: We use -C (clean env) + --nv (nvidia) + correct Binds
singularity exec -C --nv \
  --bind "$BINDS" \
  --pwd "$REPO" \
  "$IMG" \
  bash -c "
    echo 'Inside Container: checking GPU...'
    nvidia-smi -L
    
    export PYTHONPATH=$REPO:\$PYTHONPATH
    
    ***REMOVED*** Python Command
    ***REMOVED*** Changes made:
    ***REMOVED*** 1. --batch_size 2048:  Saturate the H100 (it has 80GB RAM, use it!)
    ***REMOVED*** 2. --num_workers 12:   Feed the beast. Now works because of /dev/shm bind.
    ***REMOVED*** 3. --precision 32-true: Safe, stable, and still crazy fast on H100.
    
    python3 -m src.training.train_tft_v1 \
      --train_parquet $LOCAL_DIR/train.parquet \
      --val_parquet   $LOCAL_DIR/val.parquet \
      --use_lstm_encodings \
      --enc_lag 96 \
      --max_epochs 30 \
      --batch_size 4096 \
      --num_workers 8 \
      --precision "16-mixed" \
      --lr 2e-3
  "

echo "=== 4. CLEANUP ==="
rm -rf "$LOCAL_DIR"
echo "Job Complete."