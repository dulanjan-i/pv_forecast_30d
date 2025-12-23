***REMOVED***!/bin/bash
***REMOVED***SBATCH --job-name=tft_v1_h100
***REMOVED***SBATCH --partition=gpuh100
***REMOVED***SBATCH --gres=gpu:1
***REMOVED***SBATCH --cpus-per-task=6
***REMOVED***SBATCH --mem=48G
***REMOVED***SBATCH --time=06:00:00
***REMOVED***SBATCH --output=/shared/%u/miracle/logs/%x_%j.out
***REMOVED***SBATCH --error=/shared/%u/miracle/logs/%x_%j.err

***REMOVED*** 1. STOP ON ERROR
set -euo pipefail

***REMOVED*** 2. LOAD MODULES (Crucial!)
***REMOVED*** Try 'apptainer' first, if that fails, try 'singularity', or check 'module spider singularity'
module load tools/apptainer || module load apptainer || module load singularity

***REMOVED*** 3. VERIFY GPU VISIBILITY (Sanity Check)
echo "Job started on $(hostname)"
echo "GPU info:"
nvidia-smi

***REMOVED*** 4. DEFINE PATHS
REPO=/shared/$USER/miracle/pv_forecast_30d
IMG=/shared/$USER/miracle/containers/tft_env_v1.sif
TRAIN_P=$REPO/data/processed/pretraining/germany/global/tft_inputs/regional_train_tft_full.parquet
VAL_P=$REPO/data/processed/pretraining/germany/global/tft_inputs/regional_val_tft_full.parquet

***REMOVED*** 5. EXECUTE
***REMOVED*** Note: Added 'apptainer' as command, fall back to 'singularity' if needed
apptainer exec -c --nv \
  --bind /shared/$USER:/shared/$USER \
  --bind /home/$USER:/home/$USER \
  --pwd "$REPO" \
  "$IMG" \
  bash -lc "
    export PYTHONPATH=$REPO:\$PYTHONPATH
    python3 -m src.training.train_tft_v1 \
      --train_parquet $TRAIN_P \
      --val_parquet   $VAL_P \
      --use_lstm_encodings \
      --enc_lag 96 \
      --max_epochs 30 \
      --batch_size 256 \
      --num_workers 6 \
      --gpus 1 \
      --precision bf16-mixed
  "