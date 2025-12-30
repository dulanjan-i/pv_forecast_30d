#!/bin/bash
#SBATCH --job-name=tft_ablate
#SBATCH --partition=gpuh100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-3
#SBATCH --output=/shared/%u/miracle/logs/%x_%A_%a.out
#SBATCH --error=/shared/%u/miracle/logs/%x_%A_%a.err

set -euo pipefail

echo "HOST=$(hostname)"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-NA}"
date

REPO="/shared/$USER/miracle/pv_forecast_30d"
IMG="/shared/$USER/miracle/containers/tft_env_v1.sif"

ABL_DIR="$REPO/data/processed/pretraining/germany/global/tft_inputs_pca32_ablations"

MODES=("full" "tft_only" "tft_lstm" "tft_pvlib")
MODE="${MODES[$SLURM_ARRAY_TASK_ID]}"

SRC_TRAIN="$ABL_DIR/train_${MODE}.parquet"
SRC_VAL="$ABL_DIR/val_${MODE}.parquet"

# ---- hard checks so we fail with a useful message (instead of silent ExitCode=1) ----
echo "MODE=$MODE"
echo "REPO=$REPO"
echo "IMG=$IMG"
echo "SRC_TRAIN=$SRC_TRAIN"
echo "SRC_VAL=$SRC_VAL"

[[ -f "$IMG" ]] || { echo "MISSING IMG: $IMG"; exit 2; }
[[ -f "$SRC_TRAIN" ]] || { echo "MISSING TRAIN: $SRC_TRAIN"; ls -lah "$ABL_DIR" || true; exit 2; }
[[ -f "$SRC_VAL"   ]] || { echo "MISSING VAL:   $SRC_VAL";   ls -lah "$ABL_DIR" || true; exit 2; }

LOCAL_DIR="/tmp/$USER/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$LOCAL_DIR"

cp "$SRC_TRAIN" "$LOCAL_DIR/train.parquet"
cp "$SRC_VAL"   "$LOCAL_DIR/val.parquet"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export SINGULARITY_BINDPATH="/shared:/shared,/home/$USER:/home/$USER,/tmp:/tmp,/dev/shm:/dev/shm"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_ROOT="experiments/tft/runs/germany/ablations/${MODE}"

EXTRA=""
if [[ "$MODE" == "full" || "$MODE" == "tft_lstm" ]]; then
  EXTRA="--use_lstm_encodings --enc_lag 96"
fi

# IMPORTANT: expand $EXTRA in the outer shell (do NOT escape it with a backslash)
singularity exec --nv "$IMG" bash -lc "
  set -euo pipefail
  cd $REPO
  export PYTHONPATH=\"$REPO:\${PYTHONPATH:-}\"

  python3 -m src.training.train_tft_v1 \
    --train_parquet $LOCAL_DIR/train.parquet \
    --val_parquet   $LOCAL_DIR/val.parquet \
    --run_root      $RUN_ROOT \
    --max_epochs 30 \
    --batch_size 512 \
    --grad_accum_steps 8 \
    --num_workers 16 \
    --prefetch_factor 2 \
    --precision bf16-mixed \
    --enable_amp \
    --lr 2e-3 \
    $EXTRA
"

rm -rf "$LOCAL_DIR"
echo "DONE mode=$MODE"
