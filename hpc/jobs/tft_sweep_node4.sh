#!/bin/bash
#SBATCH --job-name=tft_sweep
#SBATCH --partition=gpuh100
#SBATCH --nodelist=dbfz-hpc23-gnode4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-7%4
#SBATCH --output=/shared/%u/miracle/logs/%x_%A_%a.out
#SBATCH --error=/shared/%u/miracle/logs/%x_%A_%a.err

set -euo pipefail

# You submit with: sbatch --export=ALL,MODE=tft_only  ...  or MODE=tft_pvlib
MODE="${MODE:-}"
if [[ "$MODE" != "tft_only" && "$MODE" != "tft_pvlib" ]]; then
  echo "ERROR: MODE must be tft_only or tft_pvlib, got: '$MODE'"
  exit 2
fi

REPO="/shared/$USER/miracle/pv_forecast_30d"
IMG="/shared/$USER/miracle/containers/tft_env_v1.sif"
ABL_DIR="$REPO/data/processed/pretraining/germany/global/tft_inputs_pca32_ablations"

SRC_TRAIN="$ABL_DIR/train_${MODE}.parquet"
SRC_VAL="$ABL_DIR/val_${MODE}.parquet"

[[ -f "$IMG" ]] || { echo "MISSING IMG: $IMG"; exit 2; }
[[ -f "$SRC_TRAIN" ]] || { echo "MISSING TRAIN: $SRC_TRAIN"; exit 2; }
[[ -f "$SRC_VAL"   ]] || { echo "MISSING VAL:   $SRC_VAL"; exit 2; }

echo "HOST=$(hostname)"
echo "MODE=$MODE"
echo "JOB=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-NA}"
date

# Sweep grid: 4 learning rates x 2 dropout values = 8 configs
LRS=(8e-4 1.2e-3 2e-3 3e-3  8e-4 1.2e-3 2e-3 3e-3)
DROPS=(0.05 0.05 0.05 0.05  0.15 0.15 0.15 0.15)

LR="${LRS[$SLURM_ARRAY_TASK_ID]}"
DROPOUT="${DROPS[$SLURM_ARRAY_TASK_ID]}"

# Keep proven stable training pipeline knobs fixed for the sweep
BATCH_SIZE=512
ACCUM=8
NUM_WORKERS=12
PREFETCH=2
PREC="bf16-mixed"
WD=1e-4
SEED=42
MAX_EPOCHS=15
PATIENCE=3

LOCAL_DIR="/tmp/$USER/sweep_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$LOCAL_DIR"
cp "$SRC_TRAIN" "$LOCAL_DIR/train.parquet"
cp "$SRC_VAL"   "$LOCAL_DIR/val.parquet"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export SINGULARITY_BINDPATH="/shared:/shared,/home/$USER:/home/$USER,/tmp:/tmp,/dev/shm:/dev/shm"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_ROOT="experiments/tft/runs/germany/sweeps/${MODE}/job${SLURM_JOB_ID}/lr${LR}_do${DROPOUT}_bs${BATCH_SIZE}_acc${ACCUM}_seed${SEED}"

singularity exec --nv "$IMG" bash -lc "
  set -euo pipefail
  cd $REPO
  export PYTHONPATH=\"$REPO:\${PYTHONPATH:-}\"

  python3 -m src.training.train_tft_v1 \
    --train_parquet $LOCAL_DIR/train.parquet \
    --val_parquet   $LOCAL_DIR/val.parquet \
    --run_root      $RUN_ROOT \
    --max_epochs    $MAX_EPOCHS \
    --batch_size    $BATCH_SIZE \
    --grad_accum_steps $ACCUM \
    --num_workers   $NUM_WORKERS \
    --prefetch_factor $PREFETCH \
    --precision     $PREC \
    --enable_amp \
    --lr            $LR \
    --weight_decay  $WD \
    --dropout       $DROPOUT \
    --patience      $PATIENCE \
    --seed          $SEED
"

rm -rf "$LOCAL_DIR"
echo "DONE MODE=$MODE LR=$LR DROPOUT=$DROPOUT"
