#!/bin/bash
#SBATCH --job-name=tft_ft_p03_pvlib
#SBATCH --partition=gpuh100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-2
#SBATCH --output=/shared/%u/miracle/logs/%x_%A_%a.out
#SBATCH --error=/shared/%u/miracle/logs/%x_%A_%a.err

set -euo pipefail

REPO="/shared/$USER/miracle/pv_forecast_30d"
IMG="/shared/$USER/miracle/containers/tft_env_v1.sif"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export SINGULARITY_BINDPATH="/shared:/shared,/home/$USER:/home/$USER,/tmp:/tmp,/dev/shm:/dev/shm"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- Plant-level parquets (you just created) ----
PPL_DIR="$REPO/data/processed/plant_level/plant_03/15min_pca32"
SRC_TRAIN="$PPL_DIR/train.parquet"
SRC_VAL="$PPL_DIR/val.parquet"

# ---- Seeds ----
SEEDS=(21 42 84)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

# ---- Best global config we found (tft_pvlib winner) ----
LR="1.2e-3"
DROPOUT="0.15"
BS="512"
ACC="8"

# ---- Local copy to node tmp (reduces shared FS IO) ----
LOCAL_DIR="/tmp/$USER/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$LOCAL_DIR"
cp "$SRC_TRAIN" "$LOCAL_DIR/train.parquet"
cp "$SRC_VAL"   "$LOCAL_DIR/val.parquet"

# ---- Output ----
RUN_ROOT="experiments/tft/runs/germany/finetune/plant_03/tft_pvlib/seed${SEED}"

# ---- Run ----
/usr/bin/singularity exec --nv "$IMG" bash -lc "
  set -euo pipefail
  cd \"$REPO\"
  export PYTHONPATH=\"$REPO:\${PYTHONPATH:-}\"

  python3 -m src.training.train_tft_v1 \
    --train_parquet \"$LOCAL_DIR/train.parquet\" \
    --val_parquet   \"$LOCAL_DIR/val.parquet\" \
    --run_root      \"$RUN_ROOT\" \
    --max_epochs 30 \
    --batch_size $BS \
    --grad_accum_steps $ACC \
    --num_workers 16 \
    --prefetch_factor 2 \
    --precision bf16-mixed \
    --enable_amp \
    --lr $LR \
    --dropout $DROPOUT \
    --seed $SEED
"

rm -rf "$LOCAL_DIR"
echo "DONE seed=$SEED"
