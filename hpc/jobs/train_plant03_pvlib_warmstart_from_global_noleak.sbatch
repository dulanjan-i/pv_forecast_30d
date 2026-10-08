#!/bin/bash
#SBATCH -J plant03_pvlib_warm
#SBATCH -p gpuh100
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH -o /shared/%u/miracle/logs/%x_%j.out
#SBATCH -e /shared/%u/miracle/logs/%x_%j.err
# Optional pin:
#SBATC H --nodelist=dbfz-hpc23-gnode4

set -euo pipefail
export OMP_NUM_THREADS=1

REPO="/shared/$USER/miracle/pv_forecast_30d"
IMG="/shared/$USER/miracle/containers/tft_env_v1.sif"
export SINGULARITY_BINDPATH="/shared:/shared,/home/$USER:/home/$USER,/tmp:/tmp,/dev/shm:/dev/shm"

PLANT_DIR="$REPO/data/processed/plant_level/plant_03/15min_pca32"
TRAIN="$PLANT_DIR/train.parquet"
VAL="$PLANT_DIR/val.parquet"

# TODO: fill this after global job ends
GLOBAL_CKPT="$REPO/experiments/tft/runs/germany/global_noleak/target03_excluded/20251229_134852/checkpoints/best.ckpt"
GLOBAL_DIR="$REPO/experiments/tft/runs/germany/global_noleak/target03_excluded/20251229_134852"
STATE_DICT="$GLOBAL_DIR/checkpoints/best_state_dict.pt"

cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

singularity exec --nv "$IMG" python3 -m src.training.train_tft_v1 \
  --train_parquet "$TRAIN" \
  --val_parquet   "$VAL" \
  --run_root "experiments/tft/runs/germany/plant_03/15min/pvlib_warmstart_from_global_noleak" \
  --init_state_dict "$STATE_DICT" \
  --lr 6e-4 \
  --dropout 0.15 \
  --batch_size 512 \
  --grad_accum_steps 8 \
  --max_epochs 50 \
  --patience 5 \
  --seed 42 \
  --precision bf16-mixed
