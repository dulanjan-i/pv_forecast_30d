#!/bin/bash
#SBATCH -J global_pvlib_noleak03
#SBATCH -p gpuh100
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH -o /shared/%u/miracle/logs/%x_%j.out
#SBATCH -e /shared/%u/miracle/logs/%x_%j.err
# Optional pin:
#SBATC H --nodelist=dbfz-hpc23-gnode4

set -euo pipefail

export OMP_NUM_THREADS=1

REPO="/shared/$USER/miracle/pv_forecast_30d"
IMG="/shared/$USER/miracle/containers/tft_env_v1.sif"
export SINGULARITY_BINDPATH="/shared:/shared,/home/$USER:/home/$USER,/tmp:/tmp,/dev/shm:/dev/shm"

SRC_TRAIN="$REPO/data/processed/pretraining/germany/global/tft_inputs_pca32_ablations/train_tft_pvlib.parquet"
SRC_VAL="$REPO/data/processed/pretraining/germany/global/tft_inputs_pca32_ablations/val_tft_pvlib.parquet"
OUT_DIR="$REPO/data/processed/pretraining/germany/global/tft_inputs_pvlib_noleak_target03"

cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

singularity exec --nv "$IMG" python3 -m src.data.make_global_noleak_parquets \
  --src_train "$SRC_TRAIN" \
  --src_val   "$SRC_VAL" \
  --exclude_plant_id "plant_03" \
  --out_dir   "$OUT_DIR"

singularity exec --nv "$IMG" python3 -m src.training.train_tft_v1 \
  --train_parquet "$OUT_DIR/train.parquet" \
  --val_parquet   "$OUT_DIR/val.parquet" \
  --run_root "experiments/tft/runs/germany/global_noleak/target03_excluded" \
  --lr 1.2e-3 \
  --dropout 0.15 \
  --batch_size 512 \
  --grad_accum_steps 8 \
  --max_epochs 30 \
  --patience 3 \
  --seed 42 \
  --precision bf16-mixed
