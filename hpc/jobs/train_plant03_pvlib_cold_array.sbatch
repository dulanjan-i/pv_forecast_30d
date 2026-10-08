#!/bin/bash
#SBATCH -J plant03_pvlib_cold
#SBATCH -p gpuh100
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=42-44
#SBATCH -o /shared/%u/miracle/logs/%x_%A_%a.out
#SBATCH -e /shared/%u/miracle/logs/%x_%A_%a.err
# Optional pin (uncomment if you want):
##SBATCH --nodelist=dbfz-hpc23-gnode4

set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

REPO="/shared/$USER/miracle/pv_forecast_30d"
IMG="/shared/$USER/miracle/containers/tft_env_v1.sif"
export SINGULARITY_BINDPATH="/shared:/shared,/home/$USER:/home/$USER,/tmp:/tmp,/dev/shm:/dev/shm"

SEED="${SLURM_ARRAY_TASK_ID}"

PLANT_DIR="$REPO/data/processed/plant_level/plant_03/15min_pca32"
TRAIN="$PLANT_DIR/train.parquet"
VAL="$PLANT_DIR/val.parquet"

# fail fast if paths wrong
[[ -f "$TRAIN" ]] || { echo "Missing TRAIN: $TRAIN"; exit 2; }
[[ -f "$VAL"   ]] || { echo "Missing VAL:   $VAL";   exit 2; }

cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

echo "============================================="
echo "MODE=plant03_pvlib_coldstart"
echo "SEED=$SEED"
echo "TRAIN=$TRAIN"
echo "VAL=$VAL"
echo "RUN_ROOT=experiments/tft/runs/germany/plant_03/15min/pvlib_coldstart/seed_${SEED}"
echo "============================================="

singularity exec --nv "$IMG" python3 -m src.training.train_tft_v1 \
  --train_parquet "$TRAIN" \
  --val_parquet   "$VAL" \
  --run_root "experiments/tft/runs/germany/plant_03/15min/pvlib_coldstart/seed_${SEED}" \
  --lr 1.2e-3 \
  --dropout 0.15 \
  --batch_size 512 \
  --grad_accum_steps 8 \
  --max_epochs 50 \
  --patience 5 \
  --seed "$SEED" \
  --precision bf16-mixed \
  --num_workers 8 \
  --prefetch_factor 4
