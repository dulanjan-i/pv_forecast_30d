#!/bin/bash
#SBATCH --job-name=eval_short_head
#SBATCH --partition=gpuh100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/shared/%u/miracle/logs/%x_%j.out
#SBATCH --error=/shared/%u/miracle/logs/%x_%j.err

set -e

REPO="/shared/$USER/miracle/pv_forecast_30d"
IMG="/shared/$USER/miracle/containers/tft_env_v1.sif"

export SINGULARITY_BINDPATH="/shared:/shared,/home/$USER:/home/$USER,/tmp:/tmp,/dev/shm:/dev/shm"

cd "$REPO"

/usr/bin/singularity exec --nv "$IMG" bash -c "
  set -e
  cd \"$REPO\"
  export PYTHONPATH=\"$REPO:\${PYTHONPATH:-}\"

  python3 -m src.validation.eval_short_head \
    --run_tft_only  experiments/tft/runs/germany/sweeps/tft_only/job24461/lr8e-4_do0.05_bs512_acc8_seed42/20251227_173728 \
    --run_tft_pvlib experiments/tft/runs/germany/sweeps/tft_pvlib/job24473/lr1.2e-3_do0.15_bs512_acc8_seed42/20251227_205027 \
    --train_tft_only  data/processed/pretraining/germany/global/tft_inputs_pca32_ablations/train_tft_only.parquet \
    --val_tft_only    data/processed/pretraining/germany/global/tft_inputs_pca32_ablations/val_tft_only.parquet \
    --train_tft_pvlib data/processed/pretraining/germany/global/tft_inputs_pca32_ablations/train_tft_pvlib.parquet \
    --val_tft_pvlib   data/processed/pretraining/germany/global/tft_inputs_pca32_ablations/val_tft_pvlib.parquet \
    --out_dir experiments/tft/notes
"
