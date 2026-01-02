#!/bin/bash
#SBATCH -J longhead_p03_720
#SBATCH -p gpuh100
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-5
#SBATCH -o /shared/%u/miracle/logs/%x_%A_%a.out
#SBATCH -e /shared/%u/miracle/logs/%x_%A_%a.err

set -euo pipefail
export OMP_NUM_THREADS=1

REPO="/shared/$USER/miracle/pv_forecast_30d"
IMG="/shared/$USER/miracle/containers/tft_env_v1.sif"
export SINGULARITY_BINDPATH="/shared:/shared,/home/$USER:/home/$USER,/tmp:/tmp,/dev/shm:/dev/shm"

cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

# Plant03 15min source splits
P15_DIR="$REPO/data/processed/plant_level/plant_03/15min_pca32"
P15_TRAIN="$P15_DIR/train.parquet"
P15_VAL="$P15_DIR/val.parquet"

# Hourly plant03 output
P1H_DIR="$REPO/data/processed/plant_level/plant_03/1h_pca32"
P1H_TRAIN="$P1H_DIR/train.parquet"
P1H_VAL="$P1H_DIR/val.parquet"

# Build hourly plant parquets if missing
if [[ ! -f "$P1H_TRAIN" || ! -f "$P1H_VAL" ]]; then
  echo "[INFO] Building plant_03 hourly parquets at: $P1H_DIR"
  singularity exec --nv "$IMG" python3 -m src.data.make_hourly_from_15min_parquets \
    --src_train "$P15_TRAIN" \
    --src_val   "$P15_VAL" \
    --out_dir   "$P1H_DIR" \
    --freq "1H"
else
  echo "[INFO] Plant_03 hourly parquets already exist. Skipping build."
fi

# Long head lengths
ENC_LEN=168
PRED_LEN=720

# Train knobs
BS=64
ACCUM=8
MAX_EPOCHS=40
PATIENCE=5

# Seeds for cold/warm
SEEDS=(42 43 44)
IDX=$SLURM_ARRAY_TASK_ID

if [[ $IDX -lt 3 ]]; then
  REGIME="cold"
  SEED="${SEEDS[$IDX]}"
  INIT_SD=""
else
  REGIME="warm"
  SEED="${SEEDS[$((IDX-3))]}"

  # TODO: set this to the winner of global sweep (best_state_dict.pt from that run)
  GLOBAL_SD="$REPO/experiments/tft/runs/germany/longhead/global_noleak_target03/hourly720/<WINNER_RUN>/checkpoints/best_state_dict.pt"
  INIT_SD="$GLOBAL_SD"
fi

# Use your proven LR/dropout as starting point for finetune
LR=6e-4
DROP=0.15

RUN_ROOT="experiments/tft/runs/germany/longhead/plant_03/hourly720/${REGIME}/seed_${SEED}"

CMD=(python3 -m src.training.train_tft_longhead_v1
  --train_parquet "$P1H_TRAIN"
  --val_parquet   "$P1H_VAL"
  --run_root "$RUN_ROOT"
  --enc_len "$ENC_LEN"
  --pred_len "$PRED_LEN"
  --lr "$LR"
  --dropout "$DROP"
  --batch_size "$BS"
  --grad_accum_steps "$ACCUM"
  --max_epochs "$MAX_EPOCHS"
  --patience "$PATIENCE"
  --seed "$SEED"
  --precision "bf16-mixed"
  --num_workers 8
  --prefetch_factor 2
)

if [[ -n "$INIT_SD" ]]; then
  CMD+=(--init_state_dict "$INIT_SD")
fi

singularity exec --nv "$IMG" "${CMD[@]}"
