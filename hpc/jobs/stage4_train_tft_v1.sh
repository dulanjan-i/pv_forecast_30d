#!/usr/bin/env bash
set -euo pipefail

: '
Stage 4: Train TFT v1.0 (Germany regional)

This runs TFT training on the already-built "full" TFT parquets.

Expected inputs (adjust if your filenames differ):
- data/processed/pretraining/germany/global/tft_inputs/regional_train_tft_full.parquet
- data/processed/pretraining/germany/global/tft_inputs/regional_val_tft_full.parquet

Outputs:
- experiments/tft/runs/germany/v1_0/<run_id>/
'

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT"

TRAIN_P="data/processed/pretraining/germany/global/tft_inputs/regional_train_tft_full.parquet"
VAL_P="data/processed/pretraining/germany/global/tft_inputs/regional_val_tft_full.parquet"

echo "================================================================================"
echo "Stage 4: TFT v1.0 training"
echo "REPO_ROOT: $REPO_ROOT"
echo "TRAIN_P:   $TRAIN_P"
echo "VAL_P:     $VAL_P"
echo "================================================================================"

test -f "$TRAIN_P"
test -f "$VAL_P"

# If your venv path differs, set it here or just run inside activated env.
# source "$REPO_ROOT/.venvs/pvforecast/bin/activate"

python -m src.training.train_tft_v1 \
  --train_parquet "$TRAIN_P" \
  --val_parquet "$VAL_P" \
  --max_epochs 30 \
  --batch_size 384 \
  --num_workers 8 \
  --gpus 1 \
  --encoder_len 96 \
  --pred_len 96 \
  --lr 1e-3 \
  --hidden_size 64 \
  --lstm_layers 2 \
  --attn_heads 4 \
  --dropout 0.1 \
  --weight_decay 1e-4 \
  --precision 32 \
  --patience 5
