#!/usr/bin/env bash
set -Eeuo pipefail

trap 'echo "[ERROR] Failed at line $LINENO. Last command: $BASH_COMMAND"' ERR

: '
STAGE 3.5: Build "regional" train/val split and train one canonical Germany-adapted LSTM encoder.

What it does:
1) Remove old regional parquets and regional run folder (safe cleanup).
2) Copy fold_4 train/val into regional_train/val.
3) Copy fold_4 scaler into regional_scaler.json.
4) Run the regional LSTM trainer to produce canonical encoder weights.
'

# Robust repo-root discovery
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$REPO_ROOT" ]]; then
  echo "[ERROR] Not inside a git repo. cd into pv_forecast_30d and rerun."
  exit 1
fi
cd "$REPO_ROOT"

DATA_DIR="$REPO_ROOT/data/processed/pretraining/germany/global"
RUN_DIR="$REPO_ROOT/experiments/lstm/runs/germany/global_v3/regional"
VENV_ACT="$HOME/.venvs/pvforecast/bin/activate"

echo "================================================================================"
echo "STAGE 3.5: Regional encoder build and training"
echo "REPO_ROOT: $REPO_ROOT"
echo "DATA_DIR:  $DATA_DIR"
echo "RUN_DIR:   $RUN_DIR"
echo "================================================================================"

echo "[INFO] 0) Sanity checks..."
test -d "$DATA_DIR"
test -f "$DATA_DIR/fold_4_train.parquet"
test -f "$DATA_DIR/fold_4_val.parquet"
test -f "$DATA_DIR/fold_4_scaler.json"
test -f "$REPO_ROOT/src/training/train_regional_lstm.py"
test -f "$VENV_ACT"

echo "[INFO] 1) Cleaning old regional artifacts (not touching folds)..."
rm -f "$DATA_DIR/regional_train.parquet" \
      "$DATA_DIR/regional_val.parquet" \
      "$DATA_DIR/regional_scaler.json"
rm -rf "$RUN_DIR"

echo "[INFO] 2) Creating regional_train/val from fold_4..."
cp -f "$DATA_DIR/fold_4_train.parquet" "$DATA_DIR/regional_train.parquet"
cp -f "$DATA_DIR/fold_4_val.parquet"   "$DATA_DIR/regional_val.parquet"
cp -f "$DATA_DIR/fold_4_scaler.json"   "$DATA_DIR/regional_scaler.json"

echo "[SUCCESS] regional split created:"
ls -lh "$DATA_DIR/regional_train.parquet" "$DATA_DIR/regional_val.parquet" "$DATA_DIR/regional_scaler.json"

echo "[INFO] 3) Activating venv..."
# shellcheck disable=SC1090
source "$VENV_ACT"

echo "[INFO] 4) Training canonical regional encoder..."
python "$REPO_ROOT/src/training/train_regional_lstm.py" \
  --window_size 96 \
  --batch_size 256 \
  --hidden_size 64 \
  --num_layers 2 \
  --dropout 0.1 \
  --lr 1e-4 \
  --max_epochs 30 \
  --patience 5 \
  --gpus 1 \
  --num_workers 2 \
  --precision 16-mixed

echo "[DONE] Regional encoder training complete."
