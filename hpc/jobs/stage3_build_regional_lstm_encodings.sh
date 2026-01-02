#!/usr/bin/env bash
set -euo pipefail

: '
STAGE 3.6: Build LSTM encodings for Germany regional train/val.

What it does:
- Uses canonical Germany regional encoder weights
  experiments/lstm/encoders/lstm_encoder_germany_regional_CANONICAL.pt
- Generates LSTM encodings for:
  data/processed/pretraining/germany/global/regional_train.parquet
  data/processed/pretraining/germany/global/regional_val.parquet
- Writes:
  data/processed/pretraining/germany/global/encodings/regional_*_lstm_encodings.parquet

Notes:
- If a venv is already active (VIRTUAL_ENV set), we do not re-source.
- Otherwise we try ~/.venvs/pvforecast/bin/activate, then ./.venvs/pvforecast/bin/activate
'

must_exist () {
  local p="$1"
  if [[ -f "$p" ]]; then
    echo "OK   $p"
  else
    echo "MISS $p"
    return 1
  fi
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

DATA_DIR="$REPO_ROOT/data/processed/pretraining/germany/global"
ENC_DIR="$DATA_DIR/encodings"
ENCODER="$REPO_ROOT/experiments/lstm/encoders/lstm_encoder_germany_regional_CANONICAL.pt"
SCRIPT="$REPO_ROOT/src/features/germany_build_lstm_encodings.py"

TRAIN_IN="$DATA_DIR/regional_train.parquet"
VAL_IN="$DATA_DIR/regional_val.parquet"
TRAIN_OUT="$ENC_DIR/regional_train_lstm_encodings.parquet"
VAL_OUT="$ENC_DIR/regional_val_lstm_encodings.parquet"

echo "================================================================================"
echo "STAGE 3.6: Build regional LSTM encodings"
echo "REPO_ROOT: $REPO_ROOT"
echo "================================================================================"

echo "[INFO] Sanity checks..."
all_ok=1
must_exist "$SCRIPT"   || all_ok=0
must_exist "$ENCODER"  || all_ok=0
must_exist "$TRAIN_IN" || all_ok=0
must_exist "$VAL_IN"   || all_ok=0

if [[ "$all_ok" -ne 1 ]]; then
  echo "[ERROR] Missing required files. Fix paths above and rerun."
  exit 1
fi

mkdir -p "$ENC_DIR"

# Activate venv if not already active
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  if [[ -f "$HOME/.venvs/pvforecast/bin/activate" ]]; then
    echo "[INFO] Activating venv: $HOME/.venvs/pvforecast"
    # shellcheck disable=SC1090
    source "$HOME/.venvs/pvforecast/bin/activate"
  elif [[ -f "$REPO_ROOT/.venvs/pvforecast/bin/activate" ]]; then
    echo "[INFO] Activating venv: $REPO_ROOT/.venvs/pvforecast"
    # shellcheck disable=SC1090
    source "$REPO_ROOT/.venvs/pvforecast/bin/activate"
  else
    echo "[ERROR] No venv found. Activate pvforecast manually, then rerun."
    exit 1
  fi
else
  echo "[INFO] Using already-active venv: $VIRTUAL_ENV"
fi

export PYTHONPATH="$REPO_ROOT"

echo "[INFO] Building encodings for TRAIN..."
python "$SCRIPT" \
  --input_parquet "$TRAIN_IN" \
  --encoder_ckpt "$ENCODER" \
  --output_parquet "$TRAIN_OUT"

echo "[INFO] Building encodings for VAL..."
python "$SCRIPT" \
  --input_parquet "$VAL_IN" \
  --encoder_ckpt "$ENCODER" \
  --output_parquet "$VAL_OUT"

echo "[SUCCESS] Wrote:"
ls -lh "$TRAIN_OUT" "$VAL_OUT"
echo "[DONE]"
