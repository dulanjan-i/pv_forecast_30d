#!/usr/bin/env bash
set -euo pipefail

: '
STAGE 3.6: Build TFT base tables by merging:
- regional_{train,val}.parquet (LSTM-ready inputs, weather likely z-scored)
- regional_{train,val}_lstm_encodings.parquet (lstm_enc_* columns)

Also optionally adds *_raw weather columns using regional_scaler.json.
This is a safe step before PVLib integration.
'

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

#source "$REPO_ROOT/.venvs/pvforecast/bin/activate"

DATA_DIR="$REPO_ROOT/data/processed/pretraining/germany/global"
ENC_DIR="$DATA_DIR/encodings"
OUT_DIR="$DATA_DIR/tft_inputs"

mkdir -p "$OUT_DIR"

echo "[INFO] Cleaning previous TFT base outputs..."
rm -f \
  "$OUT_DIR/regional_train_tft_base.parquet" \
  "$OUT_DIR/regional_val_tft_base.parquet"

echo "[INFO] Building TRAIN TFT base..."
python src/features/germany_merge_lstm_encodings_for_tft.py \
  --base_parquet "$DATA_DIR/regional_train.parquet" \
  --enc_parquet  "$ENC_DIR/regional_train_lstm_encodings.parquet" \
  --output_parquet "$OUT_DIR/regional_train_tft_base.parquet" \
  --scaler_json "$DATA_DIR/regional_scaler.json"

echo "[INFO] Building VAL TFT base..."
python src/features/germany_merge_lstm_encodings_for_tft.py \
  --base_parquet "$DATA_DIR/regional_val.parquet" \
  --enc_parquet  "$ENC_DIR/regional_val_lstm_encodings.parquet" \
  --output_parquet "$OUT_DIR/regional_val_tft_base.parquet" \
  --scaler_json "$DATA_DIR/regional_scaler.json"

echo "[DONE] Stage 3.6 complete."
ls -lh "$OUT_DIR/regional_train_tft_base.parquet" "$OUT_DIR/regional_val_tft_base.parquet"
