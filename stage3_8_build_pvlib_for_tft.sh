#!/usr/bin/env bash
set -euo pipefail

: '
STAGE 3.8: Build PVLib features for TFT (regional split)

Inputs (from Stage 3.7):
- data/processed/pretraining/germany/global/weather_tft/regional_train_weather_tft.parquet
- data/processed/pretraining/germany/global/weather_tft/regional_val_weather_tft.parquet

Plant metadata:
- data/metadata/germany/<plant_id>.json

Outputs:
- data/processed/pretraining/germany/global/pvlib_tft/regional_train_pvlib.parquet
- data/processed/pretraining/germany/global/pvlib_tft/regional_val_pvlib.parquet
'

REPO_ROOT="$(pwd)"
cd "$REPO_ROOT"

TRAIN_W="$REPO_ROOT/data/processed/pretraining/germany/global/weather_tft/regional_train_weather_tft.parquet"
VAL_W="$REPO_ROOT/data/processed/pretraining/germany/global/weather_tft/regional_val_weather_tft.parquet"
META_DIR="$REPO_ROOT/data/metadata/germany"
OUT_DIR="$REPO_ROOT/data/processed/pretraining/germany/global/pvlib_tft"

echo "================================================================================"
echo "STAGE 3.8: PVLib build"
echo "TRAIN_W:  $TRAIN_W"
echo "VAL_W:    $VAL_W"
echo "META_DIR: $META_DIR"
echo "OUT_DIR:  $OUT_DIR"
echo "================================================================================"

test -f "$TRAIN_W"
test -f "$VAL_W"
test -d "$META_DIR"

echo "[INFO] Cleaning previous outputs..."
rm -f "$OUT_DIR/regional_train_pvlib.parquet" "$OUT_DIR/regional_val_pvlib.parquet" || true
mkdir -p "$OUT_DIR"

export PYTHONPATH="$REPO_ROOT"

echo "[INFO] Running builder..."
python "$REPO_ROOT/src/features/germany_build_pvlib_for_tft.py" \
  --train_weather "$TRAIN_W" \
  --val_weather "$VAL_W" \
  --meta_dir "$META_DIR" \
  --out_dir "$OUT_DIR"

echo "[DONE]"
ls -lh \
  "$OUT_DIR/regional_train_pvlib_tft.parquet" \
  "$OUT_DIR/regional_val_pvlib_tft.parquet"
