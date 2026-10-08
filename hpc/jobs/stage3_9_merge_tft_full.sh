#!/usr/bin/env bash
set -euo pipefail

: '
STAGE 3.9: Merge TFT base + weather + PVLib into final TFT tables.

Outputs:
- data/processed/pretraining/germany/global/tft_inputs/regional_train_tft_full.parquet
- data/processed/pretraining/germany/global/tft_inputs/regional_val_tft_full.parquet
'

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

BASE_DIR="$REPO_ROOT/data/processed/pretraining/germany/global/tft_inputs"
WEATHER_DIR="$REPO_ROOT/data/processed/pretraining/germany/global/weather_tft"
PVLIB_DIR="$REPO_ROOT/data/processed/pretraining/germany/global/pvlib_tft"
OUT_DIR="$REPO_ROOT/data/processed/pretraining/germany/global/tft_inputs"

echo "================================================================================"
echo "STAGE 3.9: Merge TFT full tables"
echo "REPO_ROOT:   $REPO_ROOT"
echo "BASE_DIR:    $BASE_DIR"
echo "WEATHER_DIR: $WEATHER_DIR"
echo "PVLIB_DIR:   $PVLIB_DIR"
echo "OUT_DIR:     $OUT_DIR"
echo "================================================================================"

export PYTHONPATH="$REPO_ROOT"

python "$REPO_ROOT/src/features/germany_merge_tft_full.py" \
  --base_dir "$BASE_DIR" \
  --weather_dir "$WEATHER_DIR" \
  --pvlib_dir "$PVLIB_DIR" \
  --out_dir "$OUT_DIR"

echo "[DONE] Stage 3.9 complete."
ls -lh "$OUT_DIR/regional_train_tft_full.parquet" "$OUT_DIR/regional_val_tft_full.parquet"