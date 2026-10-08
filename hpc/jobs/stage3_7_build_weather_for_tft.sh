#!/usr/bin/env bash
set -euo pipefail

: '
STAGE 3.7: Build TFT weather tables aligned to the TFT base tables.

What this does:
- Reads TFT base parquets (regional_train_tft_base, regional_val_tft_base) which already contain the
  exact timestamp_utc and plant_id rows that downstream TFT will see.
- Builds matching "weather tables" by loading per-plant weather_15min parquets from data/interim/germany,
  adding plant_id (derived from filename), and inner-joining on (plant_id, timestamp_utc).
- Does NOT require poa_irradiance. We use global_tilted_irradiance_instant (GTI) as irradiance proxy.
  POA is derived later via PVLib.

Outputs:
- data/processed/pretraining/germany/global/weather_tft/regional_train_weather_tft.parquet
- data/processed/pretraining/germany/global/weather_tft/regional_val_weather_tft.parquet
'

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

TRAIN_BASE="$REPO_ROOT/data/processed/pretraining/germany/global/tft_inputs/regional_train_tft_base.parquet"
VAL_BASE="$REPO_ROOT/data/processed/pretraining/germany/global/tft_inputs/regional_val_tft_base.parquet"
WEATHER_DIR="$REPO_ROOT/data/interim/germany"
OUT_DIR="$REPO_ROOT/data/processed/pretraining/germany/global/weather_tft"

echo "================================================================================"
echo "STAGE 3.7: Build TFT weather tables"
echo "REPO_ROOT:   $REPO_ROOT"
echo "TRAIN_BASE:  $TRAIN_BASE"
echo "VAL_BASE:    $VAL_BASE"
echo "WEATHER_DIR: $WEATHER_DIR"
echo "OUT_DIR:     $OUT_DIR"
echo "================================================================================"

# Optional venv activation (do not fail if missing)
VENV_ACT="$REPO_ROOT/.venvs/pvforecast/bin/activate"
if [[ -f "$VENV_ACT" ]]; then
  echo "[INFO] Activating venv: $VENV_ACT"
  # shellcheck disable=SC1090
  source "$VENV_ACT"
else
  echo "[WARN] venv activate not found at $VENV_ACT, continuing (assuming env already active)."
fi

# Critical: allow "from src...." imports
export PYTHONPATH="$REPO_ROOT"

echo "[INFO] Cleaning previous outputs..."
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

echo "[INFO] Running builder..."
python "$REPO_ROOT/src/features/germany_build_tft_weather.py" \
  --train_base "$TRAIN_BASE" \
  --val_base "$VAL_BASE" \
  --weather_dir "$WEATHER_DIR" \
  --out_dir "$OUT_DIR"

echo "[DONE] Stage 3.7 complete."
ls -lh "$OUT_DIR"/*.parquet
