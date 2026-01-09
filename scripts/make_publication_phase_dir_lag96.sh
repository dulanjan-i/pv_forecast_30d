#!/usr/bin/env bash
set -euo pipefail

ORIG_PHASE_DIR="freeze/final_thesis_v1/phase1_2024daily_final"
PUB_PHASE_DIR="freeze/final_thesis_v1/publication_phase_lag96"

WEATHER_NAME="weather_with_pvlib_15min.parquet"
ORIG_WEATHER="${ORIG_PHASE_DIR}/${WEATHER_NAME}"
PUB_WEATHER="${PUB_PHASE_DIR}/${WEATHER_NAME}"

mkdir -p "$PUB_PHASE_DIR"

# Symlink everything from ORIG into PUB except the weather parquet
for item in "${ORIG_PHASE_DIR}"/*; do
  base="$(basename "$item")"
  if [[ "$base" == "$WEATHER_NAME" ]]; then
    continue
  fi
  # If link already exists, skip
  if [[ -e "${PUB_PHASE_DIR}/${base}" ]]; then
    continue
  fi
  ln -s "$(realpath "$item")" "${PUB_PHASE_DIR}/${base}"
done

# Create lagged weather parquet inside PUB dir (publication only)
PYTHONPATH=. python -m src.inference.make_weather_lagged_lstm_pca \
  --in-weather "$ORIG_WEATHER" \
  --out-weather "$PUB_WEATHER" \
  --lag 96

echo "[OK] Publication phase dir ready: $PUB_PHASE_DIR"
echo "     Swapped weather file: $PUB_WEATHER"
