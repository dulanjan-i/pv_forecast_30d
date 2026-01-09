#!/usr/bin/env bash
set -euo pipefail

# -----------------
# Common inputs
# -----------------
ORIG_PHASE_DIR="freeze/final_thesis_v1/phase1_2024daily_final"
LAG_PHASE_DIR="freeze/final_thesis_v1/publication_phase_lag96"

PLANT_META="V1.0_FINAL_TFT/plant_metadata/plant_03.json"

START="2024-01-01"
END="2024-12-02"
STRIDE="1"

SHORT_TRAIN="data/processed/plant_level/plant_03/15min_pca32/train.parquet"
LONG_TRAIN="data/processed/plant_level/plant_03/hourly_longhead/train.parquet"
HIST_ENCODER="data/processed/plant_level/plant_03/hist_weather_gt_15min_utc.parquet"

LONG_CKPT="/home/dwijenayake/pv_forecast_30d/experiments/tft/runs/germany/plant_03/longhead/hourly720/BEST/checkpoints/best.ckpt"

OUT_DIR="freeze/final_thesis_v1/ablations_shorthead_inference"
mkdir -p "$OUT_DIR"

# -----------------
# Ablation short-head ckpts
# -----------------
BASE="/home/dwijenayake/pv_forecast_30d/experiments/tft/runs/germany/ablations"
TS="20251226_165225"

CKPT_TFT_ONLY="${BASE}/tft_only/${TS}/checkpoints/best.ckpt"
CKPT_TFT_PVLIB="${BASE}/tft_pvlib/${TS}/checkpoints/best.ckpt"
CKPT_TFT_LSTM="${BASE}/tft_lstm/${TS}/checkpoints/best.ckpt"
CKPT_FULL="${BASE}/full/${TS}/checkpoints/best.ckpt"

# -----------------
# Sanity checks
# -----------------
for p in "$CKPT_TFT_ONLY" "$CKPT_TFT_PVLIB" "$CKPT_TFT_LSTM" "$CKPT_FULL" "$LONG_CKPT"; do
  if [[ ! -f "$p" ]]; then
    echo "MISSING: $p"
    exit 1
  fi
done

run_one () {
  local gpu="$1"
  local phase_dir="$2"
  local name="$3"
  local short_ckpt="$4"
  local out="$OUT_DIR/${name}.parquet"

  echo ""
  echo "============================================================"
  echo "RUN: $name"
  echo "GPU: $gpu"
  echo "phase1-dir: $phase_dir"
  echo "short-ckpt: $short_ckpt"
  echo "long-ckpt:  $LONG_CKPT"
  echo "pred-mode: short_only"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=. python -m src.inference.phase1_inference_pipeline_v3 \
    --weather-source historical \
    --start-date "$START" --end-date "$END" --stride-days "$STRIDE" \
    --phase1-dir "$phase_dir" \
    --out "$out" \
    --plant-meta "$PLANT_META" \
    --short-ckpt "$short_ckpt" --long-ckpt "$LONG_CKPT" \
    --short-train "$SHORT_TRAIN" --long-train "$LONG_TRAIN" \
    --hist-encoder "$HIST_ENCODER" \
    --pred-mode short_only \
    --save-components 0 \
    --log-level INFO
}

# First pair (no LSTM cols required)
run_one 0 "$ORIG_PHASE_DIR" "ablate_tft_only"  "$CKPT_TFT_ONLY"  &
pid0=$!
run_one 1 "$ORIG_PHASE_DIR" "ablate_tft_pvlib" "$CKPT_TFT_PVLIB" &
pid1=$!
wait $pid0
wait $pid1

# Second pair (needs lagged PCA cols)
run_one 0 "$LAG_PHASE_DIR" "ablate_tft_lstm" "$CKPT_TFT_LSTM" &
pid0=$!
run_one 1 "$LAG_PHASE_DIR" "ablate_full"     "$CKPT_FULL"     &
pid1=$!
wait $pid0
wait $pid1

echo ""
echo "DONE. Outputs:"
ls -lah "$OUT_DIR"/*.parquet
