#!/usr/bin/env bash
set -euo pipefail

PHASE_DIR="freeze/final_thesis_v1/phase1_2024daily_final"
PLANT_META="V1.0_FINAL_TFT/plant_metadata/plant_03.json"

START="2024-01-01"
END="2024-12-02"
STRIDE="1"

SHORT_TRAIN="data/processed/plant_level/plant_03/15min_pca32/train.parquet"
LONG_TRAIN="data/processed/plant_level/plant_03/hourly_longhead/train.parquet"
HIST_ENCODER="data/processed/plant_level/plant_03/hist_weather_gt_15min_utc.parquet"

OUT_DIR="freeze/final_thesis_v1/inference_v3_runs"
mkdir -p "$OUT_DIR"

# -----------------------------
# WARM (production) ckpts
# -----------------------------
WARM_SHORT="/home/dwijenayake/pv_forecast_30d/experiments/tft/runs/germany/plant_03/15min/BEST/checkpoints/best.ckpt"
WARM_LONG="/home/dwijenayake/pv_forecast_30d/experiments/tft/runs/germany/plant_03/longhead/hourly720/BEST/checkpoints/best.ckpt"

# -----------------------------
# COLD ckpts 
# -----------------------------
COLD_SHORT="/home/dwijenayake/pv_forecast_30d/experiments/tft/runs/germany/plant_03/15min/pvlib_coldstart/20251229_134850/checkpoints/best.ckpt"
COLD_LONG="/home/dwijenayake/pv_forecast_30d/experiments/tft/runs/germany/plant_03/longhead/hourly720/cold/lr2e-3_do0.15_bs64_acc8_seed43/20251231_104406/checkpoints/best.ckpt"

must_exist () { [[ -f "$1" ]] || { echo "MISSING: $1"; exit 1; }; }

must_exist "${PHASE_DIR}/weather_with_pvlib_15min.parquet"
must_exist "$PLANT_META"
must_exist "$SHORT_TRAIN"
must_exist "$LONG_TRAIN"
must_exist "$HIST_ENCODER"
must_exist "$WARM_SHORT"
must_exist "$WARM_LONG"
must_exist "$COLD_SHORT"
must_exist "$COLD_LONG"

echo "Launching 2 independent runs:"
echo " GPU0: warm hybrid with components"
echo " GPU1: cold hybrid glued"
echo "Outputs -> $OUT_DIR"

# GPU0: warm hybrid components
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m src.inference.phase1_inference_pipeline_v3 \
  --weather-source historical \
  --start-date "$START" --end-date "$END" --stride-days "$STRIDE" \
  --phase1-dir "$PHASE_DIR" \
  --plant-meta "$PLANT_META" \
  --short-ckpt "$WARM_SHORT" --long-ckpt "$WARM_LONG" \
  --short-train "$SHORT_TRAIN" --long-train "$LONG_TRAIN" \
  --hist-encoder "$HIST_ENCODER" \
  --pred-mode "hybrid" --save-components 1 \
  --out "${OUT_DIR}/warm_hybrid_components.parquet" \
  --log-level INFO &

pid0=$!

# GPU1: cold hybrid glued
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python -m src.inference.phase1_inference_pipeline_v3 \
  --weather-source historical \
  --start-date "$START" --end-date "$END" --stride-days "$STRIDE" \
  --phase1-dir "$PHASE_DIR" \
  --plant-meta "$PLANT_META" \
  --short-ckpt "$COLD_SHORT" --long-ckpt "$COLD_LONG" \
  --short-train "$SHORT_TRAIN" --long-train "$LONG_TRAIN" \
  --hist-encoder "$HIST_ENCODER" \
  --pred-mode "hybrid" --save-components 0 \
  --out "${OUT_DIR}/cold_hybrid_glued.parquet" \
  --log-level INFO &

pid1=$!

wait $pid0
wait $pid1

echo "DONE:"
ls -lah "${OUT_DIR}/warm_hybrid_components.parquet" "${OUT_DIR}/cold_hybrid_glued.parquet"
