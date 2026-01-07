#!/usr/bin/env bash
set -euo pipefail

# --------
# Config
# --------
PHASE_DIR="freeze/final_thesis_v1/phase1_2024daily_final"
PLANT_META="V1.0_FINAL_TFT/plant_metadata/plant_03.json"

START="2024-01-01"
END="2024-12-02"
STRIDE="1"

SHORT_TRAIN="data/processed/plant_level/plant_03/15min_pca32/train.parquet"
LONG_TRAIN="data/processed/plant_level/plant_03/hourly_longhead/train.parquet"
HIST_ENCODER="data/processed/plant_level/plant_03/hist_weather_gt_15min_utc.parquet"

# Keep long head constant unless you want warm vs cold later
LONG_CKPT="/home/dwijenayake/pv_forecast_30d/experiments/tft/runs/germany/plant_03/longhead/hourly720/BEST/checkpoints/best.ckpt"


OUT_DIR="freeze/final_thesis_v1/ablations_inference_2gpu"
mkdir -p "$OUT_DIR"

# Which prediction type to output: hybrid, pvlib_only, tft_only
PRED_MODE="hybrid"
SAVE_COMPONENTS="0"

# --------
# Your ablation short-head checkpoints
# Edit these paths to match your repo
# --------
BASE_DIR="/home/dwijenayake/pv_forecast_30d/experiments/tft/runs/germany/ablations"
TS="20251226_165225"

declare -A SHORT_CKPTS
SHORT_CKPTS["tft_only"]="${BASE_DIR}/tft_only/${TS}/checkpoints/best.ckpt"
SHORT_CKPTS["tft_pvlib"]="${BASE_DIR}/tft_pvlib/${TS}/checkpoints/best.ckpt"
SHORT_CKPTS["tft_lstm"]="${BASE_DIR}/tft_lstm/${TS}/checkpoints/best.ckpt"
SHORT_CKPTS["full"]="${BASE_DIR}/full/${TS}/checkpoints/best.ckpt"

# --------
# Helpers
# --------
merge_two_parquets () {
  local in1="$1"
  local in2="$2"
  local out="$3"

  python - <<PY
import pyarrow.parquet as pq
import pyarrow as pa

t1 = pq.read_table("${in1}")
t2 = pq.read_table("${in2}")

t = pa.concat_tables([t1, t2], promote=True)

# Sort for deterministic output if keys exist
cols = t.column_names
sort_keys = [c for c in ["forecast_start", "step_ahead", "timestamp_utc"] if c in cols]
if sort_keys:
    df = t.to_pandas()
    df = df.sort_values(sort_keys)
    df.to_parquet("${out}", index=False)
else:
    pq.write_table(t, "${out}")

print("MERGED:", "${out}", "rows=", t.num_rows)
PY
}

run_one_model () {
  local name="$1"
  local short_ckpt="$2"

  if [[ ! -f "$short_ckpt" ]]; then
    echo "MISSING short ckpt for $name: $short_ckpt"
    return 0
  fi
  if [[ ! -f "$LONG_CKPT" ]]; then
    echo "MISSING long ckpt: $LONG_CKPT"
    exit 1
  fi

  local out0="${OUT_DIR}/${name}.shard0.parquet"
  local out1="${OUT_DIR}/${name}.shard1.parquet"
  local out="${OUT_DIR}/ablation_${name}.parquet"

  echo ""
  echo "============================================================"
  echo "RUN: ${name}"
  echo "short_ckpt: ${short_ckpt}"
  echo "long_ckpt:  ${LONG_CKPT}"
  echo "pred_mode:  ${PRED_MODE}"
  echo "============================================================"

  # shard 0 on GPU 0
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m src.inference.phase1_inference_pipeline_v2 \
    --weather-source historical \
    --start-date "$START" --end-date "$END" --stride-days "$STRIDE" \
    --phase1-dir "$PHASE_DIR" \
    --plant-meta "$PLANT_META" \
    --short-ckpt "$short_ckpt" --long-ckpt "$LONG_CKPT" \
    --short-train "$SHORT_TRAIN" --long-train "$LONG_TRAIN" \
    --hist-encoder "$HIST_ENCODER" \
    --pred-mode "$PRED_MODE" --save-components "$SAVE_COMPONENTS" \
    --num-shards 2 --shard-idx 0 \
    --out "$out0" \
    --log-level INFO &

  pid0=$!

  # shard 1 on GPU 1
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python -m src.inference.phase1_inference_pipeline_v2 \
    --weather-source historical \
    --start-date "$START" --end-date "$END" --stride-days "$STRIDE" \
    --phase1-dir "$PHASE_DIR" \
    --plant-meta "$PLANT_META" \
    --short-ckpt "$short_ckpt" --long-ckpt "$LONG_CKPT" \
    --short-train "$SHORT_TRAIN" --long-train "$LONG_TRAIN" \
    --hist-encoder "$HIST_ENCODER" \
    --pred-mode "$PRED_MODE" --save-components "$SAVE_COMPONENTS" \
    --num-shards 2 --shard-idx 1 \
    --out "$out1" \
    --log-level INFO &

  pid1=$!

  wait $pid0
  wait $pid1

  merge_two_parquets "$out0" "$out1" "$out"

  rm -f "$out0" "$out1"
  echo "DONE: $out"
}

# --------
# Run all
# --------
for name in "${!SHORT_CKPTS[@]}"; do
  run_one_model "$name" "${SHORT_CKPTS[$name]}"
done

echo ""
echo "ALL DONE. Outputs in: $OUT_DIR"
