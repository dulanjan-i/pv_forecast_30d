#!/bin/bash
# Benchmark Suite v3 - FORMATTED for thesis
# Same plots as v1, just FORMATTING FIXES (labels, DPI, legend clarity)
# Output to NEW directory: thesis_formatted_v3

set -e

echo "=== Benchmark Suite v3 - FORMATTED ==="
echo "Changes from v1:"
echo "  - 300 DPI (was 200)"
echo "  - Clear legend labels: 'Ground Truth Plant 03', 'MiRACLE v1.0 Core'"
echo "  - Larger fonts, better grid styling"
echo "  - Improved date formatting on x-axis"
echo ""

TRUTH="/home/dwijenayake/pv_forecast_30d/freeze/final_thesis_v1/phase1_2024daily_final/processed/ground_truth_15min_utc_capnorm.parquet"
BASELINE="/home/dwijenayake/pv_forecast_30d/freeze/final_thesis_v1/phase1_2024daily_final/processed/predictions_phase1_baseline_rerun.parquet"
PVLIB="/home/dwijenayake/pv_forecast_30d/freeze/final_thesis_v1/inference_v3_runs/derived_only/pvlib_only.parquet"
TFT="/home/dwijenayake/pv_forecast_30d/freeze/final_thesis_v1/inference_v3_runs/derived_only/tft_only.parquet"
SHORT="/home/dwijenayake/pv_forecast_30d/freeze/final_thesis_v1/inference_v3_runs/derived_only/short_only.parquet"
LONG="/home/dwijenayake/pv_forecast_30d/freeze/final_thesis_v1/inference_v3_runs/derived_only/long_only.parquet"

OUT="/home/dwijenayake/pv_forecast_30d/freeze/final_thesis_v1/benchmarks/thesis_formatted_v3"

echo "Running formatted benchmark suite..."
echo "  Truth: $TRUTH"
echo "  Baseline: MiRACLE v1.0 Core"
echo "  Models: PVLib-Physics-Only, TFT-Only, Short-TFT-Only, Long-TFT-Only"
echo "  Output: $OUT"
echo ""

PYTHONUNBUFFERED=1 PYTHONPATH=. python -m src.evaluation.run_benchmark_suite_v3_formatted \
  --truth "$TRUTH" \
  --baseline-name "MiRACLE v1.0 Core" \
  --baseline "$BASELINE" \
  --truth-label "Ground Truth Plant 03" \
  --model "PVLib-Physics-Only:$PVLIB" \
  --model "TFT-Only:$TFT" \
  --model "Short-TFT-Only:$SHORT" \
  --model "Long-TFT-Only:$LONG" \
  --out "$OUT" \
  --daylight-threshold 0.01

echo ""
echo "=== SUCCESS ==="
echo "Formatted plots saved to: $OUT/figures/"
echo ""
echo "Generated figures (SAME as v2, but FORMATTED):"
echo "  1. facets_case_summer_week.png - 300 DPI, clear labels"
echo "  2. facets_case_winter_week.png - 300 DPI, clear labels"  
echo "  3. facets_abs_error_hist.png - Improved legend"
echo "  4. facets_leadtime_rmse_curve_0_24h.png - Better styling"
echo "  5. monthly_rmse_all_models.png - Larger figure"
echo ""
echo "Compare with original:"
echo "  Original: freeze/final_thesis_v1/benchmarks/final_suite_v2/figures/"
echo "  Formatted: $OUT/figures/"
