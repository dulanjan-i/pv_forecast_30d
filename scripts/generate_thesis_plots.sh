#!/bin/bash
# Script to generate thesis-ready benchmark plots with proper labels
# Usage: bash scripts/generate_thesis_plots.sh

set -e

echo "=== Generating Thesis-Ready Benchmark Plots ==="
echo ""

# Paths (actual data locations from your working command)
TRUTH="/home/dwijenayake/pv_forecast_30d/freeze/final_thesis_v1/phase1_2024daily_final/processed/ground_truth_15min_utc_capnorm.parquet"
BASELINE="/home/dwijenayake/pv_forecast_30d/freeze/final_thesis_v1/phase1_2024daily_final/processed/predictions_phase1_baseline_rerun.parquet"
OUT_DIR="/home/dwijenayake/pv_forecast_30d/freeze/final_thesis_v1/benchmarks/thesis_ready"

# Check if files exist
if [ ! -f "$TRUTH" ]; then
    echo "ERROR: Ground truth file not found: $TRUTH"
    echo "Please update the TRUTH path in this script"
    exit 1
fi

if [ ! -f "$BASELINE" ]; then
    echo "ERROR: Baseline predictions file not found: $BASELINE"
    echo "Please update the BASELINE path in this script"
    exit 1
fi

# Additional model paths for comparison
PVLIB_ONLY="/home/dwijenayake/pv_forecast_30d/freeze/final_thesis_v1/inference_v3_runs/derived_only/pvlib_only.parquet"
TFT_ONLY="/home/dwijenayake/pv_forecast_30d/freeze/final_thesis_v1/inference_v3_runs/derived_only/tft_only.parquet"
SHORT_ONLY="/home/dwijenayake/pv_forecast_30d/freeze/final_thesis_v1/inference_v3_runs/derived_only/short_only.parquet"
LONG_ONLY="/home/dwijenayake/pv_forecast_30d/freeze/final_thesis_v1/inference_v3_runs/derived_only/long_only.parquet"

# Run thesis-ready benchmark suite
echo "Running thesis-ready benchmark suite..."
echo "  - Truth: $TRUTH"
echo "  - Baseline (Core): $BASELINE"
echo "  - Comparison Models:"
echo "    - PVLib-Only (physics baseline)"
echo "    - TFT-Only (pure ML, no physics)"
echo "    - Short-TFT-Only (tactical 24h)"
echo "    - Long-TFT-Only (strategic 30d)"
echo "  - Output: $OUT_DIR"
echo ""

PYTHONUNBUFFERED=1 PYTHONPATH=. python -m src.evaluation.run_benchmark_suite_thesis_ready \
  --truth "$TRUTH" \
  --baseline-name "MiRACLE v1.0 Core" \
  --baseline "$BASELINE" \
  --model "PVLib-Physics-Only:$PVLIB_ONLY" \
  --model "TFT-Only:$TFT_ONLY" \
  --model "Short-TFT-Only:$SHORT_ONLY" \
  --model "Long-TFT-Only:$LONG_ONLY" \
  --out "$OUT_DIR" \
  --truth-label "Ground Truth Plant 03" \
  --case-summer-start "2024-07-01T00:00:00Z" \
  --case-summer-end "2024-07-08T00:00:00Z" \
  --case-winter-start "2024-01-10T00:00:00Z" \
  --case-winter-end "2024-01-17T00:00:00Z"

echo ""
echo "=== SUCCESS ==="
echo "Thesis-ready plots saved to: $OUT_DIR/figures/"
echo ""
echo "Generated plots:"
echo "  1. thesis_case_summer_week.png - Summer week comparison (FIXED LABELS)"
echo "  2. thesis_case_winter_week.png - Winter week comparison (FIXED LABELS)"
echo "  3. thesis_scatter_predicted_vs_actual.png - Scatter plot with R² (NEW)"
echo "  4. thesis_residuals_histogram.png - Residual distribution (NEW)"
echo "  5. thesis_qq_plot.png - Q-Q normality check (NEW)"
echo ""
echo "Next steps:"
echo "  - Review plots in: $OUT_DIR/figures/"
echo "  - Check VISUALIZATION_CHECKLIST.md for missing plots"
echo "  - Add comparison models using --model flag if available"
