#!/bin/bash
# Regenerate all RQ evaluations with thesis-consistent colors
# Ground truth = light grey, Model A (baseline/warm) = bold green, Model B (comparison) = light blue

set -e

cd /home/dwijenayake/pv_forecast_30d

# Define paths
TRUTH="freeze/final_thesis_v1/phase1_2024daily_final/processed/ground_truth_15min_utc_capnorm.parquet"
WARM="freeze/final_thesis_v1/inference_v3_runs/warm_hybrid_components.parquet"
COLD="freeze/final_thesis_v1/inference_v3_runs/cold_hybrid_glued.parquet"
PVLIB="freeze/final_thesis_v1/inference_v3_runs/derived_only/pvlib_only.parquet"
SHORT="freeze/final_thesis_v1/inference_v3_runs/derived_only/short_only.parquet"
LONG="freeze/final_thesis_v1/inference_v3_runs/derived_only/long_only.parquet"
TFT="freeze/final_thesis_v1/inference_v3_runs/derived_only/tft_only.parquet"

echo "=== Regenerating RQ1 evaluations (warm hybrid vs ablations) ==="

echo "RQ1a: Warm vs PVLib"
PYTHONPATH=. python -m src.evaluation.run_pair_eval \
  --truth "$TRUTH" \
  --a "$WARM" \
  --b "$PVLIB" \
  --a-name "MiRACLE v1.0 (Core)" \
  --b-name "PVLib-Physics-Only" \
  --out freeze/final_thesis_v1/eval/rq1_warm_vs_pvlib

echo "RQ1b: Warm vs TFT"
PYTHONPATH=. python -m src.evaluation.run_pair_eval \
  --truth "$TRUTH" \
  --a "$WARM" \
  --b "$TFT" \
  --a-name "MiRACLE v1.0 (Core)" \
  --b-name "TFT-Only" \
  --out freeze/final_thesis_v1/eval/rq1_warm_vs_tft

echo "RQ1c: Warm vs Short"
PYTHONPATH=. python -m src.evaluation.run_pair_eval \
  --truth "$TRUTH" \
  --a "$WARM" \
  --b "$SHORT" \
  --a-name "MiRACLE v1.0 (Core)" \
  --b-name "Short-TFT-Only" \
  --out freeze/final_thesis_v1/eval/rq1_warm_vs_short

echo "RQ1d: Warm vs Long"
PYTHONPATH=. python -m src.evaluation.run_pair_eval \
  --truth "$TRUTH" \
  --a "$WARM" \
  --b "$LONG" \
  --a-name "MiRACLE v1.0 (Core)" \
  --b-name "Long-TFT-Only" \
  --out freeze/final_thesis_v1/eval/rq1_warm_vs_long

echo "=== Regenerating RQ2 evaluation (warm vs cold start) ==="
PYTHONPATH=. python -m src.evaluation.run_pair_eval \
  --truth "$TRUTH" \
  --a "$WARM" \
  --b "$COLD" \
  --a-name "MiRACLE v1.0 (Core)" \
  --b-name "MiRACLE v1.0 (Core, cold-start)" \
  --out freeze/final_thesis_v1/eval/rq2_warm_vs_cold

echo "=== Regenerating RQ4 evaluation (baseline vs policy) ==="
PYTHONPATH=. python -m src.evaluation.run_full_eval \
  --truth "$TRUTH" \
  --baseline freeze/final_thesis_v1/phase1_2024daily_final/processed/predictions_phase1_baseline_rerun.parquet \
  --policy freeze/final_thesis_v1/phase1_2024daily_final/processed/predictions_phase1_policy_rerun.parquet \
  --out freeze/final_thesis_v1/eval/rq4_baseline_vs_policy

echo ""
echo "=== SUCCESS ==="
echo "All RQ evaluations regenerated with thesis-consistent colors:"
echo "  - Ground Truth: Light grey (#888888), linewidth 1.5, alpha 0.7"
echo "  - Model A/Baseline (MiRACLE v1.0 (Core)): Bold green (#00AA00), linewidth 2.5, alpha 1.0"
echo "  - Model B/Comparison: Light blue (#6BA3D8), linewidth 1.5, alpha 0.9"
echo "  - Resolution: 300 DPI"
echo ""
echo "Updated directories:"
echo "  - freeze/final_thesis_v1/eval/rq1_warm_vs_pvlib/"
echo "  - freeze/final_thesis_v1/eval/rq1_warm_vs_tft/"
echo "  - freeze/final_thesis_v1/eval/rq1_warm_vs_short/"
echo "  - freeze/final_thesis_v1/eval/rq1_warm_vs_long/"
echo "  - freeze/final_thesis_v1/eval/rq2_warm_vs_cold/"
echo "  - freeze/final_thesis_v1/eval/rq4_baseline_vs_policy/"
