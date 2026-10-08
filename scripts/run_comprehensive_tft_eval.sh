#!/bin/bash
# Comprehensive TFT RMSE/MAE evaluation across all phases
set -e

cd ~/pv_forecast_30d
source ~/.venvs/pvforecast/bin/activate

echo "========================================="
echo "Phase 1: Ablation Study (Short-head only)"
echo "========================================="

# Short-head ablation already exists at experiments/tft/notes/short_head_eval.csv
echo "✓ Ablation short-head eval already exists"
cat experiments/tft/notes/short_head_eval.csv

echo ""
echo "========================================="
echo "Phase 2: Plant_03 Fine-tuning (Short-head)"
echo "========================================="

python -m src.validation.eval_short_head \
  --modes "warm_seed42" "warm_seed43" "warm_seed44" "cold_seed42" "cold_seed43" "cold_seed44" \
  --run_dirs \
    "experiments/tft/runs/germany/plant_03/15min/pvlib_warmstart_from_global_noleak/20251229_151100" \
    "experiments/tft/runs/germany/plant_03/15min/pvlib_warmstart_from_global_noleak/seed_43/20251229_151105" \
    "experiments/tft/runs/germany/plant_03/15min/pvlib_warmstart_from_global_noleak/seed_44/20251229_151107" \
    "experiments/tft/runs/germany/plant_03/15min/pvlib_coldstart/20251229_134850" \
    "experiments/tft/runs/germany/plant_03/15min/pvlib_coldstart/seed_43/20251229_155059" \
    "experiments/tft/runs/germany/plant_03/15min/pvlib_coldstart/seed_44/20251229_155103" \
  --train_parquet "data/processed/plant_level/plant_03/15min_pca32/train.parquet" \
  --val_parquet "data/processed/plant_level/plant_03/15min_pca32/val.parquet" \
  --out_dir "experiments/tft/runs/germany/plant_03/15min" \
  --batch_size 1024

echo ""
echo "========================================="
echo "Phase 3: Plant_03 Fine-tuning (Long-head)"
echo "========================================="

python -m src.validation.eval_long_head \
  --modes "warm_seed42" "warm_seed43" "warm_seed44" "cold_seed42" "cold_seed43" "cold_seed44" \
  --run_dirs \
    "experiments/tft/runs/germany/plant_03/longhead/hourly720/warm/lr8e-4_do0.15_bs64_acc8_seed42/20251231_104406" \
    "experiments/tft/runs/germany/plant_03/longhead/hourly720/warm/lr8e-4_do0.15_bs64_acc8_seed43/20251231_104405" \
    "experiments/tft/runs/germany/plant_03/longhead/hourly720/warm/lr8e-4_do0.15_bs64_acc8_seed44/20251231_104405" \
    "experiments/tft/runs/germany/plant_03/longhead/hourly720/cold/lr2e-3_do0.15_bs64_acc8_seed42/20251231_104406" \
    "experiments/tft/runs/germany/plant_03/longhead/hourly720/cold/lr2e-3_do0.15_bs64_acc8_seed43/20251231_104406" \
    "experiments/tft/runs/germany/plant_03/longhead/hourly720/cold/lr2e-3_do0.15_bs64_acc8_seed44/20251231_104406" \
  --train_parquet "data/processed/plant_level/plant_03/hourly_longhead/train.parquet" \
  --val_parquet "data/processed/plant_level/plant_03/hourly_longhead/val.parquet" \
  --out_dir "experiments/tft/runs/germany/plant_03/longhead/hourly720" \
  --batch_size 512

echo ""
echo "========================================="
echo "All evaluations complete!"
echo "========================================="
echo "Results saved to:"
echo "  - experiments/tft/notes/short_head_eval.csv (ablation)"
echo "  - experiments/tft/runs/germany/plant_03/15min/short_head_eval.csv (finetune short)"
echo "  - experiments/tft/runs/germany/plant_03/longhead/hourly720/long_head_eval.csv (finetune long)"
