#!/bin/bash
# Quick Reference: Version 3 Execution Commands

# ============================================================================
# PREPROCESSING (Run these first - ~3 minutes total)
# ============================================================================

# Step 1: Build Super Matrix (~2 min)
python src/preprocessing/germany_build_global_supermatrix.py

# Step 2: Create Rolling Origin Splits (~1 min)
python src/preprocessing/germany_global_rolling_origin_split.py

# ============================================================================
# TRAINING (Run these second - ~2-3 hours total)
# ============================================================================

# Option A: Train all 4 folds sequentially (recommended)
./run_stage3_global_training.sh

# Option B: Train individual folds (for testing/debugging)
python src/training/train_global_lstm_v3.py --fold 1 --gpus 1
python src/training/train_global_lstm_v3.py --fold 2 --gpus 1
python src/training/train_global_lstm_v3.py --fold 3 --gpus 1
python src/training/train_global_lstm_v3.py --fold 4 --gpus 1

# Option C: CPU-only training (if no GPU available)
python src/training/train_global_lstm_v3.py --fold 1 --gpus 0

# Option D: Adjust batch size if OOM errors
python src/training/train_global_lstm_v3.py --fold 1 --batch_size 64 --gpus 1

# ============================================================================
# VALIDATION (Run after training)
# ============================================================================

# Quick check: View metrics for fold 1
cat experiments/lstm/runs/germany/global_v3/fold_1/metrics.csv | tail -n 5

# Calculate average train/val ratio across all folds
python -c "
import pandas as pd
folds = [1, 2, 3, 4]
ratios = []
for f in folds:
    df = pd.read_csv(f'experiments/lstm/runs/germany/global_v3/fold_{f}/metrics.csv')
    final = df.iloc[-1]
    ratio = final['val_loss'] / final['train_loss']
    ratios.append(ratio)
    print(f'Fold {f}: train={final[\"train_loss\"]:.4f}, val={final[\"val_loss\"]:.4f}, ratio={ratio:.2f}')
print(f'\nAverage ratio: {sum(ratios)/len(ratios):.2f}')
print(f'Target: <1.5 (Version 02 was 2.0-2.5)')
"

# ============================================================================
# TROUBLESHOOTING
# ============================================================================

# Check if supermatrix was created
ls -lh data/processed/pretraining/germany/global/supermatrix_base.parquet

# Check if fold data was created
ls -lh data/processed/pretraining/germany/global/fold_*

# Check if Farm2107 checkpoint exists
ls -lh experiments/lstm/encoders/lstm_encoder_farm2107_CANONICAL.pt

# View training logs for fold 1
tail -f experiments/lstm/runs/germany/global_v3/fold_1/version_0/metrics.csv

# Check GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check GPU memory usage
nvidia-smi

# ============================================================================
# GIT WORKFLOW
# ============================================================================

# Commit Version 3 implementation
git add src/data/schema.py
git add src/preprocessing/germany_build_global_supermatrix.py
git add src/preprocessing/germany_global_rolling_origin_split.py
git add src/models/global_lstm_encoder.py
git add src/training/train_global_lstm_v3.py
git add run_stage3_global_training.sh
git add reports/version03_global_model_implementation.md
git commit -m "feat: Version 3 Global Model - Super Matrix + Rolling Origin CV + Zero-padding Transfer Learning"
git push origin lstm-build

# ============================================================================
# EXPECTED OUTPUTS
# ============================================================================

# Preprocessing outputs:
# - data/processed/pretraining/germany/global/supermatrix_base.parquet (~150K rows, 21 cols)
# - data/processed/pretraining/germany/global/fold_X_{train,val}.parquet (12 files)
# - data/processed/pretraining/germany/global/fold_X_scaler.json (4 files)

# Training outputs (per fold):
# - experiments/lstm/runs/germany/global_v3/fold_X/lstm_encoder_global_fold_X.pt
# - experiments/lstm/runs/germany/global_v3/fold_X/best_checkpoint.ckpt
# - experiments/lstm/runs/germany/global_v3/fold_X/metrics.csv
# - experiments/lstm/runs/germany/global_v3/fold_X/hparams.yaml
