#!/bin/bash

################################################################################
# run_stage3_global_training.sh
#
# Version 3 - Global Forecasting Model: Rolling Origin Training Wrapper
#
# PURPOSE
# -------
# Sequential training of all 4 rolling origin folds for the Global LSTM Encoder.
# This script automates the training pipeline by calling train_global_lstm_v3.py
# four times (once per fold).
#
# FOLDS
# -----
# Fold 1 (Spring): Val = Mar-May 2023, Train = all before Mar 2023
# Fold 2 (Summer): Val = Jun-Aug 2023, Train = all before Jun 2023
# Fold 3 (Fall): Val = Sep-Nov 2023, Train = all before Sep 2023
# Fold 4 (Winter): Val = Dec 2023 - Feb 2024, Train = all before Dec 2023
#
# USAGE
# -----
# Make executable:
#   chmod +x run_stage3_global_training.sh
#
# Run all folds:
#   ./run_stage3_global_training.sh
#
# Run specific fold (optional):
#   python src/training/train_global_lstm_v3.py --fold 1
#
# PREREQUISITES
# -------------
# 1. Preprocessing completed:
#    - data/processed/pretraining/germany/global/supermatrix_base.parquet
#    - data/processed/pretraining/germany/global/fold_{1,2,3,4}_{train,val}.parquet
# 2. Farm2107 checkpoint exists:
#    - experiments/lstm/encoders/lstm_encoder_farm2107_CANONICAL.pt
# 3. Python environment activated with PyTorch, Lightning, pandas
#
# OUTPUTS
# -------
# For each fold, saves to: experiments/lstm/runs/germany/global_v3/fold_X/
#   - lstm_encoder_global_fold_X.pt  # Final model weights
#   - best_checkpoint.ckpt            # Best val_loss checkpoint
#   - metrics.csv                     # Train/val loss per epoch
#   - hparams.yaml                    # Logged hyperparameters
#
# ESTIMATED RUNTIME
# -----------------
# Per fold: ~30-45 minutes on GPU (RTX 3090)
# Total: ~2-3 hours for all 4 folds (with early stopping)
#
# GPU MEMORY
# ----------
# ~2-3 GB per fold (batch_size=128, hidden_size=64)
# Adjust --batch_size if OOM errors occur
#
# Author: PV Forecast Team
# Date: December 2024
# Version: 3.0 (Global Model with Rolling Origin CV)
################################################################################

# Exit on any error
set -e

# Color codes for pretty output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_separator() {
    echo "================================================================================"
}

# Print header
print_separator
echo -e "${GREEN}STAGE 3: GLOBAL LSTM ENCODER TRAINING${NC}"
echo -e "${GREEN}Version 3 - Rolling Origin Cross-Validation${NC}"
print_separator
echo ""

# Step 1: Activate Python environment
print_info "Activating Python environment..."
if [ -f "$HOME/.venvs/pvforecast/bin/activate" ]; then
    source "$HOME/.venvs/pvforecast/bin/activate"
    print_success "Python environment activated: pvforecast"
elif [ -f "$HOME/miniconda3/bin/activate" ]; then
    source "$HOME/miniconda3/bin/activate"
    conda activate pv_forecast_30d
    print_success "Conda environment activated: pv_forecast_30d"
else
    print_error "Python environment not found!"
    print_info "Expected locations:"
    print_info "  - $HOME/.venvs/pvforecast/bin/activate"
    print_info "  - $HOME/miniconda3/bin/activate"
    exit 1
fi

# Step 2: Set PYTHONPATH
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"
print_info "PYTHONPATH set to: $REPO_ROOT"
echo ""

# Step 3: Validate prerequisites
print_info "Validating prerequisites..."

# Check supermatrix
SUPERMATRIX="$REPO_ROOT/data/processed/pretraining/germany/global/supermatrix_base.parquet"
if [ ! -f "$SUPERMATRIX" ]; then
    print_error "Super matrix not found: $SUPERMATRIX"
    print_info "Run preprocessing first:"
    print_info "  python src/preprocessing/germany_build_global_supermatrix.py"
    exit 1
fi
print_success "Super matrix found: $SUPERMATRIX"

# Check fold data
for fold in 1 2 3 4; do
    TRAIN_FILE="$REPO_ROOT/data/processed/pretraining/germany/global/fold_${fold}_train.parquet"
    VAL_FILE="$REPO_ROOT/data/processed/pretraining/germany/global/fold_${fold}_val.parquet"
    
    if [ ! -f "$TRAIN_FILE" ]; then
        print_error "Fold $fold train data not found: $TRAIN_FILE"
        print_info "Run preprocessing first:"
        print_info "  python src/preprocessing/germany_global_rolling_origin_split.py"
        exit 1
    fi
    
    if [ ! -f "$VAL_FILE" ]; then
        print_error "Fold $fold val data not found: $VAL_FILE"
        exit 1
    fi
done
print_success "All fold data files found (folds 1-4)"

# Check Farm2107 checkpoint
FARM2107_CKPT="$REPO_ROOT/experiments/lstm/encoders/lstm_encoder_farm2107_CANONICAL.pt"
if [ ! -f "$FARM2107_CKPT" ]; then
    print_warning "Farm2107 checkpoint not found: $FARM2107_CKPT"
    print_warning "Training will proceed from scratch (no transfer learning)"
else
    print_success "Farm2107 checkpoint found: $FARM2107_CKPT"
fi

echo ""
print_separator

# Step 4: Train all folds (1-4 CV, 5 test)
TOTAL_FOLDS=5
FAILED_FOLDS=()

for fold in 1 2 3 4; do
    print_separator
    echo -e "${GREEN}TRAINING FOLD $fold / $TOTAL_FOLDS${NC}"
    print_separator
    echo ""
    
    # Start timestamp
    START_TIME=$(date +%s)
    
    # Run training
    if python "$REPO_ROOT/src/training/train_global_lstm_v3.py" \
        --fold "$fold" \
        --window_size 96 \
        --batch_size 256 \
        --hidden_size 64 \
        --num_layers 2 \
        --dropout 0.1 \
        --lr 1e-4 \
        --max_epochs 30 \
        --patience 5 \
        --gpus 2 \
        --num_workers 2 \
        --precision 16-mixed \
        --precision_override high; then
        
        # Calculate elapsed time
        END_TIME=$(date +%s)
        ELAPSED=$((END_TIME - START_TIME))
        MINUTES=$((ELAPSED / 60))
        SECONDS=$((ELAPSED % 60))
        
        print_success "Fold $fold training completed in ${MINUTES}m ${SECONDS}s"
        echo ""
    else
        print_error "Fold $fold training FAILED!"
        FAILED_FOLDS+=("$fold")
        echo ""
    fi
done

# Step 5: Summary
print_separator
echo -e "${GREEN}TRAINING SUMMARY${NC}"
print_separator
echo ""

if [ ${#FAILED_FOLDS[@]} -eq 0 ]; then
    print_success "All 5 folds trained successfully! ✅"
    echo ""
    print_info "Results saved to:"
    print_info "  experiments/lstm/runs/germany/global_v3/fold_1/ (CV - Spring)"
    print_info "  experiments/lstm/runs/germany/global_v3/fold_2/ (CV - Summer)"
    print_info "  experiments/lstm/runs/germany/global_v3/fold_3/ (CV - Fall)"
    print_info "  experiments/lstm/runs/germany/global_v3/fold_4/ (CV - Winter)"
    print_info "  experiments/lstm/runs/germany/global_v3/fold_5/ (TEST - Held-out)"
    echo ""
    print_info "Next steps:"
    print_info "  1. Analyze CV metrics: check metrics.csv in folds 1-4 (average)"
    print_info "  2. Evaluate test set: check fold 5 metrics (final evaluation)"
    print_info "  3. Compare to Version 02 baseline (ratio ~2.0-2.5)"
    print_info "  4. If successful (CV ratio <1.5, test ratio ~1.2-1.4), proceed to Stage 2B"
else
    print_error "Training failed for folds: ${FAILED_FOLDS[*]}"
    print_info "Check error logs above for details"
    exit 1
fi

print_separator
echo ""
