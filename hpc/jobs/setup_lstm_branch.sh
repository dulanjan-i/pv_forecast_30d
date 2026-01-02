#!/bin/bash
# setup_lstm_branch.sh
# Script to scaffold LSTM branch with sample files and commits

set -e

echo "⚡ Setting up LSTM branch structure..."

# Create directories
mkdir -p notebooks/lstm
mkdir -p src/models
mkdir -p src/training
mkdir -p src/features
mkdir -p src/utils
mkdir -p experiments/lstm
mkdir -p reports

# Add __init__.py so folders are packages
touch src/__init__.py src/models/__init__.py src/training/__init__.py src/features/__init__.py src/utils/__init__.py

# Notebooks (placeholders)
cat > notebooks/lstm/01_lstm_baseline.ipynb <<EOL
{
 "cells": [
  {"cell_type": "markdown", "metadata": {}, "source": ["# Baseline LSTM Model Notebook"]},
  {"cell_type": "markdown", "metadata": {}, "source": ["This notebook explores a simple LSTM baseline on sample PV data."]}
 ],
 "metadata": {"kernelspec": {"name": "python3", "language": "python"}},
 "nbformat": 4,
 "nbformat_minor": 2
}
EOL

# Source files (placeholders with docstrings)
cat > src/models/lstm_model.py <<EOL
"""
LSTM model definition for PV power forecasting.
"""
import torch
import torch.nn as nn

class LSTMForecast(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # last time step
        return out
EOL

cat > src/training/train_lstm.py <<EOL
"""
Training loop for LSTM model.
"""
# Placeholder: load data, define model, optimizer, loss, train, save checkpoints
EOL

cat > src/features/sequence_generator.py <<EOL
"""
Utility to create sliding windows for time series.
"""
# Placeholder: function to generate input/output sequences
EOL

cat > src/utils/metrics.py <<EOL
"""
Custom metrics for PV forecasting.
"""
# Placeholder: RMSE, MAE, R²
EOL

# Experiment config
cat > experiments/lstm/exp01.yaml <<EOL
# Experiment 01: Baseline LSTM
model:
  input_size: 10
  hidden_size: 64
  num_layers: 2
  output_size: 1
training:
  batch_size: 32
  epochs: 50
  learning_rate: 0.001
data:
  source: data/processed/sample.csv
EOL

# Reports placeholder
cat > reports/lstm_results.md <<EOL
# LSTM Experiment Results

This document will track results from LSTM experiments.

- **exp01**: Baseline LSTM (config in experiments/lstm/exp01.yaml)
EOL

# README specific to LSTM branch
cat > README_LSTM.md <<EOL
# LSTM Branch Overview

This branch contains all files related to the development of the LSTM forecasting model.

## Structure
- \`notebooks/lstm/\`: Prototyping and exploration notebooks
- \`src/models/lstm_model.py\`: LSTM model definition
- \`src/training/train_lstm.py\`: Training loop for LSTM
- \`src/features/sequence_generator.py\`: Sliding window generator
- \`src/utils/metrics.py\`: Custom metrics (RMSE, MAE, R²)
- \`experiments/lstm/\`: Config files for reproducible runs
- \`reports/lstm_results.md\`: Results and findings

## First Sprint Goal
1. Build baseline LSTM (notebook + simple training loop)
2. Save experiment config + results
3. Refactor into \`src/\` for reusability
4. Merge back to \`main\` when baseline is stable
EOL

# Git add + commit step-by-step
git add notebooks/lstm/01_lstm_baseline.ipynb
git commit -m "notebooks: add baseline LSTM exploration notebook"

git add src/models/lstm_model.py
git commit -m "feat(models): add LSTM model definition"

git add src/training/train_lstm.py
git commit -m "feat(training): add placeholder training loop for LSTM"

git add src/features/sequence_generator.py
git commit -m "feat(features): add sequence generator utility for time series"

git add src/utils/metrics.py
git commit -m "feat(utils): add placeholder metrics (RMSE, MAE, R²)"

git add experiments/lstm/exp01.yaml
git commit -m "chore(experiments): add baseline LSTM experiment config"

git add reports/lstm_results.md
git commit -m "docs(reports): add LSTM experiment results placeholder"

git add README_LSTM.md
git commit -m "docs: add LSTM branch overview README"

echo "✅ LSTM branch scaffold complete!"
