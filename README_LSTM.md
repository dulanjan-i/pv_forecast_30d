# LSTM Branch Overview

This branch contains all files related to the development of the LSTM forecasting model.

## Structure
- `notebooks/lstm/`: Prototyping and exploration notebooks
- `src/models/lstm_model.py`: LSTM model definition
- `src/training/train_lstm.py`: Training loop for LSTM
- `src/features/sequence_generator.py`: Sliding window generator
- `src/utils/metrics.py`: Custom metrics (RMSE, MAE, R²)
- `experiments/lstm/`: Config files for reproducible runs
- `reports/lstm_results.md`: Results and findings

## First Sprint Goal
1. Build baseline LSTM (notebook + simple training loop)
2. Save experiment config + results
3. Refactor into `src/` for reusability
4. Merge back to `main` when baseline is stable
