# Copilot instructions for pv_forecast_30d

This repo supports a time-series forecasting workflow for PV output, centered on an LSTM baseline and a Lightning-based encoder. Use these notes to be productive quickly and follow existing patterns.

## Big picture
- Goals: prototype and train PV power forecasting models (initially LSTM), track experiments, and refactor into reusable `src/` modules.
- Data layout: `data/{raw,interim,processed}` with sliding-window inputs expected as tensors of shape (B, T, F).
- Code layout:
  - `src/models/lstm_model.py`: minimal PyTorch LSTM that predicts last step.
  - `src/models/lstm_encoder.py`: PyTorch Lightning encoder (LSTM -> embedding [+ optional next-step auxiliary head]). Includes `LSTMEncoderConfig`, `LSTMEncoder`, `SimpleWindowDataset`, and `make_trainer()`.
  - `src/features/sequence_generator.py`: intended sliding-window builder (currently a placeholder).
  - `src/training/train_lstm.py`: training script stub; use `LSTMEncoder` or `LSTMForecast` as needed.
  - `src/utils/metrics.py`: placeholder for RMSE/MAE/R².
  - `experiments/lstm/*.yaml`: experiment configs (reference for hyperparameters and data source paths).
  - `notebooks/lstm/`: prototyping; results summarized in `reports/lstm_results.md`.

## Key patterns and conventions
- Tensors/shapes: windows are (B, T, F). `LSTMEncoder.forward` returns a dict with `embedding: (B, D)` and optional `next_pred: (B,)`.
- Configs: prefer typed configs via `dataclasses` (see `LSTMEncoderConfig`). If mapping YAML to code, parse YAML and populate this dataclass explicitly.
- Training: for Lightning flows, use `make_trainer(...)` from `lstm_encoder.py` and `SimpleWindowDataset` for quick loops. Non-Lightning baseline exists in `lstm_model.py`.
- Experiments: keep per-run YAML under `experiments/lstm/` and append results to `reports/lstm_results.md`. Use run IDs like `exp01`, `exp02`, etc.
- Packaging: every module folder has `__init__.py` for importable packages; prefer `src`-relative imports.

## Dev environment and tooling
- Env: create a Conda env with `environment.yml` (Python 3.11, PyTorch stack, sklearn, Optuna, PVLib, etc.). Note: `pytorch_lightning` is used in `lstm_encoder.py` but not pinned in `environment.yml`—install it if missing.
- Notebooks: JupyterLab is included; prototype in `notebooks/lstm/` and backport stable code into `src/`.
- CI: `.github/workflows/` exists but has no active workflows yet.

## How to run a quick local experiment (example)
- Minimal Lightning training loop using the encoder and synthetic data:
  ```python
  import torch
  from src.models.lstm_encoder import LSTMEncoderConfig, LSTMEncoder, SimpleWindowDataset, make_trainer

  cfg = LSTMEncoderConfig(input_size=10, hidden_size=64, num_layers=1, dropout=0.1, lr=1e-3)
  model = LSTMEncoder(cfg)
  x = torch.randn(256, 24, 10)  # 256 windows, 24 timesteps, 10 features
  y = torch.randn(256)          # next-step target
  ds = SimpleWindowDataset(x, y)
  trainer = make_trainer(max_epochs=2, gpus=0)
  trainer.fit(model, torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True))
  ```
- For classic PyTorch (no Lightning), adapt `src/models/lstm_model.py` and implement the training loop in `src/training/train_lstm.py`.

## Integration points
- Data -> windows: implement `sequence_generator.py` to produce (X, y) windows consistent with shapes above.
- Configs -> runs: read from `experiments/lstm/*.yaml` and map to model/training params (ensure keys match your chosen class: `LSTMEncoderConfig` or `LSTMForecast`).
- Metrics: add RMSE/MAE/R² to `src/utils/metrics.py` and use them in evaluation/validation logging.

## Gotchas
- Maintain shape discipline (B, T, F). `LSTMEncoder` uses the final hidden state; ensure `batch_first=True` semantics.
- When `aux_predict=True`, `training_step` expects targets `y` shaped `(B,)` and logs MSE loss.
- If using GPU, ensure CUDA availability and pass `gpus>0` into `make_trainer()`; otherwise CPU is used.

---
Questions or gaps to clarify:
- Confirm the intended YAML->code mapping (which model/config should be authoritative?).
- Define the canonical data pipeline for generating (B, T, F) windows from CSV/Parquet in `sequence_generator.py`.
- Decide on experiment logging (e.g., CSV, TensorBoard, Lightning loggers) and integrate into training.
