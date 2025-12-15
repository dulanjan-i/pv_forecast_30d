# LSTM Experiment Results

This document tracks results from LSTM pretraining experiments on PVDAQ Data [Farm2107] for PV power forecasting.

---

## Farm 2107 Hyperparameter Sweep (2024-11-24)

**Objective:** Establish baseline LSTM encoder configuration for thesis comparison.

**Methodology:**
- Grid search over 12 configurations:
  - `hidden_size`: [32, 64, 128]
  - `num_layers`: [1, 2]
  - `learning_rate`: [5e-4, 1e-3]
- Fixed parameters:
  - `batch_size`: 256
  - `max_epochs`: 20
  - `dropout`: 0.1
  - `window_size`: 96 (24 hours @ 15-min intervals)
  - `horizon`: 1 (next-step prediction)
- Parallel execution on 4× NVIDIA L4 GPUs (3 hours total runtime)

**Results Summary:**

| Rank | Configuration | Hidden Size | Layers | Learning Rate | Best Val RMSE | Final Val RMSE |
|------|---------------|-------------|--------|---------------|---------------|----------------|
| 🥇 1 | h64_l2_lr1em03 | 64 | 2 | 0.0010 | **0.040388** | 0.040463 |
| 🥈 2 | h128_l2_lr5em04 | 128 | 2 | 0.0005 | 0.040498 | 0.040948 |
| 🥉 3 | h128_l1_lr5em04 | 128 | 1 | 0.0005 | 0.040534 | 0.040709 |
| 4 | h64_l1_lr1em03 | 64 | 1 | 0.0010 | 0.040547 | 0.040825 |
| 5 | h32_l1_lr1em03 | 32 | 1 | 0.0010 | 0.040567 | 0.040567 |

**Full results:** `experiments/lstm/pretrain_hparam_results.csv`

**Key Findings:**
- All configurations performed within ~2% of each other (RMSE range: 0.0404-0.0412)
- Robust architecture: no catastrophic failures across parameter space
- Best config: **h64_l2_lr1em03** → RMSE = **0.040388** (normalized power units)
- Moderate model size (64 hidden units, 2 layers) outperformed both smaller and larger variants
- Higher learning rate (1e-3) slightly preferred over 5e-4

**Canonical Configuration:**
- File: `experiments/lstm/pretrain_farm2107_CANONICAL.yaml`
- Weights: `experiments/lstm/encoders/lstm_encoder_farm2107_CANONICAL.pt`
- Use this as baseline for all downstream tasks and thesis comparisons

---

## Previous Experiments

- **exp01**: Initial baseline LSTM (config in experiments/lstm/exp01.yaml)
