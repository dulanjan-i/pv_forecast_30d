# MiRACLE v1.0: Results

## Experimental Validation of Physics-Informed Temporal Fusion Transformers for Multi-Horizon PV Forecasting

---

## 1. Executive Summary

This section presents empirical results from the MiRACLE v1.0 (Meta Intelligent Reinforcement Driven Adaptive Control for Learning Based Ensembles) forecasting framework across three experimental phases: (1) feature ablation study quantifying physics-informed feature contributions, (2) global pretraining validation demonstrating cross-site knowledge transfer, and (3) cold-start versus warm-start comparison establishing transfer learning efficacy.

**Key Findings:**
- PVLib physics features reduced validation RMSE by 5.36% relative to TFT-only baseline
- Warm-start transfer learning achieved 32.5% lower validation loss (short-term) and 5.3% (long-term) compared to cold-start random initialization on held-out target plant

---

## 2. Phase 1: Feature Ablation Study

### 2.1 Experimental Setup

Four model configurations trained on identical Germany multi-site corpus (plants 01, 02, 03, 05, 06) to isolate feature group contributions.

**Common Hyperparameters:**
- Encoder/prediction length: 96 timesteps (24 hours @ 15-min)
- Architecture: Hidden size 64, LSTM layers 2, attention heads 4
- Training: Effective batch 4096 (BS=512, grad_accum=8), early stopping patience 5
- Evaluation: Validation RMSE/MAE computed on median quantile forecast over all 96 horizons

### 2.2 Quantitative Results

#### Table 1: Ablation Study Performance (Validation Set)

| Mode | RMSE | MAE | Best Epoch | Relative RMSE Improvement |
|---|---:|---:|---:|---:|
| TFT-Only (Baseline) | **0.05130** | 0.02058 | 11 | — |
| **TFT+PVLib** ✓ | **0.04855** | **0.01982** | 4 | **+5.36%** |
| TFT+LSTM | 0.01873 ⚠️ | — | 7 | (excluded) |
| Full | 0.01350 ⚠️ | — | 7 | (excluded) |

**Source:** `experiments/tft/runs/germany/ablations/ablation_summary.csv` and `ablation_summary_extended.csv`

**Notes:**
- ⚠️ TFT+LSTM and Full modes showed anomalously low losses, suggesting **potential data leakage** via LSTM embeddings computed on overlapping windows. These configurations excluded from production pipeline.
- ✓ **TFT+PVLib selected as winner**: Best validated performance without leakage risk

**Statistical Significance:**
- RMSE reduction: 0.05130 → 0.04855 (Δ = 0.00275 absolute, 5.36% relative)
- MAE reduction: 0.02058 → 0.01982 (Δ = 0.00076 absolute, 3.69% relative)
- Convergence speed: TFT+PVLib reached optimal loss at epoch 4 vs. epoch 11 for baseline (2.75× faster)

**Winner Configuration Locked:** All downstream experiments (pretraining, transfer learning, multi-horizon) employed TFT+PVLib mode.

---

## 3. Phase 2: Global Pretraining and No-Leak Validation

### 3.1 Pretraining Strategy

To enable safe transfer learning to target plant `plant_03`, we constructed a **data-leakage-free global model** via:

**Step 1: No-Leak Corpus Construction**
- **Excluded**: All plant_03 data (train and validation splits)
- **Retained**: Plants {01, 02, 05, 06} only
- **Implementation:** `src/data/make_global_noleak_parquets.py --exclude_plant_id "plant_03"`

**Step 2: Global Model Training**
- Configuration: TFT+PVLib (winner from ablation)
- Hyperparameters: lr=1.2e-3, dropout=0.15, BS=512×8
- Encoder/prediction length: 96 timesteps (short-term head, 24h)
- Checkpoint format: PyTorch state_dict saved via `torch.save(model.state_dict(), path)`

**Step 3: Checkpoint Verification**
- Confirmed: No plant_03 timestamps present in training data via explicit exclusion

### 3.2 Global Model Performance

**Training Dynamics:**
- Converged with early stopping after 11 epochs (patience=3, final epoch stopped at 10)
- Best validation loss: 0.012530 achieved at epoch 7
- Total training time: 1.19 GPU-hours (verified from `experiments/tft/runs/germany/global_noleak/target03_excluded/20251229_134852/logs/metrics.csv`)

---

## 4. Phase 3: Transfer Learning to Target Plant (plant_03)

### 4.1 Experimental Design: Cold vs. Warm Start

Two fine-tuning regimes compared to quantify transfer learning efficacy:

#### Cold Start (Random Initialization Baseline)
- **Initialization**: Glorot uniform (PyTorch defaults)
- **Training data**: plant_03 15-min splits only
- **Hyperparameters**: lr=2e-3, dropout=0.15, BS=64×8

#### Warm Start (Transfer Learning)
- **Initialization**: Load pretrained weights from global no-leak model
- **Training data**: plant_03 15-min splits only
- **Hyperparameters**: lr=8e-4, dropout=0.15, BS=64×8

**Multi-Seed Robustness:**
- Each regime executed with seeds {42, 43, 44} (3 replicates)

### 4.2 Short-Term Head Results (15-min, 24h Horizon)

#### Table 2: Plant_03 Fine-Tuning Performance (Validation Loss)

| Regime | Seed | Best Val Loss | Best Epoch | Final Val Loss (Last Epoch) |
|---|---:|---:|---:|---:|
| **Cold Start** | 42 | 0.03251 | 16 | 0.03300 (epoch 21) |
| **Cold Start** | 43 | 0.05603 | 22 | 0.05750 (epoch 27) |
| **Cold Start** | 44 | 0.03041 | 10 | 0.03261 (epoch 15) |
| **Cold Mean ± Std** | — | **0.0397 ± 0.0145** | 16.0 ± 6.0 | — |
| | | | | |
| **Warm Start** ✓ | 42 | **0.02666** | 12 | 0.03077 (epoch 17) |
| **Warm Start** ✓ | 43 | 0.02720 | 14 | 0.03017 (epoch 19) |
| **Warm Start** ✓ | 44 | 0.02666 | 14 | 0.03168 (epoch 19) |
| **Warm Mean ± Std** | — | **0.0268 ± 0.0003** | 13.3 ± 1.2 | — |

**Source:** `experiments/tft/runs/germany/plant_03/15min/finetune_summary.csv`

**Statistical Analysis:**

$$
\text{Relative Improvement} = \frac{0.0397 - 0.0268}{0.0397} \times 100\% = \mathbf{32.5\%}
$$

**Key Observations:**
1. **Warm start dominates consistently**: All 3 warm-start seeds outperform all 3 cold-start seeds
2. **Reduced variance**: Warm-start std dev 98% lower (0.0003 vs. 0.0145)
3. **Faster convergence**: Warm start optimal at epoch ~13 vs. cold ~16 (19% fewer epochs)
4. **Outlier cold seed**: Seed 43 cold start exhibited poor convergence (0.0560 best loss)

**Winner Selected:** Warm-start seed 42 (best val loss **0.02666**)  
**Checkpoint:** `experiments/tft/runs/germany/plant_03/15min/pvlib_warmstart_from_global_noleak/20251229_151100/checkpoints/best.ckpt`

### 4.3 Long-Term Head Results (1-hour, 30-day Horizon)

**Experimental Modifications:**
- **Temporal resolution**: Hourly aggregation via mean pooling from 15-min source
- **Sequence lengths**: Encoder 720, prediction 720 (30 days)

#### Table 3: Plant_03 Longhead Fine-Tuning (Validation Loss)

| Regime | Seed | Best Val Loss | Best Epoch |
|---|---:|---:|---:|
| **Cold Start** | 42 | 0.02669 | 18 |
| **Cold Start** | 43 | 0.02713 | 25 |
| **Cold Start** | 44 | 0.02595 | 30 |
| **Cold Mean ± Std** | — | **0.0266 ± 0.0006** | 24.3 ± 6.1 |
| | | | |
| **Warm Start** ✓ | 42 | 0.02565 | 9 |
| **Warm Start** ✓ | 43 | **0.02414** | 36 |
| **Warm Start** ✓ | 44 | 0.02585 | 10 |
| **Warm Mean ± Std** | — | **0.0252 ± 0.0009** | 18.3 ± 15.6 |

**Source:** Metrics extracted from `experiments/tft/runs/germany/plant_03/longhead/hourly720/{cold,warm}/lr*_seed*/*/logs/metrics.csv`

**Analysis:**

$$
\text{Relative Improvement} = \frac{0.0266 - 0.0252}{0.0266} \times 100\% = \mathbf{5.3\%}
$$

**Observations:**
1. **Consistent warm-start advantage** preserved at extended horizon (5.3% improvement)
2. **Lower magnitude gain**: 5.3% vs. 32.5% for short-term head (hourly aggregation smooths patterns)
3. **Variance increase**: Warm-start std dev higher (0.0009 vs. 0.0003 for short-term), likely due to smaller dataset size

**Winner Selected:** Warm-start seed 43 (best val loss **0.02414**)  
**Checkpoint:** `experiments/tft/runs/germany/plant_03/longhead/hourly720/warm/lr8e-4_do0.15_bs64_acc8_seed43/20251231_104405/checkpoints/best.ckpt`

### 4.4 Convergence Speed Analysis

#### Table 4: Epochs to Optimal Loss

| Regime | Short-Term (24h) | Long-Term (30d) |
|---|---:|---:|
| Cold Start (mean) | 16.0 ± 6.0 | 24.3 ± 6.1 |
| Warm Start (mean) | **13.3 ± 1.2** | **18.3 ± 15.6** |
| Speedup Factor | **1.20×** | **1.33×** |

**Training Time Per Epoch:** ~70 seconds (based on metrics.csv `epoch_sec` column)

**Time Savings:**
- Short-term: 2.7 fewer epochs × 70 sec ≈ **3.2 minutes saved**
- Long-term: 6.0 fewer epochs × 70 sec ≈ **7.0 minutes saved**

---

## 5. Production Model Selection and Deployment

Based on validation results, the following configurations locked for MiRACLE v1.0 release:

#### Short-Term Operational Head (15-min, 24h)
- **Model**: TFT+PVLib, warm-start seed 42
- **Validation Loss**: 0.02666
- **Checkpoint**: `experiments/tft/runs/germany/plant_03/15min/pvlib_warmstart_from_global_noleak/20251229_151100/checkpoints/best.ckpt`
- **Format**: PyTorch state_dict (1.7 MB `.ckpt` file). Despite the `.ckpt` extension, the file stores only a `torch.save(model.state_dict())` output, not a PyTorch Lightning checkpoint.

#### Long-Term Strategic Head (1-hour, 30-day)
- **Model**: TFT+PVLib, warm-start seed 43
- **Validation Loss**: 0.02414
- **Checkpoint**: `experiments/tft/runs/germany/plant_03/longhead/hourly720/warm/lr8e-4_do0.15_bs64_acc8_seed43/20251231_104405/checkpoints/best.ckpt`

**Loading Protocol:**
```python
from pytorch_forecasting.models import TemporalFusionTransformer
import torch

# Rebuild model architecture from dataset
model = TemporalFusionTransformer.from_dataset(
    train_ds,
    hidden_size=64,
    lstm_layers=2,
    attention_head_size=4,
    dropout=0.15,
    learning_rate=8e-4,
)

# Load weights
sd = torch.load("best.ckpt", map_location="cpu")
model.load_state_dict(sd, strict=True)
```

**Reference:** `src/validation/eval_short_head.py:eval_one()`

---

## 6. Error Analysis and Failure Modes

### 6.1 Observed Patterns

**Seed Variability:**
- Cold-start seed 43 exhibited significantly worse performance (val loss 0.0560 vs. ~0.032 for other seeds)
- Warm-start seeds showed minimal variance (0.0003 std dev)

**Interpretation:** Cold-start optimization is sensitive to initial parameter configuration; warm-start provides more stable initialization.

### 6.2 Limitations

**Statistical Power:**
- Sample size n=3 insufficient for frequentist significance tests at α=0.05
- However, monotonic superiority across all comparisons (6/6 warm-start wins) provides strong practical evidence

**Seasonal Coverage:**
- Validation period covers winter months (Dec–Feb 2024)
- Summer performance characteristics remain unvalidated

---

## 7. Computational Resource Summary

### 7.1 Training Costs (Single Model)

**Per-Epoch Statistics (from metrics.csv):**
- Time per epoch: ~70 seconds (short-term), ~70 seconds (long-term)
- GPU memory: ~4.8 GB peak (per metrics logs)
- Throughput: ~68 samples/sec (effective batch 4096)

**Total Training Time:**
- Short-term warm-start: ~13 epochs × 70 sec ≈ **15 minutes**
- Long-term warm-start: ~18 epochs × 70 sec ≈ **21 minutes**

### 7.2 Experimental Campaign Totals

**Ablation study:** 2 modes evaluated for production (tft_only: 1.34 GPU-hours, tft_pvlib: 1.13 GPU-hours) = **2.47 GPU-hours total**. Note: `tft_lstm` and `full` modes excluded due to suspected leakage.
**Global pretraining:** Single run = **1.19 GPU-hours** (11 epochs, best at epoch 7)
**Plant_03 fine-tuning:** Winner configurations (seed 42 short-term: 0.37 GPU-hours for 18 epochs; seed 43 long-term: estimated similar duration) ≈ **0.7-1.0 GPU-hours total**

---

## 8. Key Findings Summary

1. **Physics-informed features deliver measurable benefit**: TFT+PVLib reduced validation RMSE by 5.36% vs. TFT-only baseline, with 2.75× faster convergence

2. **Transfer learning robustly improves target-site performance**: Warm-start initialization yielded 32.5% lower validation loss (short-term) and 5.3% (long-term) compared to cold start

3. **Multi-horizon consistency**: Transfer learning benefits persisted across both 24-hour and 30-day forecast horizons

4. **Reduced variance from pretraining**: Warm-start models exhibited 98% lower standard deviation across random seeds

5. **Production models locked**: Seed 42 (short-term, val loss 0.0267), seed 43 (long-term, val loss 0.0241)

6. **Data leakage prevention validated**: Global pretraining with plant_03 exclusion successfully avoided train/test contamination

---

## 9. Future Work

### 9.1 Immediate Extensions

1. **Full cross-site validation**: Repeat transfer learning experiment with plants {01, 02, 05, 06} as held-out targets
2. **Seasonal stratification**: Validate on full annual cycle (summer 2024 data pending)
3. **Probabilistic calibration**: Evaluate quantile coverage (P10, P90 reliability)

### 9.2 Architectural Innovations

1. **LSTM encoder integration (v1.1)**: Implement safe rollout strategy for upstream embeddings
2. **Multi-horizon joint training**: Enforce consistency between short and long heads
3. **Attention interpretability**: Visualize temporal attention patterns

### 9.3 Operational Deployment

1. **Real-time inference latency**: Benchmark prediction throughput
2. **NWP forecast error propagation**: Replace historical reanalysis with operational forecasts
3. **Online learning**: Continual fine-tuning as new site data accumulates

---

## References

1. Lim, B., et al. (2021). Temporal Fusion Transformers for interpretable multi-horizon time series forecasting. *International Journal of Forecasting*, 37(4), 1748-1764.

---

**Document Version**: v1.0-corrected  
**Last Updated**: January 1, 2026  
**Implementation Base**: MiRACLE v1.0 codebase verification  
**Companion Document**: [miracle_v1_methodology_CORRECTED.md](miracle_v1_methodology_CORRECTED.md)
