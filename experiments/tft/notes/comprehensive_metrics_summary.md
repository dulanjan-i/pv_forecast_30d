# MiRACLE v1.0: Comprehensive TFT Ablation and Hparam tuning Metrics Summary

**Purpose**: Defensible documentation combining:
- **Val Loss (QuantileLoss)**: Training selection criterion (early stopping)
- **RMSE/MAE**: Post-hoc interpretable metrics on validation set (median quantile)

---

## Phase A: Feature Ablation Study (Short-head, 15-min, 24h)

**Purpose**: Quantify PVLib physics contribution vs. TFT-only baseline

| Mode | Val Loss (selection) | RMSE | MAE | Best Epoch |
|---|---:|---:|---:|---:|
| tft_only | 0.012569 | 0.051300 | 0.020576 | 11 |
| tft_pvlib | 0.012741 | 0.048549 | 0.019817 | 4 |

**Winner**: TFT+PVLib (5.36% RMSE improvement vs. baseline)

---

## Phase B: Global Pretraining (Multi-site, no-leak)

**Purpose**: Learn cross-site patterns for transfer learning initialization

- **Best Val Loss**: 0.012530
- **Best Epoch**: 7
- **Training Data**: Plants {01, 02, 05, 06} (plant_03 excluded for no-leak validation)
- **Note**: RMSE not computed (multi-site aggregate; per-plant eval in Phase 3)

---

## Phase C: Plant_03 Fine-tuning (Short-head, 15-min, 24h)

**Purpose**: Validate transfer learning (warm) vs. cold-start

| Regime | Seed | Val Loss (selection) | RMSE | MAE |
|---|---:|---:|---:|---:|
| warm | seed42 | 0.026656 | 0.103378 | 0.050792 |
| cold | seed42 | 0.032514 | 0.114600 | 0.059051 |
| cold | seed43 | nan | 0.109580 | 0.057252 |
| **Warm Mean** | — | **0.026793** | **0.103378** | — |
| **Cold Mean** | — | **0.037868** | **0.112090** | — |

**Transfer Learning Benefit**:
- Val Loss: **29.2%** improvement (warm vs. cold)
- RMSE: **7.8%** improvement

---

## Phase D: Plant_03 Fine-tuning (Long-head, 1-hour, 30 days)

**Purpose**: Validate transfer learning at extended forecast horizon

| Regime | Seed | Val Loss (selection) | RMSE | MAE |
|---|---:|---:|---:|---:|
| warm | seed42 | 0.025645 | 0.091869 | 0.045621 |
| warm | seed43 | 0.024137 | 0.091848 | 0.044661 |
| warm | seed44 | 0.025846 | 0.095439 | 0.046181 |
| cold | seed42 | 0.026689 | 0.093822 | 0.048502 |
| cold | seed43 | 0.027129 | 0.096003 | 0.049277 |
| cold | seed44 | 0.025948 | 0.095261 | 0.046803 |
| **Warm Mean** | — | **0.025209** | **0.093052** | — |
| **Cold Mean** | — | **0.026589** | **0.095029** | — |

**Transfer Learning Benefit**:
- Val Loss: **5.2%** improvement (warm vs. cold)
- RMSE: **2.1%** improvement

---

## Key Findings

1. **Val Loss (QuantileLoss)**: Selection criterion used throughout training (early stopping)
2. **RMSE/MAE**: Post-hoc interpretable metrics computed on validation set using median quantile
3. **Consistency**: Transfer learning benefits observed in both selection criterion (val loss) and interpretable metrics (RMSE)
4. **Multi-horizon validation**: Benefits persist across short-term (24h) and long-term (30-day) horizons

---

**Generated**: 2026-01-02 10:05:22
