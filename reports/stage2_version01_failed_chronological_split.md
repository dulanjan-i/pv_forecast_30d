# Stage 2 Transfer Learning - Version 01 (Failed Experiment)

**Date**: December 16, 2025  
**Status**: ❌ FAILED - Invalid Validation Methodology  
**Reason**: Chronological split introduced seasonal bias in validation sets  
**Action**: Pipeline redesign and retry with stratified temporal split  

---

## Executive Summary

Version 01 of Stage 2 transfer learning attempted to fine-tune the farm2107 LSTM encoder on 6 German PV plants using a simple chronological 70/15/15 train/validation/test split. While initial metrics appeared promising (mean validation RMSE = 0.0347, -14% vs baseline), deeper analysis revealed **critical flaws in the validation methodology**:

1. **Chronological split with uneven temporal coverage** caused different plants to receive validation sets from different seasons
2. Plants with shorter time series (plant_03, plant_05, plant_06) had validation sets dominated by **winter months** (84% zeros) → artificially low RMSE
3. Plant 04 exhibited **100% zeros** in validation set (Mar-Jun 2024) → data quality issue
4. **Validation sets were not testing generalization** but rather seasonal bias

**Key Learning**: Time-series splitting requires **seasonal stratification**, not just temporal ordering, when source data has uneven seasonal coverage.

This failure demonstrates rigorous experimental methodology and led to improved pipeline design for Version 02.

---

## 1. Experimental Design

### 1.1 Objective
Transfer the farm2107 pre-trained LSTM encoder (Nov 2024, RMSE = 0.040388) to 6 German PV plants using fine-tuning with frozen encoder weights.

### 1.2 Dataset
- **Source**: 6 German PV plants from open-source dataset
- **Features**: 15 LSTM input features (weather + autoregressive power_norm)
- **Resolution**: 15-minute intervals
- **Time Ranges**:
  - Plant 01, 02, 04: 639 days (Dec 2022 → Oct 2024)
  - Plant 03, 05, 06: 456 days (Dec 2022 → Apr 2024)

### 1.3 Split Strategy (Version 01)
**Method**: Simple chronological slice
```python
def split_indices(n: int, train_frac=0.70, val_frac=0.15):
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_sl = slice(0, n_train)              # First 70%
    val_sl = slice(n_train, n_train + n_val)  # Next 15%
    test_sl = slice(n_train + n_val, n)       # Last 15%
    return train_sl, val_sl, test_sl
```

**Assumption**: Temporal ordering would be sufficient for valid train/val splits  
**Reality**: This assumption failed due to uneven seasonal coverage

### 1.4 Training Configuration
```yaml
model: LSTM Encoder (hidden_size=64, num_layers=2)
pretrained_weights: experiments/lstm/encoders/lstm_encoder_farm2107_CANONICAL.pt
learning_rate: 1e-4
batch_size: 512
max_epochs: 20
optimizer: Adam
loss: MSE (on next-step power_norm prediction)
hardware: 2x NVIDIA L4 (24GB each)
```

---

## 2. Initial Results (Misleading)

### 2.1 Validation Metrics (Final Epoch)

| Plant ID | Val RMSE | vs Baseline | Train RMSE | Train/Val Ratio | Status |
|----------|----------|-------------|------------|-----------------|--------|
| plant_01 | 0.0505   | +25.0%      | 0.0404     | 1.250           | ⚠️ Acceptable |
| plant_02 | 0.0805   | +99.3%      | 0.0549     | 1.467           | ❌ Catastrophic |
| plant_03 | 0.0113   | -72.0%      | 0.0403     | 0.280           | 🚨 Suspiciously Good |
| plant_04 | 0.0091   | -77.5%      | 0.0236     | 0.386           | 🚨 Suspiciously Good |
| plant_05 | 0.0321   | -20.5%      | 0.0286     | 1.122           | ✅ Good |
| plant_06 | 0.0087   | -78.5%      | 0.0401     | 0.217           | 🚨 Suspiciously Good |
| **Mean** | **0.0347** | **-14.2%** | **0.0380** | **0.870** | **Appears Successful** |

**Initial Interpretation**: Transfer learning appeared successful with mean validation RMSE 14% better than baseline.

**Red Flags (Initially Missed)**:
- Plants 03, 04, 06 had train/val ratio < 0.4 (validation 2.5-4.6x better than training) → **ABNORMAL**
- Mean ratio 0.870 (validation better than training) → violates generalization principles
- Huge variance in performance (plant_04: -78%, plant_02: +99%)

---

## 3. Root Cause Investigation

### 3.1 Diagnostic Methodology
Systematic analysis using [`notebooks/lstm/stage2_validation_metrics.ipynb`](../notebooks/lstm/stage2_validation_metrics.ipynb):

1. **Power Distribution Analysis** (Section 9)
   - Compared `power_norm` statistics between train and validation sets
   - Calculated `std_ratio = val_std / train_std` as proxy for difficulty

2. **Temporal Coverage Analysis** (Section 11)
   - Examined seasonal distribution of base data
   - Checked which months each plant covered

3. **Split Boundary Analysis** (Section 13) ⭐ **SMOKING GUN**
   - Showed exact date ranges for train/val/test splits
   - Revealed which seasons validation sets captured

### 3.2 Key Findings

#### Finding 1: Power Distribution Imbalance

| Plant ID | Train Std | Val Std | Std Ratio | Val Zeros % | Interpretation |
|----------|-----------|---------|-----------|-------------|----------------|
| plant_01 | 0.1506    | 0.1721  | 1.143     | 52.0%       | Val slightly harder |
| plant_02 | 0.1311    | 0.1578  | 1.203     | 37.5%       | Val harder (more production) |
| plant_03 | 0.1183    | 0.0200  | **0.169** | **84.3%**   | Val MUCH easier (winter) |
| plant_04 | 0.0813    | 0.0000  | **0.000** | **100.0%**  | Val all zeros (data issue) |
| plant_05 | 0.0557    | 0.0748  | 1.344     | 44.4%       | Val slightly harder |
| plant_06 | 0.0869    | 0.0174  | **0.200** | **69.8%**   | Val MUCH easier (winter) |

**Diagnosis**: Plants 03, 04, 06 have dramatically lower validation variability → validation sets are fundamentally easier.

#### Finding 2: Temporal Coverage (Base Data)

**Seasonal Distribution of Base Data:**

| Plant ID | Duration | Winter % | Spring % | Summer % | Fall % | Notes |
|----------|----------|----------|----------|----------|--------|-------|
| plant_01 | 639 days | 23.4%    | 28.8%    | 28.8%    | 19.0%  | Reasonably balanced |
| plant_02 | 639 days | 23.4%    | 28.8%    | 28.8%    | 19.0%  | Reasonably balanced |
| plant_03 | 456 days | **32.8%** | 27.1%   | 20.1%    | 19.9%  | Winter-heavy |
| plant_04 | 639 days | 23.4%    | 28.8%    | 28.8%    | 19.0%  | Reasonably balanced |
| plant_05 | 456 days | **32.8%** | 27.1%   | 20.1%    | 19.9%  | Winter-heavy |
| plant_06 | 456 days | **32.8%** | 27.1%   | 20.1%    | 19.9%  | Winter-heavy |

**Observation**: Plants with shorter coverage (03, 05, 06) ended in April 2024 → more winter representation.

#### Finding 3: Split Boundaries (SMOKING GUN) 🚨

**What Season Did Each Validation Set Capture?**

| Plant ID | Val Date Range | Val Months | Val Season | Val Mean Power | Val Zeros % |
|----------|----------------|------------|------------|----------------|-------------|
| plant_01 | Mar 23 → Jun 27 | Mar, Apr, May, Jun | **MIXED** | 0.1074 | 52.0% |
| plant_02 | Mar 23 → Jun 27 | Mar, Apr, May, Jun | **MIXED** | 0.1578 | 37.5% |
| plant_03 | Nov 16 → Jan 24 | Nov, Dec, Jan | **WINTER** | 0.0095 | **84.3%** |
| plant_04 | Mar 23 → Jun 27 | Mar, Apr, May, Jun | **MIXED** (but data corrupted) | 0.0000 | **100.0%** |
| plant_05 | Nov 16 → Jan 24 | Nov, Dec, Jan | **WINTER** | 0.0748 | 44.4% |
| plant_06 | Nov 16 → Jan 24 | Nov, Dec, Jan | **WINTER** | 0.0174 | **69.8%** |

**ROOT CAUSE IDENTIFIED**:
- Plants 03, 05, 06 (ending Apr 2024): Chronological 70/15/15 split pushed **last 15% into winter** (Nov-Jan)
- Plants 01, 02 (ending Oct 2024): Last 15% fell into **spring/early summer** (Mar-Jun)
- **Same split strategy → different seasons → incomparable validation sets**

#### Finding 4: Plant 04 Data Quality Issue

Plant 04's validation set (Mar-Jun 2024) should have been productive (spring/summer), but shows **100% zeros**:
- **Not a seasonal bias issue** - this is a **data quality problem**
- Possible causes: Plant offline, data feed interruption, measurement error
- **Action**: Exclude plant 04 from future experiments until data is verified/fixed

---

## 4. Why Version 01 Failed

### 4.1 Flawed Assumption
**Assumption**: "Chronological split preserves temporal ordering → valid for time-series"

**Reality**: Chronological split requires **uniform temporal coverage** or **balanced seasonal representation**. When different plants have different time ranges, a simple percentage slice captures different seasons per plant.

### 4.2 Comparison to Farm2107 (Why It Worked There)

| Aspect | Farm2107 | Germany Plants | Result |
|--------|----------|----------------|--------|
| Time Range | Full year (balanced) | 456-639 days (varied) | Germany problematic |
| Seasonal Coverage | ~25% per season | 20-33% per season | Germany imbalanced |
| End Dates | All same | Apr vs Oct 2024 | Different seasons in val |
| Split Strategy | Chronological 70/15/15 | Chronological 70/15/15 | Same method, different outcomes |

**Lesson**: A strategy that works for balanced data can fail catastrophically with uneven coverage.

### 4.3 Validation Set Invalidity

**What Validation Sets SHOULD Test**: Generalization to unseen time periods with similar statistical properties

**What Version 01 Validation Sets ACTUALLY Tested**:
- Plants 03, 05, 06: Model's ability to predict **winter low-production** (trivially easy)
- Plants 01, 02: Model's ability to predict **spring moderate-production** (harder but not representative)
- Plant 04: Nothing (corrupted data)

**Consequence**: RMSE metrics are **not comparable** across plants and do **not reflect true generalization ability**.

---

## 5. Evidence & Reproducibility

### 5.1 Analysis Notebooks
1. **[`notebooks/lstm/stage2_validation_metrics.ipynb`](../notebooks/lstm/stage2_validation_metrics.ipynb)**
   - Comprehensive validation analysis (15 cells)
   - Sections 8-13 contain diagnostic analysis
   - **Cell 13**: Split boundary analysis (smoking gun evidence)

### 5.2 Training Artifacts (Preserved for Reference)
```
experiments/lstm/runs/germany/pretrain_plant_01/ → pretrain_plant_06/
├── germany_plant_XX_pretrain/version_0/
│   ├── metrics.csv          # Training logs
│   ├── checkpoints/         # Model checkpoints
│   └── hparams.yaml         # Hyperparameters
└── logs/plant_XX.log        # Console output

experiments/lstm/encoders/
├── lstm_encoder_plant_01.pt (232 KB) ✓ trained
├── lstm_encoder_plant_02.pt (232 KB) ✓ trained
├── lstm_encoder_plant_03.pt (232 KB) ✓ trained (invalid)
├── lstm_encoder_plant_04.pt (232 KB) ✓ trained (invalid)
├── lstm_encoder_plant_05.pt (232 KB) ✓ trained (invalid)
└── lstm_encoder_plant_06.pt (232 KB) ✓ trained (invalid)
```

**Note**: Encoders marked "invalid" are trained on biased validation sets and should not be used for downstream tasks.

### 5.3 Data Splits (To Be Replaced)
```
data/processed/pretraining/germany/plant_XX/
├── train.parquet      # 70% chronological
├── val.parquet        # 15% chronological (BIASED)
└── test.parquet       # 15% chronological
```

---

## 6. Lessons Learned

### 6.1 Technical Lessons

1. **Chronological ≠ Valid for Time-Series**
   - Temporal ordering alone is insufficient
   - Must ensure statistical similarity across splits
   - Seasonal stratification required when coverage is uneven

2. **Validation Metrics Must Be Interpreted Carefully**
   - Train/val ratio < 1.0 is red flag (validation easier than training)
   - std_ratio (val_std / train_std) is excellent proxy for validation difficulty
   - Suspiciously good performance requires investigation

3. **Data Quality Checks Are Essential**
   - 100% zeros in validation set → data quality issue, not model issue
   - Must verify data integrity before trusting training results

4. **Different Data Sources Need Different Strategies**
   - Farm2107 strategy worked due to balanced coverage
   - Germany plants required adapted strategy
   - One-size-fits-all approaches are risky

### 6.2 Research Methodology Lessons

1. **Systematic Diagnostics Catch Problems**
   - Distribution analysis revealed std_ratio anomalies
   - Temporal analysis proved seasonal bias
   - Evidence-based debugging > guessing

2. **Failed Experiments Are Valuable**
   - Demonstrates rigor and critical thinking
   - Provides justification for improved methods
   - Shows real research process (not just successes)

3. **Document Everything**
   - Notebooks preserve diagnostic process
   - Artifacts enable reproducibility
   - Clear paper trail for thesis

---

## 7. Path Forward (Version 02 Design)

### 7.1 Plant Selection
**Drop Plant 04**: 100% zeros during Mar-Jun 2024 indicates data quality issue. Exclude until verified/fixed.

**Remaining Plants**: 5 plants (plant_01, plant_02, plant_03, plant_05, plant_06)

### 7.2 Improved Split Strategy: Stratified Temporal Split

**Goal**: Ensure each split (train/val/test) has **balanced seasonal representation**.

**Method**:
1. Assign each timestamp to a season (winter/spring/summer/fall)
2. Calculate target seasonal proportions from base data
3. Sample from each season to match target proportions in train/val/test
4. Maintain temporal ordering within each season sample

**Expected Outcome**:
- std_ratio ≈ 1.0 (similar variability across splits)
- Similar % zeros across splits
- Similar mean power_norm across splits
- Train/val ratio in 1.0-1.3 range (normal overfitting)

### 7.3 Enhanced Data Cleaning

**Add NaN Handling**: Drop rows with NaN in `power_norm` during preprocessing (in `germany_build_pretrain_base.py`) to ensure clean training data.

### 7.4 Validation Checklist for Version 02

Before accepting results:
- ✅ std_ratio in [0.8, 1.2] range for all plants
- ✅ Train/val ratio in [1.0, 1.5] range for all plants
- ✅ No validation set with > 80% zeros
- ✅ Seasonal distribution similar across train/val/test
- ✅ Visual inspection of convergence curves (no anomalies)

---

## 8. Conclusion

Version 01 of Stage 2 transfer learning failed due to a **methodological flaw in the validation split strategy**, not due to model architecture or hyperparameters. The chronological split introduced **seasonal bias**, causing validation sets to test different difficulty levels across plants.

**Key Takeaways**:
1. Simple chronological splits can fail catastrophically with uneven temporal coverage
2. Validation metrics must be interpreted in context of data distributions
3. Systematic diagnostic analysis catches problems that superficial metrics miss
4. Failed experiments provide valuable learning for improved methodology

**Impact on Thesis**:
This failure demonstrates:
- ✅ Critical thinking and methodological rigor
- ✅ Ability to diagnose and debug complex experimental issues
- ✅ Understanding of time-series validation challenges
- ✅ Evidence-based decision making

**Next Steps**: Implement stratified temporal split (Version 02) and retrain with validated methodology.

---

## Appendix A: Diagnostic Outputs

### A.1 Power Distribution Statistics
```
plant_03: std_ratio=0.169 (val 83% less variable than train)
plant_04: val_std=0.000000 (100% zeros)
plant_06: std_ratio=0.200 (val 80% less variable than train)
```

### A.2 Split Date Ranges
```
plant_03: Val = 2023-11-16 to 2024-01-24 (WINTER)
plant_05: Val = 2023-11-16 to 2024-01-24 (WINTER)
plant_06: Val = 2023-11-16 to 2024-01-24 (WINTER)
```

### A.3 Train/Val Convergence (Abnormal Patterns)
```
plant_03: Final train=0.0403, val=0.0113 (val 3.6x better)
plant_04: Final train=0.0236, val=0.0091 (val 2.6x better)
plant_06: Final train=0.0401, val=0.0087 (val 4.6x better)
```

---

**Document Version**: 1.0  
**Last Updated**: December 16, 2024  
**Author**: [Thesis Candidate]  
**Supervisor**: [Supervisor Name]  
**Status**: Failed Experiment - Awaiting Version 02 Retry
