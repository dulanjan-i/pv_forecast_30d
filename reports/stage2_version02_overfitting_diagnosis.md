# Stage 2 Transfer Learning - Version 02: Overfitting Diagnosis

**Date**: December 16, 2025  
**Status**: Stratified Split SUCCESS, but Severe Overfitting Revealed  
**Next Version**: Version 2.1 (Hyperparameter Tuning)

---

## Executive Summary

Version 02 **successfully resolved** the seasonal bias problem identified in Version 01 by implementing stratified temporal splitting. However, this fix revealed the **true underlying issue**: severe overfitting with train/val ratios of 2.0-2.5 (expected: 1.0-1.3). The balanced validation sets now provide trustworthy metrics, showing that the transfer learning approach from farm2107 to Germany plants requires significant regularization improvements.

**Key Achievement**: Fixed validation bias (std_ratio now 0.99-1.01, perfectly balanced)  
**Key Problem**: Severe overfitting revealed (train/val ratio 2.0-2.5)  
**Thesis Value**: Complete research narrative from bias detection → bias correction → true challenge identification

---

## Version 02 Design Changes

### 1. Stratified Temporal Split Implementation
Replaced simple chronological split with season-aware stratification:

```python
def stratified_temporal_split(df, time_col, train_frac=0.70, val_frac=0.15, random_seed=42):
    """
    Split data ensuring balanced seasonal representation across train/val/test.
    Each season (winter/spring/summer/fall) contributes proportionally to each split.
    """
    months = pd.to_datetime(df[time_col]).dt.month
    seasons = classify_by_season(months)  # winter=Dec-Feb, spring=Mar-May, etc.
    
    for season in ['winter', 'spring', 'summer', 'fall']:
        season_indices = np.where(seasons == season)[0]
        np.random.shuffle(season_indices)  # random_seed=42 for reproducibility
        # Split proportionally: 70% train, 15% val, 15% test
        ...
    
    return sorted(train_idx), sorted(val_idx), sorted(test_idx)  # Maintain temporal order
```

**Result**: All splits have identical seasonal distributions per plant
- Example (plant_01): All splits have Winter=23.4%, Spring=28.8%, Summer=28.8%, Fall=19.0%
- Example (plant_03): All splits have Winter=24.5%, Spring=25.3%, Summer=25.3%, Fall=25.0%

### 2. Plant Exclusion
- **Excluded plant_04**: 100% zeros in Mar-Jun 2024 (data quality issue, not seasonal)
- **Training on 5 plants**: plant_01, 02, 03, 05, 06

### 3. NaN Dropping
Added explicit NaN cleaning in preprocessing:
- plant_01: 0.0% dropped (2 rows)
- plant_02: 0.0% dropped (6 rows)
- plant_03: 20.3% dropped (8,920 rows)
- plant_05: 39.8% dropped (17,480 rows)
- plant_06: 0.0% dropped (0 rows)

---

## Version 02 Results

### Final Validation Metrics

| Plant    | Train RMSE | Val RMSE | Train/Val Ratio | Val vs farm2107 |
|----------|------------|----------|-----------------|-----------------|
| plant_01 | 0.0507     | 0.1180   | 2.328 ⚠️        | +192% ↑         |
| plant_02 | 0.0673     | 0.1347   | 2.002 ⚠️        | +234% ↑         |
| plant_03 | 0.0658     | 0.1536   | 2.334 ⚠️        | +280% ↑         |
| plant_05 | 0.0380     | 0.0931   | 2.451 ⚠️        | +131% ↑         |
| plant_06 | 0.0685     | 0.1489   | 2.173 ⚠️        | +269% ↑         |

**Mean Train/Val Ratio**: 2.258 (Expected: 1.0-1.3, Acceptable: <1.5)  
**Assessment**: ❌ SEVERE OVERFITTING (0/5 plants in normal range)

### Power Distribution Analysis (Validation Trustworthiness)

| Plant    | std_ratio | train_zeros % | val_zeros % | Assessment |
|----------|-----------|---------------|-------------|------------|
| plant_01 | 1.011     | 51.3%         | 51.3%       | ✅ BALANCED |
| plant_02 | 0.999     | 47.6%         | 47.8%       | ✅ BALANCED |
| plant_03 | 0.989     | 53.4%         | 52.9%       | ✅ BALANCED |
| plant_05 | 0.992     | 50.4%         | 50.1%       | ✅ BALANCED |
| plant_06 | 1.007     | 51.8%         | 51.3%       | ✅ BALANCED |

**Assessment**: ✅ **5/5 plants have balanced std_ratios (0.9-1.1)**  
**Conclusion**: Stratified split successfully eliminated seasonal bias

---

## Version 01 vs Version 02 Comparison

### What Changed?

| Metric | Version 01 (Chronological) | Version 02 (Stratified) | Improvement |
|--------|----------------------------|-------------------------|-------------|
| **std_ratio range** | 0.169-1.344 (EXTREME) | 0.989-1.011 (BALANCED) | ✅ FIXED |
| **val_zeros range** | 37-100% (BIASED) | 48-53% (CONSISTENT) | ✅ FIXED |
| **Suspiciously good plants** | 2 plants (ratio <0.5) | 0 plants | ✅ FIXED |
| **Train/val ratio issues** | 3/5 abnormal | 5/5 abnormal | ❌ WORSE |

### Train/Val Ratio Evolution

| Plant    | V01 Ratio | V02 Ratio | Change  | Interpretation |
|----------|-----------|-----------|---------|----------------|
| plant_01 | 1.250     | 2.328     | +1.078  | Was acceptable, now overfits |
| plant_02 | 1.467     | 2.002     | +0.535  | Still overfits (slightly better) |
| plant_03 | 0.280 🚨  | 2.334     | +2.054  | Was artificially good, now reveals true overfitting |
| plant_05 | 1.122     | 2.451     | +1.329  | Was acceptable, now overfits |
| plant_06 | 0.217 🚨  | 2.173     | +1.956  | Was artificially good, now reveals true overfitting |

**Key Insight**: Plants 03 & 06 appeared "good" in Version 01 (ratios 0.28, 0.22) because their validation sets were winter-only (84% zeros). This was **measurement error**, not true generalization. Version 02's higher ratios reflect **reality**.

---

## Root Cause Analysis: Why Severe Overfitting?

### 1. Transfer Learning Mismatch Hypothesis
- **farm2107 baseline**: Single site, consistent environmental patterns, full-year coverage
- **Germany plants**: 5 different sites, diverse characteristics, uneven temporal coverage
- **Mismatch**: Pretrained weights may be TOO specific to farm2107's patterns
- **Evidence**: All plants show similar overfitting degree (2.0-2.5), suggesting systematic issue

### 2. Insufficient Regularization Hypothesis
Current training configuration:
```yaml
hidden_size: 64
num_layers: 2
dropout: 0.1  # ← TOO LOW for transfer learning?
lr: 1e-4
batch_size: 512
max_epochs: 20
weight_decay: (not specified, likely 0)
early_stopping: (not implemented)
```

**Problem**: dropout=0.1 is minimal. Transfer learning often requires higher regularization (0.3-0.5).

### 3. Dataset Size Reduction Hypothesis
After NaN dropping:
- plant_03: 34,945 samples (from 43,865) → 20% loss
- plant_05: 26,385 samples (from 43,927) → 40% loss

**Impact**: Smaller datasets are easier to memorize, harder to generalize from.

### 4. Germany Data Complexity Hypothesis
Power standard deviations vary significantly:
- plant_05: 0.127 (low variability)
- plant_03: 0.223 (high variability, 75% higher)

**Impact**: Higher variability = more complex patterns = harder to learn = more overfitting without strong regularization.

---

## Why Version 02 is Actually a Success

### Scientific Success ✅
1. **Fixed the bias**: std_ratio proves seasonal balance achieved
2. **Trustworthy metrics**: Validation now tests generalization, not seasonal luck
3. **Identified real problem**: Overfitting, not validation methodology
4. **Reproducible**: Stratified split with random_seed=42

### Thesis Value ✅
Complete research narrative demonstrating:
- **Problem identification** (Version 01: seasonal bias)
- **Root cause diagnosis** (smoking gun: split boundary analysis)
- **Solution design** (stratified temporal split)
- **Solution validation** (std_ratio 0.99-1.01)
- **Secondary problem discovery** (severe overfitting revealed)
- **Critical thinking** (distinguishing measurement error from true performance)

### Performance Failure ❌
- Train/val ratios 2.0-2.5 (unacceptable for deployment)
- RMSE 2-3x worse than farm2107 baseline
- Cannot proceed to Stage 2B (TFT ensemble) with these encoders

---

## Diagnostic Evidence

### 1. Convergence Curves
(See notebook: `stage2_version02_validation_metrics.ipynb`, Section 5)
- Training loss decreases smoothly to very low values (0.001-0.005 MSE)
- Validation loss plateaus at much higher values (0.009-0.024 MSE)
- Gap widens consistently, indicating memorization

### 2. Power Distribution Confirmation
```
plant_01: train_mean=0.094, val_mean=0.096, diff=1.5% ✓
plant_02: train_mean=0.115, val_mean=0.115, diff=0.2% ✓
plant_03: train_mean=0.124, val_mean=0.122, diff=2.1% ✓
plant_05: train_mean=0.071, val_mean=0.073, diff=1.7% ✓
plant_06: train_mean=0.113, val_mean=0.116, diff=3.0% ✓
```
All mean differences <3% → splits are statistically similar → overfitting is NOT due to distribution shift.

### 3. Comparison to Baseline
All plants perform 130-280% worse than farm2107 baseline (RMSE 0.040), suggesting:
- Transfer learning is NOT effectively transferring knowledge, OR
- Germany plants are fundamentally harder to predict, OR
- Model capacity is insufficient, OR
- Regularization is inadequate

---

## Path Forward: Version 2.1 Strategy

### Decision Tree

```
Version 2.1: Hyperparameter Tuning (FIRST ATTEMPT)
├─ Keep: Stratified splits ✓
├─ Keep: 5 plants (no plant_04) ✓
├─ Keep: NaN dropping ✓
└─ Change: Training hyperparameters
   ├─ Lower LR: 1e-4 → 5e-5 or 1e-5
   ├─ Adjust batch_size: Try 256 or 1024
   ├─ Increase dropout: 0.1 → 0.3 or 0.5
   ├─ Add weight_decay: 1e-4 or 1e-5
   └─ Implement early_stopping: patience=3-5

Version 2.2: Train from Scratch (IF 2.1 FAILS)
├─ No transfer learning (initialize randomly)
├─ Same stratified splits
└─ Same hyperparameter improvements from 2.1

Version 2.3: Architecture Changes (IF 2.2 FAILS)
├─ Reduce model size (hidden_size 64→32)
├─ Add batch normalization
└─ Try different architecture (Transformer, TFT)
```

### Specific Recommendations for Version 2.1

#### Option A: Conservative Tuning (RECOMMENDED FIRST)
```yaml
# Increase regularization, reduce learning rate
dropout: 0.3         # 0.1 → 0.3 (moderate increase)
lr: 5e-5             # 1e-4 → 5e-5 (half learning rate)
weight_decay: 1e-4   # Add L2 regularization
batch_size: 512      # Keep same
early_stopping:
  monitor: val_loss
  patience: 5
  mode: min
```

**Rationale**: Gentle changes to see if overfitting is due to insufficient regularization.

#### Option B: Aggressive Tuning
```yaml
# Strong regularization for transfer learning
dropout: 0.5         # 0.1 → 0.5 (strong dropout)
lr: 1e-5             # 1e-4 → 1e-5 (very slow learning)
weight_decay: 1e-4
batch_size: 256      # 512 → 256 (smaller batches, more updates)
early_stopping:
  monitor: val_loss
  patience: 3
  mode: min
```

**Rationale**: If overfitting is severe, need strong regularization to prevent memorization.

#### Option C: Learning Rate Schedule
```yaml
# Progressive learning rate reduction
dropout: 0.3
lr: 1e-4             # Start same
lr_scheduler:
  type: ReduceLROnPlateau
  factor: 0.5
  patience: 3
  min_lr: 1e-6
batch_size: 512
early_stopping:
  monitor: val_loss
  patience: 5
```

**Rationale**: Allow fast initial learning, then reduce LR when validation plateaus.

### Additional Diagnostics to Add

1. **Gradient Clipping**: Prevent exploding gradients
   ```python
   max_grad_norm: 1.0
   ```

2. **Monitor Gradient Norms**: Track if gradients are vanishing/exploding
   ```python
   log_grad_norm: True
   ```

3. **Validation Frequency**: Check overfitting earlier
   ```python
   val_check_interval: 0.25  # Validate 4x per epoch
   ```

4. **Learning Curve Analysis**: Plot training curves every N steps
   ```python
   log_every_n_steps: 50
   ```

---

## Blocking Issues for Stage 2B

### Why We Cannot Proceed to TFT Ensemble

1. **Encoders are unreliable**: train/val gap of 2x means embeddings capture training noise, not generalizable patterns
2. **TFT requires quality features**: Garbage in (overfitted embeddings) → garbage out (poor ensemble)
3. **Computational waste**: Training TFT on bad encoders wastes GPU time
4. **Thesis integrity**: Cannot claim "Stage 2 complete" with failed validation

### Success Criteria for Stage 2B Readiness

- ✅ Stratified splits (already achieved)
- ❌ Train/val ratio < 1.5 (currently 2.0-2.5)
- ❌ Val RMSE < 0.08 (currently 0.09-0.15)
- ✅ std_ratio 0.9-1.1 (already achieved)
- ❌ Convergence without early plateauing (not achieved)

**Status**: 2/5 criteria met → **NOT READY**

---

## Lessons Learned

### Methodological Insights

1. **Fixing bias reveals truth**: Correcting measurement error exposes real performance
2. **Good metrics can hide bad models**: Version 01's "good" plants were measurement artifacts
3. **Transfer learning is not automatic**: Domain shift requires careful regularization tuning
4. **Stratified splitting is critical**: For seasonal/temporal data, random splitting is insufficient
5. **Validation trustworthiness**: Must verify split balance BEFORE trusting metrics

### Technical Insights

1. **std_ratio as diagnostic**: val_std / train_std is excellent proxy for split quality
2. **Zero percentage patterns**: Extreme differences (37% vs 84%) indicate seasonal bias
3. **Train/val ratio interpretation**: <0.5 suspicious, 1.0-1.3 normal, >2.0 severe overfitting
4. **Temporal split challenges**: Chronological split assumes uniform distribution (often false)

### Research Insights

1. **Failed experiments have value**: Version 01 + Version 02 tell complete story
2. **Iterative refinement**: V01 (identify) → V02 (fix bias) → V2.1 (fix overfitting)
3. **Documentation importance**: Detailed reports enable reproducibility and thesis writing
4. **Hypothesis testing**: Each version tests specific hypotheses about the problem

---

## Reproducibility

### Code Changes
- **Preprocessing**: `src/preprocessing/germany_pretrain_normalize_split.py` (stratified_temporal_split)
- **Training**: `run_stage2_transfer_learning.sh` (5 plants, no plant_04)
- **Analysis**: `notebooks/lstm/stage2_version02_validation_metrics.ipynb`

### Data Artifacts
- **Splits**: `data/processed/pretraining/germany/plant_XX/{train,val,test}.parquet`
- **Encoders**: `experiments/lstm/encoders/lstm_encoder_plant_XX.pt`
- **Logs**: `experiments/lstm/runs/germany/pretrain_plant_XX/germany_plant_XX_pretrain/version_0/metrics.csv`

### Configuration
```yaml
pretrained_weights: experiments/lstm/encoders/lstm_encoder_farm2107_CANONICAL.pt
hidden_size: 64
num_layers: 2
dropout: 0.1
lr: 1e-4
batch_size: 512
max_epochs: 20
random_seed: 42  # For stratified split reproducibility
```

### Random Seeds
- Stratified split: `random_seed=42` (in `stratified_temporal_split()`)
- PyTorch Lightning: (check training script for seed setting)

---

## Next Actions (Version 2.1)

### Immediate (Before Retraining)
1. ✅ Document Version 02 results (this report)
2. ⏳ Create Version 2.1 todo list with hyperparameter experiments
3. ⏳ Update training script with new hyperparameters
4. ⏳ Implement early stopping in training loop
5. ⏳ Add gradient clipping

### Short-term (Version 2.1 Execution)
1. ⏳ Experiment 1: Conservative tuning (dropout=0.3, lr=5e-5)
2. ⏳ Experiment 2: Aggressive tuning (dropout=0.5, lr=1e-5)
3. ⏳ Experiment 3: LR scheduler approach
4. ⏳ Compare results in updated validation notebook
5. ⏳ If successful: Proceed to Stage 2B
6. ⏳ If failed: Move to Version 2.2 (train from scratch)

### Medium-term (If Version 2.1 Fails)
1. ⏳ Version 2.2: Train from scratch (no transfer learning)
2. ⏳ Version 2.3: Architecture changes (smaller model, batch norm)
3. ⏳ Diagnostic: Train farm2107 with stratified split (baseline comparison)

---

## References

- **Version 01 Report**: `reports/stage2_version01_failed_chronological_split.md`
- **Version 02 Analysis**: `notebooks/lstm/stage2_version02_validation_metrics.ipynb`
- **Preprocessing Implementation**: `src/preprocessing/germany_pretrain_normalize_split.py`
- **Training Script**: `run_stage2_transfer_learning.sh`

---

## Appendices

### Appendix A: Stratified Split Verification Output

```
plant_01: train=43,000 val=9,213 test=9,218
  Train seasons: Winter=23.4% Spring=28.8% Summer=28.8% Fall=19.0%
  Val   seasons: Winter=23.4% Spring=28.8% Summer=28.8% Fall=19.0%
  Test  seasons: Winter=23.4% Spring=28.8% Summer=28.7% Fall=19.0%

plant_03: train=24,460 val=5,239 test=5,246
  Train seasons: Winter=24.5% Spring=25.3% Summer=25.3% Fall=25.0%
  Val   seasons: Winter=24.5% Spring=25.3% Summer=25.3% Fall=25.0%
  Test  seasons: Winter=24.5% Spring=25.3% Summer=25.3% Fall=25.0%
```

### Appendix B: Full Metrics Table

(See `stage2_version02_validation_metrics.ipynb` for complete output)

### Appendix C: Convergence Curve Analysis

(Plots saved in notebook output)

---

**Report Status**: Complete  
**Next Version**: 2.1 (Hyperparameter Tuning)  
**Updated**: December 16, 2024
