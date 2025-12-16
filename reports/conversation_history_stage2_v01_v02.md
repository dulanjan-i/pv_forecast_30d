# Stage 2 Transfer Learning - Conversation History
**Date**: December 16, 2024  
**Session**: Version 01 Failure Analysis → Version 02 Implementation & Overfitting Discovery

---

## Session Overview

This conversation documents the complete journey from discovering validation bias in Version 01 through implementing and validating Version 02's stratified temporal split, which revealed severe overfitting requiring Version 2.1 hyperparameter tuning.

**Timeline**:
1. **Version 01 Validation Analysis** - Discovered suspicious metrics (plants 03, 04, 06 had validation 2-5x better than training)
2. **Root Cause Investigation** - Split boundary analysis revealed seasonal bias (winter vs spring validation sets)
3. **Version 01 Documentation** - Created thesis-format failure report
4. **Version 02 Implementation** - Stratified temporal split + NaN dropping + plant_04 exclusion
5. **Version 02 Validation** - Confirmed bias fixed (std_ratio 0.99-1.01) but revealed severe overfitting (train/val ratio 2.0-2.5)
6. **Version 2.1 Planning** - Hyperparameter tuning strategy with 3 experiments

---

## Key Decisions Made

### Decision 1: Document Version 01 as Failed Experiment
**Context**: Initial metrics looked promising (mean RMSE -14% vs baseline), but deeper analysis showed this was measurement error, not true generalization.

**Evidence**:
- Plants 03, 06 had train/val ratios of 0.28, 0.22 (validation 3.5-4.6x better) - physically impossible
- Split boundary analysis showed these plants had winter-only validation (Nov-Jan, 84% zeros)
- std_ratio of 0.17-0.20 proved validation difficulty was drastically different

**Decision**: Document failure in thesis-ready markdown, preserve all artifacts for reproducibility

**Outcome**: Created `reports/stage2_version01_failed_chronological_split.md`

### Decision 2: Implement Stratified Temporal Split
**Context**: Chronological split assumes uniform temporal coverage, but Germany plants had uneven seasonal representation (456-639 days, different end dates).

**Solution Design**:
```python
def stratified_temporal_split(df, time_col, train_frac=0.70, val_frac=0.15, random_seed=42):
    # Classify timestamps by season
    # Sample proportionally from each season
    # Maintain temporal ordering
    return sorted(train_idx), sorted(val_idx), sorted(test_idx)
```

**Validation Criteria**:
- std_ratio in 0.9-1.1 (balanced difficulty)
- Similar % zeros across splits
- Identical seasonal distributions in train/val/test

**Outcome**: Perfect balance achieved (std_ratio 0.989-1.011)

### Decision 3: Exclude Plant 04
**Context**: Plant 04 showed 100% zeros during Mar-Jun 2024 (spring/summer) - not a seasonal bias but data quality issue.

**Evidence**: Should be high-production season, but got all zeros → plant offline or data corrupted

**Decision**: Exclude from all preprocessing, train on 5 plants only

**Outcome**: Updated 4 preprocessing scripts to remove plant_04

### Decision 4: Add NaN Dropping
**Context**: Some plants had significant NaN values in power_norm column.

**Implementation**: Added explicit `dropna(subset=['power_norm'])` in `germany_build_pretrain_base.py`

**Results**:
- plant_03: 20.3% dropped (8,920 rows)
- plant_05: 39.8% dropped (17,480 rows)
- Others: <0.1% dropped

### Decision 5: Retrain with Version 02 Corrections
**Context**: Version 01 artifacts invalid due to biased validation, need clean retrain.

**Cleanup Steps**:
1. Removed old encoders (`lstm_encoder_plant_*.pt`)
2. Removed old training logs (`pretrain_plant_*/`)
3. Preserved farm2107_CANONICAL baseline
4. Re-ran preprocessing pipeline
5. Trained 5 plants with corrected splits

**Outcome**: Training completed, all 5 plants finished 20 epochs

### Decision 6: Version 02 Analysis Reveals Overfitting
**Context**: With balanced validation sets, true performance became visible.

**Results**:
- ✅ std_ratio: 0.989-1.011 (PERFECT balance)
- ✅ val_zeros: 48-53% (consistent across plants)
- ❌ train/val ratio: 2.0-2.5 (SEVERE overfitting)
- ❌ Val RMSE: 0.09-0.15 (2-3x worse than farm2107 baseline)

**Interpretation**: Version 01's "good" results were measurement artifacts. Version 02 shows reality: model overfits badly.

**Thesis Value**: Complete narrative demonstrating:
- Problem identification
- Root cause diagnosis
- Solution implementation
- Secondary problem discovery
- Critical thinking about measurement vs reality

### Decision 7: Plan Version 2.1 Hyperparameter Tuning
**Context**: Cannot proceed to Stage 2B with overfitted encoders. Need to fix regularization before ensemble training.

**Strategy**:
1. **Exp 1 (Conservative)**: dropout 0.3, lr 5e-5, weight_decay 1e-4, early_stop patience=5
2. **Exp 2 (Aggressive)**: dropout 0.5, lr 1e-5, early_stop patience=3
3. **Exp 3 (LR Scheduler)**: ReduceLROnPlateau starting at 1e-4
4. **Fallback**: Version 2.2 train from scratch (no transfer learning)

**Additional Improvements**:
- Gradient clipping (max_norm=1.0)
- Validation frequency (4x per epoch)
- Gradient norm monitoring

---

## Technical Findings

### Finding 1: std_ratio as Validation Quality Metric
**Discovery**: `std_ratio = val_std / train_std` is excellent proxy for split quality.

**Thresholds**:
- 0.9-1.1: Balanced difficulty ✅
- <0.8: Val much easier (seasonal bias) 🚨
- >1.2: Val much harder

**Application**: All Version 02 plants showed 0.99-1.01, proving balance.

### Finding 2: Train/Val Ratio Interpretation
**Discovery**: Ratio alone insufficient - must consider validation trustworthiness.

**Version 01 Misinterpretation**:
- plant_03 ratio 0.28 → appeared "excellent generalization"
- Reality: validation was winter-only (84% zeros), artificially easy

**Version 02 Correct Interpretation**:
- plant_03 ratio 2.33 → appears "severe overfitting"
- Reality: validation now balanced, shows true performance gap

**Lesson**: Context matters. Good metrics on bad data = bad conclusions.

### Finding 3: Chronological Split Failure Mode
**Discovery**: Simple percentage slicing fails when:
- Source data has uneven temporal coverage
- Different entities have different time ranges
- Seasons are unevenly represented

**Example**:
- plant_03 (456 days, end Apr 2024): Last 15% → Nov-Jan (winter)
- plant_01 (639 days, end Oct 2024): Last 15% → Mar-Jun (spring)
- Same split method → different seasons → incomparable metrics

**Solution**: Stratify by season, not just time.

### Finding 4: Transfer Learning Overfitting Pattern
**Discovery**: All 5 plants show similar overfitting degree (2.0-2.5 ratio), suggesting **systematic issue**, not plant-specific.

**Hypotheses**:
1. Pretrained weights too specific to farm2107
2. Insufficient regularization (dropout=0.1 too low)
3. Dataset size reduction (20-40% NaN dropped)
4. Germany data more complex than farm2107

**Next Test**: Hyperparameter tuning will distinguish between these.

---

## Code Changes Summary

### Modified Files

1. **`src/data/preprocess_germany_pv.py`**
   - Removed plant_04 from `plant_ids` list
   - Updated docstring referencing Version 02

2. **`src/data/merge_germany_pv_weather.py`**
   - Removed plant_04 from `PLANT_IDS`
   - Added comment explaining exclusion

3. **`src/preprocessing/germany_build_pretrain_base.py`**
   - Removed plant_04
   - Added NaN dropping logic with diagnostic output
   - Results: 0-40% dropped per plant

4. **`src/preprocessing/germany_pretrain_normalize_split.py`** (MAJOR REWRITE)
   - Deleted `split_indices()` function (chronological split)
   - Implemented `stratified_temporal_split()` function
   - Added seasonal balance verification output
   - Updated `process_one()` to use new split
   - Confirmed test split creation

### Created Files

1. **`reports/stage2_version01_failed_chronological_split.md`**
   - Comprehensive thesis-format failure documentation
   - 8 sections + appendices
   - Evidence tables and diagnostic outputs

2. **`reports/stage2_version02_overfitting_diagnosis.md`**
   - Version 02 results and analysis
   - Root cause hypotheses
   - Version 2.1 strategy with 3 experiments
   - Complete decision tree for next steps

3. **`notebooks/lstm/stage2_version02_validation_metrics.ipynb`**
   - Clean analysis notebook for Version 02
   - 7 sections + deep dive interpretation
   - Comparison tables with Version 01
   - Go/no-go criteria for Stage 2B

### Preprocessing Pipeline Execution

**Commands Run**:
```bash
# Step 1: Build pretrain base (with NaN dropping)
python src/preprocessing/germany_build_pretrain_base.py

# Step 2: Stratified split + normalization
python src/preprocessing/germany_pretrain_normalize_split.py

# Output: data/processed/pretraining/germany/plant_XX/{train,val,test}.parquet
```

**Results**:
- plant_01: 43,000 train, 9,213 val, 9,218 test | Winter=23.4% Spring=28.8% Summer=28.8% Fall=19.0%
- plant_02: 42,998 train, 9,213 val, 9,216 test | Winter=23.4% Spring=28.8% Summer=28.8% Fall=19.0%
- plant_03: 24,460 train, 5,239 val, 5,246 test | Winter=24.5% Spring=25.3% Summer=25.3% Fall=25.0%
- plant_05: 18,469 train, 3,956 val, 3,960 test | Winter=32.7% Spring=11.6% Summer=22.6% Fall=33.1%
- plant_06: 30,704 train, 6,577 val, 6,584 test | Winter=32.8% Spring=27.1% Summer=20.1% Fall=19.9%

### Training Execution

**Script**: `./run_stage2_transfer_learning.sh`

**Configuration**:
```yaml
pretrained_weights: lstm_encoder_farm2107_CANONICAL.pt
hidden_size: 64
num_layers: 2
dropout: 0.1
lr: 1e-4
batch_size: 512
max_epochs: 20
```

**Status**: Completed 20 epochs for all 5 plants

**Output**:
- Encoders: `experiments/lstm/encoders/lstm_encoder_plant_{01,02,03,05,06}.pt`
- Logs: `experiments/lstm/runs/germany/pretrain_plant_XX/.../metrics.csv`

---

## Artifacts & Reproducibility

### Preserved Artifacts

**Version 01 Evidence** (Documented but artifacts cleaned):
- Analysis notebook: `notebooks/lstm/stage2_validation_metrics.ipynb` (15 cells)
- Failure report: `reports/stage2_version01_failed_chronological_split.md`

**Version 02 Results**:
- Splits: `data/processed/pretraining/germany/plant_XX/{train,val,test}.parquet`
- Encoders: `experiments/lstm/encoders/lstm_encoder_plant_XX.pt` (5 plants)
- Training logs: `experiments/lstm/runs/germany/pretrain_plant_XX/.../metrics.csv`
- Analysis notebook: `notebooks/lstm/stage2_version02_validation_metrics.ipynb`
- Diagnosis report: `reports/stage2_version02_overfitting_diagnosis.md`

**Baseline Reference**:
- Farm2107 encoder: `experiments/lstm/encoders/lstm_encoder_farm2107_CANONICAL.pt` (233KB, Nov 24, RMSE=0.040388)

### Random Seeds

**Reproducibility**:
- Stratified split: `random_seed=42` (in `stratified_temporal_split()`)
- PyTorch Lightning: (seed set in training script)

### Validation Criteria

**Version 02 Checklist**:
- ✅ std_ratio 0.9-1.1: 5/5 plants PASS
- ❌ train/val ratio 1.0-1.3: 0/5 plants PASS
- ✅ No suspiciously good plants (<0.5): 5/5 plants PASS
- ❌ Val RMSE <0.08: 0/5 plants PASS
- ✅ Balanced seasonal distributions: 5/5 plants PASS

**Stage 2B Readiness**: 3/5 criteria passed → **NOT READY** (overfitting must be fixed)

---

## Lessons for Future Work

### Methodological Lessons

1. **Always verify split balance** before trusting metrics
   - Use std_ratio, mean differences, % zeros
   - Visual inspection of distributions
   - Temporal coverage analysis

2. **Suspicious metrics require investigation**
   - Val better than train → red flag
   - Extreme variance across entities → check for bias
   - "Too good to be true" probably is

3. **Failed experiments have research value**
   - Shows rigor and critical thinking
   - Justifies improved methods
   - Demonstrates problem-solving process

4. **Documentation enables reproducibility**
   - Preserve artifacts even from failed experiments
   - Clear paper trail for thesis
   - Evidence-based conclusions

### Technical Lessons

1. **Chronological ≠ valid for time-series**
   - Needs uniform coverage or stratification
   - Simple percentage slicing can fail badly
   - Always check what seasons each split captures

2. **Transfer learning is not automatic**
   - Domain shift requires careful tuning
   - Pretrained weights can be too specific
   - Regularization often needs strengthening

3. **Fixing measurement error reveals truth**
   - Version 01: biased metrics hid overfitting
   - Version 02: balanced metrics revealed reality
   - Better to know the truth than live with artifacts

4. **Stratified splitting is powerful**
   - Ensures balanced seasonal representation
   - Makes validation sets trustworthy
   - Enables fair comparison across entities

### Research Strategy Lessons

1. **Iterative refinement works**
   - V01: Identify problem
   - V02: Fix bias, discover overfitting
   - V2.1: Fix overfitting
   - Each version tests specific hypothesis

2. **Systematic diagnostics catch problems**
   - Power distribution analysis
   - Temporal coverage analysis
   - Split boundary analysis (smoking gun)

3. **Context matters for interpretation**
   - Good metric + bad data = bad conclusion
   - Must verify data quality before trusting results
   - Ratio alone insufficient, need std_ratio too

---

## Next Session Plan (Version 2.1)

### Immediate Tasks

1. **Update training script** with hyperparameter options:
   - Add command-line args for dropout, lr, weight_decay
   - Implement early stopping callback
   - Add gradient clipping
   - Enable validation frequency control

2. **Create experiment configs** (YAML):
   - `v21_exp1_conservative.yaml`
   - `v21_exp2_aggressive.yaml`
   - `v21_exp3_lr_schedule.yaml`

3. **Run Experiment 1** (Conservative):
   - dropout=0.3, lr=5e-5, weight_decay=1e-4
   - early_stop patience=5
   - Validate with version02 notebook

4. **Decision Point**:
   - IF train/val ratio <1.5: SUCCESS → Stage 2B
   - IF not improved: Try Exp 2 (Aggressive)
   - IF still fails: Version 2.2 (train from scratch)

### Long-term Goals

- Version 2.1: Fix overfitting with hyperparameter tuning
- Version 2.2 (if needed): Train from scratch (no transfer learning)
- Version 2.3 (if needed): Architecture changes
- Stage 2B: TFT ensemble (only when encoders are trustworthy)
- Thesis: Document complete journey (V01 → V02 → V2.1 → Stage 2B)

---

## Questions Answered During Session

**Q: "What is wrong with the validation metrics?"**
A: Version 01 used chronological split which gave different plants different seasons in validation. Plants 03, 06 got winter (84% zeros) → artificially good metrics.

**Q: "How do we fix the seasonal bias?"**
A: Stratified temporal split - sample proportionally from each season for train/val/test, ensuring balanced representation.

**Q: "Should we exclude plant_04?"**
A: Yes, 100% zeros during Mar-Jun 2024 is data quality issue (should be high production in spring). Exclude until fixed.

**Q: "Did stratified split work?"**
A: Yes! std_ratio now 0.99-1.01 (perfect balance). But it revealed severe overfitting (train/val ratio 2.0-2.5).

**Q: "Is Version 02 a success or failure?"**
A: Scientific success (fixed bias, trustworthy metrics), performance failure (severe overfitting). Great for thesis!

**Q: "Can we proceed to Stage 2B?"**
A: No, encoders are overfitted (memorizing, not generalizing). Must fix with Version 2.1 first.

**Q: "What should we try first for Version 2.1?"**
A: Conservative hyperparameter tuning: dropout 0.3, lr 5e-5, weight_decay 1e-4. If that fails, train from scratch.

---

## User's Working Style Observations

**Preferences**:
- Systematic, evidence-based approach
- Thesis-ready documentation throughout
- Preserves artifacts for reproducibility
- Values complete narratives (including failures)
- Appreciates detailed technical explanations

**Notation Style**:
- Git commits: `[CATEGORY]: Description`
- Categories: REFACTOR, ADD, MOD, FIX
- Detailed, multi-line commit messages

**Work Environment**:
- calc02 HPC with 2x NVIDIA L4 GPUs
- Multiple terminal sessions for monitoring
- Conda virtual environment (`pvforecast`)
- Organized directory structure with clear separation

**Communication Style**:
- Direct questions when uncertain
- Requests clarification on technical details
- Appreciates suggestions with rationale
- Values efficiency (parallel operations)

---

**Session End**: December 16, 2024  
**Status**: Version 02 complete, Version 2.1 planned  
**Next Action**: Implement hyperparameter tuning experiments
