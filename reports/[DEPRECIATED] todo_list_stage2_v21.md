# [DEPRECIATED] → MOVED TO VERSION 03 
## which is the correct canonical version now
# Stage 2 Transfer Learning - Active Todo List


**Date**: December 16, 2025  
**Current Version**: Version 02 (Stratified Split Complete)  
**Next Version**: Version 2.1 (Hyperparameter Tuning)  
**Status**: Overfitting diagnosed, ready for regularization experiments

---

## Priority: HIGH (Blocking Stage 2B)

### ✅ Completed Tasks

1. **Document Version 02 overfitting results**
   - Created: `reports/stage2_version02_overfitting_diagnosis.md`
   - Documented stratified split success (std_ratio 0.99-1.01)
   - Documented overfitting problem (train/val ratio 2.0-2.5)
   - Outlined Version 2.1 strategy with 3 experiments

---

## Version 2.1: Hyperparameter Tuning Experiments

### ⏳ Experiment 1: Conservative Tuning (RECOMMENDED FIRST)

**Goal**: Test if gentle regularization increases reduce overfitting

**Configuration**:
```yaml
dropout: 0.3          # 0.1 → 0.3 (3x increase)
lr: 5e-5              # 1e-4 → 5e-5 (half learning rate)
weight_decay: 1e-4    # Add L2 regularization
batch_size: 512       # Keep same
max_epochs: 20
early_stopping:
  monitor: val_loss
  patience: 5
  mode: min
gradient_clip_val: 1.0
```

**Implementation Steps**:
1. Update training script with new hyperparameters
2. Implement early stopping callback
3. Add gradient clipping
4. Run training for 5 plants
5. Analyze results with `stage2_version02_validation_metrics.ipynb`

**Success Criteria**:
- Train/val ratio: 1.0-1.5 (currently 2.0-2.5)
- Val RMSE: <0.10 (currently 0.09-0.15)
- Early stopping triggered before epoch 20
- Convergence curves show val improving with train

**IF SUCCESS**: Proceed to Stage 2B (TFT ensemble)  
**IF FAILURE**: Move to Experiment 2 (Aggressive Tuning)

---

### ⏳ Experiment 2: Aggressive Tuning (IF EXP1 FAILS)

**Goal**: Strong regularization to prevent memorization

**Configuration**:
```yaml
dropout: 0.5          # 0.1 → 0.5 (5x increase, very strong)
lr: 1e-5              # 1e-4 → 1e-5 (10x slower learning)
weight_decay: 1e-4    # Same as Exp1
batch_size: 256       # 512 → 256 (smaller batches, more updates)
max_epochs: 30        # Increase epochs (slower learning)
early_stopping:
  monitor: val_loss
  patience: 3         # More aggressive (stop sooner)
  mode: min
gradient_clip_val: 1.0
```

**Rationale**: If conservative fails, overfitting is severe and needs strong regularization

**Success Criteria**: Same as Exp1

**IF SUCCESS**: Proceed to Stage 2B  
**IF FAILURE**: Move to Experiment 3 (LR Scheduler) or Version 2.2 (train from scratch)

---

### ⏳ Experiment 3: Learning Rate Scheduler (ALTERNATIVE)

**Goal**: Adaptive learning rate reduction when validation plateaus

**Configuration**:
```yaml
dropout: 0.3          # Moderate regularization
lr: 1e-4              # Start same as Version 02
lr_scheduler:
  type: ReduceLROnPlateau
  monitor: val_loss
  factor: 0.5         # Halve LR when plateau
  patience: 3         # Reduce after 3 epochs no improvement
  min_lr: 1e-6        # Stop reducing at this point
weight_decay: 1e-4
batch_size: 512
max_epochs: 30
early_stopping:
  monitor: val_loss
  patience: 5
  mode: min
gradient_clip_val: 1.0
```

**Rationale**: Allow fast initial learning, then automatically slow down when overfitting starts

**Success Criteria**: Same as Exp1

---

## Supporting Implementation Tasks

### ⏳ Implement gradient clipping
**Purpose**: Prevent exploding gradients during transfer learning

**Code**:
```python
trainer = pl.Trainer(
    gradient_clip_val=1.0,
    gradient_clip_algorithm="norm"
)
```

**Validation**: Check logs for gradient norm values

---

### ⏳ Add early stopping callback
**Purpose**: Stop training when validation stops improving

**Code**:
```python
from pytorch_lightning.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,  # or 3 for aggressive
    mode='min',
    verbose=True
)

trainer = pl.Trainer(callbacks=[early_stop])
```

**Validation**: Training should stop before max_epochs if val plateaus

---

### ⏳ Compare Version 2.1 results with Version 02 baseline
**Purpose**: Quantify improvement from hyperparameter changes

**Metrics to Compare**:
- Train/val ratio (target: <1.5, baseline: 2.0-2.5)
- Val RMSE (target: <0.10, baseline: 0.09-0.15)
- std_ratio (should maintain: 0.9-1.1)
- Convergence behavior (should improve: val following train)

**Method**: Update `stage2_version02_validation_metrics.ipynb` to include Version 2.1 results

---

## Fallback Strategy

### ⏳ IF Version 2.1 fails: Version 2.2 - Train from Scratch

**Context**: If all 3 hyperparameter experiments fail to reduce overfitting below 1.5 ratio

**Hypothesis**: Transfer learning from farm2107 may be hurting, not helping

**Design**:
- **No pretrained weights** (initialize randomly)
- **Same stratified splits** (Version 02 splits are correct)
- **Best hyperparameters from Version 2.1** (use whatever worked best)
- **Longer training** (may need 50-100 epochs from scratch)

**Implementation**:
```python
# Remove this line:
# model.load_state_dict(torch.load("lstm_encoder_farm2107_CANONICAL.pt"))

# Train from random initialization
model = LSTMEncoder(config)  # Random init
trainer.fit(model, train_loader, val_loader)
```

**Success Criteria**:
- Train/val ratio <1.5
- Val RMSE competitive with or better than Version 02
- Convergence within 50 epochs

**IF SUCCESS**: Use these encoders for Stage 2B  
**IF FAILURE**: Move to Version 2.3 (architecture changes)

---

## Additional Diagnostic Tasks

### ⏳ Monitor gradient norms during training
**Purpose**: Detect vanishing/exploding gradients

**Implementation**:
```python
trainer = pl.Trainer(
    track_grad_norm=2,  # Track L2 norm
    log_every_n_steps=50
)
```

**Analysis**: Check TensorBoard or CSV logs for gradient norm trends

---

### ⏳ Increase validation frequency
**Purpose**: Catch overfitting earlier in training

**Implementation**:
```python
trainer = pl.Trainer(
    val_check_interval=0.25,  # Validate 4x per epoch
)
```

**Benefit**: More granular convergence curves

---

### ⏳ (Optional) Diagnostic: Train farm2107 with stratified split
**Purpose**: Determine if stratified split itself causes overfitting

**Hypothesis**: If farm2107 also shows high ratios with stratified split, the split method may be too aggressive

**Method**:
1. Apply stratified temporal split to farm2107 data
2. Retrain farm2107 encoder
3. Compare train/val ratio to original (should still be 1.0-1.3)

**IF farm2107 overfits with stratified split**: Split parameters need tuning  
**IF farm2107 fine with stratified split**: Germany data or transfer learning is the issue

---

## Long-term Roadmap

### Version 2.3: Architecture Changes (IF 2.2 FAILS)

**Options**:
1. **Reduce model capacity**:
   - hidden_size: 64 → 32
   - num_layers: 2 → 1
   - Rationale: Smaller model = less capacity to memorize

2. **Add batch normalization**:
   - After each LSTM layer
   - Rationale: Stabilize training, reduce internal covariate shift

3. **Try different architecture**:
   - Transformer encoder
   - Temporal Fusion Transformer (TFT)
   - Rationale: May be better suited to multi-site data

---

## Stage 2B: TFT Ensemble (ONLY WHEN READY)

**Prerequisites**:
- ✅ Stratified splits (achieved in Version 02)
- ❌ Train/val ratio <1.5 (Version 2.1 goal)
- ❌ Val RMSE <0.08-0.10 (Version 2.1 goal)
- ✅ std_ratio 0.9-1.1 (achieved in Version 02)

**Status**: **BLOCKED** until Version 2.1 shows acceptable overfitting levels

**Rationale**: TFT ensemble requires quality LSTM embeddings. Current encoders memorize training data, so embeddings won't generalize to TFT.

---

## Thesis Integration

### Narrative Arc (For Documentation)

1. **Chapter: Stage 2 Transfer Learning**
   - Section 2.1: Initial Approach (Version 01)
     - Chronological split methodology
     - Initial promising results
     - Validation analysis reveals bias
   
   2. **Section 2.2: Root Cause Investigation**
     - Power distribution analysis
     - Split boundary analysis (smoking gun)
     - Understanding seasonal bias
   
   3. **Section 2.3: Corrected Methodology (Version 02)**
     - Stratified temporal split design
     - Implementation and validation
     - Discovery of overfitting problem
   
   4. **Section 2.4: Regularization Tuning (Version 2.1)**
     - Hyperparameter experiments
     - Results and analysis
     - Final encoder selection
   
   5. **Section 2.5: Lessons Learned**
     - Validation methodology importance
     - Transfer learning challenges
     - Iterative refinement process

**Key Takeaway**: Complete research story from initial failure → diagnosis → solution → new challenge → final solution

---

## Notes for Next Session

**Save State**:
- Version 02 encoders: `experiments/lstm/encoders/lstm_encoder_plant_{01,02,03,05,06}.pt`
- Version 02 splits: `data/processed/pretraining/germany/plant_XX/{train,val,test}.parquet`
- Analysis notebook: `notebooks/lstm/stage2_version02_validation_metrics.ipynb`
- Reports: `reports/stage2_version0{1,2}_*.md`

**Key Files to Modify for Version 2.1**:
- Training script: Update hyperparameters, add early stopping
- Experiment configs: Create YAML files for each experiment
- Analysis notebook: Add Version 2.1 comparison cells

**Monitoring**:
- Watch for early stopping trigger
- Check gradient norms
- Compare convergence curves to Version 02
- Calculate train/val ratios after each experiment

---

**Todo List Status**: 1 completed, 7 active  
**Priority**: Fix overfitting to unblock Stage 2B  
**Estimated Time**: 1-2 days for Version 2.1 experiments  
**Last Updated**: December 16, 2024
