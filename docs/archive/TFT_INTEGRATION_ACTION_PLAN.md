***REMOVED*** TFT Integration Action Plan

**Date:** 2026-01-02  
**Status:** Ready to implement  
**Estimated Time:** 2-3 hours

---

***REMOVED******REMOVED*** 📋 Study Summary: `offline_predict_tft.py`

***REMOVED******REMOVED******REMOVED*** What It Does
1. **Loads config & checkpoints** from `run_dir/`
   - `run_config.json` → model hyperparameters (encoder_len, pred_len, hidden_size, etc.)
   - `column_roles.json` → feature column mappings
   - `checkpoints/best_state_dict.pt` or `checkpoints/best.ckpt` → trained weights

2. **Creates TimeSeriesDataSet** from training data
   - Uses PyTorch Forecasting's `TimeSeriesDataSet` class
   - Handles: group IDs, time index, target, known/unknown features
   - Key parameters: `max_encoder_length`, `max_prediction_length`

3. **Creates test dataset** using `TimeSeriesDataSet.from_dataset()`
   - Inherits normalization/encoding from training dataset
   - Critical: Must use same preprocessing as training

4. **Batch inference loop**
   - Iterates through DataLoader
   - Extracts predictions: `output.prediction` → shape `(B, P, Q)`
     - B = batch size
     - P = prediction length (96 for short-head, 720 for long-head)
     - Q = number of quantiles (default: 7)
   - **Q50 quantile:** Middle quantile (index 3 if 7 quantiles)

5. **Output format:**
   ```python
   {
       'plant_id': str,
       'horizon': int (1 to P),
       'time_idx': int,
       'timestamp_utc': datetime,
       'y_true': float,
       'y_hat_q50': float,  ***REMOVED*** Median prediction
       'y_hat_q02': float,  ***REMOVED*** 2% quantile
       'y_hat_q98': float,  ***REMOVED*** 98% quantile
       ...
   }
   ```

---

***REMOVED******REMOVED*** 🔍 Key Findings & Issues

***REMOVED******REMOVED******REMOVED*** ✅ What Works
- Config loading with fallback: `encoder_len` → `max_encoder_length`, `pred_len` → `max_prediction_length`
- Automatic time_idx recomputation via `cumcount()` (prevents missing timestep errors)
- State dict loading with prefix stripping (`model.`, `tft.`, `net.`)
- Group ID encoding/decoding
- Quantile extraction

***REMOVED******REMOVED******REMOVED*** ⚠️ Critical Issue Identified: **Prediction Length**

**The User's Original Issue:**
> "it had an issue and we had to put the correct pred lengths"

**What This Means:**
```python
***REMOVED*** Config says:
pred_len = 96  ***REMOVED*** or 720 for longhead

***REMOVED*** But model actually predicts:
output.prediction.shape = (B, pred_len, num_quantiles)

***REMOVED*** Must extract correct length!
```

**Solution in `offline_predict_tft.py` (lines 221-223):**
```python
max_encoder_length = int(cfg.get("max_encoder_length", cfg.get("encoder_len", 96)))
max_prediction_length = int(cfg.get("max_prediction_length", cfg.get("pred_len", 96)))
```

**For our hierarchical forecasting:**
- **Short-head:** `encoder_len=96`, `pred_len=96` @ 15-min
- **Long-head:** `encoder_len=168`, `pred_len=720` @ 1-hour

**Must match exactly or TimeSeriesDataSet will fail!**

---

***REMOVED******REMOVED*** 🏗️ Implementation Strategy

***REMOVED******REMOVED******REMOVED*** Architecture Mapping

```
PhysicsAwareForecaster.predict_30d()
  │
  ├─> _predict_long_head()           [1 call]
  │     ├─ Load long-head TFT model
  │     ├─ Create TimeSeriesDataSet (encoder=168h, decoder=720h)
  │     ├─ Run inference
  │     └─ Extract q50 → (720,)
  │
  ├─> Rolling loop (30 days)
  │     │
  │     └─> _predict_short_head_for_day(day=0..29)  [30 calls]
  │           ├─ Extract day window (encoder=96@15min, decoder=96@15min)
  │           ├─ Create TimeSeriesDataSet batch
  │           ├─ Run inference
  │           └─ Extract q50 → (96,)
  │
  └─> blend_hierarchical() per day
```

***REMOVED******REMOVED******REMOVED*** Critical Differences from `offline_predict_tft.py`

| Aspect | Offline Script | Our Hierarchical Forecaster |
|--------|----------------|------------------------------|
| **Input** | Full test parquet | Live weather forecast DataFrame |
| **Dataset** | TimeSeriesDataSet from full df | Single-sample batch per day |
| **Batch size** | 512 samples | 1 sample (single day) |
| **Output** | Full parquet with all horizons | q50 only, shape (96,) or (720,) |
| **Normalization** | From training set | Must inherit from training set |
| **Purpose** | Offline evaluation | Real-time production |

---

***REMOVED******REMOVED*** 📝 Detailed Implementation Plan

***REMOVED******REMOVED******REMOVED*** **STEP 1: Prepare Utility Functions** (15 min)

Create `src/inference/tft_utils.py` with shared logic:

```python
def load_tft_config(run_dir: Path) -> Dict:
    """Load run_config.json and column_roles.json."""
    
def create_training_dataset(
    train_df: pd.DataFrame,
    roles: Dict,
    encoder_len: int,
    pred_len: int
) -> TimeSeriesDataSet:
    """Create TimeSeriesDataSet from training data."""

def load_tft_checkpoint(
    checkpoint_path: Path,
    model: TemporalFusionTransformer,
    strict: bool = True
) -> None:
    """Load weights into model."""

def extract_q50_prediction(output, quantiles: List[float]) -> np.ndarray:
    """Extract median (q50) quantile from model output."""
    ***REMOVED*** Find q50 index
    q50_idx = quantiles.index(0.5) if 0.5 in quantiles else len(quantiles) // 2
    pred = output.prediction  ***REMOVED*** (B, P, Q)
    if pred.ndim == 3:
        return pred[:, :, q50_idx].cpu().numpy()
    else:
        return pred.cpu().numpy()
```

---

***REMOVED******REMOVED******REMOVED*** **STEP 2: Implement `_predict_short_head_for_day()`** (45 min)

**File:** `src/inference/physics_aware_forecaster.py`

**Current Status:** Placeholder with synthetic data

**Implementation:**

```python
def _predict_short_head_for_day(
    self,
    day_start: pd.Timestamp,
    day_idx: int,
    historical_df: pd.DataFrame,  ***REMOVED*** Full history
    weather_df: pd.DataFrame      ***REMOVED*** Full 30-day forecast
) -> np.ndarray:
    """
    Predict single day using short-head TFT (96 steps @ 15-min).
    
    Returns:
        predictions: Array shape (96,), normalized [0, 1]
    """
    ***REMOVED*** 1. Extract encoder window: 96 steps BEFORE day_start
    encoder_start = day_start - pd.Timedelta(hours=24)
    encoder_end = day_start
    
    encoder_df = historical_df[
        (historical_df['timestamp_utc'] >= encoder_start) &
        (historical_df['timestamp_utc'] < encoder_end)
    ].copy()
    
    ***REMOVED*** 2. Extract decoder window: 96 steps FROM day_start
    decoder_start = day_start
    decoder_end = day_start + pd.Timedelta(hours=24)
    
    decoder_df = weather_df[
        (weather_df['timestamp_utc'] >= decoder_start) &
        (weather_df['timestamp_utc'] < decoder_end)
    ].copy()
    
    ***REMOVED*** 3. Validate window lengths
    if len(encoder_df) != 96:
        raise ValueError(f"Encoder window must be 96 steps, got {len(encoder_df)}")
    if len(decoder_df) != 96:
        raise ValueError(f"Decoder window must be 96 steps, got {len(decoder_df)}")
    
    ***REMOVED*** 4. Concatenate encoder + decoder (continuous time series)
    inference_df = pd.concat([encoder_df, decoder_df], ignore_index=True)
    
    ***REMOVED*** 5. Ensure required columns exist
    if 'plant_id' not in inference_df.columns:
        inference_df['plant_id'] = self.plant_id
    
    ***REMOVED*** Recompute time_idx (critical!)
    inference_df = inference_df.sort_values('timestamp_utc').reset_index(drop=True)
    inference_df['time_idx'] = inference_df.groupby('plant_id').cumcount()
    
    ***REMOVED*** 6. Create TimeSeriesDataSet from this window
    ***REMOVED*** Use .from_dataset() to inherit normalization from training
    test_ds = TimeSeriesDataSet.from_dataset(
        self.short_train_ds,  ***REMOVED*** Training dataset (loaded in __init__)
        inference_df,
        predict=True,  ***REMOVED*** Important: predict mode (no target needed)
        stop_randomization=True
    )
    
    ***REMOVED*** 7. Create single-sample DataLoader
    test_dl = test_ds.to_dataloader(
        train=False,
        batch_size=1,  ***REMOVED*** Single day
        num_workers=0,
        shuffle=False
    )
    
    ***REMOVED*** 8. Run inference
    self.short_model.eval()
    with torch.no_grad():
        for x, y in test_dl:
            ***REMOVED*** Move to device
            x = {k: v.to(self.device) if torch.is_tensor(v) else v 
                 for k, v in x.items()}
            
            ***REMOVED*** Forward pass
            output = self.short_model(x)
            
            ***REMOVED*** Extract q50 quantile
            predictions = extract_q50_prediction(
                output, 
                self.short_model.loss.quantiles
            )
            
            ***REMOVED*** predictions shape: (1, 96) → squeeze to (96,)
            return predictions.squeeze()
    
    raise RuntimeError("No predictions generated")
```

**Key Considerations:**
- ✅ **Window alignment:** Encoder ends exactly when decoder starts
- ✅ **Time index continuity:** `time_idx` must be continuous (no gaps)
- ✅ **Normalization inheritance:** Use `.from_dataset()` not fresh `TimeSeriesDataSet()`
- ✅ **Predict mode:** `predict=True` → no target column needed
- ✅ **Single sample:** `batch_size=1` for one day

---

***REMOVED******REMOVED******REMOVED*** **STEP 3: Implement `_predict_long_head()`** (45 min)

**File:** `src/inference/physics_aware_forecaster.py`

**Current Status:** Placeholder with synthetic data

**Implementation:**

```python
def _predict_long_head(
    self,
    forecast_start: pd.Timestamp,
    historical_df: pd.DataFrame,  ***REMOVED*** Full history @ 1h
    weather_df: pd.DataFrame      ***REMOVED*** Full 30-day forecast @ 1h
) -> np.ndarray:
    """
    Predict all 30 days using long-head TFT (720 steps @ 1-hour).
    
    Returns:
        predictions: Array shape (720,), normalized [0, 1]
    """
    ***REMOVED*** 1. Extract encoder window: 168 hours (7 days) BEFORE forecast_start
    encoder_start = forecast_start - pd.Timedelta(hours=168)
    encoder_end = forecast_start
    
    encoder_df = historical_df[
        (historical_df['timestamp_utc'] >= encoder_start) &
        (historical_df['timestamp_utc'] < encoder_end)
    ].copy()
    
    ***REMOVED*** 2. Extract decoder window: 720 hours (30 days) FROM forecast_start
    decoder_start = forecast_start
    decoder_end = forecast_start + pd.Timedelta(hours=720)
    
    decoder_df = weather_df[
        (weather_df['timestamp_utc'] >= decoder_start) &
        (weather_df['timestamp_utc'] < decoder_end)
    ].copy()
    
    ***REMOVED*** 3. Validate window lengths
    if len(encoder_df) != 168:
        raise ValueError(f"Encoder window must be 168 steps, got {len(encoder_df)}")
    if len(decoder_df) != 720:
        raise ValueError(f"Decoder window must be 720 steps, got {len(decoder_df)}")
    
    ***REMOVED*** 4. Concatenate encoder + decoder
    inference_df = pd.concat([encoder_df, decoder_df], ignore_index=True)
    
    ***REMOVED*** 5. Ensure required columns exist
    if 'plant_id' not in inference_df.columns:
        inference_df['plant_id'] = self.plant_id
    
    ***REMOVED*** Recompute time_idx
    inference_df = inference_df.sort_values('timestamp_utc').reset_index(drop=True)
    inference_df['time_idx'] = inference_df.groupby('plant_id').cumcount()
    
    ***REMOVED*** 6. Create TimeSeriesDataSet
    test_ds = TimeSeriesDataSet.from_dataset(
        self.long_train_ds,  ***REMOVED*** Training dataset (loaded in __init__)
        inference_df,
        predict=True,
        stop_randomization=True
    )
    
    ***REMOVED*** 7. Create single-sample DataLoader
    test_dl = test_ds.to_dataloader(
        train=False,
        batch_size=1,
        num_workers=0,
        shuffle=False
    )
    
    ***REMOVED*** 8. Run inference
    self.long_model.eval()
    with torch.no_grad():
        for x, y in test_dl:
            x = {k: v.to(self.device) if torch.is_tensor(v) else v 
                 for k, v in x.items()}
            
            output = self.long_model(x)
            
            predictions = extract_q50_prediction(
                output,
                self.long_model.loss.quantiles
            )
            
            ***REMOVED*** predictions shape: (1, 720) → squeeze to (720,)
            return predictions.squeeze()
    
    raise RuntimeError("No predictions generated")
```

---

***REMOVED******REMOVED******REMOVED*** **STEP 4: Update `__init__()` to Load Training Datasets** (20 min)

**Problem:** Need training datasets for `.from_dataset()` normalization inheritance

**Solution:**

```python
def __init__(
    self,
    short_ckpt: Path,
    long_ckpt: Path,
    plant_metadata: Path,
    short_train_parquet: Path,  ***REMOVED*** NEW
    long_train_parquet: Path,   ***REMOVED*** NEW
    device: Optional[str] = None
):
    ***REMOVED*** Existing init...
    self.pvlib_predictor = PVLibPredictor(plant_metadata)
    self.rl_controller = RLMetaController(mode="heuristic")
    
    ***REMOVED*** NEW: Load training datasets
    from src.inference.tft_utils import load_tft_config, create_training_dataset
    
    ***REMOVED*** Short-head training dataset
    short_config = load_tft_config(short_ckpt.parent.parent)
    short_train_df = pd.read_parquet(short_train_parquet)
    self.short_train_ds = create_training_dataset(
        short_train_df,
        short_config['roles'],
        encoder_len=96,
        pred_len=96
    )
    
    ***REMOVED*** Long-head training dataset
    long_config = load_tft_config(long_ckpt.parent.parent)
    long_train_df = pd.read_parquet(long_train_parquet)
    self.long_train_ds = create_training_dataset(
        long_train_df,
        long_config['roles'],
        encoder_len=168,
        pred_len=720
    )
    
    ***REMOVED*** Load models and weights...
    ***REMOVED*** (existing code)
```

---

***REMOVED******REMOVED******REMOVED*** **STEP 5: Testing Strategy** (30 min)

***REMOVED******REMOVED******REMOVED******REMOVED*** Test 1: Single Day Short-Head
```python
***REMOVED*** Test _predict_short_head_for_day() on Day 0
forecaster = PhysicsAwareForecaster(
    short_ckpt=Path("experiments/tft/.../shorthead/checkpoints/best_state_dict.pt"),
    long_ckpt=Path("experiments/tft/.../longhead/checkpoints/best_state_dict.pt"),
    plant_metadata=Path("data/metadata/germany/plant_03.json"),
    short_train_parquet=Path("data/processed/plant_level/plant_03/15min_pca32/train.parquet"),
    long_train_parquet=Path("data/processed/plant_level/plant_03/hourly_longhead/train.parquet")
)

***REMOVED*** Load test data
test_df = pd.read_parquet("data/processed/plant_level/plant_03/15min_pca32/test.parquet")
day0_start = pd.Timestamp("2023-10-12 00:00:00", tz="UTC")

pred_day0 = forecaster._predict_short_head_for_day(
    day_start=day0_start,
    day_idx=0,
    historical_df=test_df,  ***REMOVED*** Use historical portion
    weather_df=test_df      ***REMOVED*** Use forecast portion
)

print(f"Day 0 prediction: shape={pred_day0.shape}, range=[{pred_day0.min():.3f}, {pred_day0.max():.3f}]")

***REMOVED*** Compare vs offline_predict_tft.py output
offline_preds = pd.read_parquet("outputs/plant03_shorthead_test_preds.parquet")
offline_day0 = offline_preds[offline_preds['horizon'] <= 96].head(96)['y_hat_q50'].values

mae = np.abs(pred_day0 - offline_day0).mean()
print(f"MAE vs offline baseline: {mae:.6f}")  ***REMOVED*** Should be ~0.0 (identical)
```

***REMOVED******REMOVED******REMOVED******REMOVED*** Test 2: Long-Head
```python
pred_long = forecaster._predict_long_head(
    forecast_start=day0_start,
    historical_df=long_historical,
    weather_df=long_forecast
)

print(f"Long-head prediction: shape={pred_long.shape}, range=[{pred_long.min():.3f}, {pred_long.max():.3f}]")
```

***REMOVED******REMOVED******REMOVED******REMOVED*** Test 3: Full 30-Day Hierarchical
```python
result = forecaster.predict_30d(
    forecast_start=day0_start,
    weather_df=weather_forecast_30d,
    historical_df=historical_weather
)

assert result['final'].shape == (2880,)
assert (result['final'] >= 0).all()
assert (result['final'][pvlib < 0.01] < 0.01).all()  ***REMOVED*** Night check
```

---

***REMOVED******REMOVED*** ⚠️ Common Pitfalls to Avoid

***REMOVED******REMOVED******REMOVED*** 1. **Time Index Gaps**
```python
***REMOVED*** ❌ WRONG: Gaps in time_idx cause assertion errors
df['time_idx'] = range(len(df))  ***REMOVED*** May have gaps from filtering

***REMOVED*** ✅ CORRECT: Use cumcount per group
df['time_idx'] = df.groupby('plant_id').cumcount()
```

***REMOVED******REMOVED******REMOVED*** 2. **Prediction Length Mismatch**
```python
***REMOVED*** ❌ WRONG: Config says 96 but model loaded with 720
model = load_checkpoint(...)  ***REMOVED*** Trained on pred_len=720
dataset = TimeSeriesDataSet(..., max_prediction_length=96)  ***REMOVED*** Mismatch!

***REMOVED*** ✅ CORRECT: Match config exactly
encoder_len, pred_len = load_from_config(run_dir)
dataset = TimeSeriesDataSet(..., max_prediction_length=pred_len)
```

***REMOVED******REMOVED******REMOVED*** 3. **Normalization Inconsistency**
```python
***REMOVED*** ❌ WRONG: Fresh dataset (different normalization than training)
test_ds = TimeSeriesDataSet(test_df, ...)  ***REMOVED*** New normalizer!

***REMOVED*** ✅ CORRECT: Inherit from training dataset
test_ds = TimeSeriesDataSet.from_dataset(train_ds, test_df, predict=True)
```

***REMOVED******REMOVED******REMOVED*** 4. **Missing Columns**
```python
***REMOVED*** ❌ WRONG: Inference df missing features
inference_df = weather_df[['timestamp_utc', 'ghi', 'dni', 'dhi']]  ***REMOVED*** Missing pvlib_* columns!

***REMOVED*** ✅ CORRECT: Compute all training features
inference_df = add_pvlib_features(weather_df)  ***REMOVED*** Match training preprocessing
```

***REMOVED******REMOVED******REMOVED*** 5. **Quantile Index**
```python
***REMOVED*** ❌ WRONG: Assume q50 is at index 3
pred_q50 = output.prediction[:, :, 3]  ***REMOVED*** May not be q50!

***REMOVED*** ✅ CORRECT: Find q50 dynamically
q50_idx = model.loss.quantiles.index(0.5)
pred_q50 = output.prediction[:, :, q50_idx]
```

---

***REMOVED******REMOVED*** 📦 Dependencies Needed

```python
***REMOVED*** Already installed:
import torch
import pandas as pd
import numpy as np
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.data.encoders import GroupNormalizer

***REMOVED*** May need to verify:
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
```

---

***REMOVED******REMOVED*** 🎯 Success Criteria

1. ✅ **Short-head prediction matches offline baseline** (MAE < 1e-6)
2. ✅ **Long-head prediction shape correct** (720,)
3. ✅ **Full 30-day pipeline produces (2880,)** output
4. ✅ **All physics constraints enforced** (night=0, capacity≤120%)
5. ✅ **No assertion errors** from TimeSeriesDataSet
6. ✅ **Predictions are reasonable** (range [0, 1], daylight pattern)

---

***REMOVED******REMOVED*** 📊 Estimated Timeline

| Task | Time | Complexity |
|------|------|------------|
| Create tft_utils.py | 15 min | Low |
| Implement _predict_short_head_for_day() | 45 min | Medium |
| Implement _predict_long_head() | 45 min | Medium |
| Update __init__() | 20 min | Low |
| Test single day | 15 min | Low |
| Test full pipeline | 15 min | Medium |
| Debug & fixes | 20 min | Variable |
| **TOTAL** | **~2.5 hours** | |

---

***REMOVED******REMOVED*** 🚀 Ready to Start?

**Next action:** Create `src/inference/tft_utils.py` with utility functions, then implement the two prediction methods.

**Confidence:** High - we have:
- ✅ Working offline_predict_tft.py reference
- ✅ Test data with real predictions to validate against
- ✅ Clean hierarchical architecture ready
- ✅ All config files and checkpoints available
