# Physics-Aware Dual-Head Glue Code Architecture

## Overview Pipeline

**INPUT:** Weather forecast (next 30 days)

↓

**STEP 1** → Generate three prediction streams

↓

**STEP 2** → Physics-aware blending (rolling window)

↓

**STEP 3** → RL meta-controller adjusts weights

↓

**OUTPUT:** 2880 steps @ 15-min resolution (30 days)

---

## STEP 1: Generate Three Prediction Streams

### A. Short Head (ML)
- **Output:** 96 steps @ 15-min (24h)
- **Uses:** Historical data + forecast weather (21 features)
- **Purpose:** Accurate near-term predictions

### B. Long Head (ML)
- **Output:** 96 steps @ 1-hour (96h = 4 days)
- **Uses:** Historical data + forecast weather (21 features)
- **Purpose:** Coarse strategic predictions

### C. PVLib (Physics)
- **Output:** 2880 steps @ 15-min (30 days)
- **Method:** Pure physics calculation (weather → DC power)
- **Purpose:** Physics-based baseline (ground truth constraint)

---

## STEP 2: Physics-Aware Blending (Rolling Window)

### For each day in 30-day horizon:

#### Day 1 (first 24h):
1. Use **SHORT HEAD** (high accuracy, 15-min resolution)
2. Constrain with PVLib baseline
3. Blend formula: blend = α₁ × short_ml + (1-α₁) × pvlib

#### Days 2-30:
1. Use **LONG HEAD** (strategic, 1-hour resolution)
2. Constrain with PVLib baseline
3. Blend formula: blend = α₂ × long_ml + (1-α₂) × pvlib
4. **Upsample:** 1-hour → 15-min (cubic interpolation)

### Physics Constraints Applied:
- If ML predicts > PVLib max → cap at PVLib × 1.2
- If ML predicts < 0 → clip to 0
- Ensure daily sum consistency (optional reconciliation)

---

## STEP 3: RL Meta-Controller (Adjusts Blend Weights)

### State (what RL observes):
- Recent forecast error (RMSE last 7 days)
- Weather stability (variance of forecasts)
- API health (latency, missing data flags)
- Compute budget remaining

### Action (what RL controls):
- α₁: Short-head blend weight, range [0.5, 1.0]
- α₂: Long-head blend weight, range [0.3, 0.8]
- quantile: Which quantile to use, range [0.25, 0.75]

### Reward:
reward = -RMSE(forecast, actual) - λ × compute_cost

### Learning Algorithm:
- Q-learning or DQN to optimize blend weights
- Update policy every 24h based on actual outcomes
- Store experiences in replay buffer

---

## Key Design Decisions

| Aspect | Choice | Rationale |
|--------|--------|-----------|
| Day 1 source | Short head | Highest accuracy for near-term |
| Day 2-30 source | Long head | Strategic horizon coverage |
| Physics role | Constraint + blend | Prevents unrealistic predictions |
| Upsampling | Cubic interpolation | Smooth transitions, preserves trends |
| RL frequency | Daily updates | Balance learning vs stability |
| Blend ranges | α₁: [0.5,1.0], α₂: [0.3,0.8] | Never fully ignore ML or physics |

---

## Implementation Notes

### Model Checkpoints:
- Short head: experiments/tft/runs/germany/plant_03/15min/pvlib_warmstart_from_global_noleak/20251229_151100/checkpoints/best.ckpt
- Long head: experiments/tft/runs/germany/plant_03/longhead/hourly720/warm/lr8e-4_do0.15_bs64_acc8_seed43/20251231_104405/checkpoints/best.ckpt

### Validation Results:
- Short head test RMSE: 0.087
- Long head test RMSE: 0.072
- Both models use 21 known future covariates (weather + PVLib features)

### Test Set Coverage:
- Short head: 4,606 prediction windows covering Oct 12 - Nov 30, 2023
- Long head: 1,009 prediction windows covering Oct 16 - Nov 30, 2023

---

## Next Implementation Steps

Day 1 Remaining (4 hours):
- Implement _pvlib_predict() function
- Test PVLib on historical weather data

Day 2 (10 hours):
- Create PhysicsAwareForecaster class
- Implement blending logic with rolling windows
- Add upsampling (hourly → 15-min)
- Test end-to-end with dummy RL (fixed α weights)

Day 3 (10 hours):
- Design RL state/action/reward interface
- Implement mock RL controller (heuristic rules)
- Integration test with weather API
- Documentation + demo script
