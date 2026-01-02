# Physics-Aware Dual-Head Glue Code Architecture

## Overview Pipeline

**INPUT:** Weather forecast (next 30 days)

↓

**STEP 1** → Generate three prediction streams

↓

**STEP 2** → Physics-aware blending (single-call, NO rolling windows!)

↓

**STEP 3** → RL meta-controller adjusts weights

↓

**OUTPUT:** 2880 steps @ 15-min resolution (30 days)

---

## STEP 1: Generate Three Prediction Streams

### A. Short Head (ML)
- **Output:** 96 steps @ 15-min (24h)
- **Encoder:** 96 steps @ 15-min historical data
- **Uses:** Historical data + forecast weather (21 features)
- **Purpose:** Accurate near-term predictions (Day 1 only)

### B. Long Head (ML)
- **Output:** 720 steps @ 1-hour (30 days) ← **SINGLE INFERENCE CALL**
- **Encoder:** 168 hours (7 days) historical context
- **Uses:** Historical data + forecast weather (21 features)
- **Purpose:** Full 30-day strategic forecast in one pass

### C. PVLib (Physics)
- **Output:** 2880 steps @ 15-min (30 days)
- **Method:** Pure physics calculation (weather → DC power)
- **Purpose:** Physics-based baseline (ground truth constraint) + intra-hour shape

---

## STEP 2: Physics-Aware Blending (Simplified Architecture)

### Day 1 (first 24h = 96 steps @ 15-min):
1. Use **SHORT HEAD** (high accuracy, matches production time resolution)
2. Constrain with PVLib baseline
3. Blend formula: `day1 = α₁ × short_ml + (1-α₁) × pvlib`
   - α₁ ∈ [0.5, 1.0] (controlled by RL)

### Days 2-30 (remaining 696h = 2784 steps @ 15-min):
1. Use **LONG HEAD** single call → 720 steps @ 1-hour
2. **Upsample** 1-hour → 15-min using PVLib intra-hour shape:
   - For each hour prediction, distribute into 4×15-min intervals
   - Weight distribution by PVLib clear-sky curve (preserves physics-informed shape)
   - Result: 720 hours × 4 = 2880 quarter-hours
3. Blend with PVLib:
   - `days2_30 = α₂ × long_upsampled + (1-α₂) × pvlib`
   - α₂ ∈ [0.3, 0.8] (controlled by RL)
4. Take first 2784 steps (discard overlap with Day 1)
### Final Assembly:
- Concatenate: `[day1 (96 steps), days2_30 (2784 steps)]` = **2880 steps @ 15-min**
- Apply physics constraints globally:
  - If any prediction > PVLib × 1.2 → cap at PVLib × 1.2
  - If PVLib = 0 (night) → force prediction = 0
  - Optional: reconcile daily energy totals

**Key Advantage:** NO rolling windows! Single long-head inference covers full 30 days.

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
| Day 1 source | Short head | Highest accuracy for near-term (RMSE 0.087) |
| Days 2-30 source | Long head single call | 720-step model trained for full horizon |
| Rolling windows? | **NO** | Long head covers 30 days natively |
| Upsampling | PVLib-weighted distribution | Physics-informed intra-hour shape |
| Blend ranges | α₁: [0.5,1.0], α₂: [0.3,0.8] | Never fully ignore ML or physics |
| RL frequency | Per inference call | Adapt to current weather regime |
| Physics role | Constraint + blend + shape | Multi-purpose baseline |

---

## Implementation Notes

### Model Checkpoints:
- Short head: `experiments/tft/runs/germany/plant_03/15min/pvlib_warmstart_from_global_noleak/20251229_151100/checkpoints/best.ckpt`
- Long head: `experiments/tft/runs/germany/plant_03/longhead/hourly720/warm/lr8e-4_do0.15_bs64_acc8_seed43/20251231_104405/checkpoints/best.ckpt`

### Validation Results:
- Short head: RMSE 0.087 (test, 24h horizon, 4,606 windows)
- Long head: RMSE 0.076 (test, 720h horizon, 313 windows)
- Long head error by day: Day 1 (0.079) → Day 15 (0.093) → Day 30 (0.059)
- Both models use 21 known future covariates (weather + PVLib features)

### Test Set Coverage:
- Short head: 442,176 predictions covering Oct 12 - Nov 30, 2023
- Long head: 225,360 predictions (313 windows × 720 horizons) covering Oct 16 - Nov 30, 2023
- Validated with proper sliding windows (predict=False mode)

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
