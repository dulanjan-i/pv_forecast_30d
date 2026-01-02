# SAR SPACE COMPARISON: Original MiRACLE Design vs Current Implementation

## ORIGINAL MiRACLE DESIGN (From Thesis)

### Architecture
- **Local Agents (Role 1-3):** Q-learning for local optimization
- **Meta-Controller (Role 4):** DDQN for global coordination
- **Philosophy:** "Policy-over-policies" hierarchical RL

### Original Reward Function
```
R_t = w₁(−RMSE_t) + w₂(−Drift_t) + w₃(−ComputeCost_t) + w₄(−RetrainFreq_t)
```

### Original State Space (Inputs to Meta-Controller)

**Role 1 Advisory (LSTM Encoder):**
- Rolling RMSE, MAE, bias
- Physics deviation from PVLib
- Temporal feature quality

**Role 2 Advisory (PVLib Physics):**
- Time since last calibration update
- Forecast volatility
- Compute load

**Role 3 Advisory (TFT Forecaster):**
- Data quality scores
- Distribution drift metrics
- Flags for suspect periods (outliers, missing data)

**Meta-Controller State (Role 4):**
- Aggregated signals from Role 1-3
- RMSE trends across horizons
- Data drift statistics
- Retraining costs and frequency
- API agreement metrics
- Compute budget

### Original Action Space (Meta-Controller Only)

**A0:** Keep current configuration (maintain)  
**A1:** Switch to ensemble-style weather API route  
**A2:** Switch to direct NWP-based API route  
**A3:** Fine-tune TFT with bounded hyperparameter changes  
**A4:** Fine-tune LSTM encoder  
**A5:** Adjust selected PVLib parameters (inside safe ranges)  
**A6:** Trigger full horizon reforecast  
**A7:** Request hard retrain (human approval required)  

### Original Local Agent Actions (Rule-Based)
- Agents provide **advisory signals** (state features)
- No direct action selection by local agents
- Meta-controller decides all actions

---

## CURRENT IMPLEMENTATION (What I Coded - WRONG)

### Architecture Problem
- **3 Local Agents:** Each with DDQN (Short-TFT, Long-TFT, PVLib)
- **1 Meta-Agent:** DDQN for blending weights only
- **Issue:** 4 separate learning agents = overfitting risk

### Current State Space

**Local Agent 1 (Short-TFT): 15 dims**
1. short_rmse_1h, short_rmse_24h
2. short_confidence (prediction variance)
3. short_drift (input distribution shift)
4. hour_of_day, is_night, cloud_cover
5. retrain_count_24h, compute_budget
6. forecast_age_hours, weather_quality
7. data_drift_score, last_action
8. ensemble_rmse, short_long_mismatch

**Local Agent 2 (Long-TFT): 15 dims**
1. long_rmse_24h, long_rmse_7d, long_rmse_30d
2. long_confidence, long_drift
3. forecast_horizon, weather_api_used, api_agreement
4. retrain_count_24h, compute_budget, last_action
5. ensemble_rmse, short_long_mismatch
6. hour_of_day, season

**Local Agent 3 (PVLib): 10 dims**
1. physics_residual (TFT vs Physics error)
2. ghi, dni, temperature
3. is_night, cloud_cover
4. tilt_angle, azimuth
5. last_calibration, ensemble_rmse

**Meta-Agent: 25 dims**
1. ensemble_rmse, short_rmse_24h, long_rmse_24h
2. physics_residual, short_long_mismatch
3. data_drift_score, hour_of_day, is_night
4. forecast_horizon, season
5. weather_api_used, api_agreement
6. cloud_cover, weather_quality
7. current_weight_short, current_weight_long, current_weight_physics
8. last_meta_action, compute_budget
9. retrain_count_short_24h, retrain_count_long_24h
10. short_confidence, long_confidence
11. forecast_age_hours, ghi

### Current Action Space

**Local Agents (5 actions each - WRONG, should be rule-based):**
- A0: MAINTAIN (no changes)
- A1: FINE_TUNE (adjust hyperparams - automated)
- A2: SUGGEST_RETRAIN (request retrain - human confirms)
- A3: ROLLBACK (revert checkpoint)
- A4: DEFER (let others compensate)

**Meta-Agent (27 actions - TOO COMPLEX):**
- 27 discrete weight combinations: [0.1, 0.5, 0.9]³
- Only controls blend weights, not system actions

### Current Reward Function
```
R = w₁(−RMSE) + w₂(−Mismatch) + w₃(−Drift) + w₄(−Cost) + Bonus
w₁=1.0 (accuracy), w₂=0.3 (consistency), w₃=0.2 (stability), w₄=0.1 (efficiency)
```

**Problems:**
1. Local agents learn independently (overfitting risk)
2. Meta-agent only controls weights, not system actions
3. Weather API routing is missing from RL actions
4. PVLib treated as tunable model (should be base truth)

---

## PROPOSED REFACTOR (Back to Original Philosophy)

### Architecture (CORRECTED)
- **3 Rule-Based Advisors:** Short-TFT, Long-TFT, PVLib (no learning)
- **1 DDQN Meta-Controller:** Global coordination and action selection
- **Weather Router:** Rule-based (as decided by user)

### Proposed State Space (Meta-Controller Only)

**Total: ~35 dimensions** (aggregated from advisors)

**Short-TFT Advisory (10 dims):**
1. short_rmse_1h, short_rmse_24h
2. short_confidence, short_drift
3. forecast_age_hours, retrain_count_24h
4. last_fine_tune_success (binary)
5. hourly_rmse_trend (slope over last 7 days)
6. night_performance_gap (day vs night RMSE delta)
7. weather_quality

**Long-TFT Advisory (10 dims):**
1. long_rmse_24h, long_rmse_7d, long_rmse_30d
2. long_confidence, long_drift
3. forecast_horizon, api_agreement
4. retrain_count_24h
5. horizon_rmse_trend (degradation 1d→7d→30d)
6. api_switch_count_24h

**PVLib Advisory (8 dims):**
1. physics_residual (TFT ensemble vs PVLib)
2. ghi, dni, temperature
3. last_calibration_hours
4. calibration_drift (measured vs expected)
5. is_night, cloud_cover

**Meta-Controller Context (7 dims):**
1. ensemble_rmse
2. short_long_mismatch
3. data_drift_score (global)
4. compute_budget
5. hour_of_day, season
6. total_retrain_count_7d

### Proposed Action Space (Meta-Controller DDQN)

**8 discrete actions** (inspired by original A0-A7):

**A0: MAINTAIN**
- Keep current configuration
- No changes to any model
- Baseline action (cost: 0)

**A1: FINE_TUNE_SHORT_TFT**
- Adjust short-head learning rate (0.8x or 1.2x)
- Update dropout slightly
- Automated, no human approval (cost: 0.1)

**A2: FINE_TUNE_LONG_TFT**
- Adjust long-head learning rate
- Update attention weights
- Automated, no human approval (cost: 0.15)

**A3: RECALIBRATE_PVLIB**
- Update panel metadata (tilt, azimuth) if calibration drift high
- Adjust soiling/degradation factors
- Automated within safe ranges (cost: 0.05)

**A4: ADJUST_BLEND_WEIGHTS_HIGH_SHORT**
- Set weights: short=0.7, long=0.2, physics=0.1
- Use when short-term accuracy critical (cost: 0)

**A5: ADJUST_BLEND_WEIGHTS_HIGH_LONG**
- Set weights: short=0.2, long=0.7, physics=0.1
- Use when long-term planning needed (cost: 0)

**A6: ADJUST_BLEND_WEIGHTS_HIGH_PHYSICS**
- Set weights: short=0.2, long=0.2, physics=0.6
- Use when TFTs show high uncertainty/drift (cost: 0)

**A7: SUGGEST_RETRAIN**
- Flag both TFT heads for retrain
- Requires human approval (human-in-the-loop)
- Only when drift severe or performance collapse (cost: 1.0)

**Removed Actions (from original):**
- ~~A1/A2: Weather API switching~~ → Now rule-based (user decision)
- ~~A4: Fine-tune LSTM~~ → LSTM removed in ablations
- ~~A6: Trigger reforecast~~ → Implicit in every step

### Proposed Reward Function (Aligned with Original)

```
R_t = w₁(−RMSE_t) + w₂(−Drift_t) + w₃(−Cost_t) + w₄(−RetrainFreq_t)
```

**Component Breakdown:**

**w₁ = 1.0: Accuracy**
```
r_accuracy = (RMSE_prev - RMSE_current) / 0.01
Reward improvement, penalize degradation
```

**w₂ = 0.5: Drift Control**
```
r_drift = −(data_drift_score + short_long_mismatch) / 2
Penalize distribution shift and model disagreement
```

**w₃ = 0.2: Computational Cost**
```
action_costs = {
    A0: 0,      # maintain (free)
    A1: 0.1,    # fine_tune_short
    A2: 0.15,   # fine_tune_long
    A3: 0.05,   # recalibrate_pvlib
    A4-A6: 0,   # blend weight changes (free)
    A7: 1.0     # suggest_retrain (expensive)
}
r_cost = −action_costs[action_t]
```

**w₄ = 0.3: Retrain Frequency**
```
r_retrain = −retrain_count_7d / 10.0
Penalize excessive retraining requests
```

**Bonus:**
```
+0.1 if api_agreement > 0.9 (high weather consensus)
```

### Proposed Rule-Based Advisors (No Learning)

**Short-TFT Advisor:**
```python
def get_advisory_state():
    """Returns 10-dim state vector for meta-controller"""
    return np.array([
        compute_rmse(horizon=1),
        compute_rmse(horizon=24),
        compute_confidence(),
        compute_drift(),
        get_forecast_age_hours(),
        get_retrain_count_24h(),
        get_last_fine_tune_success(),
        compute_hourly_rmse_trend(),
        compute_night_performance_gap(),
        get_weather_quality()
    ])

def should_alert_meta_controller():
    """Rule-based alert for meta-controller attention"""
    if rmse_1h > 0.15 and drift_score > 0.5:
        return "high_rmse_and_drift"
    if night_performance_gap > 0.10:
        return "night_degradation"
    return None
```

**Long-TFT Advisor:**
```python
def get_advisory_state():
    """Returns 10-dim state vector"""
    return np.array([
        compute_rmse(horizon=24),
        compute_rmse(horizon=168),  # 7d
        compute_rmse(horizon=720),  # 30d
        compute_confidence(),
        compute_drift(),
        get_forecast_horizon(),
        get_api_agreement(),
        get_retrain_count_24h(),
        compute_horizon_rmse_trend(),
        get_api_switch_count_24h()
    ])

def should_alert_meta_controller():
    """Rule-based alert"""
    if (rmse_30d - rmse_24h) > 0.05:  # Degradation over horizon
        return "horizon_degradation"
    if api_agreement < 0.6:
        return "weather_api_disagreement"
    return None
```

**PVLib Advisor:**
```python
def get_advisory_state():
    """Returns 8-dim state vector"""
    return np.array([
        compute_physics_residual(),
        get_ghi(),
        get_dni(),
        get_temperature(),
        get_last_calibration_hours(),
        compute_calibration_drift(),
        is_night(),
        get_cloud_cover()
    ])

def should_alert_meta_controller():
    """Rule-based alert for calibration drift"""
    if physics_residual > 0.20 and last_calibration > 168:  # 1 week
        return "calibration_drift"
    if physics_residual > 0.30:
        return "severe_physics_mismatch"
    return None
```

---

## KEY DIFFERENCES SUMMARY

| Component | Original Design | Current (WRONG) | Proposed (FIXED) |
|-----------|----------------|-----------------|------------------|
| **Learning Agents** | 1 DDQN (Role 4) | 4 DDQN (all) | 1 DDQN (meta) |
| **Local Agents** | Rule-based advisors | DDQN learners | Rule-based advisors |
| **State Dim** | ~30-40 dims | 65 dims (fragmented) | ~35 dims (aggregated) |
| **Actions** | 8 system actions | 3,375 combinations | 8 system actions |
| **Meta Control** | Global coordination | Weight blending only | Global coordination |
| **PVLib Role** | Base truth + advisory | Tunable model | Base truth + advisory |
| **Weather Router** | RL action (A1/A2) | Rule-based | Rule-based (user decision) |
| **Overfitting Risk** | LOW (1 agent) | HIGH (4 agents) | LOW (1 agent) |
| **Training Time** | Fast (1 agent) | Slow (4 agents) | Fast (1 agent) |
| **Interpretability** | High (single policy) | Low (4 policies) | High (single policy) |

---

## AGENTS WITHOUT JOBS IN V1.0

**1. LSTM Encoder Agent (Role 1 in original):**
- **Original Job:** Fine-tune LSTM encoder, manage temporal embeddings
- **Current Status:** LSTM removed in ablations
- **V1.0 Solution:** Short-TFT advisor absorbs role (temporal feature monitoring)
- **V1.1 Option:** If LSTM reintroduced, create separate advisor

**2. Database/Cadence Manager (Role 5 in original thesis?):**
- **Original Job:** Manage data ingestion, caching, update schedules
- **Current Status:** Not implemented in V1.0
- **V1.0 Solution:** Human operator handles data updates manually
- **V1.1 Option:** Add database advisor (monitors data freshness, triggers fetches)

**Recommendation:** Keep 3 advisors (Short-TFT, Long-TFT, PVLib) for V1.0 simplicity.

---

## REFACTOR PLAN

### Step 1: Simplify LocalAgent (Rule-Based Only)
- Remove: `policy_net`, `target_net`, `optimizer`, `replay_buffer`
- Keep: `_heuristic_action()`, `get_advisory_state()`
- Add: `should_alert_meta_controller()` for critical conditions

### Step 2: Expand MetaAgent Actions
- Change from 27 weight combos → 8 system actions (A0-A7)
- Update `action_dim = 8`
- Map actions to system behaviors (fine-tune, recalibrate, blend, retrain)

### Step 3: Consolidate State Space
- Merge local states into single 35-dim meta state
- Advisors build state vectors, meta-controller aggregates

### Step 4: Update Reward Function
- Align with original: `R_t = w₁(−RMSE) + w₂(−Drift) + w₃(−Cost) + w₄(−RetrainFreq)`
- Adjust weights: w₁=1.0, w₂=0.5, w₃=0.2, w₄=0.3

### Step 5: Update Tests
- Test single DDQN meta-controller
- Test rule-based advisors
- Test action execution (fine-tune, recalibrate, blend, retrain)

**Estimated Time:** ~1 hour refactor + 30 min testing = 1.5 hours total

---

## NEXT STEPS

1. **User Approval:** Confirm proposed refactor matches original vision
2. **Refactor Code:** Implement 1 DDQN + 3 rule-based advisors
3. **Test Integration:** Validate with mock forecaster
4. **Wire Real Forecaster:** Connect to PhysicsAwareForecaster
5. **Experience Collection:** Run heuristic baseline (2k-5k episodes)
6. **Train DDQN:** Single agent training on 2xL4 GPUs (2-3 hours)
7. **Deploy V1.0:** Production inference with trained meta-controller
