# MiRACLE V1.0 - State-Action-Reward (SAR) Space

**Date:** 2026-01-02  
**Architecture:** 1 DDQN Meta-Controller + 3 Rule-Based Advisors  
**Total State Dimensions:** 35  
**Total Actions:** 8 discrete system actions  

---

## 🏗️ Architecture Overview

```
┌────────────────────────────────────────────────────────────┐
│         META-CONTROLLER (DDQN - LEARNS POLICY)             │
│  • State: 35 dimensions (aggregated from advisors)        │
│  • Actions: 8 system actions                              │
│  • Reward: Multi-objective (accuracy+drift+cost+retrain)  │
│  • Network: 3-layer MLP (35 → 256 → 256 → 8)             │
└───────────┬────────────────────────────────────────────────┘
            │
   ┌────────┴────────┬─────────────────┬──────────────────┐
   │                 │                 │                  │
┌──▼──────────┐  ┌──▼──────────┐  ┌──▼──────────┐
│ SHORT-TFT   │  │ LONG-TFT    │  │  PVLIB      │
│  ADVISOR    │  │  ADVISOR    │  │  ADVISOR    │
│             │  │             │  │             │
│ Rule-Based  │  │ Rule-Based  │  │ Rule-Based  │
│ (NO LEARN)  │  │ (NO LEARN)  │  │ (NO LEARN)  │
│             │  │             │  │             │
│ 10-dim      │  │ 10-dim      │  │  8-dim      │
│ State       │  │ State       │  │  State      │
└─────────────┘  └─────────────┘  └─────────────┘
     │                │                │
     └────────────────┴────────────────┘
              Provide state signals
              to meta-controller
```

**Key Design Choice:** Only the meta-controller learns via DDQN. Advisors are rule-based monitors that build state vectors, eliminating overfitting risk with limited data.

---

## 📊 STATE SPACE (S) - 35 Dimensions

State is built by aggregating advisor observations + meta-context.

```python
# Code Location: src/rl/rl_meta_controller.py, line ~720
def build_meta_state(self, metrics: Dict) -> np.ndarray:
    """
    Layout: [short_advisory(10), long_advisory(10), pvlib_advisory(8), context(7)]
    """
    short_state = self.advisor_short_tft.get_advisory_state(metrics)
    long_state = self.advisor_long_tft.get_advisory_state(metrics)
    pvlib_state = self.advisor_pvlib.get_advisory_state(metrics)
    meta_context = ...  # 7 dims
    
    return np.concatenate([short_state, long_state, pvlib_state, meta_context])
```

### SHORT-TFT ADVISOR (10 dimensions)

**Purpose:** Monitor short-term forecast quality (1-24h horizon)

| # | Feature | Range | Normalization | Code Reference |
|---|---------|-------|---------------|----------------|
| 1 | `short_rmse_1h` | [0.0, 0.3] | Raw kW | Line 235 |
| 2 | `short_rmse_24h` | [0.0, 0.3] | Raw kW | Line 236 |
| 3 | `short_confidence` | [0.0, 1.0] | Inverse variance | Line 237 |
| 4 | `short_drift` | [0.0, 1.0] | KL divergence | Line 238 |
| 5 | `forecast_age_hours` | [0, 24] | / 24.0 | Line 239 |
| 6 | `retrain_count_24h` | [0, 10+] | / 10.0 | Line 240 |
| 7 | `last_fine_tune_success` | {0, 1} | Boolean | Line 241 |
| 8 | `hourly_rmse_trend` | [-0.1, 0.1] | Polyfit slope | Line 242 |
| 9 | `night_performance_gap` | [0.0, 0.2] | Day-night RMSE Δ | Line 243 |
| 10 | `weather_quality` | [0.0, 1.0] | API reliability | Line 244 |

**Code Location:**
```python
# src/rl/rl_meta_controller.py, lines 205-245
class LocalAdvisor:
    def _build_short_tft_state(self, metrics: Dict) -> np.ndarray:
        rmse_1h = metrics.get('short_rmse_1h', 0.0)
        rmse_24h = metrics.get('short_rmse_24h', 0.0)
        
        # Compute trend (slope of last 20 samples)
        rmse_trend = np.polyfit(x, recent, 1)[0] if len(history) >= 20 else 0.0
        
        return np.array([rmse_1h, rmse_24h, confidence, drift, age/24, ...])
```

### LONG-TFT ADVISOR (10 dimensions)

**Purpose:** Monitor long-term forecast quality (1-30 day horizon)

| # | Feature | Range | Normalization | Code Reference |
|---|---------|-------|---------------|----------------|
| 11 | `long_rmse_24h` | [0.0, 0.4] | Raw kW | Line 251 |
| 12 | `long_rmse_7d` | [0.0, 0.5] | Raw kW | Line 252 |
| 13 | `long_rmse_30d` | [0.0, 0.6] | Raw kW | Line 253 |
| 14 | `long_confidence` | [0.0, 1.0] | Inverse variance | Line 254 |
| 15 | `long_drift` | [0.0, 1.0] | KL divergence | Line 255 |
| 16 | `forecast_horizon` | [1, 30] | / 30.0 | Line 256 |
| 17 | `api_agreement` | [0.0, 1.0] | Weather API consensus | Line 257 |
| 18 | `retrain_count_24h` | [0, 10+] | / 10.0 | Line 258 |
| 19 | `horizon_rmse_trend` | [-0.2, 0.2] | RMSE degradation | Line 259 |
| 20 | `api_switch_count_24h` | [0, 10+] | / 10.0 | Line 260 |

**Code Location:**
```python
# src/rl/rl_meta_controller.py, lines 248-263
def _build_long_tft_state(self, metrics: Dict) -> np.ndarray:
    rmse_24h = metrics.get('long_rmse_24h', 0.0)
    rmse_7d = metrics.get('long_rmse_7d', 0.0)
    rmse_30d = metrics.get('long_rmse_30d', 0.0)
    
    # Horizon degradation: how much worse is 30d vs 24h?
    horizon_rmse_trend = (rmse_30d - rmse_24h) if rmse_24h > 0 else 0.0
    
    return np.array([rmse_24h, rmse_7d, rmse_30d, confidence, drift, ...])
```

### PVLIB ADVISOR (8 dimensions)

**Purpose:** Monitor physics-based model alignment and calibration

| # | Feature | Range | Normalization | Code Reference |
|---|---------|-------|---------------|----------------|
| 21 | `physics_residual` | [0.0, 0.5] | TFT-PVLib RMSE | Line 268 |
| 22 | `ghi` | [0, 1500] | / 1500.0 | Line 269 |
| 23 | `dni` | [0, 1200] | / 1200.0 | Line 270 |
| 24 | `temperature` | [-20, 60] | / 60.0 | Line 271 |
| 25 | `last_calibration_hours` | [0, 168+] | / 168.0 (week) | Line 272 |
| 26 | `calibration_drift` | [0.0, 0.3] | Measured-expected Δ | Line 273 |
| 27 | `is_night` | {0, 1} | Boolean | Line 274 |
| 28 | `cloud_cover` | [0, 100] | / 100.0 | Line 275 |

**Code Location:**
```python
# src/rl/rl_meta_controller.py, lines 266-277
def _build_pvlib_state(self, metrics: Dict) -> np.ndarray:
    return np.array([
        metrics.get('physics_residual', 0.0),
        metrics.get('ghi', 0.0) / 1500.0,
        metrics.get('dni', 0.0) / 1200.0,
        metrics.get('temperature', 20.0) / 60.0,
        metrics.get('last_calibration_hours', 0) / 168.0,
        metrics.get('calibration_drift', 0.0),
        1.0 if metrics.get('is_night', False) else 0.0,
        metrics.get('cloud_cover', 0.0) / 100.0
    ])
```

### META-CONTEXT (7 dimensions)

**Purpose:** Global system state for meta-controller decision-making

| # | Feature | Range | Normalization | Code Reference |
|---|---------|-------|---------------|----------------|
| 29 | `ensemble_rmse` | [0.0, 0.5] | Combined RMSE | Line 737 |
| 30 | `short_long_mismatch` | [0.0, 0.3] | Head disagreement | Line 738 |
| 31 | `data_drift_score` | [0.0, 1.0] | Global KL div | Line 739 |
| 32 | `compute_budget` | [0.0, 1.0] | Available resources | Line 740 |
| 33 | `hour_of_day` | [0, 23] | / 24.0 | Line 741 |
| 34 | `season` | {0, 1, 2, 3} | / 3.0 | Line 742 |
| 35 | `total_retrain_count_7d` | [0, 10+] | / 10.0 | Line 743 |

**Code Location:**
```python
# src/rl/rl_meta_controller.py, lines 736-745
meta_context = np.array([
    metrics.get('ensemble_rmse', 0.0),
    metrics.get('short_long_mismatch', 0.0),
    metrics.get('data_drift_score', 0.0),
    metrics.get('compute_budget', 1.0),
    metrics.get('hour_of_day', 12) / 24.0,
    metrics.get('season', 0) / 3.0,
    metrics.get('total_retrain_count_7d', 0) / 10.0
])
```

---

## ⚡ ACTION SPACE (A) - 8 Discrete Actions

Only the meta-controller selects actions. Advisors do not act.

```python
# Code Location: src/rl/rl_meta_controller.py, lines 380-402
class MetaController:
    ACTION_MAINTAIN = 0
    ACTION_FINE_TUNE_SHORT = 1
    ACTION_FINE_TUNE_LONG = 2
    ACTION_RECALIBRATE_PVLIB = 3
    ACTION_BLEND_HIGH_SHORT = 4
    ACTION_BLEND_HIGH_LONG = 5
    ACTION_BLEND_HIGH_PHYSICS = 6
    ACTION_SUGGEST_RETRAIN = 7
```

### A0: MAINTAIN

**Purpose:** No changes, keep current configuration  
**Cost:** 0.0 (free)  
**When Used:** System performing well, no drift detected  
**Heuristic Rule:** Default action when all metrics nominal  

**Code Location:**
```python
# src/rl/rl_meta_controller.py, line 537
if everything_nominal:
    return self.ACTION_MAINTAIN
```

### A1: FINE_TUNE_SHORT_TFT

**Purpose:** Adjust short-head TFT hyperparameters (automated)  
**Cost:** 0.1  
**Parameters Adjusted:**
- Learning rate: ×0.8 or ×1.2  
- Dropout: ±0.05  

**When Used:** Short-term RMSE elevated, drift detected  
**Heuristic Rule:** `short_rmse_1h > 0.10 AND short_drift > 0.5`  

**Code Location:**
```python
# src/rl/rl_meta_controller.py, line 549
if short_rmse_1h > 0.10 and short_drift > 0.5:
    return self.ACTION_FINE_TUNE_SHORT
```

### A2: FINE_TUNE_LONG_TFT

**Purpose:** Adjust long-head TFT hyperparameters (automated)  
**Cost:** 0.15  
**Parameters Adjusted:**
- Learning rate: ×0.8 or ×1.2  
- Attention weights: Re-normalize  

**When Used:** Long-term horizon degradation detected  
**Heuristic Rule:** `horizon_degradation > 0.05 OR long_rmse_30d > 0.12`  

**Code Location:**
```python
# src/rl/rl_meta_controller.py, line 553
if horizon_degradation > 0.05 or long_rmse_30d > 0.12:
    return self.ACTION_FINE_TUNE_LONG
```

### A3: RECALIBRATE_PVLIB

**Purpose:** Update panel metadata (automated within safe ranges)  
**Cost:** 0.05  
**Parameters Adjusted:**
- Tilt angle: ±5°  
- Azimuth: ±10°  
- Soiling factor: ±0.05  
- Degradation rate: ±0.005  

**When Used:** Physics residual high, calibration drift detected  
**Heuristic Rule:** `physics_residual > 0.25`  

**Code Location:**
```python
# src/rl/rl_meta_controller.py, line 542
if physics_residual > 0.25:
    return self.ACTION_RECALIBRATE_PVLIB
```

### A4: ADJUST_BLEND_WEIGHTS_HIGH_SHORT

**Purpose:** Favor short-term TFT in ensemble  
**Cost:** 0.0 (free)  
**Weights:** `short=0.7, long=0.2, physics=0.1`  

**When Used:** Short-term accuracy critical, high drift (short-term adapts faster)  
**Heuristic Rule:** `data_drift_global > 0.6`  

**Code Location:**
```python
# src/rl/rl_meta_controller.py, lines 420-424
BLEND_PRESETS = {
    4: {'short': 0.7, 'long': 0.2, 'physics': 0.1},  # High short
    ...
}
```

### A5: ADJUST_BLEND_WEIGHTS_HIGH_LONG

**Purpose:** Favor long-term TFT in ensemble  
**Cost:** 0.0 (free)  
**Weights:** `short=0.2, long=0.7, physics=0.1`  

**When Used:** Planning horizon > 7 days, stable conditions  
**Heuristic Rule:** Rarely used in heuristic mode (RL learns when)  

### A6: ADJUST_BLEND_WEIGHTS_HIGH_PHYSICS

**Purpose:** Favor PVLib physics model in ensemble  
**Cost:** 0.0 (free)  
**Weights:** `short=0.2, long=0.2, physics=0.6`  

**When Used:** Nighttime, high TFT uncertainty, physics residual low  
**Heuristic Rule:** `is_night AND physics_residual < 0.15`  

**Code Location:**
```python
# src/rl/rl_meta_controller.py, line 538
if is_night > 0.5 and physics_residual < 0.15:
    return self.ACTION_BLEND_HIGH_PHYSICS
```

### A7: SUGGEST_RETRAIN

**Purpose:** Request full TFT retrain (human approval required)  
**Cost:** 1.0 (expensive)  
**When Used:** Severe performance collapse, excessive drift  
**Heuristic Rule:** `short_rmse_1h > 0.15 AND total_retrain_count < 2`  

**Human-in-the-Loop Workflow:**
```python
# src/rl/rl_meta_controller.py, lines 780-790
if action == self.ACTION_SUGGEST_RETRAIN:
    self.retrain_queue.append({
        'timestamp': now,
        'reason': f"RMSE={rmse:.3f}, drift={drift:.2f}",
        'state': current_state
    })
    logger.warning("Retrain suggested - awaiting human confirmation")
    # Human reviews queue and approves/rejects
```

**Code Location:**
```python
# src/rl/rl_meta_controller.py, line 545
if short_rmse_1h > 0.15 and total_retrain_count < 2:
    return self.ACTION_SUGGEST_RETRAIN
```

---

## 🎁 REWARD FUNCTION (R)

Aligned with original MiRACLE paper formulation.

```python
R_t = w₁(−RMSE_t) + w₂(−Drift_t) + w₃(−Cost_t) + w₄(−RetrainFreq_t) + Bonus
```

**Code Location:** `src/rl/rl_meta_controller.py`, lines 805-835

### Component Breakdown

#### 1. Accuracy (w₁ = 1.0)

```python
# lines 815-818
rmse_prev = metrics.get('ensemble_rmse', 0.0)
rmse_next = metrics_next.get('ensemble_rmse', 0.0)
r_accuracy = 1.0 * (rmse_prev - rmse_next) / 0.01  # Normalize by 10W
```

**Interpretation:** Reward RMSE improvement, penalize degradation. Normalized by 10W (0.01 kW) for numerical stability.

#### 2. Drift Control (w₂ = 0.5)

```python
# lines 821-823
drift_score = metrics_next.get('data_drift_score', 0.0)
short_long_mismatch = metrics_next.get('short_long_mismatch', 0.0)
r_drift = -0.5 * (drift_score + short_long_mismatch) / 2.0
```

**Interpretation:** Penalize distribution shift AND model disagreement. Higher weight than original (0.5 vs paper's w₂) because drift is critical indicator.

#### 3. Computational Cost (w₃ = 0.2)

```python
# lines 826-828
action_cost = self.meta_controller.ACTION_COSTS.get(action, 0.0)
r_cost = -0.2 * action_cost
```

**Action Costs:**
- Maintain: 0.0
- Fine-tune short: 0.1
- Fine-tune long: 0.15
- Recalibrate PVLib: 0.05
- Blend adjustments: 0.0
- Suggest retrain: 1.0

#### 4. Retrain Frequency (w₄ = 0.3)

```python
# lines 831-832
retrain_count = metrics_next.get('total_retrain_count_7d', 0)
r_retrain = -0.3 * retrain_count / 10.0
```

**Interpretation:** Penalize excessive retraining requests. Normalized by 10 retrains/week.

#### 5. Bonus: API Agreement

```python
# lines 835-836
api_agreement = metrics_next.get('api_agreement', 1.0)
bonus = 0.1 if api_agreement > 0.9 else 0.0
```

**Interpretation:** Small reward when weather APIs agree (high confidence).

### Reward Range

**Typical Range:** `[-2.0, +2.0]`  
**Best Case:** RMSE improvement + high API agreement = +2.1  
**Worst Case:** RMSE degradation + high drift + retrain request = -2.0  

---

## 🧠 Learning Algorithm (DDQN)

**Code Location:** `src/rl/rl_meta_controller.py`, lines 570-622

### Hyperparameters (from MiRACLE paper)

```python
# src/rl/rl_meta_controller.py, lines 25-40
class RLConfig:
    learning_rate: float = 1e-4
    gamma: float = 0.95
    batch_size: int = 64
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay: int = 10000  # steps
    buffer_capacity: int = 20000
    target_update_freq: int = 1  # Soft update every step (smooth weather)
    tau: float = 0.005  # Soft update (0.5% per step)
    alpha: float = 0.6  # Prioritized replay
    beta_start: float = 0.4
    beta_frames: int = 100000
```

### Q-Network Architecture

```python
# src/rl/rl_meta_controller.py, lines 124-138
class DQN(nn.Module):
    def __init__(self, state_dim=35, action_dim=8, hidden_dim=256):
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # Q-values for each action
```

### DDQN Update Rule

```python
# src/rl/rl_meta_controller.py, lines 586-610
def update(self) -> float:
    # Sample batch with prioritization
    transitions, indices, weights = self.replay_buffer.sample(batch_size, beta)
    
    # Compute Q(s,a) from policy network
    q_values = self.policy_net(state_batch).gather(1, action_batch)
    
    # Compute target: r + γ * Q_target(s', argmax_a' Q_policy(s', a'))
    with torch.no_grad():
        next_q_values = self.target_net(next_state_batch).max(1)[0]
        target_q_values = reward_batch + gamma * next_q_values
    
    # TD-errors for priority update
    td_errors = abs(q_values - target_q_values)
    self.replay_buffer.update_priorities(indices, td_errors)
    
    # Weighted MSE loss (importance sampling)
    loss = (weights_batch * (q_values - target_q_values)**2).mean()
    
    # Gradient descent
    self.optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
    self.optimizer.step()
    
    # Soft target update every step for smooth weather tracking
    # θ_target = τ*θ_policy + (1-τ)*θ_target
    if steps % target_update_freq == 0:  # target_update_freq = 1
        soft_update_target()
```

---

## 📍 Code Map Summary

| Component | File | Lines | Description |
|-----------|------|-------|-------------|
| **RLConfig** | `rl_meta_controller.py` | 25-40 | Hyperparameter dataclass |
| **DQN Network** | `rl_meta_controller.py` | 124-138 | 3-layer MLP Q-function |
| **PrioritizedReplayBuffer** | `rl_meta_controller.py` | 147-195 | TD-error prioritized sampling |
| **LocalAdvisor** | `rl_meta_controller.py` | 205-328 | Rule-based state builders |
| **MetaController** | `rl_meta_controller.py` | 366-658 | DDQN agent (8 actions) |
| **RLMetaControllerSystem** | `rl_meta_controller.py` | 666-910 | Main coordinator |
| **build_meta_state()** | `rl_meta_controller.py` | 720-745 | 35-dim state aggregation |
| **step()** | `rl_meta_controller.py` | 747-780 | One control step |
| **compute_reward()** | `rl_meta_controller.py` | 805-835 | Multi-objective reward |
| **update()** | `rl_meta_controller.py` | 837-850 | DDQN learning |

---

## 🎯 Usage Example

```python
from src.rl import RLMetaControllerSystem, RLConfig

# Initialize system
config = RLConfig(mode="heuristic")  # Start with rule-based
rl_system = RLMetaControllerSystem(config=config)

# Collect metrics from forecaster
metrics = {
    'short_rmse_1h': 0.08,
    'short_rmse_24h': 0.12,
    'long_rmse_24h': 0.15,
    'long_rmse_7d': 0.18,
    'long_rmse_30d': 0.22,
    'physics_residual': 0.10,
    'ensemble_rmse': 0.11,
    'short_long_mismatch': 0.03,
    'data_drift_score': 0.4,
    'is_night': False,
    'hour_of_day': 14,
    'season': 2,
    # ... (35 features total)
}

# Meta-controller selects action
action_info = rl_system.step(metrics)

print(f"Action: {action_info['action_name']}")
print(f"Blend Weights: {action_info['blend_weights']}")
print(f"Advisor Alerts: {action_info['advisor_alerts']}")

# After forecasting, update with reward
metrics_next = {...}  # New metrics after action
rl_system.update(metrics_next, done=False)

# Switch to learned policy after training
config.mode = "rl"
rl_system.load_checkpoint(Path("checkpoints/rl/meta_controller.pt"))
```

---

## 📚 References

1. Original MiRACLE Paper: Hierarchical RL formulation (R_t = w₁(−RMSE) + w₂(−Drift) + w₃(−Cost) + w₄(−RetrainFreq))
2. DDQN Paper: van Hasselt et al., "Deep Reinforcement Learning with Double Q-learning" (2015)
3. Prioritized Experience Replay: Schaul et al., "Prioritized Experience Replay" (2015)
4. Soft Target Updates: Lillicrap et al., "Continuous control with deep reinforcement learning" (2015)

---

**Last Updated:** 2026-01-02  
**Version:** 1.0 (Refactored to 1 DDQN + 3 Advisors)  
**Author:** MiRACLE Team
