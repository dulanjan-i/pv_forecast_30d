# MiRACLE: A Hierarchical Reinforcement Learning Framework for Multi-Horizon Photovoltaic Power Forecasting

**Scientific Workflow Documentation**

**Date:** January 2, 2026  
**Version:** 1.0  
**Framework:** MiRACLE (Multi-Resolution Intelligent Renewable Adaptive Computational Learning Engine)

---

## Abstract

This document presents the complete scientific workflow for MiRACLE, a hierarchical deep learning framework that combines dual-resolution Temporal Fusion Transformers (TFT), physics-based modeling (PVLib), and a Double Deep Q-Network (DDQN) meta-controller for adaptive ensemble forecasting of photovoltaic power output across horizons ranging from 1 hour to 30 days. The system employs a policy-over-policies architecture where three rule-based advisors monitor subsystem performance while a single DDQN agent learns optimal control policies for model management and ensemble weighting.

---

## 1. System Architecture

### 1.1 Hierarchical Design

```
┌───────────────────────────────────────────────────────────────────┐
│                    MIRACLE SYSTEM ARCHITECTURE                    │
└───────────────────────────────────────────────────────────────────┘

                        ┌─────────────────────┐
                        │  RL Meta-Controller │
                        │  (DDQN Policy)      │
                        │  • State: 35 dims   │
                        │  • Actions: 8       │
                        └──────────┬──────────┘
                                   │
                   ┌───────────────┼───────────────┐
                   │               │               │
         ┌─────────▼────────┐ ┌───▼───────┐ ┌────▼─────────┐
         │  Short-TFT       │ │ Long-TFT  │ │   PVLib      │
         │  Advisor         │ │ Advisor   │ │   Advisor    │
         │  (Rule-Based)    │ │ (Rule)    │ │   (Rule)     │
         └─────────┬────────┘ └───┬───────┘ └────┬─────────┘
                   │              │               │
         ┌─────────▼────────┐ ┌───▼────────┐ ┌───▼──────────┐
         │  Short-Head TFT  │ │ Long-Head  │ │ PVLib        │
         │  • 96 steps      │ │ TFT        │ │ Physics      │
         │  • 15-min res    │ │ • 720 step │ │ Deterministic│
         │  • 24h horizon   │ │ • 1h res   │ │ First-Prin.  │
         └─────────┬────────┘ └───┬────────┘ └───┬──────────┘
                   │              │               │
                   └──────────────┴───────────────┘
                                  │
                        ┌─────────▼─────────┐
                        │  Ensemble Blend   │
                        │  (RL-Controlled)  │
                        └───────────────────┘
                                  │
                        ┌─────────▼─────────┐
                        │  Final Forecast   │
                        │  [1h → 30d]       │
                        └───────────────────┘

┌───────────────────────────────────────────────────────────────────┐
│                    WEATHER DATA PIPELINE                          │
└───────────────────────────────────────────────────────────────────┘

         ┌──────────────────────────────────────┐
         │   Multi-API Weather Router           │
         │   (Rule-Based Selection)             │
         └──────────┬───────────────────────────┘
                    │
         ┌──────────┼──────────┬────────────────┐
         │          │          │                │
    ┌────▼─────┐ ┌─▼────────┐ ┌▼──────────┐   │
    │Forecast  │ │  ECMWF   │ │    GFS    │   │
    │API       │ │  ERA5    │ │  OpenMeteo│   │
    │(0-7d)    │ │  (8-15d) │ │  (backup) │   │
    └──────────┘ └──────────┘ └───────────┘   │
         │          │          │                │
         └──────────┴──────────┴────────────────┘
                    │
         ┌──────────▼──────────────┐
         │  Weather Preprocessing  │
         │  • Dual resolution      │
         │  • 15-min / 1-hour      │
         │  • Feature engineering  │
         └─────────────────────────┘
```

### 1.2 Component Specifications

| Component | Type | Input Dim | Output Dim | Parameters | Training |
|-----------|------|-----------|------------|------------|----------|
| Short-Head TFT | Transformer | 96×20 | 96 | 1.2M | Supervised |
| Long-Head TFT | Transformer | 720×20 | 360 | 1.8M | Supervised |
| PVLib Model | Physics | Solar geometry | P_DC | 0 (deterministic) | N/A |
| Short Advisor | Rule-Based | Metrics | 10-dim state | 0 | N/A |
| Long Advisor | Rule-Based | Metrics | 10-dim state | 0 | N/A |
| PVLib Advisor | Rule-Based | Metrics | 8-dim state | 0 | N/A |
| Meta-Controller | DDQN | 35-dim | 8 actions | 0.4M | Reinforcement |

---

## 2. Data Processing Pipeline

### 2.1 Input Data Sources

#### 2.1.1 Historical PV Power Data
- **Source:** PVDAQ (US), HSEPM (Germany)
- **Temporal Resolution:** 15-minute intervals
- **Features:** P_AC (kW), timestamp
- **Preprocessing:**
  - Normalization: P_norm = P_AC / P_rated
  - Missing value imputation: Linear interpolation (<5% gaps), forward-fill (>5%)
  - Outlier removal: Z-score > 3.5 flagged and replaced

#### 2.1.2 Numerical Weather Prediction (NWP)
- **Primary Sources:** 
  - Forecast.solar API (0-7 days, hourly)
  - ECMWF ERA5 (8-15 days, 0.25° resolution)
  - GFS via OpenMeteo (backup, 16-day limit)
- **Variables:** GHI, DNI, DHI, temperature, wind speed, humidity, pressure, cloud cover
- **Preprocessing:**
  - Spatial interpolation: Bilinear to plant coordinates
  - Temporal resampling: 15-min (short-head), 1-hour (long-head)
  - Feature scaling: Min-max normalization per variable

#### 2.1.3 Static Metadata
- **Panel Configuration:** Tilt angle, azimuth, rated capacity
- **Geographic:** Latitude, longitude, elevation
- **System:** Inverter efficiency, temperature coefficients

### 2.2 Feature Engineering

#### 2.2.1 Temporal Features
```python
# Cyclical encoding for periodicity
hour_sin = sin(2π × hour / 24)
hour_cos = cos(2π × hour / 24)
day_sin = sin(2π × day_of_year / 365)
day_cos = cos(2π × day_of_year / 365)
```

#### 2.2.2 Solar Geometry (PVLib)
```python
# Computed via pvlib.solarposition
solar_zenith = f(lat, lon, timestamp)
solar_azimuth = f(lat, lon, timestamp)
air_mass = f(solar_zenith)
extraterrestrial_dni = 1367 W/m² (solar constant)
```

#### 2.2.3 Sequence Construction

**Short-Head (15-minute resolution):**
- Input window: 96 steps (24 hours lookback)
- Forecast horizon: 96 steps (24 hours ahead)
- Stride: 4 steps (1 hour) for training
- Total features: 20 (NWP + temporal + solar)

**Long-Head (1-hour resolution):**
- Input window: 720 steps (30 days lookback)
- Forecast horizon: 360 steps (15 days ahead, extendable to 30d)
- Stride: 24 steps (1 day) for training
- Total features: 20 (NWP + temporal + solar)

---

## 3. Model Training Workflow

### 3.1 Phase 1: TFT Pretraining (Supervised Learning)

#### 3.1.1 Short-Head TFT Training

**Objective:**
Minimize mean squared error (MSE) between predicted and observed power:

$$
\mathcal{L}_{\text{short}} = \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{96} (y_{i,t} - \hat{y}_{i,t})^2
$$

**Training Protocol:**
```python
# Hyperparameters (optimized via Optuna)
learning_rate = 1e-3
batch_size = 32
hidden_size = 128
num_heads = 4
dropout = 0.1
gradient_clip_val = 1.0

# Training schedule
max_epochs = 100
early_stopping_patience = 10
lr_scheduler = ReduceLROnPlateau(patience=5, factor=0.5)

# Data split
train: 70% (chronological)
validation: 15%
test: 15%

# Augmentation
- Random noise injection: σ = 0.01
- Temporal jittering: ±1 timestep
- Dropout: 0.1 on attention layers
```

**Training Loop:**
```
FOR epoch = 1 TO max_epochs:
    FOR batch IN train_loader:
        # Forward pass
        y_pred = short_tft(x_past, x_future, static_features)
        loss = MSE(y_pred, y_true)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(parameters, max_norm=1.0)
        optimizer.step()
    
    # Validation
    val_loss = evaluate(val_loader)
    lr_scheduler.step(val_loss)
    
    # Early stopping check
    IF val_loss not improved for 10 epochs:
        BREAK
    
    # Checkpoint
    save_checkpoint(f"short_tft_epoch{epoch}.pt")
END FOR
```

**Convergence Criteria:**
- Validation loss plateau: ΔL < 10⁻⁴ for 10 epochs
- Maximum epochs: 100
- Gradient norm stability: ||∇L|| < 1.0

**Performance Metrics:**
- RMSE (kW): Root mean squared error
- MAE (kW): Mean absolute error
- R²: Coefficient of determination
- sMAPE: Symmetric mean absolute percentage error

#### 3.1.2 Long-Head TFT Training

**Objective:**
Multi-horizon quantile loss for uncertainty quantification:

$$
\mathcal{L}_{\text{long}} = \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{360} \sum_{q \in Q} \rho_q(y_{i,t} - \hat{y}_{i,t}^{(q)})
$$

where $\rho_q(u) = u(q - \mathbb{1}_{u < 0})$ is the quantile loss, and $Q = \{0.1, 0.5, 0.9\}$.

**Training Protocol:**
```python
# Hyperparameters (different from short-head)
learning_rate = 5e-4  # Lower for stability
batch_size = 16       # Smaller due to longer sequences
hidden_size = 160     # Larger for long-term dependencies
num_heads = 8
dropout = 0.15        # Higher to prevent overfitting
gradient_clip_val = 0.5

# Training schedule
max_epochs = 150
early_stopping_patience = 15
lr_scheduler = CosineAnnealingWarmRestarts(T_0=10, T_mult=2)

# Data split (same as short-head)
train: 70%, validation: 15%, test: 15%
```

**Multi-Horizon Loss Weighting:**
```python
# Exponential decay to prioritize near-term accuracy
horizon_weights = exp(-0.01 * t)  # t ∈ [0, 360]

loss = sum(horizon_weights[t] * quantile_loss(y_pred[t], y_true[t]))
```

### 3.2 Phase 2: PVLib Calibration

**Objective:**
Minimize residual between PVLib physics model and observed power:

$$
\mathcal{L}_{\text{physics}} = \frac{1}{N} \sum_{i=1}^{N} (P_{\text{obs},i} - P_{\text{pvlib},i}(\theta))^2
$$

where $\theta = \{\text{tilt}, \text{azimuth}, \text{soiling}, \text{degradation}\}$.

**Calibration Procedure:**
```python
# Grid search over panel parameters
tilt_range = [optimal - 5°, optimal + 5°]
azimuth_range = [optimal - 10°, optimal + 10°]
soiling_range = [0.90, 1.00]  # Loss factor
degradation_range = [0.005, 0.015]  # %/year

# Bayesian optimization (faster than grid search)
from skopt import gp_minimize

def objective(params):
    tilt, azimuth, soiling, degradation = params
    mc = ModelChain(system=PVSystem(
        surface_tilt=tilt,
        surface_azimuth=azimuth,
        module_parameters={'pdc0': rated_power * soiling},
        temperature_model_parameters={'a': -3.56, 'b': -0.075}
    ), location=location)
    
    predictions = mc.run_model(weather_data)
    rmse = sqrt(mean((predictions - observations)**2))
    return rmse

# Optimize
result = gp_minimize(objective, search_space, n_calls=100)
optimal_params = result.x
```

**Validation:**
- Cross-validation on held-out months
- Seasonal performance check (winter/summer)
- Edge case handling (snow, dust storms)

### 3.3 Phase 3: RL Meta-Controller Training

#### 3.3.1 Experience Collection (Heuristic Policy)

**Objective:**
Collect diverse state-action-reward trajectories using rule-based baseline.

**Procedure:**
```
INITIALIZE rl_system with mode="heuristic"
INITIALIZE replay_buffer with capacity=20,000
LOAD pretrained TFT checkpoints (short + long)
LOAD calibrated PVLib model

FOR episode = 1 TO 5000:
    # Sample historical timestamp
    timestamp = sample_uniform(train_period)
    
    # Initialize environment
    weather_data = fetch_weather(timestamp, horizon=30d)
    ground_truth = load_ground_truth(timestamp + horizon)
    
    state_t = build_meta_state(initial_metrics)
    done = False
    episode_reward = 0
    
    WHILE NOT done:
        # Heuristic action selection
        action_t = rl_system.meta_controller.select_action(
            state_t, mode="heuristic"
        )
        
        # Execute action in environment
        action_info = rl_system.step(metrics_t)
        
        # Generate forecasts with current weights
        forecast_short = short_tft.predict(weather_data)
        forecast_long = long_tft.predict(weather_data)
        forecast_physics = pvlib.predict(weather_data)
        
        # Ensemble blending (RL-controlled weights)
        forecast_ensemble = (
            action_info['blend_weights']['short'] * forecast_short +
            action_info['blend_weights']['long'] * forecast_long +
            action_info['blend_weights']['physics'] * forecast_physics
        )
        
        # Compute reward
        metrics_next = evaluate_forecast(forecast_ensemble, ground_truth)
        reward_t = rl_system.compute_reward(metrics_t, metrics_next)
        
        # Store transition
        state_next = build_meta_state(metrics_next)
        replay_buffer.store(state_t, action_t, reward_t, state_next, done)
        
        # Update state
        state_t = state_next
        episode_reward += reward_t
        
        # Episode termination (1 forecast cycle)
        done = True
    
    # Log episode statistics
    log(f"Episode {episode}: Reward={episode_reward:.2f}, "
        f"RMSE={metrics_next['ensemble_rmse']:.3f}")
    
    IF episode % 100 == 0:
        save_replay_buffer(f"experience_ep{episode}.pkl")
END FOR

# Save final experience dataset
save_replay_buffer("experience_heuristic_5k.pkl")
```

**Data Collection Targets:**
- Episodes: 5,000 minimum
- Transitions: ~5,000 (1 per episode, can be extended)
- Coverage: All seasons, weather conditions, system states
- Diversity: Ensure exploration of all 8 actions

#### 3.3.2 DDQN Training (Off-Policy Learning)

**Theoretical Foundation:**

The RL meta-controller implements **Double Deep Q-Networks (DDQN)** with **Prioritized Experience Replay** based on established RL theory (Zhao, 2023; Sutton & Barto, 2018). This section addresses key theoretical components:

**1. Q-Learning and Value Functions (Watkins, 1989):**

The agent learns an action-value function $Q(s, a)$ representing expected cumulative reward:

$$
Q^*(s, a) = \mathbb{E}[r_t + \gamma \max_{a'} Q^*(s_{t+1}, a') | s_t = s, a_t = a]
$$

Where $Q^*(s,a)$ is the optimal action-value function satisfying the Bellman optimality equation.

**2. Experience Replay (Lin, 1993):**

Stores transitions $(s_t, a_t, r_t, s_{t+1})$ in a replay buffer $\mathcal{D}$ and samples mini-batches for training. This technique:
- **Breaks temporal correlations:** Weather data exhibits strong autocorrelation (6-24h), causing instability in online learning
- **Improves sample efficiency:** Reuses past experiences multiple times
- **Stabilizes training:** Reduces variance in gradient updates

**3. Prioritized Experience Replay (Schaul et al., 2016):**

Samples transitions with probability proportional to TD-error $\delta_t$:

$$
P(i) = \frac{p_i^{\alpha}}{\sum_k p_k^{\alpha}}, \quad p_i = |\delta_i| + \epsilon
$$

Where:
- $\alpha$ = 0.6 (prioritization strength): Controls how much prioritization affects sampling
- $\epsilon$ = $10^{-6}$: Ensures non-zero probability for all transitions
- $\beta_t$ = 0.4 → 1.0 (importance sampling correction): Annealed over 100k steps to correct sampling bias

**Why critical for weather forecasting:** High-TD-error transitions often correspond to rare weather events (storms, rapid cloud transients) that are most informative for learning robust policies.

**4. Double DQN (van Hasselt et al., 2016):**

Addresses **overestimation bias** in standard DQN by decoupling action selection and evaluation:

$$
y_t^{\text{DDQN}} = r_t + \gamma Q_{\theta^-}(s_{t+1}, \arg\max_{a'} Q_{\theta}(s_{t+1}, a'))
$$

Where:
- $Q_{\theta}$: **Policy network** (updated every step) - selects actions
- $Q_{\theta^-}$: **Target network** (updated slowly) - evaluates actions

**Standard DQN overestimation problem:** Using $\max_{a'} Q_{\theta}(s_{t+1}, a')$ for both selection and evaluation causes positive bias because noise in Q-values tends to push maximum upward.

**5. Target Network Updates (Soft Polyak Averaging):**

Instead of periodic hard updates, we use **soft updates every step** (Lillicrap et al., 2016):

$$
\theta^- \leftarrow \tau \theta + (1 - \tau) \theta^-
$$

Where $\tau = 0.005$ (0.5% update per step). This provides:
- **Smooth tracking:** Target network gradually follows policy network
- **Weather-appropriate:** Smooth weather evolution requires smooth target updates
- **Stability:** Prevents "moving target" problem while maintaining responsiveness

**Convergence time:** With $\tau = 0.005$, target network reaches 50% convergence in $\approx 1386$ steps: $0.5 = (1 - \tau)^n \Rightarrow n = \frac{\ln(0.5)}{\ln(0.995)} \approx 1386$

**6. Exploration-Exploitation (ε-greedy):**

$$
a_t = \begin{cases}
\arg\max_a Q_{\theta}(s_t, a) & \text{with probability } 1 - \epsilon_t \\
\text{random action} & \text{with probability } \epsilon_t
\end{cases}
$$

**Linear decay schedule:**
$$
\epsilon_t = \max(0.1, 1.0 - t / 10000)
$$

- **No per-episode reset:** Continuous decay across all steps (suitable for online deployment)
- **Minimum floor:** $\epsilon_{\min} = 0.1$ ensures continued exploration (never fully greedy)

**7. Reward Function and Cost Shaping:**

The reward includes **action costs** as negative penalties to prevent overfitting:

$$
r_t = w_1 \cdot \Delta\text{RMSE}_t - w_2 \cdot \text{Drift}_t - w_3 \cdot \text{Cost}(a_t) - w_4 \cdot \text{RetrainFreq}_t + \text{Bonus}_t
$$

**Why costs matter:** Without cost penalties, RL would over-optimize by triggering expensive actions (retraining, fine-tuning) excessively, leading to:
- **Computational overfitting:** Models overfit to recent data through excessive updates
- **Resource waste:** Unnecessary retraining costs
- **Instability:** Constant hyperparameter changes prevent convergence

**Cost values:** MAINTAIN=0.0, FINE_TUNE=0.1-0.15, RECALIBRATE=0.05, BLEND=0.0, RETRAIN=1.0

**8. Gradient Clipping:**

Apply L2-norm clipping to prevent exploding gradients (Pascanu et al., 2013):

$$
\nabla_{\theta} \leftarrow \begin{cases}
\nabla_{\theta} & \text{if } ||\nabla_{\theta}||_2 \leq 1.0 \\
\frac{\nabla_{\theta}}{||\nabla_{\theta}||_2} & \text{otherwise}
\end{cases}
$$

**Why necessary:** Weather data contains outliers (storms, sensor errors) that can cause large TD-errors and unstable gradients.

---

**Training Protocol:**
```python
# DDQN hyperparameters (from MiRACLE paper)
learning_rate = 1e-4
gamma = 0.95                  # Discount factor
batch_size = 64
buffer_capacity = 20000
epsilon_start = 1.0           # Initial exploration rate
epsilon_end = 0.1             # Minimum exploration (never fully greedy)
epsilon_decay = 10000         # Linear decay over 10k steps
target_update_freq = 1        # Soft update every step (smooth weather tracking)
tau = 0.005                   # Soft update coefficient (0.5% per step, 50% convergence in 1386 steps)
alpha = 0.6                   # Prioritization exponent
beta_start = 0.4              # Importance sampling
beta_end = 1.0
beta_frames = 100000

# Network architecture
state_dim = 35
action_dim = 8
hidden_dim = 256

policy_net = DQN(state_dim, action_dim, hidden_dim)
target_net = DQN(state_dim, action_dim, hidden_dim)
target_net.load_state_dict(policy_net.state_dict())
optimizer = Adam(policy_net.parameters(), lr=learning_rate)
```

**Training Loop:**
```
LOAD replay_buffer from experience collection
INITIALIZE policy_net, target_net
INITIALIZE optimizer

FOR step = 1 TO 100000:
    # Sample prioritized batch
    beta = min(1.0, beta_start + step * (1.0 - beta_start) / beta_frames)
    batch, indices, weights = replay_buffer.sample(batch_size, beta)
    
    # Unpack batch
    states, actions, rewards, next_states, dones = batch
    
    # Compute current Q-values
    q_values = policy_net(states).gather(1, actions)
    
    # Compute target Q-values (DDQN)
    with torch.no_grad():
        # Select action with policy network
        next_actions = policy_net(next_states).argmax(dim=1, keepdim=True)
        # Evaluate with target network
        next_q_values = target_net(next_states).gather(1, next_actions)
        target_q_values = rewards + gamma * next_q_values * (1 - dones)
    
    # TD-error for priority update
    td_errors = abs(q_values - target_q_values)
    replay_buffer.update_priorities(indices, td_errors + 1e-6)
    
    # Weighted MSE loss (importance sampling)
    loss = (weights * (q_values - target_q_values)**2).mean()
    
    # Optimize
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
    optimizer.step()
    
    # Soft target network update (every step for smooth weather tracking)
    # Polyak averaging: θ⁻ ← τθ + (1-τ)θ⁻
    IF step % target_update_freq == 0:  # target_update_freq = 1
        FOR param_target, param_policy IN zip(
            target_net.parameters(), policy_net.parameters()
        ):
            param_target.data.copy_(
                tau * param_policy.data + (1 - tau) * param_target.data
            )
        # Note: With τ=0.005, target lags policy by ~1400 steps (50% convergence)
    
    # Logging
    IF step % 1000 == 0:
        log(f"Step {step}: Loss={loss:.4f}, "
            f"Epsilon={epsilon:.3f}, Beta={beta:.3f}")
    
    # Validation every 10k steps
    IF step % 10000 == 0:
        val_performance = evaluate_policy(policy_net, val_episodes)
        save_checkpoint(f"meta_controller_step{step}.pt")
    
    # Convergence check
    IF moving_avg_loss_delta < 1e-3 FOR 5000 steps:
        log("Converged!")
        BREAK
END FOR
```

**Convergence Criteria:**
- Q-value stability: |ΔQ| < 10⁻³ for 50 consecutive episodes
- Policy stability: Action distribution unchanged for 100 episodes
- Validation performance: RMSE improvement > 5% vs heuristic baseline
- Maximum steps: 100,000

**Validation Metrics:**
- Average reward per episode (validation set)
- RMSE improvement: (RMSE_heuristic - RMSE_rl) / RMSE_heuristic
- Action diversity: Entropy of action distribution
- Convergence speed: Steps to reach 95% final performance

---

## 4. Inference Workflow

### 4.1 Real-Time Forecasting Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│              REAL-TIME INFERENCE WORKFLOW                   │
└─────────────────────────────────────────────────────────────┘

[1] WEATHER DATA ACQUISITION (t = 0s)
    │
    ├─ Query timestamp and location
    ├─ Call smart weather router
    │   ├─ IF horizon ≤ 7d:  Forecast.solar API
    │   ├─ IF 8d ≤ horizon ≤ 15d: ECMWF ERA5
    │   └─ IF horizon > 15d: GFS (OpenMeteo)
    │
    └─ Output: weather_data (dual resolution)
              • 15-min: (672, 20) for short-head
              • 1-hour: (360, 20) for long-head

[2] FEATURE PREPROCESSING (t = 0.5s)
    │
    ├─ Compute solar geometry (PVLib)
    ├─ Encode temporal features (sin/cos)
    ├─ Normalize weather variables
    └─ Construct input tensors
    │
    └─ Output: X_short (96, 20), X_long (720, 20)

[3] TFT FORWARD PASSES (t = 1.0s)
    │
    ├─ Short-head inference
    │   └─ forecast_short = short_tft.predict(X_short)
    │       • Shape: (96,) @ 15-min = 24h ahead
    │       • Latency: ~200ms (GPU)
    │
    ├─ Long-head inference
    │   └─ forecast_long = long_tft.predict(X_long)
    │       • Shape: (360,) @ 1-hour = 15d ahead
    │       • Latency: ~400ms (GPU)
    │
    └─ PVLib physics inference
        └─ forecast_physics = pvlib.predict(weather_data, metadata)
            • Shape: (360,) @ 1-hour
            • Latency: ~50ms (CPU, deterministic)

[4] PERFORMANCE METRIC COLLECTION (t = 1.5s)
    │
    ├─ Load recent ground truth (if available)
    ├─ Compute RMSE @ multiple horizons
    │   • short_rmse_1h, short_rmse_24h
    │   • long_rmse_24h, long_rmse_7d, long_rmse_30d
    │   • physics_residual
    ├─ Compute confidence scores (quantile spread)
    ├─ Detect drift (KL divergence vs training distribution)
    └─ Collect system context
        • hour_of_day, season, weather_quality
        • retrain_count_24h, compute_budget
    │
    └─ Output: metrics (39 features)

[5] RL META-CONTROLLER DECISION (t = 1.6s)
    │
    ├─ Build state vector (35 dims)
    │   └─ state = rl_system.build_meta_state(metrics)
    │
    ├─ Query advisors
    │   ├─ short_alert = short_advisor.check_alert(state[0:10])
    │   ├─ long_alert = long_advisor.check_alert(state[10:20])
    │   └─ pvlib_alert = pvlib_advisor.check_alert(state[20:28])
    │
    ├─ Meta-controller action selection
    │   └─ action = meta_controller.select_action(state, mode="rl")
    │       • ε-greedy (ε=0.0 in production)
    │       • Forward pass: ~5ms
    │
    └─ Execute action
        ├─ IF action = FINE_TUNE_SHORT: adjust_lr(short_tft, ±20%)
        ├─ IF action = RECALIBRATE_PVLIB: update_panel_params()
        ├─ IF action = BLEND_HIGH_SHORT: weights = {0.7, 0.2, 0.1}
        ├─ IF action = SUGGEST_RETRAIN: add_to_queue()
        └─ ...
    │
    └─ Output: action_info (action, weights, alerts)

[6] ENSEMBLE BLENDING (t = 1.7s)
    │
    ├─ Resample to common resolution (1-hour)
    │   └─ forecast_short_1h = downsample(forecast_short, 15min→1h)
    │
    ├─ Weighted ensemble
    │   └─ forecast_ensemble = (
    │       w_short * forecast_short_1h +
    │       w_long * forecast_long +
    │       w_physics * forecast_physics
    │   )
    │       • Shape: (360,) @ 1-hour = 15 days
    │
    └─ Apply physics constraints
        ├─ Clip negative values: max(0, forecast)
        ├─ Cap at rated capacity: min(P_rated, forecast)
        └─ Smooth discontinuities: Savitzky-Golay filter
    │
    └─ Output: forecast_final (360,) @ 1-hour

[7] POST-PROCESSING & OUTPUT (t = 1.8s)
    │
    ├─ Generate uncertainty bands (quantiles from long-head)
    ├─ Format timestamps (timezone-aware)
    ├─ Compute aggregate statistics
    │   • Daily energy (kWh), peak power (kW)
    │   • Capacity factor, ramp rates
    │
    └─ Return forecast object
        {
            'timestamp_start': '2026-01-02T14:00:00Z',
            'forecast_kw': array([...]),  # (360,)
            'horizon_hours': array([1, 2, ..., 360]),
            'quantiles': {0.1: [...], 0.5: [...], 0.9: [...]},
            'blend_weights': {short: 0.5, long: 0.4, physics: 0.1},
            'rl_action': 'MAINTAIN',
            'advisor_alerts': {short: 'ok', long: 'ok', pvlib: 'ok'},
            'metadata': {...}
        }

[8] MONITORING & LOGGING (t = 2.0s)
    │
    ├─ Log forecast to database (Prometheus/InfluxDB)
    ├─ Update performance dashboard
    ├─ Check alert conditions
    │   • IF rmse_1h > 0.15: trigger_alert()
    │   • IF action = SUGGEST_RETRAIN: notify_operator()
    └─ Store for RL replay buffer (online learning)

TOTAL LATENCY: ~2.0 seconds (end-to-end)
```

### 4.2 Batch Forecasting (Historical Backtesting)

```python
def batch_forecast_pipeline(
    timestamps: List[datetime],
    plant_metadata: Dict,
    rl_checkpoint: Path
) -> pd.DataFrame:
    """
    Generate forecasts for multiple timestamps (backtesting).
    """
    # Load models
    short_tft = load_tft_model("V1.0_FINAL_TFT/shorthead_seed42")
    long_tft = load_tft_model("V1.0_FINAL_TFT/longhead_seed43")
    pvlib_model = initialize_pvlib(plant_metadata)
    rl_system = RLMetaControllerSystem()
    rl_system.load_checkpoint(rl_checkpoint)
    
    results = []
    
    for ts in timestamps:
        # Fetch weather
        weather = fetch_weather_batch(ts, horizon_days=15)
        
        # Preprocess
        X_short, X_long = preprocess_inputs(weather, ts, plant_metadata)
        
        # TFT inference
        forecast_short = short_tft.predict(X_short)
        forecast_long = long_tft.predict(X_long)
        forecast_physics = pvlib_model.predict(weather)
        
        # Collect metrics (use last known RMSE)
        metrics = build_metrics(ts, forecast_short, forecast_long, 
                                forecast_physics, ground_truth=None)
        
        # RL decision
        action_info = rl_system.step(metrics)
        
        # Ensemble
        forecast_final = ensemble_blend(
            forecast_short, forecast_long, forecast_physics,
            weights=action_info['blend_weights']
        )
        
        # Store result
        results.append({
            'timestamp': ts,
            'forecast': forecast_final,
            'action': action_info['action_name'],
            'weights': action_info['blend_weights']
        })
    
    return pd.DataFrame(results)
```

---

## 5. Experimental Validation Workflow

### 5.1 Cross-Validation Strategy

**Temporal Cross-Validation (Forward Chaining):**
```
Training Set                Validation Set   Test Set
├───────────────────────────┼────────────────┼──────────┤
│ 70% (Jan 2023 - Sept 2024)│ 15% (Oct-Nov)  │ 15% (Dec)│
└───────────────────────────┴────────────────┴──────────┘

FOR fold = 1 TO 5:
    train_end = start_date + fold * (total_period / 5)
    val_start = train_end
    val_end = val_start + (total_period * 0.15)
    
    train_models(data[start:train_end])
    validate_models(data[val_start:val_end])
    
    log_metrics(fold, train_rmse, val_rmse)
```

### 5.2 Ablation Studies

**Experimental Matrix:**

| Experiment | Short-TFT | Long-TFT | PVLib | RL | Baseline |
|------------|-----------|----------|-------|----|----|
| E1: Persistence | ❌ | ❌ | ❌ | ❌ | Last value |
| E2: PVLib Only | ❌ | ❌ | ✅ | ❌ | Physics |
| E3: Short-TFT Only | ✅ | ❌ | ❌ | ❌ | ML only |
| E4: Long-TFT Only | ❌ | ✅ | ❌ | ❌ | ML only |
| E5: Short+Long (Fixed) | ✅ | ✅ | ❌ | ❌ | 0.5/0.5 |
| E6: Short+Long+PVLib (Fixed) | ✅ | ✅ | ✅ | ❌ | 0.4/0.4/0.2 |
| E7: **MiRACLE (Heuristic)** | ✅ | ✅ | ✅ | Heuristic | Rule-based |
| E8: **MiRACLE (RL)** | ✅ | ✅ | ✅ | ✅ Learned | Full system |

**Evaluation Metrics:**

For each experiment, compute:

$$
\text{RMSE}_h = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_{i,h} - \hat{y}_{i,h})^2}
$$

$$
\text{MAE}_h = \frac{1}{N} \sum_{i=1}^{N} |y_{i,h} - \hat{y}_{i,h}|
$$

$$
\text{sMAPE}_h = \frac{100\%}{N} \sum_{i=1}^{N} \frac{|y_{i,h} - \hat{y}_{i,h}|}{(|y_{i,h}| + |\hat{y}_{i,h}|) / 2}
$$

for horizons $h \in \{1, 6, 24, 168, 360\}$ hours.

**Statistical Testing:**
- Diebold-Mariano test for forecast accuracy comparison
- Paired t-test for mean RMSE differences (α = 0.05)
- Wilcoxon signed-rank test for non-normal distributions

### 5.3 Computational Performance Benchmarking

**Hardware Configuration:**
- Training: 2× NVIDIA L4 GPUs (48GB VRAM total), 64-core AMD EPYC
- Inference: Single NVIDIA L4 GPU or 8-core CPU
- Storage: NVMe SSD (3.5 GB/s read)

**Latency Breakdown:**

| Stage | GPU (ms) | CPU (ms) | Bottleneck |
|-------|----------|----------|------------|
| Weather fetch | 200 | 200 | Network I/O |
| Preprocessing | 50 | 150 | Feature engineering |
| Short-TFT | 180 | 1200 | Transformer attention |
| Long-TFT | 350 | 2500 | Sequence length |
| PVLib | 30 | 50 | Numpy operations |
| RL decision | 5 | 10 | Small network |
| Ensemble | 20 | 30 | Numpy operations |
| **Total** | **835** | **4140** | Transformer inference |

**Throughput:**
- Single forecast: 1.2 forecasts/second (GPU)
- Batch (32): 15 forecasts/second (GPU)
- Training: 2-3 hours for 10k RL episodes (2×L4)

---

## 6. Continuous Learning & Deployment

### 6.1 Online Learning Protocol

**Trigger Conditions for Model Updates:**
1. **Scheduled Retraining:** Weekly (Sundays, 00:00 UTC)
2. **Performance Degradation:** RMSE_7d > 1.2 × RMSE_baseline
3. **Data Drift:** KL(P_current || P_train) > 0.5
4. **RL Suggestion:** Meta-controller action = SUGGEST_RETRAIN

**Incremental Learning Workflow:**
```python
def incremental_training_cycle():
    """
    Weekly incremental update of TFT models.
    """
    # Fetch new data (last 7 days)
    new_data = fetch_recent_data(days=7)
    
    # Validate data quality
    IF data_quality_check(new_data) < 0.9:
        log("Poor data quality, skipping update")
        RETURN
    
    # Load current model
    model = load_checkpoint("current_model.pt")
    
    # Fine-tune with small learning rate
    optimizer = Adam(model.parameters(), lr=1e-5)  # 10× lower
    
    FOR epoch IN range(10):  # Short fine-tuning
        loss = train_epoch(model, new_data)
        log(f"Fine-tune epoch {epoch}: loss={loss:.4f}")
    
    # Validate on held-out data
    val_rmse = evaluate(model, validation_set)
    
    # A/B test: compare old vs new
    old_rmse = evaluate(old_model, test_set)
    new_rmse = evaluate(model, test_set)
    
    IF new_rmse < old_rmse * 0.95:  # 5% improvement threshold
        save_checkpoint(model, "current_model.pt")
        log("Model updated successfully")
    ELSE:
        log("New model not better, keeping old model")
        rollback()
```

### 6.2 A/B Testing Framework

**Deployment Strategy:**
```
┌─────────────────────────────────────┐
│         Traffic Split (50/50)       │
└─────────┬─────────────────┬─────────┘
          │                 │
    ┌─────▼─────┐     ┌─────▼─────┐
    │  Model A  │     │  Model B  │
    │ (Current) │     │   (New)   │
    └─────┬─────┘     └─────┬─────┘
          │                 │
    ┌─────▼─────────────────▼─────┐
    │    Performance Tracker      │
    │  • RMSE A vs B              │
    │  • Latency A vs B           │
    │  • Drift detection          │
    └─────────────┬───────────────┘
                  │
            ┌─────▼─────┐
            │  Decision │
            │  Engine   │
            └─────┬─────┘
                  │
        ┌─────────┴──────────┐
        │                    │
   ┌────▼────┐         ┌────▼────┐
   │ Promote │         │ Reject  │
   │ Model B │         │ Model B │
   └─────────┘         └─────────┘
```

**Statistical Test:**
```python
from scipy.stats import ttest_ind

# Collect forecasts over 7 days
errors_A = [compute_error(forecast_A, truth) for t in test_period]
errors_B = [compute_error(forecast_B, truth) for t in test_period]

# Two-sample t-test
t_stat, p_value = ttest_ind(errors_A, errors_B)

IF p_value < 0.05 AND mean(errors_B) < mean(errors_A):
    promote_model_B()
ELSE:
    keep_model_A()
```

### 6.3 Monitoring Dashboard

**Key Performance Indicators (KPIs):**

1. **Forecast Accuracy:**
   - RMSE @ 1h, 6h, 24h, 7d, 15d
   - Rolling 7-day average RMSE
   - Quantile calibration score

2. **RL Meta-Controller:**
   - Action distribution (histogram)
   - Average reward per episode
   - Blend weight evolution (time series)
   - Retrain request frequency

3. **System Health:**
   - Inference latency (p50, p95, p99)
   - API uptime (Forecast, ECMWF, GFS)
   - Data drift score (daily)
   - GPU/CPU utilization

4. **Alerting Rules:**
   ```yaml
   alerts:
     - name: high_rmse
       condition: rmse_1h > 0.15
       severity: warning
       action: notify_oncall
     
     - name: critical_rmse
       condition: rmse_24h > 0.25
       severity: critical
       action: trigger_retraining
     
     - name: data_drift
       condition: kl_divergence > 0.7
       severity: warning
       action: log_and_monitor
     
     - name: api_failure
       condition: weather_api_timeout > 10s
       severity: critical
       action: switch_to_backup_api
   ```

---

## 7. Reproducibility & Documentation

### 7.1 Experiment Tracking

**MLflow Integration:**
```python
import mlflow

# Start experiment
mlflow.set_experiment("MiRACLE_TFT_Training")

with mlflow.start_run():
    # Log parameters
    mlflow.log_params({
        "model": "short_tft",
        "learning_rate": 1e-3,
        "batch_size": 32,
        "hidden_size": 128,
        "num_heads": 4,
        "dropout": 0.1,
        "seed": 42
    })
    
    # Training loop
    for epoch in range(max_epochs):
        train_loss = train_epoch(model, train_loader)
        val_loss = evaluate(model, val_loader)
        
        # Log metrics
        mlflow.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss
        }, step=epoch)
    
    # Log final model
    mlflow.pytorch.log_model(model, "model")
    
    # Log artifacts
    mlflow.log_artifact("config.yaml")
    mlflow.log_artifact("training_log.txt")
```

### 7.2 Version Control

**Git Structure:**
```
pv_forecast_30d/
├── .git/
├── src/
│   ├── models/
│   │   ├── tft_model.py (v1.0)
│   │   └── pvlib_predictor.py (v1.0)
│   ├── rl/
│   │   ├── rl_meta_controller.py (v1.0-refactored)
│   │   └── rl_integrated_forecaster.py (v1.0)
│   └── inference/
│       ├── weather_client.py (v1.0)
│       └── physics_aware_forecaster.py (v1.0)
├── V1.0_FINAL_TFT/
│   ├── shorthead_seed42/
│   └── longhead_seed43/
├── checkpoints/
│   └── rl/
│       └── meta_controller_100k.pt (v1.0)
└── docs/
    ├── MIRACLE_SAR_SPACE_CLEAN.md
    └── MIRACLE_SCIENTIFIC_WORKFLOW.md (this file)

# Git tags for releases
git tag -a v1.0.0-miracle -m "MiRACLE V1.0: Initial production release"
git push origin v1.0.0-miracle
```

### 7.3 Environment Management

**Docker Container:**
```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install Python 3.11
RUN apt-get update && apt-get install -y python3.11 python3-pip

# Install dependencies
COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt

# Copy code
COPY src/ /app/src/
COPY V1.0_FINAL_TFT/ /app/V1.0_FINAL_TFT/
COPY checkpoints/ /app/checkpoints/

# Set environment variables
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
ENV OMP_NUM_THREADS=8

# Expose API port
EXPOSE 8000

# Run inference server
CMD ["python3", "/app/src/api/server.py"]
```

**Singularity Container (HPC):**
```
Bootstrap: docker
From: nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

%files
    requirements.txt /opt/
    src/ /opt/src/
    V1.0_FINAL_TFT/ /opt/V1.0_FINAL_TFT/
    checkpoints/ /opt/checkpoints/

%post
    apt-get update && apt-get install -y python3.11 python3-pip
    pip3 install -r /opt/requirements.txt

%environment
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
    export OMP_NUM_THREADS=8

%runscript
    python3 /opt/src/training/train_rl_meta.py "$@"
```

---

## 8. Future Extensions

### 8.1 Planned Enhancements (V1.1+)

1. **LSTM Encoder Reintroduction:**
   - Role: Temporal feature extraction from raw time series
   - Architecture: Bidirectional LSTM (128 hidden) → embeddings
   - Integration: Feed embeddings to both TFT heads
   - Benefit: Capture complex temporal patterns not seen by TFT

2. **Database Manager Advisor:**
   - Role: Monitor data quality, trigger fetches, manage cache
   - State: 8 dims (freshness, quality, coverage, etc.)
   - Actions: fetch_weather, invalidate_cache, request_cleanup
   - Benefit: Autonomous data pipeline management

3. **Multi-Site Ensemble:**
   - Aggregate forecasts from nearby plants
   - Spatial correlation modeling
   - Transfer learning across sites

4. **Probabilistic RL:**
   - Replace DDQN with Soft Actor-Critic (SAC)
   - Continuous action space for blend weights
   - Better exploration in high-dimensional space

### 8.2 Research Directions

1. **Causal Inference:**
   - Identify causal weather-power relationships
   - Reduce spurious correlations
   - Improve generalization to unseen conditions

2. **Federated Learning:**
   - Train on distributed sites without sharing raw data
   - Privacy-preserving model updates
   - Scalable to thousands of installations

3. **Interpretability:**
   - SHAP values for feature importance
   - Attention weight visualization
   - Counterfactual explanations for RL actions

---

## 9. References

### Reinforcement Learning Theory

1. **Foundational RL:**
   - Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
   - Zhao, Z. (2023). *Reinforcement Learning: Theory and Python Implementation*. Springer Nature.
   - Watkins, C. J. C. H. (1989). "Learning from Delayed Rewards," PhD Thesis, Cambridge University.

2. **Experience Replay:**
   - Lin, L.-J. (1993). "Reinforcement Learning for Robots Using Neural Networks," PhD Thesis, Carnegie Mellon University.
   - Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2016). "Prioritized Experience Replay," *ICLR 2016*.

3. **Deep Q-Networks:**
   - Mnih, V., et al. (2015). "Human-level control through deep reinforcement learning," *Nature*, 518(7540), 529-533.
   - van Hasselt, H., Guez, A., & Silver, D. (2016). "Deep Reinforcement Learning with Double Q-learning," *AAAI 2016*.
   - Lillicrap, T. P., et al. (2016). "Continuous control with deep reinforcement learning," *ICLR 2016*.

4. **Gradient Methods:**
   - Pascanu, R., Mikolov, T., & Bengio, Y. (2013). "On the difficulty of training recurrent neural networks," *ICML 2013*.
   - Kingma, D. P., & Ba, J. (2015). "Adam: A Method for Stochastic Optimization," *ICLR 2015*.

5. **Hierarchical RL:**
   - Bacon, P.-L., Harb, J., & Precup, D. (2017). "The Option-Critic Architecture," *AAAI 2017*.
   - Vezhnevets, A. S., et al. (2017). "FeUdal Networks for Hierarchical Reinforcement Learning," *ICML 2017*.

### Transformer Architecture

6. **Attention Mechanisms:**
   - Vaswani, A., et al. (2017). "Attention Is All You Need," *NeurIPS 2017*.
   - Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T. (2021). "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting," *International Journal of Forecasting*, 37(4), 1748-1764.

### Physics-Based Modeling

7. **Solar Energy:**
   - Holmgren, W. F., Hansen, C. W., & Mikofski, M. A. (2018). "pvlib python: A python package for modeling solar energy systems," *Journal of Open Source Software*, 3(29), 884.
   - Duffie, J. A., & Beckman, W. A. (2013). *Solar Engineering of Thermal Processes* (4th ed.). Wiley.
   - Perez, R., et al. (1990). "Modeling daylight availability and irradiance components from direct and global irradiance," *Solar Energy*, 44(5), 271-289.

### Time Series Forecasting

8. **Statistical Methods:**
   - Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts.
   - Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2018). "The M4 Competition: Results, findings, conclusions and way forward," *International Journal of Forecasting*, 34(4), 802-808.

### Domain Applications

9. **PV Forecasting:**
   - Antonanzas, J., et al. (2016). "Review of photovoltaic power forecasting," *Solar Energy*, 136, 78-111.
   - Voyant, C., et al. (2017). "Machine learning methods for solar radiation forecasting: A review," *Renewable Energy*, 105, 569-582.
   - Das, U. K., et al. (2018). "Forecasting of photovoltaic power generation and model optimization: A review," *Renewable and Sustainable Energy Reviews*, 81, 912-928

---

## Appendix A: Notation & Terminology

| Symbol | Description | Dimension |
|--------|-------------|-----------|
| $s_t$ | State at time $t$ | $\mathbb{R}^{35}$ |
| $a_t$ | Action at time $t$ | $\{0, 1, ..., 7\}$ |
| $r_t$ | Reward at time $t$ | $\mathbb{R}$ |
| $Q(s, a)$ | Action-value function | $\mathbb{R}$ |
| $\pi(a|s)$ | Policy (action distribution) | $[0, 1]$ |
| $\gamma$ | Discount factor | 0.95 |
| $\epsilon$ | Exploration rate | $[0.1, 1.0]$ |
| $\tau$ | Soft update parameter (0.5% per step) | 0.005 |
| $y_t$ | Ground truth power at $t$ | $\mathbb{R}_+$ (kW) |
| $\hat{y}_t$ | Predicted power at $t$ | $\mathbb{R}_+$ (kW) |
| $\mathbf{w}$ | Ensemble blend weights | $\mathbb{R}^3$, $\sum w_i = 1$ |

---

## Appendix B: Computational Resources

**Training Resource Requirements:**

| Phase | Hardware | Duration | Cost (AWS p3.2xlarge) |
|-------|----------|----------|-----------------------|
| TFT Pretraining (both heads) | 2× V100 (16GB) | 6-8 hours | $18-24 |
| PVLib Calibration | 1× CPU (8-core) | 1 hour | $0.50 |
| Experience Collection | 1× V100 | 12 hours | $18 |
| DDQN Training | 1× V100 | 2-3 hours | $4.50-6.75 |
| **Total** | | **21-24 hours** | **$41-49** |

**Inference Resource Requirements:**
- GPU: NVIDIA L4 or T4 (4GB VRAM minimum)
- CPU: 4 cores @ 2.5GHz (fallback)
- RAM: 8GB
- Storage: 10GB (models + cache)

---

**Document Version:** 1.0  
**Last Updated:** January 2, 2026  
**Authors:** MiRACLE Research Team  
**Contact:** [Your Institution/Email]  
**License:** Academic Use Only

---

**End of Scientific Workflow Documentation**
