***REMOVED*** RL Meta-Controller Integration Guide

***REMOVED******REMOVED*** Overview

The RL meta-controller provides adaptive ensemble management for the PV forecasting system with:
- **3 Local Agents:** Short-TFT, Long-TFT, PVLib
- **1 Meta-Agent:** Dynamic ensemble blending
- **Weather Router:** Rule-based (not RL)
- **Human-in-the-loop:** Safe retrain confirmations

***REMOVED******REMOVED*** Quick Start

***REMOVED******REMOVED******REMOVED*** 1. Basic Usage

```python
from src.rl.rl_integrated_forecaster import RLIntegratedForecaster
from src.rl.rl_meta_controller import RLConfig

***REMOVED*** Initialize forecaster
forecaster = PhysicsAwareForecaster(...)

***REMOVED*** Wrap with RL
rl_forecaster = RLIntegratedForecaster(
    forecaster=forecaster,
    rl_mode="heuristic",  ***REMOVED*** Start with rule-based
    rl_config=RLConfig()
)

***REMOVED*** Generate forecast
forecast, info = rl_forecaster.forecast_with_rl(weather_data)

***REMOVED*** Check actions
print(f"Blend weights: {info['blend_weights']}")
print(f"Actions: {info['actions']}")
```

***REMOVED******REMOVED******REMOVED*** 2. Operating Modes

**Heuristic Mode** (Recommended for initial deployment):
```python
RLConfig(mode="heuristic")
```
- Uses rule-based policies (no ML)
- Baseline performance
- Collect experience for training

**RL Mode** (After training):
```python
RLConfig(mode="rl")
```
- Uses learned DQN policies
- Requires trained checkpoints
- Adaptive blending

**Hybrid Mode** (Advanced):
```python
RLConfig(mode="hybrid")
```
- Local agents use RL
- Meta-agent uses heuristic
- Balance safety + adaptation

***REMOVED******REMOVED*** Architecture

***REMOVED******REMOVED******REMOVED*** Local Agents (Per-Model Control)

**Actions** (5 per agent):
- `0`: **Maintain** - No changes
- `1`: **Fine-tune** - Adjust hyperparameters (automated)
- `2`: **Retrain** - Full retrain (human confirmation)
- `3`: **Rollback** - Revert to previous checkpoint
- `4`: **Defer** - Let others handle

**State Features** (~15 dims):
- Performance: RMSE @ 1h, 24h, 7d, 30d
- Confidence: Prediction variance
- Drift: Input distribution shift
- Context: Hour, season, night/day
- History: Recent retrain count

***REMOVED******REMOVED******REMOVED*** Meta-Agent (Ensemble Blending)

**Actions** (27 discrete):
- Weight combinations: `[0.1, 0.5, 0.9]` per model
- Example: `(0.5, 0.1, 0.9)` → Short=0.357, Long=0.071, Physics=0.571
- Normalized to sum = 1.0

**State Features** (~25 dims):
- All local agent states
- Short-long mismatch
- Ensemble RMSE
- Weather quality
- API agreement

***REMOVED******REMOVED******REMOVED*** Reward Function

```
R = w₁(−RMSE) + w₂(−Mismatch) + w₃(−Drift) + w₄(−Cost)
```

**Default Weights:**
- Accuracy (w₁): 1.0
- Consistency (w₂): 0.3
- Stability (w₃): 0.2
- Efficiency (w₄): 0.1

***REMOVED******REMOVED*** Deployment Strategy

***REMOVED******REMOVED******REMOVED*** Phase 1: Heuristic Baseline (Day 1-2)

**Goal:** Collect 2k-5k experience episodes (run parallel forecasts to speed up)

```python
***REMOVED*** Deploy in heuristic mode
rl_forecaster = RLIntegratedForecaster(
    forecaster=forecaster,
    rl_mode="heuristic"
)

***REMOVED*** Run production forecasts
for _ in range(10000):
    forecast, info = rl_forecaster.forecast_with_rl(weather_data)
    
    ***REMOVED*** When ground truth available:
    rl_forecaster.rl_controller.update(
        metrics_prev=prev_metrics,
        actions=prev_actions,
        metrics_next=curr_metrics,
        done=False
    )

***REMOVED*** Save experience
rl_forecaster.save_checkpoint(Path("checkpoints/heuristic_experience.pt"))
```

**Metrics to Track:**
- Ensemble RMSE (baseline)
- Blend weight distributions
- Retrain suggestions
- Computational cost

***REMOVED******REMOVED******REMOVED*** Phase 2: Offline DQN Training (Day 2-3)

**Hardware:** H100 GPU on HPC (1-2 hours), 2x L4 local (2-3 hours)

```bash
***REMOVED*** Train DQN
python scripts/train_dqn_offline.py \
  --experience checkpoints/heuristic_experience.pt \
  --episodes 10000 \
  --gpus 1 \
  --checkpoint_every 100

***REMOVED*** Monitor convergence: |ΔQ| < 10⁻³ for 50 episodes
```

**Expected Training Time:**
- 10³ episodes: ~1-2 hours on H100, ~2-3 hours on 2x L4
- Validation every 100 episodes
- Early stopping if converged

***REMOVED******REMOVED******REMOVED*** Phase 3: A/B Testing (Day 3-4)

**Design:**
- 50% traffic: Heuristic
- 50% traffic: RL
- Duration: 24-48 hours minimum (statistical power depends on forecast volume)

```python
import random

***REMOVED*** Route traffic
if random.random() < 0.5:
    mode = "heuristic"
else:
    mode = "rl"

rl_forecaster = RLIntegratedForecaster(
    forecaster=forecaster,
    rl_mode=mode
)
```

**Decision Criteria:**
- RL wins if: RMSE < heuristic AND stable
- Heuristic wins if: RL unstable or no improvement
- Statistical test: t-test on RMSE distributions

***REMOVED******REMOVED******REMOVED*** Phase 4: Production Deployment (Day 4+)

```python
***REMOVED*** Full RL mode
rl_forecaster = RLIntegratedForecaster(
    forecaster=forecaster,
    rl_mode="rl",
    checkpoint_dir=Path("models/rl_production")
)

***REMOVED*** Load best checkpoint
rl_forecaster.load_checkpoint(Path("models/rl_production/best.pt"))
```

***REMOVED******REMOVED*** Human-in-the-Loop Workflow

***REMOVED******REMOVED******REMOVED*** Retrain Suggestions

When RL suggests retrain (action=2):

```python
***REMOVED*** Check queue
status = rl_forecaster.get_status()
pending = status['pending_retrains']

if pending['short_tft'] > 0:
    print("Short-TFT retrain suggested:")
    queue = rl_forecaster.rl_controller.retrain_queue['short_tft']
    for req in queue:
        print(f"  Reason: {req['reason']}")
        print(f"  Timestamp: {req['timestamp']}")
```

***REMOVED******REMOVED******REMOVED*** Human Approval

```python
***REMOVED*** Approve retrain
rl_forecaster.confirm_retrain(model='short_tft', approve=True)

***REMOVED*** Or reject
rl_forecaster.confirm_retrain(model='short_tft', approve=False)
```

***REMOVED******REMOVED******REMOVED*** Automated Actions

**Fine-tuning (action=1)** is automated:
- Adjusts learning rate
- Modifies batch size
- No human confirmation needed

***REMOVED******REMOVED*** Hyperparameter Tuning

***REMOVED******REMOVED******REMOVED*** Default (from Paper)

```python
RLConfig(
    learning_rate=1e-4,
    gamma=0.95,
    batch_size=64,
    epsilon_start=1.0,
    epsilon_end=0.1,
    epsilon_decay=10000,
    buffer_capacity=10000
)
```

***REMOVED******REMOVED******REMOVED*** Conservative (Slower Learning)

```python
RLConfig(
    learning_rate=5e-5,  ***REMOVED*** Lower LR
    gamma=0.99,          ***REMOVED*** More future weight
    epsilon_decay=20000  ***REMOVED*** Slower exploration decay
)
```

***REMOVED******REMOVED******REMOVED*** Aggressive (Faster Adaptation)

```python
RLConfig(
    learning_rate=3e-4,  ***REMOVED*** Higher LR
    gamma=0.90,          ***REMOVED*** Less future weight
    epsilon_decay=5000   ***REMOVED*** Faster exploration decay
)
```

***REMOVED******REMOVED*** Monitoring

***REMOVED******REMOVED******REMOVED*** Key Metrics

```python
diag = rl_forecaster.rl_controller.get_diagnostics()

print(f"Episode: {diag['episode']}")
print(f"Avg reward: {diag['avg_reward_100ep']:.4f}")
print(f"Buffer size: {diag['agents']['short_tft']['buffer_size']}")
print(f"Epsilon: {diag['agents']['short_tft']['epsilon']:.4f}")
```

***REMOVED******REMOVED******REMOVED*** Alerts

**Critical:**
- Ensemble RMSE > 2x baseline
- Buffer size = 0 (no learning)
- Epsilon stuck at 1.0 (no exploitation)

**Warning:**
- Retrain queue backlog > 10
- Q-value divergence
- Reward trend negative

***REMOVED******REMOVED*** Troubleshooting

***REMOVED******REMOVED******REMOVED*** Issue: RL Not Learning

**Symptoms:**
- Avg reward not improving
- Epsilon stuck at 1.0
- Buffer size = 0

**Solutions:**
1. Check if updates enabled: `config.mode == "rl"`
2. Verify buffer has data: `agent.replay_buffer.size() > batch_size`
3. Increase training episodes
4. Lower epsilon decay

***REMOVED******REMOVED******REMOVED*** Issue: Unstable Blend Weights

**Symptoms:**
- Weights oscillate rapidly
- Ensemble RMSE worse than single models

**Solutions:**
1. Switch to hybrid mode: `config.mode = "hybrid"`
2. Increase gamma (more stability): `gamma = 0.99`
3. Use heuristic meta-agent
4. Reduce w_consistency weight

***REMOVED******REMOVED******REMOVED*** Issue: Too Many Retrain Suggestions

**Symptoms:**
- Queue length > 10
- Frequent action=2

**Solutions:**
1. Increase RMSE thresholds in heuristic policy
2. Penalize retrains in reward: increase w_efficiency
3. Add cooldown period between retrains
4. Reject aggressive suggestions

***REMOVED******REMOVED*** Advanced Configuration

***REMOVED******REMOVED******REMOVED*** Custom Reward Weights

```python
RLConfig(
    w_accuracy=2.0,    ***REMOVED*** Prioritize RMSE
    w_consistency=0.1,  ***REMOVED*** Relax short-long alignment
    w_stability=0.5,    ***REMOVED*** Increase drift penalty
    w_efficiency=0.0    ***REMOVED*** Ignore compute cost
)
```

***REMOVED******REMOVED******REMOVED*** Prioritized Replay

```python
RLConfig(
    prioritized_replay=True,
    alpha=0.6,           ***REMOVED*** Prioritization strength
    beta_start=0.4,      ***REMOVED*** Importance sampling
    beta_frames=100000   ***REMOVED*** Anneal to 1.0
)
```

***REMOVED******REMOVED******REMOVED*** Soft Target Updates

```python
RLConfig(
    target_update_freq=1000,  ***REMOVED*** Update every N steps
    tau=0.005                  ***REMOVED*** Soft update coefficient
)
```

***REMOVED******REMOVED*** Testing

Run integration tests:

```bash
python tests/test_rl_integration.py
```

Expected output:
```
✅ All 8 tests PASSED
```

Run unit tests:

```bash
pytest tests/test_rl_meta_controller.py -v
```

***REMOVED******REMOVED*** References

- **Paper:** MiRACLE - Hierarchical RL for PV Forecasting
- **Code:** [src/rl/rl_meta_controller.py](../src/rl/rl_meta_controller.py)
- **Integration:** [src/rl/rl_integrated_forecaster.py](../src/rl/rl_integrated_forecaster.py)
- **Weather Routing:** [WEATHER_API_SMART_ROUTING.md](../WEATHER_API_SMART_ROUTING.md)

***REMOVED******REMOVED*** Contact

For questions or issues, contact the author.
