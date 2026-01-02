***REMOVED*** Hierarchical Physics-Constrained Inference: Complete Technical Specification

**Version:** 2.0 (Hierarchical Refinement Architecture)  
**Date:** 2026-01-02  
**Status:** 🔒 LOCKED - Production Ready (TFT stubs remaining)

---

***REMOVED******REMOVED*** Table of Contents

1. [Architecture Overview](***REMOVED***architecture-overview)
2. [Hierarchical Refinement Concept](***REMOVED***hierarchical-refinement-concept)
3. [Data Flow Pipeline](***REMOVED***data-flow-pipeline)
4. [Component Specifications](***REMOVED***component-specifications)
5. [Mathematical Formulations](***REMOVED***mathematical-formulations)
6. [RL Meta-Controller](***REMOVED***rl-meta-controller)
7. [Weather-Only Encoding](***REMOVED***weather-only-encoding)
8. [Implementation Details](***REMOVED***implementation-details)
9. [Testing & Validation](***REMOVED***testing--validation)
10. [TFT Integration Roadmap](***REMOVED***tft-integration-roadmap)

---

***REMOVED******REMOVED*** Architecture Overview

***REMOVED******REMOVED******REMOVED*** High-Level Design (Hierarchical Refinement v2.0)

```
┌──────────────────────────────────────────────────────────────────────┐
│           Hierarchical 30-Day PV Power Forecasting System             │
│                "Drone + Fighter Jet" Architecture                      │
└──────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
    LONG-HEAD                   SHORT-HEAD                  PVLib PHYSICS
    (Strategic)                 (Tactical)                  (Constraints)
        │                           │                           │
    1 TFT call                  30 TFT calls               Computed from
    720 hours @ 1h              96 steps @ 15min / day     weather forecast
    Rough 30-day view           Daily precision            Physics baseline
        │                           │                           │
        │                           │                           │
        ├─────> Upsample to 15-min──┤                           │
        │         (PVLib shape)      │                           │
        │                           │                           │
        └───────────────────────────┴───────────────────────────┘
                                    │
                          PER-DAY HIERARCHICAL BLEND
                      (RL-adaptive 3-way combination)
                                    │
                          ┌─────────┴─────────┐
                          │   Layer 1: ML     │
                          │ short + long      │
                          └─────────┬─────────┘
                                    │
                          ┌─────────┴─────────┐
                          │ Layer 2: Physics  │
                          │   ML + PVLib      │
                          └─────────┬─────────┘
                                    │
                          ┌─────────┴─────────┐
                          │ Layer 3: Hard     │
                          │   Constraints     │
                          └─────────┬─────────┘
                                    │
                                    ▼
                        Final Forecast: 2880 steps @ 15-min
```

***REMOVED******REMOVED******REMOVED*** Key Properties

| Property | Value | Rationale |
|----------|-------|-----------|
| **Total Duration** | 30 days | Project requirement |
| **Resolution** | 15 minutes | High granularity |
| **Total Steps** | 2880 | 30 × 24 × 4 |
| **Long-head** | 720 hours (1 call) | Strategic 30-day overview |
| **Short-head** | 96 steps × 30 days | Tactical daily refinement |
| **Total TFT Calls** | 31 (1 long + 30 short) | Optimized for thesis demo |
| **Physics Model** | PVLib (weather→power) | Solar geometry baseline |
| **Blending** | 3-way hierarchical | Short + Long + Physics |
| **RL Controller** | Adaptive weights | By day/horizon/confidence |
| **Encoder** | Weather-only (novelty) | No PV sensor dependency |

---

***REMOVED******REMOVED*** Hierarchical Refinement Concept

***REMOVED******REMOVED******REMOVED*** "Drone + Fighter Jet" Analogy

**Drone (Long-head):**
- Flies high, sees whole battlefield
- Rough strategic overview of 30 days
- Identifies patterns, trends, weather regimes
- Contribution: 40% weight in blend
- **NOT DISCARDED** - provides strategic context throughout

**Fighter Jet (Short-head):**
- Flies low, precise tactical strikes
- Refines each day to 96×15-min granularity
- Adjusts for intra-day dynamics, ramps
- Contribution: 60% weight in blend
- Called 30 times (once per day)

**Ground Truth (PVLib Physics):**
- Reality check from solar geometry
- Constraints what's physically possible
- Contribution: 30% in blend + hard limits
- Prevents unphysical predictions

***REMOVED******REMOVED******REMOVED*** Why Hierarchical vs Simple?

**Old Simple Architecture (v1.0) ❌:**
- Day 1: Short-head (96@15min)
- Days 2-30: Long-head upsampled (2784@15min)
- Problem: Days 2-30 lose high-frequency details

**New Hierarchical Architecture (v2.0) ✅:**
- Day 0-29: Short-head refines EACH day (30×96)
- Plus: Long-head provides strategic context (40%)
- Benefit: All days get precision + strategy

---

***REMOVED******REMOVED*** Data Flow Pipeline

***REMOVED******REMOVED******REMOVED*** INPUT PHASE

```python
***REMOVED*** Required inputs for hierarchical forecasting
inputs = {
    'forecast_start': pd.Timestamp("2023-11-01 00:00:00", tz="UTC"),
    
    'weather_forecast': pd.DataFrame({
        ***REMOVED*** 30 days @ 15-min (2880 steps)
        'timestamp_utc': pd.date_range(..., periods=2880, freq='15min'),
        'ghi': [...],            ***REMOVED*** Global Horizontal Irradiance (W/m²)
        'dni': [...],            ***REMOVED*** Direct Normal Irradiance (W/m²)
        'dhi': [...],            ***REMOVED*** Diffuse Horizontal Irradiance (W/m²)
        'temp_air': [...],       ***REMOVED*** Air temperature (°C)
        'wind_speed': [...],     ***REMOVED*** Wind speed (m/s)
        'temp_dew': [...],       ***REMOVED*** Dew point (°C)
        'pressure_sea': [...],   ***REMOVED*** Sea level pressure (hPa)
        ***REMOVED*** ... 21 known future covariates total
    }),
    
    'historical_data': pd.DataFrame({
        ***REMOVED*** Historical weather for encoder windows
        ***REMOVED*** Short-head: 96 steps @ 15-min before forecast_start
        ***REMOVED*** Long-head: 168 hours @ 1-hour before forecast_start
        'timestamp_utc': [...],
        ***REMOVED*** Same weather features + PVLib computed from weather
        ***REMOVED*** NO PV power measurements (weather-only encoding)
    })
}
```

***REMOVED******REMOVED******REMOVED*** STEP 1: Generate Physics Baseline

```python
***REMOVED*** Initialize PVLib predictor
from src.inference.pvlib_predictor import PVLibPredictor

predictor = PVLibPredictor("data/metadata/plant_03_metadata.json")
***REMOVED*** Metadata: lat=50.95, lon=6.96, tilt=25°, azimuth=180°, capacity=7358.9kW

***REMOVED*** Generate full 30-day baseline
pvlib_15min = predictor.predict_from_weather(
    weather_df=inputs['weather_forecast']
)
***REMOVED*** Output shape: (2880,)
***REMOVED*** Content: Normalized PV power [0, 1] from solar physics
```

**PVLib Calculation (per timestamp):**

```python
***REMOVED*** 1. Solar Position
solar_pos = pvlib.solarposition.get_solarposition(
    time=timestamp,
    latitude=50.95,
    longitude=6.96,
    altitude=50.0  ***REMOVED*** meters
)
***REMOVED*** Returns: zenith, azimuth, elevation

***REMOVED*** 2. POA Irradiance (tilted plane)
poa = pvlib.irradiance.get_total_irradiance(
    surface_tilt=25.0,       ***REMOVED*** degrees
    surface_azimuth=180.0,   ***REMOVED*** South-facing
    dni=weather['dni'],
    ghi=weather['ghi'],
    dhi=weather['dhi'],
    solar_zenith=solar_pos['apparent_zenith'],
    solar_azimuth=solar_pos['azimuth']
)
***REMOVED*** Returns: poa_global (W/m² on tilted surface)

***REMOVED*** 3. DC Power Conversion
dc_power_kw = (poa['poa_global'] / 1000.0) * 7358.9  ***REMOVED*** capacity
power_normalized = np.clip(dc_power_kw / 7358.9, 0, 1)  ***REMOVED*** [0,1] scale

***REMOVED*** 4. Handle night (sun below horizon)
if solar_pos['elevation'] < 0:
    power_normalized = 0.0
```

***REMOVED******REMOVED******REMOVED*** STEP 2: Long-Head Strategic Overview (1 Call)

```python
***REMOVED*** Model: Long-head TFT checkpoint
***REMOVED*** Config: encoder=168 hours, decoder=720 hours, freq='1h'
***REMOVED*** Purpose: Predict all 30 days in ONE call (strategic view)

long_head_pred = forecaster._predict_long_head(
    forecast_start=inputs['forecast_start'],
    historical_df=inputs['historical_data'],
    weather_df=inputs['weather_forecast']
)
***REMOVED*** Output shape: (720,) @ 1-hour
***REMOVED*** Content: Hourly predictions for 30 days (rough strategic overview)

***REMOVED*** Upsample to 15-min using PVLib solar curve shape
from src.inference.physics_glue import upsample_with_pvlib_shape

long_upsampled = upsample_with_pvlib_shape(
    hourly_predictions=long_head_pred,  ***REMOVED*** (720,)
    pvlib_15min=pvlib_15min,            ***REMOVED*** (2880,)
    method="proportional"
)
***REMOVED*** Output shape: (2880,) @ 15-min
***REMOVED*** Energy preserved: sum(long_upsampled[h*4:(h+1)*4]) == long_head_pred[h]
```

**Upsampling Algorithm:**

```python
def upsample_with_pvlib_shape(hourly_predictions, pvlib_15min):
    """
    Distribute each hourly value across 4 quarter-hours
    proportional to PVLib intra-hour solar curve.
    """
    upsampled_15min = np.zeros(len(pvlib_15min))
    
    for h in range(len(hourly_predictions)):
        ***REMOVED*** Extract PVLib shape for this hour (4 steps)
        pvlib_hour = pvlib_15min[h*4:(h+1)*4]
        
        ***REMOVED*** Calculate proportions
        if pvlib_hour.sum() > 1e-6:  ***REMOVED*** Daytime
            proportions = pvlib_hour / pvlib_hour.sum()
        else:  ***REMOVED*** Nighttime (avoid divide-by-zero)
            proportions = np.ones(4) / 4.0
        
        ***REMOVED*** Distribute hourly value
        upsampled_15min[h*4:(h+1)*4] = hourly_predictions[h] * proportions
    
    return upsampled_15min
```

***REMOVED******REMOVED******REMOVED*** STEP 3: Rolling Daily Refinement (30 Calls)

```python
***REMOVED*** Initialize RL meta-controller
from src.inference.rl_controller import RLMetaController

rl_controller = RLMetaController(mode="heuristic")

***REMOVED*** Initialize output array
forecast_15min = np.zeros(2880)
short_head_daily = []  ***REMOVED*** Store for analysis
blend_weights_daily = []

***REMOVED*** Loop over 30 days
for day in range(30):
    ***REMOVED*** Day time bounds
    day_start = inputs['forecast_start'] + pd.Timedelta(days=day)
    day_start_idx = day * 96
    day_end_idx = (day + 1) * 96
    
    ***REMOVED*** STEP 3A: Get RL-adaptive blend weights
    weights = rl_controller.get_blend_weights(
        day=day,
        weather_confidence=0.8  ***REMOVED*** From weather API
    )
    ***REMOVED*** Returns: {alpha_short, alpha_long, alpha_ml, alpha_pvlib}
    ***REMOVED*** Example Day 0: {0.65, 0.35, 0.71, 0.29}
    ***REMOVED*** Example Day 15: {0.35, 0.65, 0.635, 0.365}
    blend_weights_daily.append(weights)
    
    ***REMOVED*** STEP 3B: Short-head tactical refinement for this day
    short_day_pred = forecaster._predict_short_head_for_day(
        day_start=day_start,
        day_idx=day,
        historical_df=inputs['historical_data'],
        weather_df=inputs['weather_forecast']
    )
    ***REMOVED*** Output shape: (96,) @ 15-min
    ***REMOVED*** Content: High-precision forecast for this specific day
    short_head_daily.append(short_day_pred)
    
    ***REMOVED*** STEP 3C: Extract corresponding slices
    long_slice = long_upsampled[day_start_idx:day_end_idx]    ***REMOVED*** (96,)
    pvlib_slice = pvlib_15min[day_start_idx:day_end_idx]      ***REMOVED*** (96,)
    
    ***REMOVED*** STEP 3D: Hierarchical 3-way blend
    from src.inference.physics_glue import blend_hierarchical
    
    day_forecast = blend_hierarchical(
        short_pred=short_day_pred,          ***REMOVED*** Tactical (60%)
        long_upsampled=long_slice,          ***REMOVED*** Strategic (40%)
        pvlib_baseline=pvlib_slice,         ***REMOVED*** Physics (30% blend + hard limits)
        alpha_short=weights['alpha_short'],  ***REMOVED*** 0.65 → 0.35 over 30 days
        alpha_long=weights['alpha_long'],    ***REMOVED*** 0.35 → 0.65 over 30 days
        alpha_ml=weights['alpha_ml'],        ***REMOVED*** 0.71 → 0.57 over 30 days
        constraints=True                     ***REMOVED*** Apply hard physics limits
    )
    ***REMOVED*** Output shape: (96,)
    ***REMOVED*** Content: Final blended forecast for this day
    
    ***REMOVED*** Store in output array
    forecast_15min[day_start_idx:day_end_idx] = day_forecast
```

***REMOVED******REMOVED******REMOVED*** STEP 4: Validation

```python
def validate_forecast(forecast, pvlib_baseline):
    """Run sanity checks on final forecast."""
    
    checks = []
    
    ***REMOVED*** Check 1: Shape
    assert forecast.shape == (2880,), f"Shape {forecast.shape} != (2880,)"
    checks.append("✓ Shape correct")
    
    ***REMOVED*** Check 2: Range
    assert forecast.min() >= 0, f"Negative values: {forecast.min()}"
    assert forecast.max() <= 1.0, f"Over 100%: {forecast.max()}"
    checks.append(f"✓ Range valid [{forecast.min():.3f}, {forecast.max():.3f}]")
    
    ***REMOVED*** Check 3: Night constraint
    night_mask = pvlib_baseline < 0.01
    night_violations = (forecast[night_mask] > 0.01).sum()
    assert night_violations == 0, f"{night_violations} night violations"
    checks.append("✓ All night hours zero")
    
    ***REMOVED*** Check 4: Capacity constraint
    max_allowed = pvlib_baseline * 1.2  ***REMOVED*** 120% of PVLib
    capacity_violations = (forecast > max_allowed).sum()
    assert capacity_violations == 0, f"{capacity_violations} capacity violations"
    checks.append("✓ Capacity limits respected")
    
    return checks
```

***REMOVED******REMOVED******REMOVED*** OUTPUT

```python
***REMOVED*** Final hierarchical forecast
result = {
    'final': forecast_15min,                 ***REMOVED*** (2880,) - Final blended
    'short_head_daily': np.array(short_head_daily),  ***REMOVED*** (30, 96)
    'long_head': long_head_pred,             ***REMOVED*** (720,)
    'pvlib_15min': pvlib_15min,              ***REMOVED*** (2880,)
    'long_upsampled': long_upsampled,        ***REMOVED*** (2880,)
    'blend_weights': blend_weights_daily     ***REMOVED*** List[Dict] - 30 days
}
```

---

***REMOVED******REMOVED*** Component Specifications

***REMOVED******REMOVED******REMOVED*** 1. PVLibPredictor Class

**File:** `src/inference/pvlib_predictor.py` (374 lines)  
**Status:** ✅ Complete and tested

**Methods:**

```python
class PVLibPredictor:
    def __init__(self, metadata_path: Path):
        """Load plant metadata (lat, lon, tilt, azimuth, capacity)"""
        
    def predict_from_weather(self, weather_df: pd.DataFrame) -> np.ndarray:
        """
        Generate physics-based forecast from weather data.
        
        Args:
            weather_df: DataFrame with columns [timestamp_utc, ghi, dni, dhi, 
                        temp_air, wind_speed, ...]
        
        Returns:
            predictions: Array shape (N,), normalized [0,1]
        """
        
    def predict_clear_sky(self, start_date: str, num_steps: int, 
                         freq: str = "15min") -> np.ndarray:
        """
        Generate clear-sky baseline (Ineichen model).
        Fallback when weather forecast unavailable.
        """
```

**Test Results:**
```
Clear-sky 30-day @ 15-min:
  Shape: (2880,)
  Range: [0.0, 0.639]
  Mean: 0.128 (12.8% capacity)
  Daylight: 1022/2880 steps (35.5%)
  Peak: 0.639 (63.9% capacity at solar noon)
```

***REMOVED******REMOVED******REMOVED*** 2. Physics Glue Functions

**File:** `src/inference/physics_glue.py` (402 lines)  
**Status:** ✅ Complete and tested

***REMOVED******REMOVED******REMOVED******REMOVED*** Function: `upsample_with_pvlib_shape()`

```python
def upsample_with_pvlib_shape(
    hourly_predictions: np.ndarray,  ***REMOVED*** (H,)
    pvlib_15min: np.ndarray,         ***REMOVED*** (H*4,)
    method: str = "proportional"
) -> np.ndarray:                     ***REMOVED*** (H*4,)
    """
    Upsample hourly to 15-min using PVLib solar curve shape.
    Preserves energy: sum(output[h*4:(h+1)*4]) == input[h]
    """
```

**Test:**
```python
Input:  [0.5, 0.6, 0.7] (3 hours)
Output: [12 values @ 15-min]
✓ Hour 0 sum: 0.500 == 0.500
✓ Hour 1 sum: 0.600 == 0.600
✓ Hour 2 sum: 0.700 == 0.700
```

***REMOVED******REMOVED******REMOVED******REMOVED*** Function: `blend_hierarchical()` ⭐ NEW in v2.0

```python
def blend_hierarchical(
    short_pred: np.ndarray,           ***REMOVED*** (N,) - Tactical precision
    long_upsampled: np.ndarray,       ***REMOVED*** (N,) - Strategic context
    pvlib_baseline: np.ndarray,       ***REMOVED*** (N,) - Physics reference
    alpha_short: float = 0.6,         ***REMOVED*** Short-head weight
    alpha_long: float = 0.4,          ***REMOVED*** Long-head weight
    alpha_ml: float = 0.7,            ***REMOVED*** ML vs physics weight
    constraints: bool = True,
    max_capacity_multiplier: float = 1.2
) -> np.ndarray:                      ***REMOVED*** (N,) - Final blended
    """
    3-way hierarchical blend: short + long + physics.
    
    Architecture:
        Layer 1: ML Ensemble = α_short × short + α_long × long
        Layer 2: Physics Blend = α_ml × ML + (1-α_ml) × PVLib
        Layer 3: Hard Constraints = night=0, capacity≤120%, non-negative
    """
    
    ***REMOVED*** Layer 1: ML Ensemble
    ml_blend = alpha_short * short_pred + alpha_long * long_upsampled
    
    ***REMOVED*** Layer 2: Physics-Aware Blend
    physics_blend = alpha_ml * ml_blend + (1.0 - alpha_ml) * pvlib_baseline
    
    ***REMOVED*** Layer 3: Hard Constraints
    if constraints:
        final = apply_physics_constraints(
            physics_blend, pvlib_baseline, max_capacity_multiplier
        )
    else:
        final = physics_blend
    
    return final
```

**Mathematical Formulation:**

Let:
- $S(t)$ = Short-head prediction at time $t$
- $L(t)$ = Long-head upsampled prediction at time $t$
- $P(t)$ = PVLib physics baseline at time $t$
- $\alpha_s$, $\alpha_l$ = ML ensemble weights ($\alpha_s + \alpha_l = 1$)
- $\alpha_m$ = ML vs physics weight

Then:
$$
\begin{aligned}
\text{ML}(t) &= \alpha_s \cdot S(t) + \alpha_l \cdot L(t) \\
\text{Blend}(t) &= \alpha_m \cdot \text{ML}(t) + (1-\alpha_m) \cdot P(t) \\
\text{Final}(t) &= \text{Constrain}(\text{Blend}(t), P(t))
\end{aligned}
$$

Where $\text{Constrain}$ applies:
$$
\text{Final}(t) = \begin{cases}
0 & \text{if } P(t) < 0.01 \text{ (night)} \\
\min(\text{Blend}(t), 1.2 \cdot P(t)) & \text{if } P(t) \geq 0.01 \\
\max(\text{Blend}(t), 0) & \text{(non-negative)}
\end{cases}
$$

**Test:**
```python
short = [0.8, 0.6, 0.4, 0.2, 0.0]
long  = [0.7, 0.5, 0.3, 0.1, 0.0]
pvlib = [0.65, 0.48, 0.25, 0.05, 0.0]

result = blend_hierarchical(short, long, pvlib, 
                           alpha_short=0.6, alpha_long=0.4, alpha_ml=0.7)

***REMOVED*** Expected calculation:
***REMOVED*** ML[0] = 0.6*0.8 + 0.4*0.7 = 0.76
***REMOVED*** Blend[0] = 0.7*0.76 + 0.3*0.65 = 0.727
***REMOVED*** Final[0] = constrain(0.727, 0.65) = 0.727 (≤ 0.78 = 120% of 0.65) ✓

Output: [0.727, 0.536, 0.300, 0.060, 0.000]
✓ Layer 1: ML ensemble computed
✓ Layer 2: Physics blend applied
✓ Layer 3: Constraints enforced
```

***REMOVED******REMOVED******REMOVED******REMOVED*** Function: `apply_physics_constraints()`

```python
def apply_physics_constraints(
    predictions: np.ndarray,
    pvlib_baseline: np.ndarray,
    max_capacity_multiplier: float = 1.2
) -> np.ndarray:
    """
    Enforce physical reality:
    1. Night constraint: PVLib < 0.01 → force to 0
    2. Capacity constraint: Clip to [0, PVLib × 1.2]
    3. Non-negative constraint
    """
    constrained = predictions.copy()
    
    ***REMOVED*** Constraint 1: Night
    night_mask = pvlib_baseline < 0.01
    constrained[night_mask] = 0.0
    
    ***REMOVED*** Constraint 2: Capacity
    max_allowed = pvlib_baseline * max_capacity_multiplier
    constrained = np.clip(constrained, 0.0, max_allowed)
    
    return constrained
```

***REMOVED******REMOVED******REMOVED*** 3. RLMetaController Class

**File:** `src/inference/rl_controller.py` (282 lines)  
**Status:** ✅ Complete (heuristic v1.0, RL training v2.0 future)

**Purpose:** Adaptive blend weight optimization based on forecast horizon and conditions

**Key Method:**

```python
class RLMetaController:
    def get_blend_weights(
        self,
        day: int,                         ***REMOVED*** Forecast day (0-29)
        weather_confidence: float = 0.8,  ***REMOVED*** [0,1] from weather API
        recent_accuracy: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Get RL-adaptive blend weights for hierarchical forecasting.
        
        Current (v1.0): Heuristic rules
        Future (v2.0): RL policy network
        
        Returns:
            {
                'alpha_short': float,   ***REMOVED*** Short-head weight in ML ensemble
                'alpha_long': float,    ***REMOVED*** Long-head weight in ML ensemble
                'alpha_ml': float,      ***REMOVED*** ML weight vs physics
                'alpha_pvlib': float    ***REMOVED*** Physics weight (1 - alpha_ml)
            }
        """
        ***REMOVED*** Rule 1: Short-head weight decreases with horizon
        if day < 7:
            alpha_short, alpha_long = 0.65, 0.35  ***REMOVED*** Near: trust short
        elif day < 14:
            alpha_short, alpha_long = 0.5, 0.5    ***REMOVED*** Mid: balanced
        else:
            alpha_short, alpha_long = 0.35, 0.65  ***REMOVED*** Far: trust long
        
        ***REMOVED*** Rule 2: ML weight decreases with horizon and low confidence
        base_ml_weight = 0.75
        confidence_penalty = (1.0 - weather_confidence) * 0.2  ***REMOVED*** Max -20%
        horizon_penalty = day * 0.005  ***REMOVED*** -0.5% per day
        
        alpha_ml = base_ml_weight - confidence_penalty - horizon_penalty
        alpha_ml = np.clip(alpha_ml, 0.45, 0.85)
        
        return {
            'alpha_short': alpha_short,
            'alpha_long': alpha_long,
            'alpha_ml': alpha_ml,
            'alpha_pvlib': 1.0 - alpha_ml
        }
```

**Weight Evolution:**

| Day | Horizon | α_short | α_long | α_ml | α_pvlib | Interpretation |
|-----|---------|---------|--------|------|---------|----------------|
| 0 | Near | 0.65 | 0.35 | 0.71 | 0.29 | Trust short-head precision |
| 3 | Near | 0.65 | 0.35 | 0.695 | 0.305 | Still near-term confident |
| 7 | Mid | 0.50 | 0.50 | 0.675 | 0.325 | Balanced, slight physics shift |
| 14 | Mid-Far | 0.35 | 0.65 | 0.640 | 0.360 | Trust long-head strategic |
| 21 | Far | 0.35 | 0.65 | 0.605 | 0.395 | Weather uncertain, shift physics |
| 29 | Very Far | 0.35 | 0.65 | 0.565 | 0.435 | High physics weight |

**Weather Confidence Impact (Day 7):**

| Confidence | α_ml | Reasoning |
|------------|------|-----------|
| 0.9 (high) | 0.695 | Trust ML predictions |
| 0.7 (med) | 0.655 | Moderate shift to physics |
| 0.5 (low) | 0.615 | Significant physics reliance |

**Future RL Training (v2.0):**

```python
***REMOVED*** State space: [day, weather_conf, recent_rmse_short, recent_rmse_long, 
***REMOVED***               pvlib_deviation, season_encoding]
***REMOVED*** Action space: [alpha_short, alpha_long, alpha_ml] (continuous [0,1])
***REMOVED*** Reward: R = -RMSE - λ × API_cost + β × computational_efficiency

***REMOVED*** When ground truth available:
controller.record_forecast(day, forecast, metadata)
***REMOVED*** ... accumulate data ...
controller.train_rl_policy(ground_truth_df)  ***REMOVED*** PPO/SAC training
```

***REMOVED******REMOVED******REMOVED*** 4. PhysicsAwareForecaster Class

**File:** `src/inference/physics_aware_forecaster.py` (492 lines)  
**Status:** ⚠️ Architecture complete, TFT inference placeholders

**Class Structure:**

```python
class PhysicsAwareForecaster:
    def __init__(
        self,
        short_ckpt: Path,      ***REMOVED*** Short-head TFT checkpoint
        long_ckpt: Path,       ***REMOVED*** Long-head TFT checkpoint
        plant_metadata: Path,  ***REMOVED*** Plant config JSON
        device: Optional[str] = None
    ):
        ***REMOVED*** Load TFT models
        self.short_model = TemporalFusionTransformer.load_from_checkpoint(short_ckpt)
        self.long_model = TemporalFusionTransformer.load_from_checkpoint(long_ckpt)
        
        ***REMOVED*** Initialize PVLib & RL controller
        self.pvlib_predictor = PVLibPredictor(plant_metadata)
        self.rl_controller = RLMetaController(mode="heuristic")
        
    def predict_30d(
        self,
        forecast_start: pd.Timestamp,
        weather_df: pd.DataFrame,
        historical_df: pd.DataFrame,
        return_components: bool = False
    ) -> Union[np.ndarray, Dict]:
        """
        Generate hierarchical 30-day forecast.
        See STEP 1-4 in Data Flow Pipeline above.
        """
        
    def _predict_long_head(self, ...) -> np.ndarray:
        """
        Long-head TFT inference (1 call → 720 hours).
        TODO: Implement real TFT inference.
        Currently: Synthetic placeholder.
        """
        
    def _predict_short_head_for_day(self, day_start, day_idx, ...) -> np.ndarray:
        """
        Short-head TFT inference for single day (96 steps @ 15-min).
        TODO: Implement real TFT inference.
        Currently: Synthetic placeholder.
        """
```

---

***REMOVED******REMOVED*** Weather-Only Encoding (Thesis Novelty)

***REMOVED******REMOVED******REMOVED*** Key Innovation

**Traditional Approach:**
- Encoder: Historical PV power measurements (SCADA data)
- Decoder: Future weather → predict PV power
- Problem: Requires working SCADA, error accumulation

**Our Approach (Weather-Only):**
- Encoder: Historical weather + PVLib(weather)
- Decoder: Future weather + PVLib(weather_forecast)
- NO PV power measurements in encoder!
- Benefits: Robust to missing SCADA, no error accumulation

***REMOVED******REMOVED******REMOVED*** Encoder/Decoder Structure

**Short-Head (96@15min):**
```python
***REMOVED*** Encoder input (96 steps before day_start):
encoder_features = [
    'ghi', 'dni', 'dhi',              ***REMOVED*** Weather (measured)
    'temp_air', 'wind_speed', ...,
    'pvlib_dc_normalized',            ***REMOVED*** PVLib computed from weather
    'hour_of_day', 'day_of_year',    ***REMOVED*** Time encodings
    ***REMOVED*** NO 'pv_measurement' column!
]

***REMOVED*** Decoder input (96 steps from day_start):
decoder_features = [
    'ghi', 'dni', 'dhi',              ***REMOVED*** Weather (forecast)
    'temp_air', 'wind_speed', ...,
    'pvlib_dc_normalized',            ***REMOVED*** PVLib computed from forecast
    'hour_of_day', 'day_of_year',
]
```

**Long-Head (720@1h):**
```python
***REMOVED*** Encoder input (168 hours before forecast_start):
***REMOVED*** Same features as short-head but hourly frequency

***REMOVED*** Decoder input (720 hours from forecast_start):
***REMOVED*** Same features as short-head but hourly frequency
```

***REMOVED******REMOVED******REMOVED*** Training vs Inference

**Training (with historical SCADA):**
```python
***REMOVED*** Target: Historical PV power (messy but usable)
train_df = pd.DataFrame({
    'timestamp_utc': [...],
    'pv_measurement': [...],  ***REMOVED*** Target for training
    'ghi': [...],             ***REMOVED*** Encoder features (weather only)
    'pvlib_dc_normalized': compute_pvlib(ghi, dni, dhi),
    ...
})

***REMOVED*** Loss: MSE(predicted, pv_measurement)
```

**Inference (weather-only, no SCADA):**
```python
***REMOVED*** Input: Pure weather forecast
inference_df = pd.DataFrame({
    'timestamp_utc': [...],
    ***REMOVED*** NO 'pv_measurement' column!
    'ghi': [...],             ***REMOVED*** From OpenMeteo API
    'pvlib_dc_normalized': compute_pvlib(ghi, dni, dhi),
    ...
})

***REMOVED*** Model predicts: PV power from weather + physics
predictions = model.predict(inference_df)
```

***REMOVED******REMOVED******REMOVED*** Production Data Flow

```
OpenMeteo Weather API
  (14-day forecast)
         ↓
Historical Weather DB
  (for encoder window)
         ↓
   Compute PVLib
 (from weather only)
         ↓
TFT Encoder (weather + pvlib)
         ↓
   TFT Decoder
   (future weather + pvlib)
         ↓
    TFT Predictions
         ↓
Hierarchical Blend + Physics
         ↓
  30-day PV Forecast
```

**No PV sensors needed at inference time!**

---

***REMOVED******REMOVED*** Implementation Status

***REMOVED******REMOVED******REMOVED*** ✅ Completed Components

| Component | File | Lines | Status |
|-----------|------|-------|--------|
| PVLib Predictor | `pvlib_predictor.py` | 374 | ✅ Complete |
| Physics Glue | `physics_glue.py` | 402 | ✅ Complete |
| RL Controller | `rl_controller.py` | 282 | ✅ Heuristic v1.0 |
| Forecaster Architecture | `physics_aware_forecaster.py` | 492 | ✅ Structure |
| Hierarchical Blend | `blend_hierarchical()` | - | ✅ Tested |
| Upsampling | `upsample_with_pvlib_shape()` | - | ✅ Tested |
| Constraints | `apply_physics_constraints()` | - | ✅ Tested |

***REMOVED******REMOVED******REMOVED*** ⚠️ TODO: TFT Integration

**Remaining Work (2-3 hours):**

1. Study `src/inference/offline_predict_tft.py`
   - Understand TimeSeriesDataSet creation
   - Batch preparation from DataFrames
   - Encoder/decoder window extraction
   - Feature alignment

2. Implement `_predict_short_head_for_day()`:
```python
def _predict_short_head_for_day(self, day_start, day_idx, historical_df, weather_df):
    ***REMOVED*** 1. Extract encoder window (96 steps before day_start, weather only)
    encoder_end = day_start
    encoder_start = day_start - pd.Timedelta(hours=24)
    encoder_data = historical_df[
        (historical_df['timestamp_utc'] >= encoder_start) &
        (historical_df['timestamp_utc'] < encoder_end)
    ]
    
    ***REMOVED*** 2. Extract decoder window (96 steps from day_start, weather forecast)
    decoder_start = day_start
    decoder_end = day_start + pd.Timedelta(hours=24)
    decoder_data = weather_df[
        (weather_df['timestamp_utc'] >= decoder_start) &
        (weather_df['timestamp_utc'] < decoder_end)
    ]
    
    ***REMOVED*** 3. Add PVLib features
    encoder_data = add_pvlib_features(encoder_data)
    decoder_data = add_pvlib_features(decoder_data)
    
    ***REMOVED*** 4. Combine into model input format
    batch = prepare_tft_batch(encoder_data, decoder_data, self.short_model)
    
    ***REMOVED*** 5. Run inference
    with torch.no_grad():
        outputs = self.short_model.predict(batch)
    
    ***REMOVED*** 6. Extract q50 quantile
    predictions = outputs[:, :, self.short_model.loss.quantiles.index(0.5)]
    
    return predictions.cpu().numpy()  ***REMOVED*** (96,)
```

3. Implement `_predict_long_head()`:
```python
def _predict_long_head(self, forecast_start, historical_df, weather_df):
    ***REMOVED*** Similar to short-head but:
    ***REMOVED*** - Encoder: 168 hours (7 days) before forecast_start
    ***REMOVED*** - Decoder: 720 hours (30 days) from forecast_start
    ***REMOVED*** - Hourly frequency instead of 15-min
    ...
    return predictions.cpu().numpy()  ***REMOVED*** (720,)
```

---

***REMOVED******REMOVED*** Testing & Validation

***REMOVED******REMOVED******REMOVED*** Test 1: Component Tests (All Passing ✅)

```bash
$ python -m src.inference.physics_glue

[TEST 1] Upsampling: ✅
    Energy preserved: ✓
    Shape correct: ✓

[TEST 2] Hierarchical Blend: ✅
    Layer 1 (ML): ✓
    Layer 2 (Physics): ✓
    Layer 3 (Constraints): ✓

[TEST 3] RL Controller: ✅
    Weight adaptation by day: ✓
    Confidence impact: ✓
    API selection: ✓
```

***REMOVED******REMOVED******REMOVED*** Test 2: End-to-End (Synthetic Data Passing ✅)

```python
***REMOVED*** Test with synthetic TFT predictions
forecast = forecaster.predict_30d(
    forecast_start="2023-11-01 00:00:00",
    weather_df=synthetic_weather,
    historical_df=synthetic_history,
    return_components=True
)

***REMOVED*** Validation results:
assert forecast['final'].shape == (2880,)  ✓
assert (forecast['final'] >= 0).all()  ✓
assert (forecast['final'][pvlib < 0.01] < 0.01).all()  ✓ Night=0
assert forecast['short_head_daily'].shape == (30, 96)  ✓
assert forecast['long_head'].shape == (720,)  ✓
```

***REMOVED******REMOVED******REMOVED*** Test 3: Production Validation (Pending TFT)

```python
***REMOVED*** With real TFT inference:
- [ ] Load test set from parquet
- [ ] Run hierarchical forecast
- [ ] Compare vs offline_predict_tft.py baseline
- [ ] Validate RMSE, MAE, R²
- [ ] Check constraint violations (should be 0)
- [ ] Verify blend weights evolution
```

---

***REMOVED******REMOVED*** TFT Integration Roadmap

***REMOVED******REMOVED******REMOVED*** Phase 1: Study Reference Implementation (30 min)

```bash
***REMOVED*** Study offline batch prediction
$ python -m src.inference.offline_predict_tft \
    --train_parquet data/.../train.parquet \
    --test_parquet data/.../test.parquet \
    --run_dir experiments/.../run_dir \
    --out_parquet outputs/test_preds.parquet
```

**Key learnings needed:**
- How TimeSeriesDataSet is created from DataFrame
- How encoder/decoder windows are extracted
- Feature column naming and ordering
- Batch dict structure for model.predict()
- Quantile extraction from output tensor

***REMOVED******REMOVED******REMOVED*** Phase 2: Implement Short-Head (1 hour)

```python
***REMOVED*** In physics_aware_forecaster.py
def _predict_short_head_for_day(self, ...):
    ***REMOVED*** TODO items:
    ***REMOVED*** 1. Window extraction (96 encoder + 96 decoder)
    ***REMOVED*** 2. Feature engineering (match training format)
    ***REMOVED*** 3. TimeSeriesDataSet creation
    ***REMOVED*** 4. Batch preparation
    ***REMOVED*** 5. Model inference
    ***REMOVED*** 6. Quantile extraction
    pass
```

***REMOVED******REMOVED******REMOVED*** Phase 3: Implement Long-Head (1 hour)

```python
def _predict_long_head(self, ...):
    ***REMOVED*** Similar to short-head but hourly
    ***REMOVED*** Encoder: 168 hours
    ***REMOVED*** Decoder: 720 hours
    pass
```

***REMOVED******REMOVED******REMOVED*** Phase 4: Integration Testing (30 min)

```python
***REMOVED*** Test with real checkpoints
forecaster = PhysicsAwareForecaster(
    short_ckpt=Path("experiments/.../shorthead/best.ckpt"),
    long_ckpt=Path("experiments/.../longhead/best.ckpt"),
    ...
)

***REMOVED*** Run on test set
forecast = forecaster.predict_30d(...)

***REMOVED*** Validate against ground truth
rmse = compute_rmse(forecast, ground_truth)
***REMOVED*** Target: RMSE < 0.10 (normalized)
```

---

***REMOVED******REMOVED*** Production Deployment

***REMOVED******REMOVED******REMOVED*** API Integration

```python
***REMOVED*** Pseudo-code for production system
import requests
from src.inference.physics_aware_forecaster import PhysicsAwareForecaster

***REMOVED*** 1. Fetch weather forecast
weather_df = requests.get("https://api.openmeteo.com/...").json_to_df()

***REMOVED*** 2. Load historical weather (no SCADA!)
historical_df = db.query("SELECT * FROM weather WHERE ...")

***REMOVED*** 3. Generate forecast
forecaster = PhysicsAwareForecaster(...)
forecast = forecaster.predict_30d(
    forecast_start=pd.Timestamp.now(tz="UTC"),
    weather_df=weather_df,
    historical_df=historical_df
)

***REMOVED*** 4. Store results
db.insert("forecasts", forecast)

***REMOVED*** 5. Update RL controller (when ground truth available)
if ground_truth_available:
    forecaster.rl_controller.record_forecast(...)
```

---

***REMOVED******REMOVED*** Architecture Verification

***REMOVED******REMOVED******REMOVED*** ✅ Requirements Checklist

- [x] Hierarchical refinement (long strategic + short tactical)
- [x] Long-head preserved (40% weight, not discarded)
- [x] Short-head refines ALL 30 days (not just Day 1)
- [x] 3-way hierarchical blend (short + long + physics)
- [x] RL adaptive weights (by day/horizon/confidence)
- [x] Total 31 TFT calls (1 long + 30 short)
- [x] Weather-only encoding (no SCADA dependency)
- [x] Physics constraints (night=0, capacity≤120%)
- [x] Output shape (2880,) @ 15-min
- [x] Energy conservation in upsampling
- [x] Validation checks (shape, range, night, capacity)
- [ ] Real TFT inference (TODO)

***REMOVED******REMOVED******REMOVED*** Novelty Claims

1. ✅ **Weather-only encoding** - No PV sensor dependency
2. ✅ **Hierarchical refinement** - Long strategic + short tactical
3. ✅ **RL meta-control** - Adaptive blend weights
4. ✅ **3-way physics blend** - ML ensemble + physics constraints

---

***REMOVED******REMOVED*** References

**Implementation Files:**
- `src/inference/pvlib_predictor.py` - Physics baseline
- `src/inference/physics_glue.py` - Blending & constraints
- `src/inference/rl_controller.py` - Adaptive weights
- `src/inference/physics_aware_forecaster.py` - Main orchestration

**Documentation:**
- `PHYSICS_GLUE_IMPLEMENTATION.md` - Quick reference
- `PHYSICS_CONSTRAINED_INFERENCE_DETAILED.md` - This document
- `HIERARCHICAL_ARCHITECTURE_AUDIT.md` - Verification report

---

**Version:** 2.0 (Hierarchical Refinement)  
**Last Updated:** 2026-01-02  
**Status:** 🔒 Locked for TFT Integration  
**Next:** Implement real TFT inference (2-3 hours)
