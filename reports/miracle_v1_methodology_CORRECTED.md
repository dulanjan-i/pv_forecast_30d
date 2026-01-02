# MiRACLE v1.0: Methodology

## Multi-Horizon Photovoltaic Forecasting via Temporal Fusion Transformers with Physics-Informed Features

---

## 1. Overview

This study presents MiRACLE v1.0 (Meta Intelligent Reinforcement Driven Adaptive Control for Learning Based Ensembles), a hybrid deep learning framework for multi-horizon photovoltaic power forecasting. The current implementation combines data-driven temporal modeling (Temporal Fusion Transformers) with physics-based solar irradiance features (PVLib library). The architecture addresses two distinct forecasting horizons: (1) short-term operational forecasting at 15-minute resolution over a 24-hour horizon (96 timesteps), and (2) long-term strategic forecasting at hourly resolution over a 30-day horizon (720 timesteps).

**Note on System Design:** MiRACLE's meta-controller architecture (reinforcement learning-based model selection and ensemble weighting) represents planned future work. The v1.0 release focuses on establishing robust TFT+PVLib baselines with validated transfer learning protocols.

---

## 2. Data Sources and Preprocessing

### 2.1 Study Site and Dataset

The analysis utilizes operational data from five commercial PV installations in Germany (designated plant_01, plant_02, plant_03, plant_05, plant_06) spanning January 2023 through February 2024. Site-specific metadata includes geographic coordinates (latitude, longitude), panel orientation (tilt angle, azimuth), and installed capacity (kW).

**Temporal Coverage:**
- Training period: January 1, 2023 23:00 UTC – November 30, 2023 23:45 UTC (142,190 samples across 5 plants)
- Validation period: December 2, 2023 00:00 UTC – February 29, 2024 23:45 UTC (36,952 samples across 5 plants)
- Temporal resolution: 15-minute granularity (96 intervals per day)

### 2.2 Target Variable Construction

Normalized AC power output serves as the prediction target:

$$
P_{\text{norm}}(t) = \frac{P_{\text{AC}}(t)}{P_{\text{nameplate}}}
$$

where $P_{\text{AC}}(t)$ represents measured AC power (kW) at time $t$, and $P_{\text{nameplate}}$ denotes the site's installed capacity (kWp). This normalization enables cross-site learning by standardizing output magnitudes while preserving temporal dynamics and efficiency variations.

### 2.3 Meteorological Inputs

Historical weather data provides horizon-known meteorological drivers at 15-minute resolution. The feature set comprises:

**Direct Solar Resource Variables:**
- `shortwave_radiation_instant` (W/m²): broadband hemispheric irradiance
- `direct_normal_irradiance_instant` (DNI, W/m²): beam component normal to solar vector
- `global_tilted_irradiance_instant` (GTI, W/m²): total irradiance on panel plane
- `diffuse_radiation_instant` (W/m²): scattered isotropic component
- `direct_radiation_instant` (W/m²): direct beam component

**Atmospheric State Variables:**
- `temperature_2m` (°C)
- `relative_humidity_2m` (%)
- `cloud_cover` (0–1 fraction)
- `surface_pressure` (hPa)
- `precipitation` (mm/hr)

**Wind Characteristics:**
- `wind_speed_10m` (m/s)
- `wind_direction_10m` (degrees)

**Encoded Cloud Conditions:**
- `weather_code`: categorical sky state descriptor

**Data Source Note:** Weather data processing pipeline is implemented in `src/features/germany_build_tft_weather.py`. The script reads pre-processed plant-specific weather parquets (already at 15-minute resolution) and aligns them to TFT base splits via inner join on `(plant_id, timestamp_utc)`. No spatial or temporal interpolation is performed; the weather data is provided as-is from OpenMeteo historical API at 15-minute granularity.

All meteorological features are treated as time-varying known reals in the TFT architecture.

### 2.4 Physics-Based Feature Engineering via PVLib

To incorporate first-principles solar physics, we compute auxiliary features using PVLib-Python (version per `environment.yml`). For each site and timestamp, the pipeline executes:

**Step 1: Solar Position Calculation**

Computed via PVLib's `pvlib.solarposition.get_solarposition()` function, which implements the Solar Position Algorithm (SPA):

$$
\theta_z, \phi_s = f_{\text{SPA}}(\lambda, \phi, t)
$$

where $\theta_z$ is solar zenith angle (degrees), $\phi_s$ is solar azimuth angle (degrees), given site longitude $\lambda$, latitude $\phi$, and UTC timestamp $t$.

**Step 2: Extraterrestrial Irradiance**

$$
I_0(t) = \text{pvlib.irradiance.get\_extra\_radiation}(t)
$$

PVLib's built-in function accounts for Earth-Sun distance variation using astronomical algorithms.

**Step 3: Plane-of-Array Irradiance Decomposition**

Using the Hay-Davies transposition model (`pvlib.irradiance.get_total_irradiance(model="haydavies")`), we decompose tilted irradiance:

```python
poa = pvlib.irradiance.get_total_irradiance(
    surface_tilt=tilt,
    surface_azimuth=azm,
    solar_zenith=solar_zenith,
    solar_azimuth=solar_azimuth,
    dni=dni,
    ghi=ghi,
    dhi=dhi,
    dni_extra=dni_extra,
    model="haydavies",
    albedo=0.2,  # ground reflectance coefficient
)
```

Returns POA components: `poa_global`, `poa_direct`, `poa_diffuse`, `poa_ground_diffuse`.

**Step 4: Cell Temperature Estimation**

Cell temperature computed using PVLib's SAPM (Sandia Array Performance Model) or PVsyst thermal models with built-in parameter sets for "open_rack_glass_glass" mounting configuration:

```python
def compute_cell_temperature(poa_global, temp_air, wind):
    # Try SAPM with version-compatible API
    params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"]["open_rack_glass_glass"]
    a, b, deltaT = params["a"], params["b"], params["deltaT"]
    return pvlib.temperature.sapm_cell(poa_global, temp_air, wind, a=a, b=b, deltaT=deltaT)
```

**Step 5: PVWatts DC-AC Conversion**

Expected power estimated using PVLib's PVWatts model (`pvlib.pvsystem.pvwatts_dc` and `pvlib.inverter.pvwatts`). Temperature coefficient is set to `gamma_pdc = -0.003` (typical crystalline silicon value per line 291 of `src/features/germany_build_pvlib_for_tft.py`). Inverter efficiency uses PVLib's PVWatts default model (no explicit eta parameter; loss calculation internal to `pvlib.inverter.pvwatts`).

**Resulting PVLib Feature Set (8 variables):**
- `pvlib_solar_zenith` (degrees)
- `pvlib_solar_azimuth` (degrees)
- `pvlib_poa_global` (W/m²)
- `pvlib_poa_direct` (W/m²)
- `pvlib_poa_diffuse` (W/m²)
- `pvlib_poa_ground_diffuse` (W/m²)
- `pvlib_dc_kw` (kW)
- `pvlib_ac_kw` (kW)

**Implementation:** `src/features/germany_build_pvlib_for_tft.py:_compute_pvlib_for_one_plant()`

---

## 3. Model Architecture: Temporal Fusion Transformer

### 3.1 TFT Framework

The Temporal Fusion Transformer (Lim et al., 2021) is a multi-horizon attention-based architecture designed for time series forecasting with heterogeneous input types. The model comprises:

1. **Gating mechanisms**: Gated Residual Networks (GRN) for adaptive feature selection
2. **Variable selection**: Static covariate encoder and temporal input selection networks
3. **Sequence processing**: LSTM encoder for historical context compression (internal TFT component, distinct from optional external LSTM embeddings pipeline)
4. **Temporal attention**: Multi-head self-attention with positional encoding

**Note:** The "LSTM encoder" here refers to TFT's internal sequence-to-sequence LSTM block (integral part of the TFT architecture). This is separate from the optional external LSTM embeddings feature pipeline, which was excluded in v1.0 to prevent data leakage during global pretraining (see Section 4.1).

The architecture handles:
- Static metadata (time-invariant site characteristics)
- Time-varying known inputs (meteorological forecasts, physics features)
- Time-varying unknown inputs (lagged targets, historical observations)

**Implementation:** PyTorch Forecasting library's `TemporalFusionTransformer.from_dataset()` method. Model configuration in `src/configs/tft_v1.py`.

### 3.2 Feature Role Assignment

**Temporal Index and Grouping:**
- `timestamp_utc`: primary time key (15-minute resolution)
- `time_idx`: integer sequence index (auto-generated per site via `TimeSeriesDataSet`)
- `plant_id`: categorical group identifier

**Target Variable:**
- `power_norm`: normalized AC power output

**Time-Varying Known Reals (Horizon-Known):**
- Raw weather variables (13 features per `src/features/germany_build_tft_weather.py:WEATHER_COLS`)
- PVLib physics features (8 variables as enumerated above)

**Time-Varying Unknown Reals (Encoder-Only):**
- Target history: `power_norm` (autoregressive context)
- LSTM embeddings: Excluded in v1.0 to prevent data leakage during global pretraining (see Section 4.3)

**Static Metadata:**
- Site identity encoded via categorical `plant_id` (embedded to learnable vector space by TFT)

### 3.3 Normalization and Scaling

- **Target normalization**: `GroupNormalizer(groups=["plant_id"], transformation="softplus")` per PyTorch Forecasting API, ensuring positive outputs and site-specific scaling
- **Input standardization**: TFT's internal preprocessing applies z-score normalization to continuous reals using training set statistics
- **Relative time encoding**: `add_relative_time_idx=True` provides position-within-sequence information

### 3.4 Loss Function and Probabilistic Output

The model optimizes quantile loss to produce distributional forecasts:

$$
\mathcal{L}_{\text{quantile}} = \frac{1}{NTQ} \sum_{i=1}^{N} \sum_{t=1}^{T} \sum_{q \in Q} \rho_q(y_{it} - \hat{y}_{it}^q)
$$

where:
$$
\rho_q(u) = \begin{cases}
q \cdot u & \text{if } u \geq 0 \\
(q - 1) \cdot u & \text{if } u < 0
\end{cases}
$$

Default quantile set: $Q = \{0.1, 0.5, 0.9\}$ (P10, median, P90). The median (0.5 quantile) serves as the point forecast for evaluation.

**Implementation:** `pytorch_forecasting.metrics.QuantileLoss`

---

## 4. Experimental Design

### 4.1 Ablation Study: Feature Contribution Analysis

To quantify the incremental value of PVLib physics features versus pure data-driven approaches, we conducted a controlled ablation study with four configurations:

**Mode A: TFT-Only (Baseline)**
- Excluded: LSTM embeddings, PVLib features
- Retained: Raw weather variables, target history, static metadata

**Mode B: TFT+PVLib (Winner)**
- Excluded: LSTM embeddings
- Retained: Raw weather, PVLib physics features, target history

**Mode C: TFT+LSTM**
- Excluded: PVLib features
- Retained: Raw weather, LSTM embeddings, target history
- **Status:** Results show anomalously low loss (potential data leakage), excluded from production

**Mode D: Full (TFT+LSTM+PVLib)**
- Retained: All feature groups
- **Status:** Results show anomalously low loss (potential data leakage), excluded from production

**Ablation Parquet Generation:** `src/features/germany_make_tft_ablation_parquets.py` creates mode-specific parquets by dropping column groups.

**Ablation Training Protocol:**
- Encoder length: 96 timesteps (24 hours)
- Prediction length: 96 timesteps (24 hours)
- Batch size: 512, gradient accumulation: 8 (effective batch 4096)
- Learning rate: Grid search {8e-4, 1.2e-3}
- Dropout: Grid search {0.05, 0.15}
- Hidden size: 64, LSTM layers: 2, attention heads: 4
- Optimizer: AdamW with weight decay 1e-4 (per `src/training/train_tft_v1.py:optimizer` construction)
- Early stopping: Patience 5 epochs on validation loss

### 4.2 Multi-Horizon Training Strategy

**Short-Term Head (15-minute, 24-hour horizon):**
- Architecture: TFT with `encoder_length=96`, `prediction_length=96`
- Data: 15-minute resolution parquets
- Hyperparameters (winner from ablation):
  - Mode: TFT+PVLib
  - Learning rate: 1.2e-3
  - Dropout: 0.15
  - Batch size: 512, accumulation: 8
  - Hidden size: 64, LSTM layers: 2, attention heads: 4

**Long-Term Head (1-hour, 30-day horizon):**
- Architecture: TFT with `encoder_length=720`, `prediction_length=720`
- Data: Hourly resampled via mean aggregation from 15-min source (script: `src/data/make_hourly_from_15min_parquets.py`)

### 4.3 Pretraining and Transfer Learning Protocol

**Global Pretraining (Data-Leakage Prevention):**

1. **Source construction**: For target plant `plant_03`, construct global training corpus by **excluding** all `plant_03` data
   - Script: `src/data/make_global_noleak_parquets.py`
   - Command: `--exclude_plant_id "plant_03"`
   - Output: Training on plants {01, 02, 05, 06} only

2. **Global model training**: Train TFT on the no-leak corpus using winner configuration (TFT+PVLib)

3. **Checkpoint extraction**: Save state_dict from best validation epoch

**Fine-Tuning on Target Plant:**

Two regimes compared via controlled experiment:

**Cold Start:**
- Initialize: Random Glorot/Xavier initialization (PyTorch defaults)
- Training data: Target plant (`plant_03`) 15-min splits only
- Hyperparameters: Learning rate 2e-3, dropout 0.15

**Warm Start (Transfer Learning):**
- Initialize: Load pretrained weights via `model.load_state_dict(torch.load(init_path))`
- Training data: Target plant (`plant_03`) 15-min splits only
- Hyperparameters: Learning rate 8e-4, dropout 0.15

**Implementation:** `src/training/train_tft_v1.py:main()` lines 410-426

**Multi-Seed Validation:**
- Each regime executed with three random seeds: {42, 43, 44}
- Seed setting: `torch.manual_seed(seed)`, `np.random.seed(seed)`, `random.seed(seed)` per `train_tft_v1.py:_seed_everything()`

---

## 5. Training Infrastructure and Implementation

### 5.1 Computational Environment

- **Hardware:** NVIDIA H100 PCIe GPUs (verified from partition `gpuh100` in SBATCH scripts and `gpu_name` column in `ablation_summary_extended.csv`)
- **Software Stack:**
  - Python 3.11
  - PyTorch (version per `environment.yml`)
  - PyTorch Forecasting (version per `environment.yml`)
  - PyTorch Lightning (version per `environment.yml`)
  - PVLib-Python (version per `environment.yml`)

- **Precision:** FP32 default, with optional mixed-precision (bf16/fp16) via `--precision` and `--enable_amp` flags (implementation: `train_tft_v1.py` lines 470-480)

### 5.2 Training Monitoring

Per-epoch metrics logged to `<run_dir>/logs/metrics.csv`:
- `train_loss`, `val_loss`: quantile loss on respective splits
- `best_val_loss`: running minimum validation loss
- `improved`: binary flag for validation improvement (based on `--min_delta` threshold)
- `bad_epochs`: patience counter for early stopping
- `lr`: current learning rate
- Throughput metrics: `samples_per_sec`, `train_it_per_sec`, `val_it_per_sec`
- Resource utilization: `gpu_peak_mem_gb`, `epoch_sec`

**Early Stopping Policy:**
- Trigger: No validation improvement for `patience` consecutive epochs (default 5)
- Restore: Best checkpoint weights saved as `best.ckpt`

**Implementation:** `src/training/train_tft_v1.py` lines 620-678

### 5.3 Checkpoint Format and Loading

**Checkpoint Save Format:**
```python
torch.save(model.state_dict(), run_dir / "checkpoints" / "best.ckpt")
```

Checkpoints contain **PyTorch state_dict only** (not full Lightning checkpoints). Saved at line 668 of `train_tft_v1.py`.

**Loading Protocol:**
```python
model = TemporalFusionTransformer.from_dataset(train_ds, <hyperparams>)
sd = torch.load(ckpt_path, map_location="cpu")
model.load_state_dict(sd, strict=True)
```

**Reference:** `src/validation/eval_short_head.py` lines 160-173

### 5.4 Data Versioning and Reproducibility

All data transformations scripted via modular Python pipeline:
- `src/features/germany_build_pvlib_for_tft.py`: PVLib feature computation
- `src/features/germany_build_tft_weather.py`: Weather alignment
- `src/features/germany_merge_tft_full.py`: Feature fusion
- `src/data/make_global_noleak_parquets.py`: Leakage-free split generation
- `src/data/make_hourly_from_15min_parquets.py`: Temporal resampling

Configuration management:
- `src/configs/tft_v1.py`: Feature role definitions
- SBATCH scripts in `hpc/jobs/`: SLURM job specifications

---

## 6. Evaluation Metrics

Performance assessed via point forecast errors on validation split:

**Root Mean Squared Error (RMSE):**
$$
\text{RMSE} = \sqrt{\frac{1}{NT} \sum_{i=1}^{N} \sum_{t=1}^{T} (y_{it} - \hat{y}_{it})^2}
$$

**Mean Absolute Error (MAE):**
$$
\text{MAE} = \frac{1}{NT} \sum_{i=1}^{N} \sum_{t=1}^{T} |y_{it} - \hat{y}_{it}|
$$

**Reporting Convention:**
- Point forecast: Median quantile (0.5) from distributional output
- Aggregation: Flattened over all horizons (96 or 720 steps)
- Comparison: Relative improvement computed as $\frac{\text{Baseline} - \text{Method}}{\text{Baseline}} \times 100\%$

**Implementation:** Streaming metrics in `src/validation/eval_short_head.py:update_streaming_metrics()`

---

## 7. Quality Assurance and Data Integrity

**Preprocessing Checks:**
- Temporal alignment verification: No duplicate timestamps per site (enforced in `germany_build_tft_weather.py` lines 109-111)
- NaN handling: Rows with NaN in required feature columns are dropped during preprocessing (verified in `germany_build_global_supermatrix.py` line 148: `df.dropna(subset=required_for_model)`). Weather pipeline raises an error if NaNs exist post-merge (line 136 of `germany_build_tft_weather.py`).
- Irradiance clipping: DNI, GHI, DHI, and POA components clipped at `lower=0.0` to enforce physical constraints (lines 251-284 of `germany_build_pvlib_for_tft.py`)

**Training Stability Diagnostics:**
- Gradient clipping: Implemented via `torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)` with default threshold 0.1 (lines 559 and 564 of `train_tft_v1.py`). Gradient norms are not logged to metrics.csv.
- Loss trajectory inspection: Monitored via metrics.csv logs
- Checkpoint validation: Best model selected by independent validation split

**Reproducibility Measures:**
- Fixed random seeds: Implementation in `train_tft_v1.py:_seed_everything()`
- Version pinning: `environment.yml` with package specifications

---

## 8. Limitations and Design Constraints

1. **Missing LSTM Encoder Integration**: v1.0 excludes upstream LSTM embeddings to prevent data leakage during global pretraining. Future versions will implement safe rollout strategies.

2. **Single-Site Transfer Validation**: Transfer learning validated on one target plant (plant_03). Cross-validation across all sites remains future work.

3. **Weather Forecast Assumptions**: Historical weather data used as proxy for operational forecasts. Spatial/temporal interpolation details require verification.

4. **Static PV System Assumptions**: Model assumes fixed panel tilt, azimuth, and capacity. Does not account for soiling, snow cover, or degradation.

5. **Horizon-Specific Models**: Short and long heads trained independently. No joint optimization or consistency enforcement.

---

## References

1. Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T. (2021). Temporal Fusion Transformers for interpretable multi-horizon time series forecasting. *International Journal of Forecasting*, 37(4), 1748-1764.

2. Holmgren, W. F., Hansen, C. W., & Mikofski, M. A. (2018). pvlib python: A python package for modeling solar energy systems. *Journal of Open Source Software*, 3(29), 884.

---

**Document Version**: v1.0-corrected  
**Last Updated**: January 1, 2026  
**Implementation Base**: MiRACLE v1.0 codebase verification
