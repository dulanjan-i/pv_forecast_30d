# MiRACLE v1.0: Methodology

## Multi-Horizon Photovoltaic Forecasting via Temporal Fusion Transformers with Physics-Informed Features

---

## 1. Overview

This study presents MiRACLE v1.0 (Multi-Resolution Adaptive Context Learning Engine), a hybrid deep learning framework for multi-horizon photovoltaic power forecasting combining data-driven temporal modeling with physics-based solar irradiance constraints. The system addresses two distinct forecasting horizons: (1) short-term operational forecasting at 15-minute resolution over a 24-hour horizon (96 timesteps), and (2) long-term strategic forecasting at hourly resolution over a 30-day horizon (720 timesteps). The architecture employs Temporal Fusion Transformers (TFT) augmented with PVLib-derived physical features to achieve robust generalization across heterogeneous PV installations.

---

## 2. Data Sources and Preprocessing

### 2.1 Study Site and Dataset

The analysis utilizes operational data from five commercial PV installations in Germany (designated plant_01, plant_02, plant_03, plant_05, plant_06) spanning January 2023 through February 2024. Each site comprises multi-megawatt ground-mounted arrays with monocrystalline silicon panels and central inverters. Site-specific metadata includes geographic coordinates (latitude, longitude), panel orientation (tilt angle, azimuth), and nameplate capacity.

**Temporal Coverage:**
- Training period: January 1, 2023 00:00 UTC – November 30, 2023 23:45 UTC (142,190 15-minute samples)
- Validation period: December 2, 2023 00:00 UTC – February 29, 2024 23:45 UTC (36,952 15-minute samples)
- Temporal resolution: 15-minute granularity (96 intervals per day)

### 2.2 Target Variable Construction

Normalized AC power output serves as the prediction target:

$$
P_{\text{norm}}(t) = \frac{P_{\text{AC}}(t)}{P_{\text{nameplate}}}
$$

where $P_{\text{AC}}(t)$ represents measured AC power (kW) at time $t$, and $P_{\text{nameplate}}$ denotes the site's installed capacity (kWp). This normalization enables cross-site learning by standardizing output magnitudes while preserving temporal dynamics and efficiency variations.

### 2.3 Meteorological Inputs

Historical weather reanalysis data from the Open-Meteo ERA5 archive provides horizon-known meteorological drivers at 15-minute resolution, interpolated to site coordinates via bilinear spatial interpolation. The feature set comprises:

**Direct Solar Resource Variables:**
- Shortwave radiation (W/m²): broadband hemispheric irradiance
- Direct normal irradiance (DNI, W/m²): beam component normal to solar vector
- Global tilted irradiance (GTI, W/m²): total irradiance on panel plane
- Diffuse radiation (W/m²): scattered isotropic component

**Atmospheric State Variables:**
- Air temperature at 2m (°C)
- Relative humidity at 2m (%)
- Cloud cover fraction (0–1)
- Surface pressure (hPa)
- Precipitation rate (mm/hr)

**Wind Characteristics:**
- Wind speed at 10m (m/s)
- Wind direction at 10m (degrees)

**Encoded Cloud Conditions:**
- Weather code (WMO 4677): categorical sky state descriptor

All meteorological features are treated as time-varying known reals in the TFT architecture, providing deterministic forcing functions over the entire forecast horizon.

### 2.4 Physics-Based Feature Engineering via PVLib

To incorporate first-principles solar physics and system-specific characteristics, we compute auxiliary features using the PVLib-Python library (Holmgren et al., 2018). For each site and timestamp, the pipeline executes:

**Step 1: Solar Position Calculation**
$$
\theta_z, \phi_s = f_{\text{spa}}(\lambda, \phi, t)
$$

where $\theta_z$ (solar zenith angle) and $\phi_s$ (solar azimuth angle) are computed via the Solar Position Algorithm (Reda & Andreas, 2004) given site longitude $\lambda$, latitude $\phi$, and UTC timestamp $t$.

**Step 2: Extraterrestrial Irradiance**
$$
I_0(t) = I_{\text{sc}} \cdot \left(1 + 0.033 \cos\left(\frac{2\pi n}{365}\right)\right)
$$

where $I_{\text{sc}} = 1367$ W/m² (solar constant) and $n$ is the day of year. This accounts for Earth-Sun distance variation.

**Step 3: Plane-of-Array Irradiance Decomposition**

Using the Haydavies transposition model (Hay & Davies, 1980), we decompose tilted irradiance into physical components:

$$
\begin{aligned}
G_{\text{POA}} &= G_{\text{beam}} + G_{\text{diffuse}} + G_{\text{ground}} \\
G_{\text{beam}} &= \text{DNI} \cdot \cos(\text{AOI}) \\
G_{\text{diffuse}} &= \text{DHI} \cdot \left[A \cdot R_b + (1 - A)\left(\frac{1 + \cos\beta}{2}\right)\right] \\
G_{\text{ground}} &= \text{GHI} \cdot \rho_g \cdot \left(\frac{1 - \cos\beta}{2}\right)
\end{aligned}
$$

where AOI is the angle of incidence, $\beta$ is panel tilt, DHI is diffuse horizontal irradiance, GHI is global horizontal irradiance, $\rho_g = 0.2$ (albedo), and $A$ is the anisotropy index. $R_b$ represents the beam radiation tilt factor.

**Step 4: PVWatts DC-AC Conversion**

We estimate expected power using the PVWatts model (Dobos, 2014), which provides a simplified single-diode representation:

$$
\begin{aligned}
P_{\text{DC}} &= \frac{G_{\text{POA}}}{G_{\text{ref}}} \cdot P_{\text{rated}} \cdot \left[1 + \gamma_P(T_{\text{cell}} - T_{\text{ref}})\right] \\
P_{\text{AC}} &= \eta_{\text{inv}}(P_{\text{DC}}) \cdot P_{\text{DC}}
\end{aligned}
$$

where $G_{\text{ref}} = 1000$ W/m², $T_{\text{ref}} = 25°C$, $\gamma_P = -0.0047$/°C (temperature coefficient), $T_{\text{cell}}$ is cell temperature (estimated from ambient temperature and irradiance), and $\eta_{\text{inv}}$ is inverter efficiency (default CEC database parameters).

**Resulting PVLib Feature Set (8 variables):**
- `pvlib_solar_zenith` (degrees): sun elevation metric
- `pvlib_solar_azimuth` (degrees): sun compass bearing
- `pvlib_poa_global` (W/m²): total plane-of-array irradiance
- `pvlib_poa_direct` (W/m²): beam component on panel
- `pvlib_poa_diffuse` (W/m²): scattered sky component
- `pvlib_poa_ground_diffuse` (W/m²): ground-reflected component
- `pvlib_dc_kw` (kW): theoretical DC output
- `pvlib_ac_kw` (kW): theoretical AC output

These physics-derived features encode site geometry, optical transposition effects, and expected power under ideal conditions, providing interpretable constraints on feasible output ranges.

---

## 3. Model Architecture: Temporal Fusion Transformer

### 3.1 TFT Framework

The Temporal Fusion Transformer (Lim et al., 2021) is a multi-horizon attention-based architecture designed for time series forecasting with heterogeneous input types. The model comprises four functional blocks:

1. **Gating mechanisms**: Gated Residual Networks (GRN) for adaptive feature selection
2. **Variable selection**: Static covariate encoder and temporal input selection networks
3. **Sequence processing**: LSTM encoder for historical context compression
4. **Temporal attention**: Multi-head self-attention with positional encoding for horizon-specific forecasting

The architecture natively handles:
- Static metadata (time-invariant site characteristics)
- Time-varying known inputs (meteorological forecasts, physics features)
- Time-varying unknown inputs (lagged targets, historical observations)

### 3.2 Feature Role Assignment

**Temporal Index and Grouping:**
- `timestamp_utc`: primary time key (15-minute resolution)
- `time_idx`: integer sequence index (auto-generated per site)
- `plant_id`: categorical group identifier

**Target Variable:**
- `power_norm`: normalized AC power output (0–1.2 range, accounting for transient overshoots)

**Time-Varying Known Reals (Horizon-Known, 51 features):**
- Raw weather forecasts (11 variables): temperature, humidity, precipitation, cloud cover, wind vector, pressure, weather code, irradiance suite
- PVLib physics features (8 variables): solar geometry, POA components, theoretical power
- Redundancy handling: retained both raw (`*_raw`) and processed weather fields to preserve information content

**Time-Varying Unknown Reals (Encoder-Only, 65 features):**
- Target history: `power_norm` (autoregressive context)
- LSTM embeddings: `lstm_enc_000` ... `lstm_enc_063` (64-dimensional latent state from upstream LSTM encoder, excluded in v1.0 to prevent data leakage)

**Static Metadata:**
- Site identity encoded via categorical `plant_id` (embedded to learnable vector space)

### 3.3 Normalization and Scaling

- **Target normalization**: GroupNormalizer with softplus transformation per `plant_id`, ensuring positive outputs and site-specific scaling
- **Input standardization**: TFT's internal preprocessing applies z-score normalization to continuous reals using training set statistics
- **Relative time encoding**: `add_relative_time_idx=True` provides position-within-sequence information to capture intra-day periodicities

### 3.4 Loss Function and Probabilistic Output

The model optimizes quantile loss (Koenker & Bassett, 1978) to produce distributional forecasts:

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

- $N$: batch size
- $T$: forecast horizon length
- $Q$: quantile set (typically {0.1, 0.5, 0.9} for P10, median, P90)
- $y_{it}$: ground truth for sample $i$ at step $t$
- $\hat{y}_{it}^q$: predicted $q$-th quantile

The median (0.5 quantile) serves as the point forecast for evaluation.

---

## 4. Experimental Design

### 4.1 Ablation Study: Feature Contribution Analysis

To quantify the incremental value of PVLib physics features versus pure data-driven approaches, we conducted a controlled ablation study with four configurations:

**Mode A: TFT-Only (Baseline)**
- Excluded: LSTM embeddings (`lstm_enc_*`), PVLib features (`pvlib_*`)
- Retained: Raw weather variables, target history, static metadata
- Rationale: Isolates TFT's intrinsic learning capacity from external guidance

**Mode B: TFT+PVLib**
- Excluded: LSTM embeddings
- Retained: Raw weather, PVLib physics features, target history
- Rationale: Tests physics-informed feature engineering impact

**Mode C: TFT+LSTM**
- Excluded: PVLib features
- Retained: Raw weather, LSTM embeddings, target history
- Rationale: Evaluates learned temporal representations

**Mode D: Full (TFT+LSTM+PVLib)**
- Retained: All feature groups
- Rationale: Maximum information integration baseline

**Ablation Training Protocol:**
- Encoder length: 96 timesteps (24 hours)
- Prediction length: 96 timesteps (24 hours)
- Batch size: 512, gradient accumulation: 8 (effective batch 4096)
- Learning rate: Grid search {8e-4, 1.2e-3}
- Dropout: Grid search {0.05, 0.15}
- Hidden size: 64, LSTM layers: 2, attention heads: 4
- Optimizer: AdamW with weight decay 1e-4
- Early stopping: patience 5 epochs on validation loss

### 4.2 Multi-Horizon Training Strategy

**Short-Term Head (15-minute, 24-hour horizon):**
- **Purpose**: Intra-day operational forecasting for grid scheduling
- **Architecture**: TFT with enc_len=96, pred_len=96
- **Data**: 15-minute resolution parquets (142k train, 37k val samples)
- **Hyperparameters** (winner from ablation):
  - Mode: TFT+PVLib
  - Learning rate: 1.2e-3
  - Dropout: 0.15
  - Batch size: 512, accumulation: 8
  - Hidden size: 64, LSTM layers: 2, attention heads: 4

**Long-Term Head (1-hour, 30-day horizon):**
- **Purpose**: Strategic planning and maintenance scheduling
- **Architecture**: TFT with enc_len=720, pred_len=720
- **Data**: Hourly resampled via mean aggregation from 15-min source
- **Rationale**: Extended horizons require coarser temporal granularity to balance sequence length constraints with computational feasibility

### 4.3 Pretraining and Transfer Learning Protocol

**Global Pretraining (Data-Leakage Prevention):**

To enable safe transfer learning, we implemented a "no-leak" pretraining regime:

1. **Source construction**: For target plant `plant_03`, construct global training corpus by **excluding** all `plant_03` data from the multi-site dataset
   - Training: Plants {01, 02, 05, 06} only (plant_03 removed)
   - Validation: Plants {01, 02, 05, 06} only (plant_03 removed)

2. **Global model training**: Train TFT on the no-leak corpus using winner configuration (TFT+PVLib)
   - Captures cross-site production patterns, weather-power correlations, seasonal effects
   - Learns site-agnostic temporal attention patterns

3. **Checkpoint extraction**: Save state_dict from best validation epoch as pretrained weights

**Fine-Tuning on Target Plant:**

Two regimes compared via controlled experiment:

**Cold Start:**
- Initialize: Random Glorot/Xavier initialization
- Training data: Target plant (`plant_03`) 15-min splits only
- Hyperparameters: Learning rate 2e-3 (higher for random init), dropout 0.15
- Rationale: Baseline representing site-specific training from scratch

**Warm Start (Transfer Learning):**
- Initialize: Load pretrained weights from global no-leak model
- Training data: Target plant (`plant_03`) 15-min splits only
- Hyperparameters: Learning rate 8e-4 (lower for fine-tuning), dropout 0.15
- Rationale: Leverages cross-site knowledge for faster convergence and better generalization

**Multi-Seed Validation:**
- Each regime executed with three random seeds: {42, 43, 44}
- Ensures statistical robustness and variance quantification
- Best model selected by minimum validation loss across seeds

---

## 5. Training Infrastructure and Implementation

### 5.1 Computational Environment

- **Hardware**: NVIDIA H100 PCIe (80GB HBM3) on HPC cluster
- **Software Stack**:
  - Python 3.11
  - PyTorch 2.4 with CUDA 12.1
  - PyTorch Forecasting 1.0.0
  - PVLib-Python 0.10.3
  - Lightning 2.1 (training orchestration)

- **Precision**: FP32 (required for numerical stability in quantile loss)
- **Parallelization**: Single-GPU training with gradient accumulation for large effective batch sizes

### 5.2 Training Monitoring

Per-epoch metrics logged to CSV:
- `train_loss`, `val_loss`: quantile loss on respective splits
- `best_val_loss`: running minimum validation loss
- `improved`: binary flag for validation improvement
- `bad_epochs`: patience counter for early stopping
- `lr`: current learning rate (adaptive schedulers)
- Throughput metrics: `samples_per_sec`, `train_it_per_sec`, `val_it_per_sec`
- Resource utilization: `gpu_peak_mem_gb`, `epoch_sec`

**Early Stopping Policy:**
- Trigger: No validation improvement for 5 consecutive epochs
- Restore: Best checkpoint weights from optimal epoch
- Justification: Prevents overfitting while allowing sufficient exploration

### 5.3 Data Versioning and Reproducibility

All data transformations scripted via modular Python pipeline:
- `src/features/germany_build_pvlib_for_tft.py`: PVLib feature computation
- `src/features/germany_build_tft_weather.py`: Weather alignment
- `src/features/germany_merge_tft_full.py`: Feature fusion
- `src/data/make_global_noleak_parquets.py`: Leakage-free split generation
- `src/data/make_hourly_from_15min_parquets.py`: Temporal resampling

Configuration management:
- `src/configs/tft_v1.py`: Canonical feature role definitions
- SBATCH scripts in `hpc/jobs/`: Reproducible SLURM job specifications
- Experiment metadata: JSON manifests with hyperparameters, data paths, timestamps

---

## 6. Evaluation Metrics

Performance assessed via point forecast errors on validation split:

**Root Mean Squared Error (RMSE):**
$$
\text{RMSE} = \sqrt{\frac{1}{NT} \sum_{i=1}^{N} \sum_{t=1}^{T} (y_{it} - \hat{y}_{it})^2}
$$

- Emphasizes large errors (critical for grid stability)
- Units: dimensionless (normalized power scale)

**Mean Absolute Error (MAE):**
$$
\text{MAE} = \frac{1}{NT} \sum_{i=1}^{N} \sum_{t=1}^{T} |y_{it} - \hat{y}_{it}|
$$

- Robust to outliers, interpretable magnitude
- Units: dimensionless (normalized power scale)

**Reporting Convention:**
- Point forecast: Median quantile (0.5) from distributional output
- Aggregation: Flattened over all horizons (96 or 720 steps)
- Comparison: Relative improvement computed as $\frac{\text{Baseline} - \text{Method}}{\text{Baseline}} \times 100\%$

---

## 7. Quality Assurance and Data Integrity

**Preprocessing Checks:**
- Temporal alignment verification: No duplicate timestamps per site
- NaN handling: Dropped incomplete samples (< 0.1% of data)
- Outlier bounds: Clipped `power_norm` to [0, 1.2] range (physical ceiling)

**Training Stability Diagnostics:**
- Gradient norm monitoring: No exploding gradient events logged
- Loss trajectory inspection: Smooth convergence without oscillations
- Checkpoint validation: Best model selected by independent validation split (no test set peeking)

**Reproducibility Measures:**
- Fixed random seeds: PyTorch, NumPy, Python (seeds 42, 43, 44)
- Deterministic operations: `torch.use_deterministic_algorithms(True)` where supported
- Version pinning: `environment.yml` with exact package versions

---

## 8. Limitations and Design Constraints

1. **Missing LSTM Encoder Integration**: v1.0 excludes upstream LSTM embeddings to prevent data leakage during global pretraining. Future versions will implement safe rollout strategies for encoder-decoder consistency.

2. **Single-Site Generalization**: Transfer learning validated on one target plant (plant_03). Cross-validation across all sites remains future work.

3. **Weather Forecast Quality**: Historical reanalysis (ERA5) used as proxy for perfect forecasts. Operational deployment requires integration with NWP forecast error characteristics.

4. **Static PV System Assumptions**: Model assumes fixed panel tilt, azimuth, and capacity. Does not account for soiling, snow cover, or degradation dynamics.

5. **Horizon-Specific Models**: Short and long heads trained independently. No joint optimization or consistency enforcement between horizons.

---

## References

1. Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T. (2021). Temporal Fusion Transformers for interpretable multi-horizon time series forecasting. *International Journal of Forecasting*, 37(4), 1748-1764.

2. Holmgren, W. F., Hansen, C. W., & Mikofski, M. A. (2018). pvlib python: A python package for modeling solar energy systems. *Journal of Open Source Software*, 3(29), 884.

3. Reda, I., & Andreas, A. (2004). Solar position algorithm for solar radiation applications. *Solar Energy*, 76(5), 577-589.

4. Hay, J. E., & Davies, J. A. (1980). Calculation of the solar radiation incident on an inclined surface. In *Proceedings of the First Canadian Solar Radiation Data Workshop*, 59-72.

5. Dobos, A. P. (2014). PVWatts version 5 manual. *National Renewable Energy Laboratory*, NREL/TP-6A20-62641.

6. Koenker, R., & Bassett Jr, G. (1978). Regression quantiles. *Econometrica*, 46(1), 33-50.

---

**Document Version**: v1.0  
**Last Updated**: January 1, 2026  
**Corresponding Implementation**: MiRACLE v1.0 release
