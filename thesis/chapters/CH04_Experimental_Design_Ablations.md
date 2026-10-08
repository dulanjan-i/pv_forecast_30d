# Chapter 4 — Experimental Design & Ablation Studies

## 4.1 Datasets

MiRACLE is validated through a combination of (i) pretraining/transfer datasets used to learn temporal representations and (ii) a target-plant backtest protocol used for thesis headline performance.

### 4.1.1 PVDAQ dataset (US, Farm2107)

**Role in the thesis:** exploratory pretraining + hyperparameter selection for the LSTM encoder.

- Data source: PVDAQ (utility-scale PV)
- Canonical pretraining configuration: `experiments/lstm/pretrain_farm2107_CANONICAL.yaml`
- Result tracking: `reports/lstm_results.md`

This dataset is used to validate that the encoder objective learns stable temporal structure, and to produce a robust initialization point for transfer.

### 4.1.2 Germany regional dataset

**Role in the thesis:** regional domain adaptation to reduce distribution shift when transferring to the target plant.

- Regional pooled data products:
  - `data/processed/pretraining/germany/global/supermatrix_base.parquet`
  - `data/processed/pretraining/germany/global/fold_{k}_{train,val}.parquet`
  - `data/processed/pretraining/germany/global/regional_{train,val}.parquet`
- Training scripts:
  - `src/training/train_global_lstm_v3.py` (rolling-origin CV)
  - `src/training/train_regional_lstm.py` (single canonical regional encoder)

**No-leak constraints:** windows are built per-plant, scalers are fold-safe, and time regularity checks exclude windows spanning gaps.

### 4.1.3 Target plant dataset

**Role in the thesis:** end-to-end system backtest for 2024 (canonical thesis headline evaluation).

- Canonical benchmark suite outputs (tables/plots/text):
  - `freeze/final_thesis_v1/benchmarks/thesis_formatted_v3/`

The benchmark suite evaluates MiRACLE and baselines against **Ground Truth Plant 03** using a fixed join/filter protocol.

### 4.1.4 Weather API data

MiRACLE consumes weather inputs via an API layer (historical reanalysis/archives for backtests, forecast endpoints for real-time). The Phase-1 report enumerates the weather variable set and derived PVLib features:

- `THESIS_RESULTS_PHASE1.md`


### 4.1.5 Dataset summary table (thesis-ready)

| Dataset | Region | Purpose | Notes |
| --- | ---: | --- | --- |
| PVDAQ Farm2107 | US | LSTM encoder hyperparameter selection | Exploratory pretraining stage |
| Germany regional pooled | DE | Regional adaptation / no-leak encoder training | Per-plant windowing prevents leakage |
| Target plant (Plant 03) | DE (Bavaria) | Canonical 2024 end-to-end backtest | Thesis headline metrics |
| Weather API / ERA5 archive | Global | Known covariates | Inputs for both training and backtesting |

(If desired, this table can be expanded with exact time ranges and sample counts extracted from the processed parquet metadata.)


## 4.2 Evaluation metrics

MiRACLE uses standard regression metrics on normalized capacity output $y$:

- Mean Absolute Error (MAE):

$$
\mathrm{MAE} = \frac{1}{N}\sum_{i=1}^{N} |y_i - \hat{y}_i|
$$

- Root Mean Squared Error (RMSE):

$$
\mathrm{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^{N} (y_i - \hat{y}_i)^2}
$$

- Normalized RMSE (nRMSE): in this repository’s canonical outputs, $\mathrm{nRMSE}=\mathrm{RMSE}$ because targets are already capacity-normalized.

- Mean Bias Error (MBE):

$$
\mathrm{MBE} = \frac{1}{N}\sum_{i=1}^{N} (\hat{y}_i - y_i)
$$

- Coefficient of determination $R^2$.

### Canonical filtering and protocol

For the thesis headline backtests:

- Night filtering is applied: **$y_{true} \ge 0.01$**.
- Canonical benchmark outputs report **N = 394,008** under this filter.

See `freeze/final_thesis_v1/benchmarks/thesis_formatted_v3/text/results.md`.


## 4.3 Systematic ablation study methodology

MiRACLE’s development is framed as systematic elimination and integration testing:

- isolate each hypothesized contributor (physics features, temporal encoding, hybrid inference),
- evaluate it under a fixed, reproducible backtest protocol,
- retain components only when they improve performance and robustness.

Ablations are therefore defined at the **system level** (end-to-end inference backtest), not only at the model training loop level.


## 4.4 Ablation Study 1: Component contribution analysis

### 4.4.1 Experiment design

We compare the following system configurations under the same 2024 inference backtest:

1. **TFT-only**: transformer forecaster without learned encoder assistance or physics glue.
2. **PVLib-Physics-only**: physics baseline without ML.
3. **Short-TFT-only**: isolated short-head forecaster (no hierarchical blending).
4. **Long-TFT-only**: isolated long-head forecaster (no hierarchical blending).
5. **MiRACLE v1.0 Core**: full integration (hybrid, physics-aware, hierarchical).

### 4.4.2 Results and analysis (canonical thesis evidence)

**Primary source (canonical):**

- `freeze/final_thesis_v1/benchmarks/thesis_formatted_v3/text/results.md`
- `freeze/final_thesis_v1/benchmarks/thesis_formatted_v3/tables/overall_metrics.csv`


![Overall RMSE ablation comparison (canonical 2024 backtest)](../figures/ablations/ablation_rmse_overall.png)

Vector/PDF version: [../figures/ablations/ablation_rmse_overall.pdf](../figures/ablations/ablation_rmse_overall.pdf)

Headline result (overall RMSE, night-filtered):

- MiRACLE v1.0 Core: RMSE = **0.11713**
- TFT-Only: RMSE = 0.140186
- PVLib-Physics-Only: RMSE = 0.163976
- Short-TFT-Only: RMSE = 0.167144
- Long-TFT-Only: RMSE = 0.223948

**Horizon-disaggregated evidence:**

- Lead-bucket table: `freeze/final_thesis_v1/benchmarks/thesis_formatted_v3/tables/lead_bucket_metrics_long.csv`

**Statistical evidence (paired daily deltas vs baseline):**

- `freeze/final_thesis_v1/benchmarks/thesis_formatted_v3/tables/paired_daily_deltas_vs_baseline.csv`

This paired-delta view provides bootstrap confidence intervals for mean daily deltas, enabling rigorous significance discussion without relying on training-loop losses.

### 4.4.3 Interpretation

- **LSTM contribution:** the encoder supplies a stable temporal representation that reduces dependence on raw autoregressive power signals alone.
- **PVLib contribution:** physics features and constraints provide strong priors (e.g., nighttime=0, plausible peak limits).
- **Synergy:** the best overall performance arises from the joint use of (i) learned temporal embeddings and (ii) physics-aware hierarchical inference.

### 4.4.4 Reproducibility note on training-loop ablation logs

The repository also contains `experiments/tft/runs/germany/ablations/ablation_summary_extended.csv`, which summarizes training job metadata (hardware, wallclock, best validation loss). These values are retained for reproducibility, but **thesis performance claims are anchored to the canonical end-to-end backtests under `freeze/`**, not to training-loop validation losses.


## 4.5 Ablation Study 2: Transfer learning impact

### 4.5.1 Experiment design

We compare **warm-start** (pretrained/regionalized encoder and system) against a **cold-start** variant under the same backtest protocol.

Canonical evaluation artifact:

- `freeze/final_thesis_v1/eval/rq2_warm_vs_cold/text/results.md`

### 4.5.2 Results

Overall (night-filtered):

- Warm-start MiRACLE v1.0 (Core): RMSE = **0.11713**
- Cold-start MiRACLE v1.0: RMSE = **0.119183**

Lead-bucket deltas show warm-start advantages across near-, mid-, and long-term buckets (see the `Lead buckets` table in the canonical RQ2 artifact).

### 4.5.3 Data efficiency and convergence

Training-time efficiency and convergence speed are supported by the LSTM experiment tracking artifacts:

- `reports/lstm_results.md`
- `docs/archive/AUDIT_LSTM_PRETRAIN.md`

(Additional learning-curve plots can be generated directly from the logged `metrics.csv` files in the corresponding run directories if required for the thesis figures list.)


## 4.6 Experimental justification summary

MiRACLE’s final architecture is justified by:

- component-level evidence that physics features and temporal encodings provide complementary gains,
- transfer-learning evidence that warm-starting improves robustness and reduces error,
- systematic elimination of leakage-prone or non-generalizing experimental variants documented in the audit trail.
