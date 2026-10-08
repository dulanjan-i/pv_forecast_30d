# Chapter 4 — Experimental Design and Ablation Studies (Print-Ready)

## 4.1 Data sources

The empirical evaluation in this thesis is based on operational PV generation and meteorological data provided by Syneco Trading GmbH under a data protection agreement. All plants are anonymized and referenced only by plant IDs. The raw PV generation series are organized as time-stamped measurements with a nominal 15-minute cadence. Meteorological covariates are available as hourly and daily time series and are treated as exogenous inputs.

A central challenge in this dataset is heterogeneous data completeness across plants and years. Because completeness affects both model training and evaluation credibility, the experimental design explicitly quantifies data coverage and uses a strict temporal split to prevent data leakage.

## 4.2 Data coverage by plant

Six anonymized PV plants are included in the Germany dataset (plant_01 through plant_06). For calendar year 2023, each plant provides an essentially complete 15-minute timestamp grid spanning the full year (35,040 records per plant-year, with minor variation in one plant). For year 2024, coverage differs: plant_01, plant_02, and plant_04 provide timestamps through early October (26,396 records), whereas plant_03, plant_05, and plant_06 provide timestamps through early April (8,828 records). These timestamp grids reflect expected quarter-hour sampling and show no evidence of duplicated timestamps in the plant-year exports.

Importantly, the presence of a timestamp grid does not imply that PV power values are present. The fraction of non-empty power measurements varies significantly across plants. For 2023, completeness ranges from approximately 23.9% (plant_05) to approximately 72.7% (plant_04), indicating that missingness is a substantial property of the raw data. For 2024, several plant-year exports contain power fields that are entirely empty, even though the timestamps are present. In particular, plant_03 (the target test plant) and plant_04 have no non-empty PV power measurements in the 2024 plant-year export.

To enable a rigorous backtesting protocol for the target plant, a separate ground-truth PV export is available for plant_03 in 2024. This ground-truth series has 5-minute cadence and spans from 1 January 2024 to 28 October 2024. The value completeness of this series is high (approximately 99.6% non-missing), although duplicated timestamps occur, as is common in industrial metering exports. In preprocessing, duplicated timestamps are resolved deterministically and the series is resampled to the 15-minute grid used in the forecasting pipeline.

These coverage properties motivate two methodological safeguards. First, training splits and preprocessing are designed to avoid using the evaluation year for any parameter estimation. Second, headline evaluation for plant_03 is anchored to a ground-truth measurement series rather than to a plant-year export with empty values.

## 4.3 Evaluation metrics

Performance is evaluated using standard regression metrics on capacity-normalized PV output. Let y_i denote the true normalized power and let y-hat_i denote the predicted normalized power. The mean absolute error and root mean squared error are defined as follows.

$$
\mathrm{MAE} = \frac{1}{N} \sum_{i=1}^{N} \left| y_{i} - \hat{y}_{i} \right|
$$

$$
\mathrm{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} \left( y_{i} - \hat{y}_{i} \right)^{2}}
$$

Mean bias error is used to characterize systematic prediction bias, and the coefficient of determination summarizes explained variance.

Because nighttime PV output is near zero, evaluation adopts a physically motivated filter that excludes points where the true normalized power is below a small threshold. This avoids metrics being dominated by trivial nighttime performance and focuses the evaluation on periods when accurate production forecasting is operationally meaningful.

## 4.4 Train–validation–backtest split and leakage prevention

The evaluation design follows a strict time-based split that simulates deployment: models are trained on historical data and evaluated on a future period that is not used during training. The target plant for headline backtesting is plant_03. The year 2024 is reserved exclusively for backtesting on this plant using the ground-truth measurement export described above. All model training, hyperparameter selection, and preprocessing parameter estimation are performed without access to any plant_03 ground-truth observations from 2024.

This temporal isolation provides a concrete guarantee against data leakage. The training phase does not ingest backtesting observations; moreover, all transformations that learn parameters from data, including normalization constants and scaling factors, are fitted on the training period only and are then applied unchanged to the backtesting period. This prevents subtle leakage mechanisms that can occur when preprocessing is computed on the full dataset.

The selection of plant_03 as the test plant is motivated by the availability of a high-quality 2024 ground-truth export and by the requirement to conduct a rigorous future-period backtest. The backtesting period is treated as a sealed benchmark. By fixing the evaluation year in advance and isolating it from training, the methodology reduces the risk of retrospective selection bias and strengthens internal validity.

## 4.5 Ablation study methodology

Ablation studies are used to attribute forecast improvements to architectural components. The ablation strategy follows a system-level view: variants are evaluated under the same backtesting protocol while removing or isolating specific components. This approach is intentionally stricter than relying on training-loop validation loss, because validation loss can be optimistic when covariate shift exists or when validation periods overlap with tuning decisions.

The primary ablation contrasts include (i) a model-only configuration without physics-informed features, (ii) a physics-only baseline, (iii) horizon-isolated forecasting variants, and (iv) the fully integrated MiRACLE configuration combining physics-informed covariates, learned temporal embeddings, and hierarchical inference.

## 4.6 Validity considerations

Three validity considerations are emphasized. First, temporal leakage is mitigated through strict year-based isolation for the test plant and by fitting preprocessing transformations only on training data. Second, heterogeneous missingness is treated as a defining characteristic of the dataset; the evaluation protocol is designed to rely on reliable ground truth for the backtest year rather than on incomplete plant-year exports. Third, anonymization limits site-specific interpretation but strengthens privacy compliance; the thesis therefore prioritizes methodological generality and statistical evidence over site narratives.
