# Chapter 4 — Experimental Design and Ablation Studies (Print-Ready Draft)

## 4.1 Data sources and structure

This thesis evaluates PV forecasting methods using operational PV generation time series and corresponding meteorological covariates provided by Syneco Trading GmbH. In accordance with data protection obligations, plants are anonymized and referred to exclusively by plant IDs. For each plant, the core signal is measured active power (or an equivalent operational generation signal). The raw plant data are organized as time-stamped series with a nominal 15-minute cadence. Meteorological data are provided in aligned time series at hourly and daily resolution, and are treated as exogenous covariates.

A distinctive feature of the dataset is that data availability is heterogeneous across plants and years. Some plant-year combinations contain a complete timestamp grid with partially missing power values (empty measurements), while other combinations contain a timestamp grid with fully missing values, reflecting differences in operational measurement availability or extraction completeness. Because these patterns can influence apparent model performance and the difficulty of learning, the experimental design explicitly quantifies data coverage and uses strict temporal splits to avoid leakage.

## 4.2 Data coverage by plant

The raw PV measurement series are available for six anonymized plants (plant_01 through plant_06). For calendar year 2023, all plants provide an essentially complete 15-minute timestamp grid spanning the full year in local German time. For year 2024, coverage differs across plants: plant_01, plant_02, and plant_04 provide timestamps through approximately early October, while plant_03, plant_05, and plant_06 provide timestamps through approximately early April.

However, the presence of timestamps does not guarantee the presence of measured power values. For 2023, the fraction of non-empty power measurements varies substantially across plants, ranging from approximately 23.9% (plant_05) to approximately 72.7% (plant_04). For 2024, several plants contain extensive missingness in the power-value column. In particular, plant_03 and plant_04 contain timestamp grids for 2024 where the power measurement field is entirely empty in the provided plant-year files, motivating the use of an alternative ground-truth export for evaluation.

For the selected test plant (plant_03), a dedicated ground-truth dataset is available for 2024 at 5-minute cadence. This ground-truth export spans from 1 January 2024 to 28 October 2024 and exhibits very high value completeness (approximately 99.6% non-missing). The ground-truth export also contains duplicated timestamps, which is consistent with operational metering exports that may repeat records under certain system conditions. In preprocessing, such duplicates are resolved deterministically and the series is resampled to the 15-minute grid used by the forecasting models.

To summarize the empirically observed coverage, the following description is based on direct inspection of the raw plant CSV exports.

For plant_01, the 2023 series spans the full calendar year at 15-minute resolution with 35,040 timestamped records, of which approximately 63.8% contain non-empty power values. For 2024, the timestamp grid spans from 1 January 2024 to 1 October 2024 (local time) with 26,396 records, and approximately 62.2% contain non-empty power values.

For plant_02, the 2023 series spans 1 January 2023 through 31 December 2023 with 35,036 records and approximately 54.4% non-empty values. For 2024, the timestamp grid again spans from 1 January 2024 through 1 October 2024 with 26,396 records and approximately 50.4% non-empty values.

For plant_03, the 2023 series spans the full year with 35,040 records and approximately 53.2% non-empty values. For 2024, the plant-year export includes 8,828 timestamped records spanning 1 January 2024 through 1 April 2024 but contains no non-empty power values. Consequently, the 2024 evaluation for plant_03 relies on the separate ground-truth export described above.

For plant_04, the 2023 series spans the full year with 35,040 records and approximately 72.7% non-empty values. For 2024, the timestamp grid spans 1 January 2024 through 1 October 2024 with 26,396 records but contains no non-empty power values.

For plant_05, the 2023 series spans the full year with 35,040 records and approximately 23.9% non-empty values, indicating particularly high missingness in the plant export for that year. For 2024, the timestamp grid spans 1 January 2024 through 1 April 2024 with 8,828 records and approximately 62.1% non-empty values.

For plant_06, the 2023 series spans the full year with 35,040 records and approximately 49.9% non-empty values. For 2024, the timestamp grid spans 1 January 2024 through 1 April 2024 with 8,828 records and approximately 59.3% non-empty values.

The heterogeneity of completeness motivates two methodological safeguards. First, training and validation are structured to avoid learning directly from missingness artifacts that could contaminate evaluation. Second, headline evaluation is based on an explicit backtesting protocol using a ground-truth measurement export for the test plant, rather than on timestamp grids with empty measurement fields.

## 4.3 Evaluation metrics

Model performance is quantified using standard regression metrics on capacity-normalized PV output. Let y_i denote the true normalized power at time i and ŷ_i denote the predicted normalized power. The mean absolute error is defined as MAE = (1/N)∑|y_i − ŷ_i|, and the root mean squared error is defined as RMSE = √((1/N)∑(y_i − ŷ_i)^2). The coefficient of determination R² is reported to summarize explained variance, and mean bias error is reported to characterize systematic over- or under-prediction.

Because PV output is identically or near-zero during nighttime, evaluation adopts a physically motivated filter that removes near-zero targets. Specifically, points with y below a small threshold are excluded from headline evaluation to prevent metrics from being dominated by trivial nighttime predictions. This choice aligns the evaluation with the operational requirement to forecast daytime production accurately.

## 4.4 Train–validation–backtest splitting strategy

The experimental design is centered on a temporally ordered split that simulates deployment. For the target plant used in headline evaluation (plant_03), the year 2024 is reserved exclusively for out-of-sample backtesting. All model training, encoder training, and hyperparameter selection are performed without access to the 2024 backtesting data. In practical terms, this means that the training phase uses data from earlier periods (principally 2023 for plant_03), and the evaluation phase is conducted on the held-out 2024 ground-truth export.

This design provides an explicit guarantee against data leakage. The training pipeline does not ingest any observations from the backtesting year; moreover, all preprocessing operations that learn parameters from data—such as scaling, normalization constants, and imputation rules—are fitted strictly on the training portion and then applied unchanged to the backtesting portion. This avoids subtle leakage pathways that can arise when preprocessing statistics are computed on the full dataset.

Selecting plant_03 as the target test plant reflects the availability of an explicit ground-truth measurement export for 2024 and the desire to perform a rigorous future-period backtest. The backtesting year was fixed prior to evaluating models, ensuring that reported performance is not the result of tuning to the evaluation period. By isolating the backtesting year, the evaluation becomes a credible estimate of real-world deployment behavior under distribution shift and weather-forecast uncertainty.

## 4.5 Ablation methodology

Ablation studies are used to attribute performance gains to architectural components rather than to implementation details. The ablation logic follows a system-level view: components are removed or isolated while holding the evaluation protocol fixed. The key ablations compare (i) a purely data-driven forecaster without physics-informed features, (ii) a purely physics-based baseline without learned models, (iii) single-component or single-horizon variants of the forecasting system, and (iv) the fully integrated MiRACLE configuration.

A critical methodological choice is that ablation conclusions are based on the same out-of-sample backtesting protocol used for headline results. Training-loop metrics, such as validation loss during model fitting, are not treated as thesis evidence for operational performance because they can be optimistic under covariate shift or when the validation period is not truly out-of-sample. System-level backtesting provides a consistent and deployment-relevant basis for comparing ablation variants.

## 4.6 Validity considerations

Three threats to validity are addressed explicitly. First, temporal leakage is mitigated through strict year-based isolation for the test plant and by fitting preprocessing transformations only on training data. Second, missingness and data availability heterogeneity are treated as part of the problem setting rather than ignored: the evaluation protocol focuses on ground-truth measurements for the backtesting year, and the interpretation of results considers the effect of incomplete measurement exports. Third, the anonymization of plants limits site-specific interpretation but strengthens privacy compliance; the thesis therefore emphasizes methodological generality and statistical evidence over site narratives.
