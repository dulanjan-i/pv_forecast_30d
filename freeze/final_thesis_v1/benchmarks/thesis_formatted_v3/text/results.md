# Benchmark suite summary (v3 - FORMATTED)

- Truth: /home/dwijenayake/pv_forecast_30d/freeze/final_thesis_v1/phase1_2024daily_final/processed/ground_truth_15min_utc_capnorm.parquet

- Baseline: MiRACLE v1.0 Core = /home/dwijenayake/pv_forecast_30d/freeze/final_thesis_v1/phase1_2024daily_final/processed/predictions_phase1_baseline_rerun.parquet

- Models: PVLib-Physics-Only, TFT-Only, Short-TFT-Only, Long-TFT-Only

- Night filtering: ON (y_true >= 0.01)

- Truth label: Ground Truth Plant 03


## Overall metrics

| model              |       MAE |     RMSE |    nRMSE |         MBE |        R2 |      N |
|:-------------------|----------:|---------:|---------:|------------:|----------:|-------:|
| Long-TFT-Only      | 0.158866  | 0.223948 | 0.223948 | -0.15345    | -0.310184 | 394008 |
| MiRACLE v1.0 Core  | 0.0841161 | 0.11713  | 0.11713  | -0.00736451 |  0.641595 | 394008 |
| PVLib-Physics-Only | 0.115044  | 0.163976 | 0.163976 |  0.084535   |  0.29758  | 394008 |
| Short-TFT-Only     | 0.118174  | 0.167144 | 0.167144 |  0.0592412  |  0.270175 | 394008 |
| TFT-Only           | 0.100812  | 0.140186 | 0.140186 | -0.0471044  |  0.486609 | 394008 |


## Stitched overall metrics

| model              |       MAE |     RMSE |    nRMSE |         MBE |        R2 |     N |
|:-------------------|----------:|---------:|---------:|------------:|----------:|------:|
| Long-TFT-Only      | 0.152194  | 0.21661  | 0.21661  | -0.146458   | -0.242353 | 13869 |
| MiRACLE v1.0 Core  | 0.0828644 | 0.117922 | 0.117922 |  0.00528544 |  0.631805 | 13869 |
| PVLib-Physics-Only | 0.116224  | 0.166253 | 0.166253 |  0.0866692  |  0.268136 | 13869 |
| Short-TFT-Only     | 0.113051  | 0.161461 | 0.161461 |  0.0462743  |  0.309722 | 13869 |
| TFT-Only           | 0.0998884 | 0.139977 | 0.139977 | -0.0500917  |  0.481198 | 13869 |


## Tail abs error

| model              |       P50 |      P90 |      P95 |      P99 |      mean |
|:-------------------|----------:|---------:|---------:|---------:|----------:|
| Long-TFT-Only      | 0.0937153 | 0.418962 | 0.47797  | 0.57093  | 0.158866  |
| MiRACLE v1.0 Core  | 0.0575713 | 0.198105 | 0.252516 | 0.358487 | 0.0841161 |
| PVLib-Physics-Only | 0.077793  | 0.271637 | 0.358961 | 0.543021 | 0.115044  |
| Short-TFT-Only     | 0.0762341 | 0.289407 | 0.359268 | 0.513335 | 0.118174  |
| TFT-Only           | 0.068616  | 0.240817 | 0.301579 | 0.424015 | 0.100812  |


## Paired daily deltas vs baseline

| model              |   mean_daily_delta_MAE |   ci95_lo |   ci95_hi |   frac_days_improved_MAE |   N_days |
|:-------------------|-----------------------:|----------:|----------:|-------------------------:|---------:|
| Long-TFT-Only      |              0.0588674 | 0.0508465 | 0.0672652 |                 0.189041 |      365 |
| PVLib-Physics-Only |              0.0327287 | 0.0279207 | 0.0376382 |                 0.221918 |      365 |
| Short-TFT-Only     |              0.0260377 | 0.0207358 | 0.0317141 |                 0.306849 |      365 |
| TFT-Only           |              0.0137278 | 0.0099267 | 0.0176841 |                 0.315068 |      365 |