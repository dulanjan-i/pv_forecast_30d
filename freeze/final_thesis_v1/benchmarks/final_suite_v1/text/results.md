# Benchmark suite summary

- Truth: /home/dwijenayake/pv_forecast_30d/freeze/final_thesis_v1/phase1_2024daily_final/processed/ground_truth_15min_utc_capnorm.parquet

- Baseline: MiRACLE_v1_core_warm = /home/dwijenayake/pv_forecast_30d/freeze/final_thesis_v1/phase1_2024daily_final/processed/predictions_phase1_baseline_rerun.parquet

- Models: MiRACLE_v1_core_cold, MiRACLE_v1_meta_policy, PVLib_only, Short_TFT_only, Long_TFT_only

- Night filtering: ON (y_true >= 0.01)


## Overall metrics

| model                  |       MAE |     RMSE |    nRMSE |         MBE |        R2 |      N |
|:-----------------------|----------:|---------:|---------:|------------:|----------:|-------:|
| Long_TFT_only          | 0.158866  | 0.223948 | 0.223948 | -0.15345    | -0.310184 | 394008 |
| MiRACLE_v1_core_cold   | 0.0875937 | 0.119183 | 0.119183 | -0.00429974 |  0.628923 | 394008 |
| MiRACLE_v1_core_warm   | 0.0841161 | 0.11713  | 0.11713  | -0.00736451 |  0.641595 | 394008 |
| MiRACLE_v1_meta_policy | 0.0841362 | 0.117161 | 0.117161 | -0.00750784 |  0.641409 | 394008 |
| PVLib_only             | 0.115044  | 0.163976 | 0.163976 |  0.084535   |  0.29758  | 394008 |
| Short_TFT_only         | 0.118174  | 0.167144 | 0.167144 |  0.0592412  |  0.270175 | 394008 |


## Stitched overall metrics

| model                  |       MAE |     RMSE |    nRMSE |         MBE |        R2 |     N |
|:-----------------------|----------:|---------:|---------:|------------:|----------:|------:|
| Long_TFT_only          | 0.152194  | 0.21661  | 0.21661  | -0.146458   | -0.242353 | 13869 |
| MiRACLE_v1_core_cold   | 0.089059  | 0.121771 | 0.121771 |  0.0131805  |  0.607375 | 13869 |
| MiRACLE_v1_core_warm   | 0.0828644 | 0.117922 | 0.117922 |  0.00528544 |  0.631805 | 13869 |
| MiRACLE_v1_meta_policy | 0.0834809 | 0.118786 | 0.118786 |  0.00144211 |  0.626389 | 13869 |
| PVLib_only             | 0.116224  | 0.166253 | 0.166253 |  0.0866692  |  0.268136 | 13869 |
| Short_TFT_only         | 0.113051  | 0.161461 | 0.161461 |  0.0462743  |  0.309722 | 13869 |


## Tail abs error

| model                  |       P50 |      P90 |      P95 |      P99 |      mean |
|:-----------------------|----------:|---------:|---------:|---------:|----------:|
| Long_TFT_only          | 0.0937153 | 0.418962 | 0.47797  | 0.57093  | 0.158866  |
| MiRACLE_v1_core_cold   | 0.063131  | 0.198641 | 0.250561 | 0.35795  | 0.0875937 |
| MiRACLE_v1_core_warm   | 0.0575713 | 0.198105 | 0.252516 | 0.358487 | 0.0841161 |
| MiRACLE_v1_meta_policy | 0.0575735 | 0.19817  | 0.25263  | 0.358487 | 0.0841362 |
| PVLib_only             | 0.077793  | 0.271637 | 0.358961 | 0.543021 | 0.115044  |
| Short_TFT_only         | 0.0762341 | 0.289407 | 0.359268 | 0.513335 | 0.118174  |


## Paired daily deltas vs baseline

| model                  |   mean_daily_delta_MAE |     ci95_lo |     ci95_hi |   frac_days_improved_MAE |   N_days |
|:-----------------------|-----------------------:|------------:|------------:|-------------------------:|---------:|
| Long_TFT_only          |            0.0588674   | 0.0508465   | 0.0672652   |                 0.189041 |      365 |
| MiRACLE_v1_core_cold   |            0.00314913  | 0.00199329  | 0.00431476  |                 0.323288 |      365 |
| MiRACLE_v1_meta_policy |            4.11246e-05 | 9.55124e-06 | 8.28904e-05 |                 0.443836 |      365 |
| PVLib_only             |            0.0327287   | 0.0279207   | 0.0376382   |                 0.221918 |      365 |
| Short_TFT_only         |            0.0260377   | 0.0207358   | 0.0317141   |                 0.306849 |      365 |