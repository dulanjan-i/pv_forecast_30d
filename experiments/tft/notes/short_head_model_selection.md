# Short-head (15-min, 24h) evaluation

Metrics are computed on validation as RMSE/MAE over all horizons (flattened).

| mode | rmse | mae | enc_len | pred_len | hidden | lstm_layers | attn_heads | dropout | lr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| warm_seed44 | 0.105161 | 0.051611 | 96 | 96 | 64 | 2 | 4 | 0.150 | 6.00e-04 |
| warm_seed43 | 0.108082 | 0.052935 | 96 | 96 | 64 | 2 | 4 | 0.150 | 6.00e-04 |
| cold_seed44 | 0.113697 | 0.058972 | 96 | 96 | 64 | 2 | 4 | 0.150 | 1.20e-03 |

## Selected model
Winner by RMSE: **warm_seed44**

- run_dir: /home/dwijenayake/pv_forecast_30d/experiments/tft/runs/germany/plant_03/15min/pvlib_warmstart_from_global_noleak/seed_44/20251229_154617
- ckpt: /home/dwijenayake/pv_forecast_30d/experiments/tft/runs/germany/plant_03/15min/pvlib_warmstart_from_global_noleak/seed_44/20251229_154617/checkpoints/best.ckpt
