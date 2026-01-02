# Short-head (15-min, 24h) evaluation

Metrics are computed on validation as RMSE/MAE over all horizons (flattened).

| mode | rmse | mae | enc_len | pred_len | hidden | lstm_layers | attn_heads | dropout | lr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tft_pvlib | 0.048549 | 0.019817 | 96 | 96 | 64 | 2 | 4 | 0.150 | 1.20e-03 |
| tft_only | 0.051300 | 0.020576 | 96 | 96 | 64 | 2 | 4 | 0.050 | 8.00e-04 |

## Selected model
Winner by RMSE: **tft_pvlib**

- run_dir: /shared/dbfz018/miracle/pv_forecast_30d/experiments/tft/runs/germany/sweeps/tft_pvlib/job24473/lr1.2e-3_do0.15_bs512_acc8_seed42/20251227_205027
- ckpt: /shared/dbfz018/miracle/pv_forecast_30d/experiments/tft/runs/germany/sweeps/tft_pvlib/job24473/lr1.2e-3_do0.15_bs512_acc8_seed42/20251227_205027/checkpoints/best.ckpt
