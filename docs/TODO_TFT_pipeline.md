# TFT Pipeline TODO

- [ ] **Step 3.7 — Build weather features (raw + scaled)**
  - Goal: produce one consistent weather feature table aligned to TFT timestamps for train/val.
  - Keys: join on `timestamp_utc, plant_id`; keep raw physical columns for interpretability; add scaled copies with `_z` suffix for stability.
  - Scaling: fit scalers on train only, apply to val; save scaler JSON.
  - Outputs: `data/processed/pretraining/germany/global/weather_tft/regional_train_weather.parquet`, `regional_val_weather.parquet`, `tft_weather_scaler.json`.

- [ ] **Step 3.8 — Build PVLib features (raw + scaled)**
  - Goal: create PVLib-derived signals per timestamp per plant.
  - Use raw weather inputs (irradiance/temp) for PVLib; if metadata incomplete, start with clear-sky + POA + temp baseline and iterate.
  - Scaling: fit on train only, apply to val; keep raw + `_z` copies; save scaler JSON.
  - Outputs: `data/processed/pretraining/germany/global/pvlib_tft/regional_train_pvlib.parquet`, `regional_val_pvlib.parquet`, `tft_pvlib_scaler.json`.

- [ ] **Step 3.9 — Final TFT input assembly**
  - Inputs: `regional_train_tft_base.parquet`, `regional_train_weather.parquet`, `regional_train_pvlib.parquet` (and val counterparts).
  - Join strategy: inner join on `timestamp_utc, plant_id`; ensure target `power_norm` preserved and unchanged.
  - Hard checks: no duplicate keys; no NaNs; row counts match base after join; save diff/audit report to `runs/.../audit.json`; emit `tft_feature_manifest.json` (column lists).
  - Outputs: `data/processed/pretraining/germany/global/tft_inputs/regional_train_tft_final.parquet`, `regional_val_tft_final.parquet`, `tft_feature_manifest.json`.

- [ ] **Step 4.0 — TFT training on the 5 plants**
  - Start with horizon 24h @ 15-min (`horizon=96`), then expand after sanity checks.
  - Use final TFT inputs and feature manifest; log configs/metrics under runs folder.
