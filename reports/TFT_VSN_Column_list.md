Temporal Fusion Transformer (VSN) — Feature Inventory
=====================================================

Dataset scope
-------------
- Train: 142,190 rows (2023-01-01 23:00 to 2023-11-30 23:45 UTC).
- Validation: 36,952 rows (2023-12-02 00:00 to 2024-02-29 23:45 UTC).
- Plants represented: `plant_01`, `plant_02`, `plant_03`, `plant_05`, `plant_06` (no `plant_04`).
- Column set: 97 fields; identical across train/val; no NaNs or duplicate keys.

Targets and keys
----------------
- Time index: `timestamp_utc`.
- Target: `power_norm` (normalized AC output).
- Plant identity: `plant_id`, plus one-hot flags `plant_01`, `plant_02`, `plant_03`, `plant_05`, `plant_06`.

Time-varying known reals
------------------------
- Raw weather:
  - `poa_irradiance`
  - `global_tilted_irradiance_instant_raw`
  - `direct_normal_irradiance_instant_raw`
  - `shortwave_radiation_instant_raw`
  - `temperature_2m`
  - `relative_humidity_2m`
  - `precipitation`
  - `cloud_cover`
  - `wind_speed_10m`
  - `wind_direction_10m`
  - `surface_pressure`
  - `weather_code`
- PVLib physics and irradiance components:
  - `shortwave_radiation_instant`
  - `direct_radiation_instant`
  - `diffuse_radiation_instant`
  - `direct_normal_irradiance_instant`
  - `global_tilted_irradiance_instant`
  - `pvlib_solar_zenith`
  - `pvlib_solar_azimuth`
  - `pvlib_poa_global`
  - `pvlib_poa_direct`
  - `pvlib_poa_diffuse`
  - `pvlib_poa_ground_diffuse`
  - `pvlib_dc_kw`
  - `pvlib_ac_kw`

Learned temporal embeddings
---------------------------
- `lstm_enc_000` ... `lstm_enc_063` (64-dim latent state summarizing long-context power/weather dynamics).

Methodology notes
-----------------
- Column parity and cleanliness (97 identical fields, no NaNs/dupes) prevent covariate shift between train/val and simplify deployment schemas.
- Targets/keys: `power_norm` is normalized to stabilize loss scaling; explicit `timestamp_utc` anchors the 15-minute granularity; plant ID + one-hot flags let the TFT share weights while learning per-site baselines.
- Time-varying known reals: weather + PVLib physics provide horizon-known drivers tied to solar resource and panel geometry, improving forecast anchoring and reducing reliance on autoregressive drift.
- Learned embeddings: the 64-dim LSTM state carries slow seasonality/soiling/operational patterns so TFT attention can prioritize medium/short-term effects without losing long-context information.

Thesis-ready methodology
------------------------
We trained a Temporal Fusion Transformer on the VSN dataset comprising 142k train and 37k validation samples (15-minute resolution, Jan 2023–Feb 2024). The supervised target is normalized AC power (`power_norm`), indexed by `timestamp_utc` and conditioned on plant identity via both a categorical key (`plant_id`) and one-hot flags for each plant, enabling shared dynamics with site-specific offsets. The input design combines horizon-known drivers (raw weather forecasts, PVLib-derived irradiance and solar-geometry terms) with a 64-dimensional upstream LSTM latent state that summarizes long-range production and weather history. This separation lets the TFT attend to medium- and short-term dependencies while retaining slow seasonal and operational effects in the latent context.

To prevent covariate shift, the train and validation splits use an identical 97-column schema with no missing values or duplicate keys, and the plant set is held constant (`plant_01`, `plant_02`, `plant_03`, `plant_05`, `plant_06`). Normalization of the target stabilizes optimization; physically grounded irradiance and atmospheric variables anchor forecasts in solar resource conditions; PVLib power proxies (`pvlib_dc_kw`, `pvlib_ac_kw`) provide consistency checks against learned representations. This feature construction yields a well-conditioned input space for the TFT while preserving interpretability of the underlying physical drivers.
