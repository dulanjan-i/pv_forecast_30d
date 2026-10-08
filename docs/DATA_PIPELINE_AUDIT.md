# Data pipeline audit (v1.1 work)

Code-level audit of `src/data`, `src/preprocessing`, `src/features`, `src/validation` and the data tests.
No real (NDA) data was used; checks that need it ship as scripts to run locally.

## Fixed in this branch
| Issue | Fix | Where |
|---|---|---|
| Hourly weather shifted by 1 h from late March to late October (uniform naive labels re-localised as Europe/Berlin). Reproduced on a synthetic series: monthly GHI-peak offset of -1 h for Apr-Oct. | Keep UTC end to end; legacy `date` CSVs are repaired exactly from the first label | `src/data/time_utils.py`, `call_openmeteo_hist_germany.py`, `preprocess_germany_weather.py` |
| Peak-based PV rescaling left no audit trail | Per-plant `<plant>_scaling.json` (factor, pre-scaling max, max/p99.9 outlier indicator), `--dry-run`, docstring now states the real behaviour | `fix_germany_pv_scaling.py` |
| "Stratified temporal split" was a random row split (leaks across overlapping windows) | Default is now chronological 70/15/15 with a 96-row purge; old behaviour behind `--legacy-random-split` | `germany_pretrain_normalize_split.py` |
| Windows could span missing rows | `SimpleWindowDataset(require_contiguous=True)` (default off to keep v1.0 reproducible); `data.require_contiguous` in the LSTM pretrain config | `sequence_generator.py`, `pretrain_lstm.py` |
| No plausibility gate | `validate_pv_frame()` (UTC, order, duplicates, grid, gaps, NaN share, range, stuck values) | `src/data/schema.py` |
| `DataPaths.germany_processed` missing, so `germany_build_pretrain_base.py` raised AttributeError | Added | `src/data/schema.py` |
| DNI taken from Open-Meteo's horizontal `direct_radiation`; wrong column names triggered a silent clear-sky fallback; `.values` crash on missing keys | Column resolver, DNI by closure when absent, warning on fallback | `pvlib_feature_generator.py` |
| `verify=False` on API calls; bare `except:`; hard-coded home path; syntactically broken duplicate fetcher | TLS on (CA bundle via env), failed chunks logged, `MIRACLE_DATA_DIR`, broken file removed | `call_openmeteo_hist_germany.py`, `weather_api_orchestrator.py`, `fetch_era5_extended.py` |
| `.gitignore` `data/` also ignored the package `src/data/` (new modules silently untracked) | Anchored to `/data/` | `.gitignore` |
| No tests on data code | 19 tests with synthetic frames | `tests/test_data_pipeline.py` |

## Not changed (needs retraining or a decision)
- The v1.0 checkpoints were trained on weather produced by the old time handling. If the real raw CSVs have the legacy
  labelling, summer training inputs were shifted by 1 h while inference (`weather_client.py`, UTC) is not. Quantify locally with
  `python scripts/check_weather_alignment.py --csv <hourly csv> --lat .. --lon ..`; fixing it properly requires retraining.
- Precipitation is forward-filled from hourly to 15 min (repeats the hourly total in each quarter hour). Changing it alters a model input, so deferred.
- Open-Meteo `*_instant` irradiance vs PV interval convention is undocumented.
- Weather imputation (`ffill().bfill()`, unlimited linear interpolation) has no imputed-flag.
- Normalisation of rescaled plants depends on one extreme sample (see the scaling record).
- Train/serve skew: ERA5 reanalysis for training, forecast APIs at inference.
- `src/rl/collect_rl_data.py` does not compile (misplaced `from __future__` import and a `return` outside a function around line 521, apparently two implementations concatenated). Only `collect_rl_data_parallel.py` compiles; it is unclear which script produced the offline-RL transitions. Left untouched pending a decision.
- Training-side RL data period and trainer of record are not documented in the repo.

## Corrections to earlier audit statements
- `germany_build_lstm_encodings.py` already checks 15-minute contiguity per window; the gap issue applies to `SimpleWindowDataset` (used by `pretrain_lstm.py`).
- `tests/test_orchestration.py` is entirely skipped, so before this branch no active test exercised any data-pipeline code.
