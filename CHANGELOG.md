# Changelog

## Unreleased (v1.1 work, branch `claude/youthful-gates-1b3ifu`)
Data-pipeline hardening; model checkpoints are unchanged from v1.0. See `docs/DATA_PIPELINE_AUDIT.md`.
- Fix: weather timestamps stay in UTC (removes a 1 h summer shift); legacy CSVs repaired on read.
- Add: `scripts/check_weather_alignment.py`, `validate_pv_frame()`, scaling provenance files, `require_contiguous` windows.
- Change: default pretrain split is chronological with purge (legacy random split behind a flag).
- Fix: PVLib feature generator DNI handling, TLS verification, error handling, `.gitignore` anchoring, `DataPaths.germany_processed`.
- Remove: `src/data/fetch_archived_forecasts.py` (syntax error, duplicate of `fetch_era5_extended.py`), empty `make_hourly_plant_parquets.py`.
- Tests: `tests/test_data_pipeline.py`.
