# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

Thesis results (MiRACLE v1.0) correspond to the tag
[`v1.0.0`](https://github.com/dulanjan-i/pv_forecast_30d/tree/v1.0.0). Model checkpoints have not changed since.

## [Unreleased]

## [1.1.0] - merged to `main`, not yet tagged
Data-pipeline hardening. Model checkpoints and the inference path are unchanged. Details and open items:
[`docs/DATA_PIPELINE_AUDIT.md`](docs/DATA_PIPELINE_AUDIT.md).

### Fixed
- Weather timestamps stay in UTC end to end. The old path shifted weather by 1 h from late March to late October;
  legacy CSVs are repaired on read. Models trained before this fix still saw the shifted weather (retraining needed).
- PVLib feature generator: DNI no longer taken from Open-Meteo's horizontal `direct_radiation`; clear-sky fallback now warns.
- TLS verification re-enabled on Open-Meteo calls; bare `except` removed from the ERA5 fetcher (failed chunks are logged).
- `DataPaths.germany_processed` was missing (`germany_build_pretrain_base.py` raised `AttributeError`).
- `.gitignore` rule `data/` also ignored the package `src/data/`; now anchored to `/data/`.
- Docker workflow: pull-request builds failed at the smoke test (`manifest unknown`) because nothing is pushed on PRs;
  PRs now smoke-test a locally loaded amd64 image.

### Added
- `scripts/check_weather_alignment.py` (UTC alignment check, synthetic or on your own CSV).
- `validate_pv_frame()` plausibility checks in `src/data/schema.py` (not yet called by the pipeline stages).
- Scaling provenance files (`<plant>_scaling.json`) and `--dry-run` for `fix_germany_pv_scaling.py`.
- Opt-in `require_contiguous` for `SimpleWindowDataset` (and `data.require_contiguous` in the LSTM pretrain config).
- `tests/test_data_pipeline.py` (19 tests on synthetic data).

### Changed
- Default pretrain split is chronological with a 96-row purge; the old random-row split is behind `--legacy-random-split`.

### Removed
- `src/data/fetch_archived_forecasts.py` (syntax error, duplicate of `fetch_era5_extended.py`) and an empty module.

## [1.0.1] - 2026-10-08 (state of `main` before the 1.1 work; tag pending)
Packaging, CI and tests on top of 1.0.0. No model or pipeline changes.

### Added
- Docker inference image (amd64 + arm64) published to GHCR, `docker-compose.yml`, and a scheduled inference workflow.
- GitHub Actions CI running the data-free pytest suite; Tier 1 unit tests for metrics, reward and data schema.
- SHA256 checkpoint manifest and TFT v1.0 final checkpoints.
- `workflow_dispatch` trigger for CI.

### Changed
- License text replaced with the official PolyForm Noncommercial 1.0.0.
- Repository cleanup: runtime logs and NDA data untracked, redundant RL checkpoints and prototypes removed.

## [1.0.0] - 2026-05-27
Thesis release: dual-head TFT with physics glue and the DDQN meta-controller.
