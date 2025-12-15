## Appendix A. German PV Data Preprocessing Workflow

### A.0 Scope and Data Flow
- **Objective:** Produce modelling-ready 15-minute PV–weather time series for six anonymised German plants.
- **Inputs:** Provider metadata workbook, raw PV CSV exports (German decimal, UTC timestamps), Open-Meteo ERA5 Seamless archive calls.
- **Outputs:** Validated interim PV (`data/interim/germany`) and weather parquets plus merged processed artefacts (`data/processed/germany`).
- **Toolchain:** Python scripts under `src/data`, exploratory verification in `notebooks/data_cleaning/german_data_sanity_check.ipynb`.

---

### A.1 Plant Metadata Harmonisation
**Source:** `src/data/generate_germany_metadata.py`  
**Inputs:** `data/metadata/germany/Base_Data_exc_Dulan.xlsx` (sheet “Basedata”).  
**Key steps:**
- Map provider-specific IDs to anonymised plant codes (`plant_01` … `plant_06`). For the two-terminal site, capacity-weighted merging combines `6_TER1` and `6_TER2`.
- Standardise numeric fields by stripping degree symbols, replacing decimal commas, and coercing tilt/azimuth to floats.
- Convert the provider’s azimuth convention (S=0, clockwise) to PVLib’s N=0 standard via `(alpha_sy + 180) mod 360`.
- Derive representative metadata per plant (installed capacity, location, tilt, azimuth, mount type, tracker flag) using capacity-weighted averages.
- Persist rich JSON descriptors (`data/metadata/germany/{plant}.json`) including raw categorical text for future audits and add `source = "syneco_base_data_v1"` for traceability.

**Outputs:** Plant-level metadata JSON files consumed throughout the pipeline.

---

### A.2 PV Signal Ingestion and Feature Derivation
**Source:** `src/data/preprocess_germany_pv.py`  
**Inputs:** Raw plant CSV dumps in `data/raw/germany/{plant}` plus metadata JSON.  
**Key steps:**
- Stack all CSVs per plant, parse semicolon-separated columns, and interpret timestamps directly in UTC (provider export already references UTC).
- Construct canonical signals: `power_kw` (robust float conversion of `power_raw`), `power_w` (`power_kw × 1000`), and `power_norm` (`power_kw / installed_capacity_kw`). Capacity zero-division cases yield `NA`.
- Enforce chronological ordering and drop duplicate timestamps to maintain a one-to-one time–power mapping.
- Persist intermediate 15-minute parquets (`data/interim/germany/{plant}_pv_15min.parquet`) that retain only the canonical power features.

**Assumptions:** Provider exports represent instantaneous AC power per 15-minute interval; any later evidence of energy readings would require updating this script alone.

---

### A.3 Interactive Sanity Checks
**Source:** `notebooks/data_cleaning/german_data_sanity_check.ipynb`  
**Purpose:** Document exploratory QA performed immediately after PV ingestion.  
**Analyses performed:**
- Inspect head/tail samples, descriptive statistics, and per-plant coverage summaries.
- Plot representative summer (e.g., 2023-06-15) and winter (e.g., 2023-01-15) diurnal curves to flag clipping, outages, or flatlines.
- Compute dominant timestamp deltas to confirm nominal 15-minute cadence and surface anomalies (e.g., a 75-minute gap in `plant_02`).
- Verify Open-Meteo hourly/15-minute parquets for timestamp continuity, DST handling, and timezone flags.
- Record conclusions and action items (scaling fix, grid enforcement) to maintain a traceable audit trail.

---

### A.4 Rule-Based Scaling Corrections
**Source:** `src/data/fix_germany_pv_scaling.py`  
**Logic:** Compare installed capacity (`installed_capacity_kw`) with each plant’s observed maximum `power_kw`.  
**Details:**
- Plants with `capacity / max_power ≤ 5` retain their original magnitude (accounts for realistic derating from temperature, losses, snow).
- Plants exceeding the threshold are interpreted as unit mismatches (e.g., micro-kW readings for multi-MW systems). A plant-specific scale factor equal to the ratio is applied to `power_kw` and `power_w`.
- `power_norm` is recomputed post-scaling to maintain consistency with capacity.
- The script emits diagnostic logs (capacity, max power, ratio, applied scale) for reproducibility.

**Outcome:** Corrected interim PV parquets with realistic magnitudes while preserving temporal dynamics.

---

### A.5 Enforcement of a Strict 15-Minute Grid
**Source:** `src/data/enforce_germany_pv_15min_grid.py`  
**Goal:** Make sampling regularity explicit before merging with weather.  
**Method:**
- For each plant, derive the full UTC range between minimum and maximum timestamps and reindex on a complete 15-minute grid.
- Existing measurements remain untouched; missing timestamps are inserted with NaNs in all power columns to expose genuine outages or telemetry gaps.
- Overwrite the same parquet paths to avoid version proliferation.

**Result:** Harmonised PV series ready for alignment operations and gap-aware modelling.

---

### A.6 Weather Forcing Retrieval
**Source:** `src/data/call_openmeteo_hist_germany.py`  
**Inputs:** Metadata JSON (lat/lon, tilt, azimuth) and PV-derived start/end dates.  
**Process:**
- Query the Open-Meteo archive API (model `era5_seamless`, timezone `Europe/Berlin`) per plant with the exact PV observation window to maximise overlap.
- Request a comprehensive hourly variable set (temperature, humidity, precipitation, cloud cover, wind, pressure, multiple irradiance components) plus daily aggregates (sunrise/sunset, daylight, precipitation sums, temperature extrema).
- Use cached, retried sessions to guarantee deterministic downloads under network variability; disable SSL verification for corporate proxy compatibility.
- Save raw CSV exports under `data/raw/germany/{plant}` and mirrored Parquet copies under `data/interim/germany/{plant}` for faster subsequent IO.

**Note:** All timestamps are returned in local time (DST-aware) and remain naive until the weather preprocessing stage.

---

### A.7 Weather Temporal Harmonisation
**Source:** `src/data/preprocess_germany_weather.py`  
**Steps:**
- Localise the naive `date` field to `Europe/Berlin`, handling DST gaps (`nonexistent="shift_forward"`) and duplicates (first occurrence kept), then convert to UTC to align with PV data.
- Enforce an hourly index before upsampling to 15-minute resolution via `asfreq("15min")`.
- Apply a hybrid interpolation strategy:
  - **Irradiance columns** (`shortwave_radiation_instant`, `direct_radiation_instant`, `diffuse_radiation_instant`, `direct_normal_irradiance_instant`, `global_tilted_irradiance_instant`): linear interpolation in time to preserve smooth solar curves.
  - **Meteorological state variables** (`temperature_2m`, `relative_humidity_2m`, `precipitation`, `cloud_cover`, `wind_speed_10m`, `wind_direction_10m`, `surface_pressure`): forward-fill with an initial backfill, reflecting quasi-steady hourly evolution without creating artificial gradients.
  - **Categorical codes** (`weather_code`): forward/backfill only, never interpolated.
- Output tidy parquets (`data/interim/germany/{plant}_weather_15min.parquet`) with timezone-aware `timestamp_utc`.

**Rationale:** Mirrors the Farm2107 reference pipeline to maintain methodological consistency across datasets.

---

### A.8 PV–Weather Integration
**Source:** `src/data/merge_germany_pv_weather.py`  
**Pipeline:**
- Load the grid-aligned PV and weather parquets, enforce UTC datetime typing on `timestamp_utc`, and drop redundant columns (e.g., local `date`).
- Perform an inner join on `timestamp_utc` (`validate="one_to_one"`) to keep only timestamps with both PV response and weather predictors, eliminating edge misalignments caused by differing coverage or DST transitions.
- Sort chronologically, reset the index, and persist merged artefacts under `data/processed/germany/{plant}_pv_weather_15min.parquet`.

**Usage:** These processed files constitute the modelling-ready tables consumed by downstream forecasting experiments.

---

### A.9 Post-Processing Validation
**Source:** `notebooks/data_cleaning/german_data_sanity_check.ipynb` (final cells).  
**Checks performed after each major script:**
- Confirm that scaling factors corrected magnitudes without distorting daily shapes.
- Re-run timestamp delta analyses to ensure the enforced 15-minute grid removed irregular gaps.
- Inspect weather 15-minute exports for uninterrupted UTC coverage and absence of DST artefacts.
- Verify that merged PV–weather tables have one-to-one joins and consistent coverage across plants.

**Documentation:** Notebook narratives capture findings, plots, and decisions, providing an auditable trail that complements the automated scripts.
