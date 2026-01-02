"""
src/configs/tft_v1.py

MiRACLE Germany TFT v1.0 feature-role config.

This config defines the exact column roles used to create the TFT TimeSeriesDataSet.

Design rules (v1.0)
- Target: power_norm
- LSTM encodings are time-varying unknown reals (encoder-only).
  Reason: for future decoder steps, we do not have ground-truth encodings unless we
  implement a separate rollout strategy. Keep it clean for v1.0.
- Weather + PVLib are time-varying known inputs.
  Reason: at inference, we will supply forecast weather and compute PVLib features.
- Drop redundant plant one-hot columns (plant_01..plant_06). plant_id is sufficient.
- Drop poa_irradiance (GTI proxy). PVLib provides POA properly (pvlib_poa_*).
- Drop duplicated irradiance columns without _raw suffix if both exist.
  Keep *_raw as the "source-of-truth" irradiance set for TFT.
"""

from __future__ import annotations

from typing import List


# Core identifiers
TIME_COL: str = "timestamp_utc"
GROUP_COL: str = "plant_id"
TARGET_COL: str = "power_norm"

# Static features
STATIC_CATEGORICALS: List[str] = [GROUP_COL]
STATIC_REALS: List[str] = []

# Time-varying known features (available for decoder at inference)
# Treat weather_code as categorical because it is an encoded condition label.
TV_KNOWN_CATEGORICALS: List[str] = ["weather_code"]

TV_KNOWN_REALS: List[str] = [
    # Raw weather reals
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",

    # Irradiance (raw set)
    "shortwave_radiation_instant_raw",
    "direct_normal_irradiance_instant_raw",
    "global_tilted_irradiance_instant_raw",

    # PVLib physics features (known if computed from forecast weather)
    "pvlib_solar_zenith",
    "pvlib_solar_azimuth",
    "pvlib_poa_global",
    "pvlib_poa_direct",
    "pvlib_poa_diffuse",
    "pvlib_poa_ground_diffuse",
    "pvlib_dc_kw",
    "pvlib_ac_kw",
]

# Time-varying unknown features (encoder-only)
# Target must be included here by PyTorch Forecasting convention.
TV_UNKNOWN_REALS_PREFIX: str = "lstm_enc_"

def lstm_encoding_cols(n: int = 64) -> List[str]:
    return [f"{TV_UNKNOWN_REALS_PREFIX}{i:03d}" for i in range(n)]

TV_UNKNOWN_REALS: List[str] = [TARGET_COL] + lstm_encoding_cols(64)

# Columns to drop before building the dataset
DROP_COLS: List[str] = [
    # Redundant plant one-hots (keep plant_id categorical)
    "plant_01", "plant_02", "plant_03", "plant_05", "plant_06",

    # GTI proxy used during LSTM stage
    "poa_irradiance",

    # Duplicated irradiance set without raw suffix (keep *_raw)
    "shortwave_radiation_instant",
    "direct_radiation_instant",
    "diffuse_radiation_instant",
    "direct_normal_irradiance_instant",
    "global_tilted_irradiance_instant",
]
