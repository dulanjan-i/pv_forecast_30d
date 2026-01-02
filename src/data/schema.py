"""
src/data/schema.py

Central schema for MiRACLE / pv_forecast_30d.

This file defines canonical column names, feature lists, and Version 3 global-model
constants (plant IDs, one-hot columns, etc.). Keep this as the single source of truth.

Key contracts used by Stage 3 (Global LSTM Encoder):
- TIME_COL: timestamp column
- POWER_NORM_COL: normalized PV power (0..1)
- TARGET_COL: prediction target (set to POWER_NORM_COL)
- LSTM_INPUT_FEATURES: 15 base features (includes POWER_NORM_COL as an input feature)
- GLOBAL_LSTM_INPUT_FEATURES: LSTM_INPUT_FEATURES + plant one-hot columns
- PLANT_ID_COL: categorical plant identifier
- PLANT_ONEHOT_COLS: one-hot columns (strings equal to PLANT_IDS)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Set


***REMOVED*** -----------------------------
***REMOVED*** Canonical column names
***REMOVED*** -----------------------------
TIME_COL: str = "timestamp_utc"
POWER_KW_COL: str = "power_kw"
POWER_NORM_COL: str = "power_norm"

***REMOVED*** Target contract (Stage 3 global encoder predicts next-step PV power)
TARGET_COL: str = POWER_NORM_COL

***REMOVED*** Expected time step for 15-min data (used by window builders)
TIME_STEP_MINUTES: int = 15


***REMOVED*** -----------------------------
***REMOVED*** LSTM base feature order (15 features)
***REMOVED*** IMPORTANT: This order must match Farm2107 pretraining feature order.
***REMOVED*** -----------------------------
LSTM_INPUT_FEATURES: List[str] = [
    POWER_NORM_COL,  ***REMOVED*** autoregressive input
    "poa_irradiance",  ***REMOVED*** required for Farm2107 feature alignment
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "weather_code",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "shortwave_radiation_instant",
    "direct_radiation_instant",
    "diffuse_radiation_instant",
    "direct_normal_irradiance_instant",
    "global_tilted_irradiance_instant",
    "surface_pressure",
]


***REMOVED*** -----------------------------
***REMOVED*** Required column sets per dataset stage
***REMOVED*** -----------------------------
REQUIRED_PV_ONLY: Set[str] = {TIME_COL, POWER_KW_COL, POWER_NORM_COL}

REQUIRED_WEATHER_15MIN: Set[str] = {
    TIME_COL,
    "poa_irradiance",
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "weather_code",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "shortwave_radiation_instant",
    "direct_radiation_instant",
    "diffuse_radiation_instant",
    "direct_normal_irradiance_instant",
    "global_tilted_irradiance_instant",
    "surface_pressure",
}

REQUIRED_MERGED: Set[str] = REQUIRED_PV_ONLY | (REQUIRED_WEATHER_15MIN - {TIME_COL})
REQUIRED_PRETRAIN_BASE: Set[str] = {TIME_COL} | set(LSTM_INPUT_FEATURES)


***REMOVED*** -----------------------------
***REMOVED*** Version 3: Global Forecasting Model constants
***REMOVED*** -----------------------------
PLANT_IDS: List[str] = [
    "plant_01",
    "plant_02",
    "plant_03",
    "plant_05",
    "plant_06",
]

PLANT_ID_COL: str = "plant_id"
PLANT_ONEHOT_COLS: List[str] = PLANT_IDS

GLOBAL_LSTM_INPUT_FEATURES: List[str] = LSTM_INPUT_FEATURES + PLANT_ONEHOT_COLS


***REMOVED*** -----------------------------
***REMOVED*** Canonical paths helper
***REMOVED*** -----------------------------
@dataclass(frozen=True)
class DataPaths:
    repo_root: Path

    @property
    def data_dir(self) -> Path:
        return self.repo_root / "data"

    @property
    def processed(self) -> Path:
        return self.data_dir / "processed"

    @property
    def germany_pretraining(self) -> Path:
        return self.processed / "pretraining" / "germany"


***REMOVED*** -----------------------------
***REMOVED*** Dataset-specific column adapters
***REMOVED*** -----------------------------
FARM2107_TO_CANONICAL: Dict[str, str] = {
    "measured_on": TIME_COL,
    "pv_power_norm": POWER_NORM_COL,
    "poa_irradiance": "poa_irradiance",
}

GERMANY_TO_CANONICAL: Dict[str, str] = {
    "timestamp_utc": TIME_COL,
    "power_kw": POWER_KW_COL,
    "power_norm": POWER_NORM_COL,
}


***REMOVED*** -----------------------------
***REMOVED*** Validation helpers
***REMOVED*** -----------------------------
def validate_required_columns(columns: Sequence[str], required: Set[str], context: str) -> None:
    cols = set(columns)
    missing = sorted(required - cols)
    if missing:
        raise ValueError(f"{context}: missing required columns: {missing}")


def canonicalize_columns(df, mapping: Dict[str, str]):
    """Rename dataset-specific columns into canonical names (no reordering)."""
    return df.rename(columns=mapping)


def enforce_lstm_feature_order(df):
    """Return a view ordered exactly as LSTM_INPUT_FEATURES."""
    return df[LSTM_INPUT_FEATURES]
