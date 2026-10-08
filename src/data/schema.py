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


# -----------------------------
# Canonical column names
# -----------------------------
TIME_COL: str = "timestamp_utc"
POWER_KW_COL: str = "power_kw"
POWER_NORM_COL: str = "power_norm"

# Target contract (Stage 3 global encoder predicts next-step PV power)
TARGET_COL: str = POWER_NORM_COL

# Expected time step for 15-min data (used by window builders)
TIME_STEP_MINUTES: int = 15


# -----------------------------
# LSTM base feature order (15 features)
# IMPORTANT: This order must match Farm2107 pretraining feature order.
# -----------------------------
LSTM_INPUT_FEATURES: List[str] = [
    POWER_NORM_COL,  # autoregressive input
    "poa_irradiance",  # required for Farm2107 feature alignment
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


# -----------------------------
# Required column sets per dataset stage
# -----------------------------
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


# -----------------------------
# Version 3: Global Forecasting Model constants
# -----------------------------
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


# -----------------------------
# Canonical paths helper
# -----------------------------
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
    def germany_processed(self) -> Path:
        return self.processed / "germany"

    @property
    def germany_pretraining(self) -> Path:
        return self.processed / "pretraining" / "germany"


# -----------------------------
# Dataset-specific column adapters
# -----------------------------
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


# -----------------------------
# Validation helpers
# -----------------------------
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


# -----------------------------
# Data quality gate for PV time series
# -----------------------------
def validate_pv_frame(
    df,
    *,
    time_col: str = TIME_COL,
    power_col: str = POWER_NORM_COL,
    step_minutes: int = TIME_STEP_MINUTES,
    max_norm: float = 1.5,
    max_nan_frac: float = 0.5,
    max_gap_steps: int = 4 * 24 * 3,
    flat_run_steps: int = 4 * 6,
    raise_on_error: bool = False,
) -> list:
    """
    Plausibility checks for one plant's PV frame. Returns a list of human-readable issues
    (empty list = clean). Nothing is modified.

    Checks: required columns, tz-aware UTC timestamps, sorted + unique timestamps, regular grid
    (reports the largest gap), share of NaN power, value range [0, max_norm], and long runs of
    identical non-zero values (stuck sensor / frozen export).

    Gaps are *reported*, not fixed: dropping or filling rows silently is what lets windows span
    missing data, so callers decide explicitly (see SimpleWindowDataset(require_contiguous=True)).
    """
    import numpy as np
    import pandas as pd

    issues: list = []
    missing = [c for c in (time_col, power_col) if c not in df.columns]
    if missing:
        issues.append(f"missing columns: {missing}")
    else:
        ts = df[time_col]
        if not pd.api.types.is_datetime64_any_dtype(ts) or getattr(ts.dt, "tz", None) is None:
            issues.append(f"{time_col} is not timezone-aware datetime")
        else:
            if str(ts.dt.tz) not in ("UTC", "utc", "datetime.timezone.utc", "+00:00"):
                issues.append(f"{time_col} is not UTC (tz={ts.dt.tz})")
            if not ts.is_monotonic_increasing:
                issues.append(f"{time_col} is not sorted ascending")
            dup = int(ts.duplicated().sum())
            if dup:
                issues.append(f"{dup} duplicate timestamps")
            step = pd.Timedelta(minutes=step_minutes)
            diffs = ts.sort_values().diff().dropna()
            off_grid = int((diffs % step != pd.Timedelta(0)).sum())
            if off_grid:
                issues.append(f"{off_grid} timestamp steps are not multiples of {step_minutes} min")
            if len(diffs):
                largest = int(diffs.max() / step) - 1
                if largest > max_gap_steps:
                    issues.append(f"largest gap is {largest} steps ({largest * step_minutes / 60:.1f} h)")

        p = pd.to_numeric(df[power_col], errors="coerce")
        nan_frac = float(p.isna().mean()) if len(p) else 1.0
        if nan_frac > max_nan_frac:
            issues.append(f"{nan_frac:.1%} of {power_col} is NaN (limit {max_nan_frac:.0%})")
        pv = p.dropna()
        if len(pv):
            if (pv < -1e-9).any():
                issues.append(f"{int((pv < -1e-9).sum())} negative {power_col} values")
            if (pv > max_norm).any():
                issues.append(f"{int((pv > max_norm).sum())} {power_col} values above {max_norm}")
            nonzero_flat = (pv > 1e-6) & (pv.diff() == 0)
            run = nonzero_flat.groupby((~nonzero_flat).cumsum()).cumsum()
            if len(run) and int(run.max()) >= flat_run_steps:
                issues.append(f"stuck value: {int(run.max())} identical non-zero {power_col} samples in a row")
            if np.isclose(pv.max(), 0.0):
                issues.append(f"{power_col} is all zeros")

    if issues and raise_on_error:
        raise ValueError("validate_pv_frame: " + "; ".join(issues))
    return issues
