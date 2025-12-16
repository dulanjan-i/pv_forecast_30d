"""
src/data/schema.py

Single source of truth for MiRACLE data contracts.

Why this exists
- Prevents column drift across branches (data-pipeline-build, pvlib-build, tft-build).
- Ensures the pretrained LSTM encoder sees the exact same input tensor definition
  in Stage 1 (Farm2107) and Stage 2 (Germany).
- Column *names* can differ per dataset, but they must be mapped into these
  canonical names and, critically, the feature *order* must be identical.

Ground truth
- Canonical LSTM input list and target are defined by:
    experiments/lstm/pretrain_farm2107_CANONICAL.yaml
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Set


# -----------------------------
# Canonical column names
# -----------------------------
TIME_COL: str = "timestamp_utc"

# Canonical target names used downstream (TFT training, evaluation).
# Note: Farm2107 YAML used "pv_power_norm" as the target; we map that into POWER_NORM.
POWER_KW_COL: str = "power_kw"
POWER_NORM_COL: str = "power_norm"


# -----------------------------
# Canonical feature order for pretrained LSTM encoder
# IMPORTANT: This order must not change unless you retrain the LSTM.
# Source: pretrain_farm2107_CANONICAL.yaml feature_cols
# -----------------------------
LSTM_INPUT_FEATURES: List[str] = [
    # autoregressive PV
    POWER_NORM_COL,
    # irradiance / POA feature
    "poa_irradiance",
    # weather features (ORDER LOCKED - matches farm2107_CANONICAL pretrained weights)
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",              # ← Index 4 (pretrained weight expects this here)
    "weather_code",               # ← Index 5
    "cloud_cover",                # ← Index 6
    "wind_speed_10m",             # ← Index 7
    "wind_direction_10m",         # ← Index 8
    # radiation features
    "shortwave_radiation_instant",
    "direct_radiation_instant",
    "diffuse_radiation_instant",
    "direct_normal_irradiance_instant",
    "global_tilted_irradiance_instant",
    "surface_pressure",           # ← Index 14 (last feature)
]


# -----------------------------
# Required column sets per dataset stage
# -----------------------------
REQUIRED_PV_ONLY: Set[str] = {TIME_COL, POWER_KW_COL, POWER_NORM_COL}

REQUIRED_WEATHER_15MIN: Set[str] = {
    TIME_COL,
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

# Pretrain base must include whatever the LSTM encoder expects
REQUIRED_PRETRAIN_BASE: Set[str] = {TIME_COL} | set(LSTM_INPUT_FEATURES)


# -----------------------------
# Canonical paths
# (These are contracts, not hard requirements. Use for consistent tooling.)
# -----------------------------
@dataclass(frozen=True)
class DataPaths:
    repo_root: Path

    @property
    def data_dir(self) -> Path:
        return self.repo_root / "data"

    @property
    def raw(self) -> Path:
        return self.data_dir / "raw"

    @property
    def interim(self) -> Path:
        return self.data_dir / "interim"

    @property
    def processed(self) -> Path:
        return self.data_dir / "processed"

    @property
    def germany_interim(self) -> Path:
        return self.interim / "germany"

    @property
    def germany_processed(self) -> Path:
        return self.processed / "germany"

    @property
    def germany_pretraining(self) -> Path:
        return self.processed / "pretraining" / "germany"


# -----------------------------
# Dataset-specific column adapters
# Map dataset-native names -> canonical names.
# Keep this minimal and explicit.
# -----------------------------
# Example: Farm2107 uses measured_on and pv_power_norm in the YAML.
FARM2107_TO_CANONICAL: Dict[str, str] = {
    "measured_on": TIME_COL,
    "pv_power_norm": POWER_NORM_COL,
    # if your farm parquet uses a different name for POA, map it here.
    "poa_irradiance": "poa_irradiance",
}

# Germany: 
GERMANY_TO_CANONICAL: Dict[str, str] = {
    "timestamp_utc": TIME_COL,
    "power_kw": POWER_KW_COL,
    "power_norm": POWER_NORM_COL,
}



# -----------------------------
# Validation helpers (used by dataloaders / wrappers)
# -----------------------------
def validate_required_columns(columns: Sequence[str], required: Set[str], context: str) -> None:
    cols = set(columns)
    missing = sorted(required - cols)
    if missing:
        raise ValueError(f"{context}: missing required columns: {missing}")


def canonicalize_columns(df, mapping: Dict[str, str]):
    """
    Rename dataset-specific columns into canonical names.
    This does NOT reorder or select features; it only renames.
    """
    return df.rename(columns=mapping)


def enforce_lstm_feature_order(df):
    """
    Return a dataframe (or view) with columns ordered exactly as LSTM_INPUT_FEATURES.
    Use this right before creating the LSTM input tensor.
    """
    return df[LSTM_INPUT_FEATURES]