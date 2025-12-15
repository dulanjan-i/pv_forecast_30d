"""
farm2107_pretrain_normalize_split.py

Prepare train/validation/test splits and feature normalization
for LSTM pretraining on PVDAQ System 2107 (Farm Solar Array).

Input:
    data/processed/pretraining/farm2107_pretrain_base.parquet

Output:
    data/processed/pretraining/farm2107_pretrain_train.parquet
    data/processed/pretraining/farm2107_pretrain_val.parquet
    data/processed/pretraining/farm2107_pretrain_test.parquet
    data/processed/pretraining/farm2107_pretrain_scalers.json

Notes:
    - Target for pretraining is `pv_power_norm` (next-step prediction).
    - `pv_power_norm` is ALSO used as an input feature (autoregressive).
    - We normalize all continuous features EXCEPT the target:
        * pv_power_norm stays in [0, 1] (no scaling).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

BASE_PATH = Path("data/processed/pretraining/farm2107_pretrain_base.parquet")

OUT_DIR = Path("data/processed/pretraining")
TRAIN_OUT = OUT_DIR / "farm2107_pretrain_train.parquet"
VAL_OUT   = OUT_DIR / "farm2107_pretrain_val.parquet"
TEST_OUT  = OUT_DIR / "farm2107_pretrain_test.parquet"
SCALERS_OUT = OUT_DIR / "farm2107_pretrain_scalers.json"


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

TIME_COL = "measured_on"
TARGET_COL = "pv_power_norm"

# All candidate feature columns (will intersect with actual columns)
# pv_power_norm appears here intentionally: we use past PV as input feature.
CANDIDATE_FEATURE_COLS: List[str] = [
    "pv_power_norm",
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
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def load_base(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Base pretraining file not found: {path}")

    df = pd.read_parquet(path)
    if TIME_COL not in df.columns:
        raise ValueError(f"Expected '{TIME_COL}' column in {path}, got {df.columns.tolist()}")

    df = df.copy()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df = df.sort_values(TIME_COL).reset_index(drop=True)

    print("[INFO] Loaded base pretraining data:")
    print(df.head())
    print(df.tail())
    print(df.info())

    return df


def select_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Select relevant columns, handle missing values."""
    available_features = [c for c in CANDIDATE_FEATURE_COLS if c in df.columns]
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in base DataFrame.")

    cols = [TIME_COL] + available_features  # includes TARGET_COL
    df = df[cols].copy()

    # Drop rows where target is missing (cannot train on missing target)
    before = len(df)
    df = df.dropna(subset=[TARGET_COL])
    after = len(df)
    print(f"[INFO] Dropped {before - after} rows with NaN target.")

    # Optional: impute missing POA using GTI if available
    if "poa_irradiance" in df.columns and "global_tilted_irradiance_instant" in df.columns:
        na_before = df["poa_irradiance"].isna().sum()
        df["poa_irradiance"] = df["poa_irradiance"].fillna(df["global_tilted_irradiance_instant"])
        na_after = df["poa_irradiance"].isna().sum()
        print(f"[INFO] Imputed {na_before - na_after} NaNs in poa_irradiance using GTI.")

    # Any remaining NaNs in features? You can drop them for pretraining simplicity.
    feature_cols = [c for c in available_features]
    na_rows = df[feature_cols].isna().any(axis=1).sum()
    if na_rows > 0:
        print(f"[INFO] Dropping {na_rows} rows with remaining NaNs in features.")
        df = df.dropna(subset=feature_cols)

    print("[INFO] Cleaned DataFrame shape:", df.shape)
    return df


def time_based_split(df: pd.DataFrame, train_frac: float = 0.7, val_frac: float = 0.15):
    """Time-ordered split into train, val, test by index proportions."""
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train = df.iloc[:train_end].reset_index(drop=True)
    val = df.iloc[train_end:val_end].reset_index(drop=True)
    test = df.iloc[val_end:].reset_index(drop=True)

    print("[INFO] Time-based splits:")
    print("  Train:", train[TIME_COL].min(), "→", train[TIME_COL].max(), f"({len(train)} rows)")
    print("  Val:  ", val[TIME_COL].min(), "→", val[TIME_COL].max(), f"({len(val)} rows)")
    print("  Test: ", test[TIME_COL].min(), "→", test[TIME_COL].max(), f"({len(test)} rows)")

    return train, val, test


def compute_scalers(train: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Compute mean/std scalers for each feature *except* the target.
    pv_power_norm is left unscaled.
    """
    to_scale = [c for c in feature_cols if c != TARGET_COL]

    feature_means = {}
    feature_stds = {}

    for col in to_scale:
        mean = float(train[col].mean())
        std = float(train[col].std(ddof=0))
        # Avoid division by zero
        if std == 0.0:
            std = 1.0

        feature_means[col] = mean
        feature_stds[col] = std

    scalers = {
        "feature_means": feature_means,
        "feature_stds": feature_stds,
        "features_scaled": to_scale,
        "target_col": TARGET_COL,
        "time_col": TIME_COL,
    }

    print("[INFO] Computed scalers for features (excluding target):")
    for col in to_scale:
        print(f"  {col}: mean={feature_means[col]:.3f}, std={feature_stds[col]:.3f}")

    return scalers


def apply_scalers(df: pd.DataFrame, scalers: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    df = df.copy()
    for col in scalers["features_scaled"]:
        mean = scalers["feature_means"][col]
        std = scalers["feature_stds"][col]
        df[col] = (df[col] - mean) / std
    return df


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_base(BASE_PATH)
    df = select_and_clean(df)

    # Figure out which features are actually present
    feature_cols = [c for c in CANDIDATE_FEATURE_COLS if c in df.columns]

    # Split into train/val/test
    train_df, val_df, test_df = time_based_split(df)

    # Compute scalers on TRAIN only
    scalers = compute_scalers(train_df, feature_cols)

    # Apply scalers
    train_scaled = apply_scalers(train_df, scalers)
    val_scaled = apply_scalers(val_df, scalers)
    test_scaled = apply_scalers(test_df, scalers)

    # Save splits
    print(f"[INFO] Saving train to {TRAIN_OUT}")
    train_scaled.to_parquet(TRAIN_OUT, index=False)

    print(f"[INFO] Saving val to {VAL_OUT}")
    val_scaled.to_parquet(VAL_OUT, index=False)

    print(f"[INFO] Saving test to {TEST_OUT}")
    test_scaled.to_parquet(TEST_OUT, index=False)

    # Save scalers metadata as JSON
    print(f"[INFO] Saving scalers to {SCALERS_OUT}")
    with SCALERS_OUT.open("w") as f:
        json.dump(scalers, f, indent=2)

    print("[INFO] Done: normalized train/val/test splits written for LSTM pretraining.")


if __name__ == "__main__":
    main()