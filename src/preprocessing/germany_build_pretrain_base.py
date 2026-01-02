"""
germany_build_pretrain_base.py - Stage 2 Transfer Learning (Version 02)

Build per-plant pretraining base parquets for Germany.

INPUT
- data/processed/germany/{plant_id}_pv_weather_15min.parquet

OUTPUT
- data/processed/pretraining/germany/{plant_id}_pretrain_base.parquet

Contract
- Output must contain at least:
  - timestamp_utc
  - power_norm (target / autoregressive input)
  - all weather features required by the canonical LSTM feature list
  - poa_irradiance column exists (filled with NaN for now, to be populated later
    by PVLib feature generation)

Why this exists
- Keeps Germany data aligned with the pretrained LSTM encoder contract defined in
  src/data/schema.py, especially feature presence and naming stability.

Version 02 Changes (Dec 2025):
- Excluded plant_04 due to data quality issues (100% zeros during Mar-Jun 2024)
- Added NaN dropping for power_norm to ensure clean training data
- See reports/stage2_version01_failed_chronological_split.md for details
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from src.data.schema import (
    DataPaths,
    GERMANY_TO_CANONICAL,
    LSTM_INPUT_FEATURES,
    REQUIRED_MERGED,
    TIME_COL,
    POWER_NORM_COL,
    validate_required_columns,
    canonicalize_columns,
)

# Version 02: Excluded plant_04 (data quality issue - 100% zeros in Mar-Jun 2024)
PLANT_IDS: List[str] = [
    "plant_01",
    "plant_02",
    "plant_03",
    # "plant_04",  # EXCLUDED: See reports/stage2_version01_failed_chronological_split.md
    "plant_05",
    "plant_06",
]


def build_one(plant_id: str, paths: DataPaths) -> Path:
    in_path = paths.germany_processed / f"{plant_id}_pv_weather_15min.parquet"
    if not in_path.exists():
        raise FileNotFoundError(f"Missing processed input: {in_path}")

    df = pd.read_parquet(in_path)

    # Validate merged contract (Germany processed should already satisfy this)
    validate_required_columns(df.columns, REQUIRED_MERGED, context=f"{plant_id}: processed(germany)")

    # Canonicalize names (mostly identity for Germany, but keep it explicit)
    df = canonicalize_columns(df, GERMANY_TO_CANONICAL)

    # Ensure time col is UTC and sorted
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], utc=True)
    df = df.sort_values(TIME_COL).reset_index(drop=True)

    # Ensure poa_irradiance exists (LSTM contract needs it, filled later by PVLib)
    if "poa_irradiance" not in df.columns:
        df["poa_irradiance"] = np.nan

    # Set poa_irradiance := globa;_tilted_irradiance as a proxy (temporary)
    if df["poa_irradiance"].isna().all() and "global_tilted_irradiance_instant" in df.columns:
        df["poa_irradiance"] = df["global_tilted_irradiance_instant"]

    # Ensure required pretrain base columns exist
    missing = [c for c in LSTM_INPUT_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"{plant_id}: missing columns required by LSTM_INPUT_FEATURES: {missing}")

    # Keep only what we need for pretraining base (time + LSTM inputs)
    out_df = df[[TIME_COL] + LSTM_INPUT_FEATURES].copy()

    # Basic sanity: target must exist
    if POWER_NORM_COL not in out_df.columns:
        raise ValueError(f"{plant_id}: missing target {POWER_NORM_COL} in pretrain base")

    # Version 02: Drop rows with NaN in power_norm (ensures clean training data)
    # This removes nighttime periods with missing production data and any data quality issues
    rows_before = len(out_df)
    out_df = out_df.dropna(subset=[POWER_NORM_COL])
    rows_after = len(out_df)
    rows_dropped = rows_before - rows_after
    if rows_dropped > 0:
        pct_dropped = (rows_dropped / rows_before) * 100
        print(f"[INFO] {plant_id}: Dropped {rows_dropped:,} rows ({pct_dropped:.1f}%) with NaN in {POWER_NORM_COL}")

    # Write
    out_dir = paths.germany_pretraining
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{plant_id}_pretrain_base.parquet"
    out_df.to_parquet(out_path, index=False)

    print(f"[OK] {plant_id}: wrote {out_path} rows={len(out_df):,} cols={len(out_df.columns)}")
    return out_path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    paths = DataPaths(repo_root=repo_root)

    for pid in PLANT_IDS:
        build_one(pid, paths)

    print("[INFO] Done: Germany pretrain base written for all plants.")


if __name__ == "__main__":
    main()
