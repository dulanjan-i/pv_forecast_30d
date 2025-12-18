"""
src/preprocessing/germany_build_global_supermatrix.py

Version 3 - Global Forecasting Model: Super Matrix Builder

PURPOSE
-------
This script concatenates all 5 Germany PV plants into a single "Super Matrix" DataFrame
for global forecasting model training. Instead of training separate models per plant
(which led to overfitting in Version 02), we train ONE model on ALL plants simultaneously.

WHY SUPER MATRIX?
-----------------
1. **More Training Data**: 5 plants × ~30K samples = ~150K total samples
   - Reduces overfitting through increased dataset size
   - Model learns generalizable patterns across diverse sites

2. **Multi-Task Learning**: Model must handle multiple plants with different:
   - Locations (lat/lon differences)
   - Capacities (different power_norm scales already normalized)
   - Weather patterns (regional variations)
   - This implicit regularization improves generalization

3. **Transfer Learning Compatible**: Still uses Farm2107 pretrained weights
   - Original 15 weather/power features preserved
   - 5 new plant_id features added (one-hot encoding)
   - Zero-padding handles dimension mismatch

APPROACH
--------
1. Load pretrain_base.parquet for each plant (5 files)
2. Add plant_id string column ('plant_01', 'plant_02', ...)
3. Create one-hot encoding (5 binary columns)
4. Concatenate all into single DataFrame
5. Save as global/supermatrix_base.parquet

INPUT FILES
-----------
data/processed/pretraining/germany/
├── plant_01_pretrain_base.parquet  (~43K rows, 15 features)
├── plant_02_pretrain_base.parquet  (~43K rows)
├── plant_03_pretrain_base.parquet  (~35K rows, 20% NaNs dropped)
├── plant_05_pretrain_base.parquet  (~26K rows, 40% NaNs dropped)
└── plant_06_pretrain_base.parquet  (~44K rows)
OUTPUT FILE
-----------
data/processed/pretraining/germany/global/supermatrix_base.parquet
    - Shape: (~150K rows, 21 columns)
    - Columns: [timestamp_utc, 15 LSTM features, plant_id, plant_01, plant_02, plant_03, plant_05, plant_06]
    - Sorted by: timestamp_utc (temporal ordering preserved)

USAGE
-----
$ cd /path/to/pv_forecast_30d
$ python src/preprocessing/germany_build_global_supermatrix.py

Expected output:
    [INFO] Loading plant_01... (43000 rows)
    [INFO] Loading plant_02... (43000 rows)
    [INFO] Loading plant_03... (35000 rows)
    [INFO] Loading plant_05... (26000 rows)
    [INFO] Loading plant_06... (44000 rows)
    [INFO] Total concatenated: 150000 rows
    [INFO] Creating one-hot encoding for plant_id...
    [INFO] Final shape: (150000, 21)
    [INFO] Saved: data/processed/pretraining/germany/global/supermatrix_base.parquet

NOTES
-----
- plant_04 excluded (data quality issue: 100% zeros Mar-Jun 2024)
- No train/val split here - that happens in germany_global_rolling_origin_split.py
- One-hot columns are binary (0 or 1), compatible with LSTM input
- plant_id string column kept for debugging/analysis

DESIGN DECISIONS
----------------
- One-Hot vs Embedding: Starting with one-hot (simpler, works with zero-padding transfer learning)
- Sort by timestamp: Ensures temporal ordering for rolling origin splits
- Keep plant_id: Useful for per-plant performance analysis later

CHANGELOG
---------

Version 3.1 - Global Forecasting Model: Super Matrix Builder

Critical fixes:
1) Sort by (plant_id, timestamp) to prevent accidental cross-plant windowing.
2) Enforce required columns and drop rows with NaNs in model-required columns.
3) Always create one-hot columns for all configured plants (PLANT_ONEHOT_COLS).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.data.schema import (
    DataPaths,
    GLOBAL_LSTM_INPUT_FEATURES,
    PLANT_ID_COL,
    PLANT_IDS,
    PLANT_ONEHOT_COLS,
    REQUIRED_PRETRAIN_BASE,
    TARGET_COL,
    TIME_COL,
    validate_required_columns,
)


def _load_one_plant(plant_id: str, base_dir: Path) -> pd.DataFrame:
    """
    Load a single plant pretrain base parquet and attach plant_id + one-hot columns.
    """
    plant_dir = base_dir / plant_id
    parquet_path = plant_dir / f"{plant_id}_pretrain_base.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing pretrain base parquet: {parquet_path}")

    df = pd.read_parquet(parquet_path)

    validate_required_columns(df.columns, REQUIRED_PRETRAIN_BASE, context=f"{plant_id} pretrain_base")

    # Canonical timestamp parsing
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], utc=True)

    # Add plant_id categorical column
    df[PLANT_ID_COL] = plant_id

    # Create one-hot columns (all plants exist as columns, only one is 1)
    for col in PLANT_ONEHOT_COLS:
        df[col] = 0.0
    df.loc[:, plant_id] = 1.0

    # Drop rows that would break the model (any NaN in required model columns)
    required_for_model = set(GLOBAL_LSTM_INPUT_FEATURES + [TARGET_COL, TIME_COL, PLANT_ID_COL])
    missing_model_cols = sorted(required_for_model - set(df.columns))
    if missing_model_cols:
        raise ValueError(f"{plant_id}: missing required model columns: {missing_model_cols}")

    before = len(df)
    df = df.dropna(subset=list(required_for_model))
    after = len(df)

    dropped = before - after
    if dropped > 0:
        pct = 100.0 * dropped / max(1, before)
        print(f"[WARN] {plant_id}: dropped {dropped:,} rows due to NaNs ({pct:.2f}%)")

    return df


def main() -> None:
    paths = DataPaths(REPO_ROOT)
    germany_dir = paths.germany_pretraining

    # Inputs live here:
    # data/processed/pretraining/germany/plant_XX/plant_XX_pretrain_base.parquet
    input_base = germany_dir
    output_dir = germany_dir / "global"
    output_dir.mkdir(parents=True, exist_ok=True)

    frames: List[pd.DataFrame] = []
    print("\n" + "=" * 80)
    print("BUILDING GERMANY SUPER MATRIX (Version 3)")
    print("=" * 80)

    for plant_id in PLANT_IDS:
        print(f"[INFO] Loading {plant_id} ...")
        df_p = _load_one_plant(plant_id, input_base)
        print(f"[INFO] {plant_id}: {len(df_p):,} rows")
        frames.append(df_p)

    df = pd.concat(frames, axis=0, ignore_index=True)

    # Sort by plant then time, so windowing cannot cross plants unless code is wrong.
    df = df.sort_values([PLANT_ID_COL, TIME_COL]).reset_index(drop=True)

    # Final sanity checks
    # 1) one-hot correctness: exactly one hot per row
    onehot_sum = df[PLANT_ONEHOT_COLS].sum(axis=1)
    bad = (onehot_sum != 1.0).sum()
    if bad > 0:
        raise ValueError(f"One-hot invalid rows: {bad} rows do not sum to 1.0")

    print("\n[INFO] Supermatrix summary")
    print(f"  Rows: {len(df):,}")
    print(f"  Date range: {df[TIME_COL].min()} to {df[TIME_COL].max()}")
    print("  Rows per plant:")
    print(df[PLANT_ID_COL].value_counts().sort_index())

    out_path = output_dir / "supermatrix_base.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\n[SUCCESS] Saved: {out_path}\n")


if __name__ == "__main__":
    main()
