"""
src/preprocessing/germany_global_rolling_origin_split.py

Version 3 - Rolling Origin split for the Germany supermatrix.

Critical fixes:
1) Do NOT z-score POWER_NORM_COL (your target and autoregressive input is already normalized).
2) Handle std == 0 safely to avoid NaNs/infs after normalization.
3) Make folds plant-aware: optionally drop plants that do not have enough rows
   in train and val for that fold.

Outputs per fold:
- fold_{k}_train.parquet
- fold_{k}_val.parquet
- fold_{k}_scaler.json  (only for normalized feature columns)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.data.schema import (
    GLOBAL_LSTM_INPUT_FEATURES,
    LSTM_INPUT_FEATURES,
    PLANT_ID_COL,
    PLANT_ONEHOT_COLS,
    POWER_NORM_COL,
    TARGET_COL,
    TIME_COL,
)

# Rolling origin folds
FOLDS = [
    {"fold_id": 1, "name": "spring", "val_start": "2023-03-01", "val_end": "2023-06-01"},
    {"fold_id": 2, "name": "summer", "val_start": "2023-06-01", "val_end": "2023-09-01"},
    {"fold_id": 3, "name": "fall",   "val_start": "2023-09-01", "val_end": "2023-12-01"},
    {"fold_id": 4, "name": "winter", "val_start": "2023-12-01", "val_end": "2024-03-01"},
]

# You used window_size=96 in training. Require at least one window worth of rows per plant.
DEFAULT_WINDOW_SIZE: int = 96
MIN_ROWS_PER_PLANT: int = DEFAULT_WINDOW_SIZE + 1

# Toggle: enforce every plant has enough rows in train and val for each fold
FILTER_PLANTS_WITH_INSUFFICIENT_ROWS: bool = True


def load_supermatrix(data_dir: Path) -> pd.DataFrame:
    supermatrix_file = data_dir / "supermatrix_base.parquet"
    if not supermatrix_file.exists():
        raise FileNotFoundError(f"Supermatrix not found: {supermatrix_file}")

    df = pd.read_parquet(supermatrix_file)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], utc=True)

    # Sort by plant then time for consistent downstream behavior
    df = df.sort_values([PLANT_ID_COL, TIME_COL]).reset_index(drop=True)
    return df


def _fit_zscore(train_df: pd.DataFrame, cols: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Fit mean/std on train only. If std is 0 or tiny, clamp to 1.0 to avoid NaNs.
    """
    stats: Dict[str, Dict[str, float]] = {}
    for c in cols:
        mu = float(np.nanmean(train_df[c].values))
        sd = float(np.nanstd(train_df[c].values))
        if not np.isfinite(sd) or sd < 1e-12:
            sd = 1.0
        stats[c] = {"mean": mu, "std": sd}
    return stats


def _apply_zscore(df: pd.DataFrame, stats: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    out = df.copy()
    for c, st in stats.items():
        out[c] = (out[c] - st["mean"]) / st["std"]
    return out


def _plant_counts(df: pd.DataFrame) -> pd.Series:
    return df.groupby(PLANT_ID_COL).size().sort_index()


def _filter_plants(train_df: pd.DataFrame, val_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Drop plants that do not have enough rows to form at least one window in both train and val.
    """
    train_counts = _plant_counts(train_df)
    val_counts = _plant_counts(val_df)

    ok_plants = []
    for plant_id in sorted(set(train_counts.index) | set(val_counts.index)):
        tr = int(train_counts.get(plant_id, 0))
        va = int(val_counts.get(plant_id, 0))
        if tr >= MIN_ROWS_PER_PLANT and va >= MIN_ROWS_PER_PLANT:
            ok_plants.append(plant_id)

    dropped = sorted(set(train_df[PLANT_ID_COL].unique()) | set(val_df[PLANT_ID_COL].unique()) - set(ok_plants))
    if dropped:
        print(f"[WARN] Dropping plants for this fold due to insufficient rows: {dropped}")
        print(f"[WARN] Requirement: train >= {MIN_ROWS_PER_PLANT}, val >= {MIN_ROWS_PER_PLANT}")

    train_f = train_df[train_df[PLANT_ID_COL].isin(ok_plants)].copy()
    val_f = val_df[val_df[PLANT_ID_COL].isin(ok_plants)].copy()
    return train_f, val_f


def process_fold(df: pd.DataFrame, fold_cfg: Dict, output_dir: Path) -> None:
    fold_id = fold_cfg["fold_id"]
    name = fold_cfg["name"]

    val_start = pd.to_datetime(fold_cfg["val_start"], utc=True)
    val_end = pd.to_datetime(fold_cfg["val_end"], utc=True)

    # Rolling origin split
    train_df = df[df[TIME_COL] < val_start].copy()
    val_df = df[(df[TIME_COL] >= val_start) & (df[TIME_COL] < val_end)].copy()

    print("\n" + "-" * 80)
    print(f"[FOLD {fold_id}] {name.upper()}  val=[{val_start} .. {val_end})")
    print(f"Train rows: {len(train_df):,} | Val rows: {len(val_df):,}")
    print("Train rows per plant:")
    print(_plant_counts(train_df))
    print("Val rows per plant:")
    print(_plant_counts(val_df))

    if FILTER_PLANTS_WITH_INSUFFICIENT_ROWS:
        train_df, val_df = _filter_plants(train_df, val_df)
        print("[INFO] After plant filtering")
        print(f"Train rows: {len(train_df):,} | Val rows: {len(val_df):,}")
        print("Train rows per plant:")
        print(_plant_counts(train_df))
        print("Val rows per plant:")
        print(_plant_counts(val_df))

    # Normalize only weather features, not POWER_NORM_COL
    # LSTM_INPUT_FEATURES includes POWER_NORM_COL, so exclude it for z-score.
    norm_cols = [c for c in LSTM_INPUT_FEATURES if c != POWER_NORM_COL]

    stats = _fit_zscore(train_df, norm_cols)
    train_norm = _apply_zscore(train_df, stats)
    val_norm = _apply_zscore(val_df, stats)

    # Drop any remaining NaNs in columns the training script will read
    required_cols = list({TIME_COL, PLANT_ID_COL, TARGET_COL} | set(GLOBAL_LSTM_INPUT_FEATURES))
    before_tr, before_va = len(train_norm), len(val_norm)
    train_norm = train_norm.dropna(subset=required_cols)
    val_norm = val_norm.dropna(subset=required_cols)
    if len(train_norm) != before_tr or len(val_norm) != before_va:
        print(f"[WARN] Dropped NaN rows after scaling. Train: {before_tr-len(train_norm)}, Val: {before_va-len(val_norm)}")

    # Ensure one-hot unchanged (still binary)
    for c in PLANT_ONEHOT_COLS:
        bad = (~train_norm[c].isin([0.0, 1.0])).sum() + (~val_norm[c].isin([0.0, 1.0])).sum()
        if bad > 0:
            raise ValueError(f"One-hot column {c} contains non-binary values after processing.")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    train_file = output_dir / f"fold_{fold_id}_train.parquet"
    val_file = output_dir / f"fold_{fold_id}_val.parquet"
    scaler_file = output_dir / f"fold_{fold_id}_scaler.json"

    train_norm.to_parquet(train_file, index=False)
    val_norm.to_parquet(val_file, index=False)

    with open(scaler_file, "w") as f:
        json.dump({"normalized_columns": norm_cols, "stats": stats}, f, indent=2)

    print(f"[SUCCESS] Saved: {train_file.name}, {val_file.name}, {scaler_file.name}")


def main() -> None:
    data_dir = REPO_ROOT / "data" / "processed" / "pretraining" / "germany" / "global"
    df = load_supermatrix(data_dir)

    for fold_cfg in FOLDS:
        process_fold(df, fold_cfg, data_dir)

    print("\n✅ Rolling origin folds complete.\n")


if __name__ == "__main__":
    main()
