"""
germany_pretrain_normalize_split.py

Time-split + normalize per-plant Germany pretrain base datasets.

INPUT
- data/processed/pretraining/germany/{plant_id}_pretrain_base.parquet

OUTPUT (per plant)
- data/processed/pretraining/germany/{plant_id}/train.parquet
- data/processed/pretraining/germany/{plant_id}/val.parquet
- data/processed/pretraining/germany/{plant_id}/test.parquet
- data/processed/pretraining/germany/{plant_id}/scaler.json

Normalization rules (match Farm2107 behavior)
- Fit scaler on TRAIN ONLY.
- Scale input features (not timestamp).
- Do NOT scale the target column power_norm (keep it in physical normalized units).
- Feature order is defined by LSTM_INPUT_FEATURES in src/data/schema.py.
- poa_irradiance may be NaN for now; scaler will skip NaNs naturally. Later PVLib
  will fill it and the pipeline can be rerun.

Splitting
- Chronological split, no shuffling.
- Defaults: train 70%, val 15%, test 15%.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.data.schema import (
    DataPaths,
    LSTM_INPUT_FEATURES,
    TIME_COL,
    POWER_NORM_COL,
    validate_required_columns,
)

PLANT_IDS: List[str] = ["plant_01","plant_02","plant_03","plant_04","plant_05","plant_06"]


def fit_scaler_train(df_train: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Return per-feature mean/std dict computed on TRAIN ONLY.
    NaNs are ignored in mean/std.
    """
    stats: Dict[str, Dict[str, float]] = {}
    for c in feature_cols:
        x = pd.to_numeric(df_train[c], errors="coerce")
        mu = float(np.nanmean(x.to_numpy()))
        sigma = float(np.nanstd(x.to_numpy()))
        if sigma == 0.0 or np.isnan(sigma):
            sigma = 1.0
        stats[c] = {"mean": mu, "std": sigma}
    return stats


def apply_scaler(df: pd.DataFrame, stats: Dict[str, Dict[str, float]], feature_cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in feature_cols:
        mu = stats[c]["mean"]
        sigma = stats[c]["std"]
        out[c] = (pd.to_numeric(out[c], errors="coerce") - mu) / sigma
    return out


def split_indices(n: int, train_frac: float = 0.70, val_frac: float = 0.15) -> Tuple[slice, slice, slice]:
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_sl = slice(0, n_train)
    val_sl = slice(n_train, n_train + n_val)
    test_sl = slice(n_train + n_val, n)
    return train_sl, val_sl, test_sl


def process_one(plant_id: str, paths: DataPaths) -> None:
    in_path = paths.germany_pretraining / f"{plant_id}_pretrain_base.parquet"
    if not in_path.exists():
        raise FileNotFoundError(f"Missing pretrain base: {in_path}")

    df = pd.read_parquet(in_path)

    # Required: time + all LSTM features
    required = {TIME_COL} | set(LSTM_INPUT_FEATURES)
    validate_required_columns(df.columns, required, context=f"{plant_id}: pretrain_base")

    # Ensure sorted time
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], utc=True)
    df = df.sort_values(TIME_COL).reset_index(drop=True)

    # Define feature cols for scaling: all LSTM inputs EXCEPT the target
    scale_cols = [c for c in LSTM_INPUT_FEATURES if c != POWER_NORM_COL]

    # Split
    n = len(df)
    tr, va, te = split_indices(n)
    df_train = df.iloc[tr].copy()
    df_val = df.iloc[va].copy()
    df_test = df.iloc[te].copy()

    # Fit scaler on train only, apply to all splits
    stats = fit_scaler_train(df_train, scale_cols)
    df_train_s = apply_scaler(df_train, stats, scale_cols)
    df_val_s = apply_scaler(df_val, stats, scale_cols)
    df_test_s = apply_scaler(df_test, stats, scale_cols)

    # Write outputs
    out_dir = paths.germany_pretraining / plant_id
    out_dir.mkdir(parents=True, exist_ok=True)

    df_train_s.to_parquet(out_dir / "train.parquet", index=False)
    df_val_s.to_parquet(out_dir / "val.parquet", index=False)
    df_test_s.to_parquet(out_dir / "test.parquet", index=False)

    with open(out_dir / "scaler.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "scaled_features": scale_cols,
                "target": POWER_NORM_COL,
                "stats": stats,
            },
            f,
            indent=2,
        )

    print(f"[OK] {plant_id}: train={len(df_train_s):,} val={len(df_val_s):,} test={len(df_test_s):,} -> {out_dir}")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    paths = DataPaths(repo_root=repo_root)

    for pid in PLANT_IDS:
        process_one(pid, paths)

    print("[INFO] Done: Germany splits + scalers written for all plants.")


if __name__ == "__main__":
    main()
