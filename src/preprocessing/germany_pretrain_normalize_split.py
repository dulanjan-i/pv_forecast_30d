"""
germany_pretrain_normalize_split.py - Stage 2 Transfer Learning (Version 02)

Time-split + normalize per-plant Germany pretrain base datasets.

INPUT
- data/processed/pretraining/germany/{plant_id}_pretrain_base.parquet

OUTPUT (per plant) - ALL THREE SPLITS
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

Old Splitting Strategy (Version 01) - DEPRECATED
- Simple chronological split (train 70%, val 15%, test 15%)
- Problem: Validation set ended up biased towards summer months, causing misleadingly low val errors.

New Splitting Strategy (Version 02)
- TODO: Implement stratified temporal split (ensures balanced seasonal representation)
- Current: Simple chronological split (train 70%, val 15%, test 15%)
- Target: Seasonal stratification to prevent validation bias
- See reports/stage2_version01_failed_chronological_split.md for methodology

Version 02 Changes (Dec 2025):
- Excluded plant_04 due to data quality issues (100% zeros during Mar-Jun 2024)
- Ensured all three splits (train/val/test) are explicitly created
- Prepared for stratified temporal split implementation (coming next)
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

# Version 02: Excluded plant_04 (data quality issue - 100% zeros in Mar-Jun 2024)
PLANT_IDS: List[str] = ["plant_01","plant_02","plant_03","plant_05","plant_06"]


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


def stratified_temporal_split(
    df: pd.DataFrame,
    time_col: str,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Stratified temporal split - ensures balanced seasonal representation across train/val/test.
    
    Version 02 Strategy (replaces Version 01 chronological split that failed):
    1. Classify each timestamp into a season (winter/spring/summer/fall)
    2. Calculate overall seasonal distribution in the full dataset
    3. For each split (train/val/test), sample from each season proportionally
    4. This ensures all splits have similar seasonal characteristics
    
    Why this works:
    - Prevents validation set from being all winter (easy) or all summer (hard)
    - Each split is representative of the full temporal distribution
    - Validation metrics become comparable across plants
    
    Args:
        df: DataFrame with timestamp column (must be sorted)
        time_col: Name of timestamp column
        train_frac: Fraction for training (default 0.70)
        val_frac: Fraction for validation (default 0.15)
        test_frac: Fraction for test (default 0.15)
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_indices, val_indices, test_indices) as numpy arrays
        Indices are NOT contiguous (stratified sampling across seasons)
    """
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError(f"Fractions must sum to 1.0, got {train_frac + val_frac + test_frac}")
    
    np.random.seed(random_seed)
    
    # Ensure timestamps are datetime
    timestamps = pd.to_datetime(df[time_col])
    
    # Classify each row by season (Northern Hemisphere)
    months = timestamps.dt.month
    seasons = np.empty(len(months), dtype='U10')
    seasons[(months == 12) | (months == 1) | (months == 2)] = 'winter'
    seasons[(months == 3) | (months == 4) | (months == 5)] = 'spring'
    seasons[(months == 6) | (months == 7) | (months == 8)] = 'summer'
    seasons[(months == 9) | (months == 10) | (months == 11)] = 'fall'
    
    # Initialize index arrays
    train_indices = []
    val_indices = []
    test_indices = []
    
    # For each season, split proportionally
    for season in ['winter', 'spring', 'summer', 'fall']:
        season_mask = (seasons == season)
        season_indices = np.where(season_mask)[0]
        n_season = len(season_indices)
        
        if n_season == 0:
            continue  # Skip seasons with no data
        
        # Shuffle season indices for random sampling
        np.random.shuffle(season_indices)
        
        # Split this season's indices proportionally
        n_train = int(n_season * train_frac)
        n_val = int(n_season * val_frac)
        # Test gets the remainder to ensure all samples are used
        
        train_indices.extend(season_indices[:n_train])
        val_indices.extend(season_indices[n_train:n_train + n_val])
        test_indices.extend(season_indices[n_train + n_val:])
    
    # Convert to numpy arrays and sort (maintains some temporal ordering within splits)
    train_indices = np.array(sorted(train_indices), dtype=int)
    val_indices = np.array(sorted(val_indices), dtype=int)
    test_indices = np.array(sorted(test_indices), dtype=int)
    
    # Diagnostic output
    total = len(train_indices) + len(val_indices) + len(test_indices)
    print(f"[SPLIT] Stratified temporal split: train={len(train_indices)} ({len(train_indices)/total*100:.1f}%), "
          f"val={len(val_indices)} ({len(val_indices)/total*100:.1f}%), "
          f"test={len(test_indices)} ({len(test_indices)/total*100:.1f}%)")
    
    # Print seasonal distribution per split for verification
    for split_name, indices in [('Train', train_indices), ('Val', val_indices), ('Test', test_indices)]:
        split_seasons = seasons[indices]
        season_counts = {s: np.sum(split_seasons == s) for s in ['winter', 'spring', 'summer', 'fall']}
        season_pcts = {s: (count / len(indices) * 100) if len(indices) > 0 else 0 
                      for s, count in season_counts.items()}
        print(f"  {split_name:5s} seasons: Winter={season_pcts['winter']:4.1f}% "
              f"Spring={season_pcts['spring']:4.1f}% "
              f"Summer={season_pcts['summer']:4.1f}% "
              f"Fall={season_pcts['fall']:4.1f}%")
    
    return train_indices, val_indices, test_indices


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

    # Split into train/val/test (ALL THREE SPLITS)
    # Version 02: Using stratified temporal split (ensures balanced seasonal representation)
    print(f"\n[INFO] {plant_id}: Performing stratified temporal split...")
    train_idx, val_idx, test_idx = stratified_temporal_split(
        df=df,
        time_col=TIME_COL,
        train_frac=0.70,
        val_frac=0.15,
        test_frac=0.15,
        random_seed=42
    )
    
    # Extract splits using indices (not slices, since stratified sampling is non-contiguous)
    df_train = df.iloc[train_idx].copy()
    df_val = df.iloc[val_idx].copy()
    df_test = df.iloc[test_idx].copy()
    
    # Sort each split by time (stratification may have shuffled within seasons)
    df_train = df_train.sort_values(TIME_COL).reset_index(drop=True)
    df_val = df_val.sort_values(TIME_COL).reset_index(drop=True)
    df_test = df_test.sort_values(TIME_COL).reset_index(drop=True)

    # Fit scaler on TRAIN ONLY, then apply to all three splits
    # This prevents data leakage from val/test into normalization
    stats = fit_scaler_train(df_train, scale_cols)
    df_train_s = apply_scaler(df_train, stats, scale_cols)
    df_val_s = apply_scaler(df_val, stats, scale_cols)
    df_test_s = apply_scaler(df_test, stats, scale_cols)  # TEST split scaled

    # Write ALL THREE splits to disk
    out_dir = paths.germany_pretraining / plant_id
    out_dir.mkdir(parents=True, exist_ok=True)

    df_train_s.to_parquet(out_dir / "train.parquet", index=False)  # TRAIN split
    df_val_s.to_parquet(out_dir / "val.parquet", index=False)      # VAL split
    df_test_s.to_parquet(out_dir / "test.parquet", index=False)    # TEST split

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
