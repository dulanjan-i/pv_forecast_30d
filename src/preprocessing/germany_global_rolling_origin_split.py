"""
src/preprocessing/germany_global_rolling_origin_split.py

Version 3 - Global Forecasting Model: Rolling Origin Cross-Validation Splits

PURPOSE
-------
Creates 4 temporal folds for rolling origin cross-validation on the Super Matrix.
Each fold tests on a different season while training on all preceding data,
respecting temporal causality (no future leakage).

WHY ROLLING ORIGIN?
-------------------
1. **Respects Temporal Causality**: Always train on past, test on future
   - Mimics production scenario: retrain periodically, forecast ahead
   - No data leakage (unlike random or stratified splits)

2. **Seasonal Robustness**: Each fold tests a different season
   - Fold 1: Spring (Mar-Jun) - moderate production
   - Fold 2: Summer (Jun-Sep) - high production  
   - Fold 3: Fall (Sep-Dec) - moderate production
   - Fold 4: Winter (Dec-Mar) - low production
   - Average performance across folds = robust estimate

3. **Realistic Evaluation**: Tests model's ability to generalize to new time periods
   - Critical for deployment: future data is fundamentally different
   - Shows which seasons are harder to predict

FOLD DESIGN
-----------
Each fold has:
- **Train Set**: All data with timestamp < validation_start_date
- **Validation Set**: Data in [validation_start_date, validation_end_date)
- **Normalization**: Z-score fitted on train, applied to both train and val

Fold Definitions:
    Fold 1 (Spring): Val = 2023-03-01 to 2023-06-01 (Train on Dec 2022 - Feb 2023)
    Fold 2 (Summer): Val = 2023-06-01 to 2023-09-01 (Train on Dec 2022 - May 2023)
    Fold 3 (Fall):   Val = 2023-09-01 to 2023-12-01 (Train on Dec 2022 - Aug 2023)
    Fold 4 (Winter): Val = 2023-12-01 to 2024-03-01 (Train on Dec 2022 - Nov 2023)

Note: Some plants end Apr 2024, so Fold 4 may have less validation data.

INPUT FILE
----------
data/processed/pretraining/germany/global/supermatrix_base.parquet
    - Shape: (~150K rows, 21 columns)
    - Columns: [timestamp_utc, 15 LSTM features, plant_id, 5 one-hot cols]

OUTPUT FILES (per fold)
-----------------------
data/processed/pretraining/germany/global/
├── fold_1_train.parquet    (normalized features)
├── fold_1_val.parquet      (normalized features)
├── fold_1_scaler.json      (z-score stats: mean, std per feature)
├── fold_2_train.parquet
├── fold_2_val.parquet
├── fold_2_scaler.json
├── fold_3_train.parquet
├── fold_3_val.parquet
├── fold_3_scaler.json
├── fold_4_train.parquet
├── fold_4_val.parquet
└── fold_4_scaler.json

Total: 12 files (3 per fold)

USAGE
-----
$ cd /path/to/pv_forecast_30d
$ python src/preprocessing/germany_global_rolling_origin_split.py

Expected output:
    [FOLD 1/4] Spring validation (2023-03-01 to 2023-06-01)
      Train: 25000 rows (Dec 2022 - Feb 2023)
      Val:   15000 rows (Mar 2023 - May 2023)
      Saved: fold_1_train.parquet, fold_1_val.parquet, fold_1_scaler.json
    
    [FOLD 2/4] Summer validation (2023-06-01 to 2023-09-01)
      Train: 40000 rows (Dec 2022 - May 2023)
      Val:   15000 rows (Jun 2023 - Aug 2023)
      ...

NORMALIZATION STRATEGY
----------------------
- **Features to normalize**: 15 original LSTM features (weather + power_norm)
- **Not normalized**: 5 one-hot plant_id columns (already binary 0/1)
- **Method**: Z-score (X - mean) / std
- **Fit on**: Train set only (prevents leakage)
- **Applied to**: Both train and val sets
- **Saved**: mean/std per feature in fold_X_scaler.json for inference

VALIDATION CHECKS
-----------------
1. No temporal overlap between train and val
2. Train timestamps < Val timestamps (causality)
3. All plants represented in both sets (check plant_id distribution)
4. No NaN values after normalization
5. One-hot columns unchanged (still binary)

Author: PV Forecast Team
Date: December 2024
Version: 3.0 (Global Forecasting Model)
"""

import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# Add repo root to path for imports
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.data.schema import (
    GLOBAL_LSTM_INPUT_FEATURES,
    LSTM_INPUT_FEATURES,
    PLANT_ID_COL,
    PLANT_ONEHOT_COLS,
    TIME_COL,
)


# Fold definitions (Rolling Origin Cross-Validation)
FOLDS = [
    {
        'fold_id': 1,
        'name': 'spring',
        'val_start': '2023-03-01',
        'val_end': '2023-06-01',
        'description': 'Spring (moderate production)',
    },
    {
        'fold_id': 2,
        'name': 'summer',
        'val_start': '2023-06-01',
        'val_end': '2023-09-01',
        'description': 'Summer (high production)',
    },
    {
        'fold_id': 3,
        'name': 'fall',
        'val_start': '2023-09-01',
        'val_end': '2023-12-01',
        'description': 'Fall (moderate production)',
    },
    {
        'fold_id': 4,
        'name': 'winter',
        'val_start': '2023-12-01',
        'val_end': '2024-03-01',
        'description': 'Winter (low production)',
    },
    {
        'fold_id': 5,
        'name': 'test',
        'val_start': '2024-03-01',
        'val_end': '2024-10-01',
        'description': 'TEST SET - Held-out final evaluation',
    },
]


def load_supermatrix(data_dir: Path) -> pd.DataFrame:
    """
    Load the Super Matrix created by germany_build_global_supermatrix.py.
    
    Parameters
    ----------
    data_dir : Path
        Directory containing supermatrix_base.parquet
        
    Returns
    -------
    pd.DataFrame
        Super Matrix with all 5 plants concatenated
        
    Raises
    ------
    FileNotFoundError
        If supermatrix_base.parquet doesn't exist
    """
    supermatrix_file = data_dir / "supermatrix_base.parquet"
    
    if not supermatrix_file.exists():
        raise FileNotFoundError(
            f"Super Matrix not found: {supermatrix_file}\n"
            f"Run germany_build_global_supermatrix.py first!"
        )
    
    df = pd.read_parquet(supermatrix_file)
    
    # Convert timestamp to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df[TIME_COL]):
        df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    
    print(f"[INFO] Loaded Super Matrix: {len(df):,} rows")
    print(f"[INFO] Date range: {df[TIME_COL].min()} to {df[TIME_COL].max()}")
    
    return df


def split_fold(
    df: pd.DataFrame,
    val_start: str,
    val_end: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split Super Matrix into train and validation for one fold.
    
    Parameters
    ----------
    df : pd.DataFrame
        Super Matrix with timestamp_utc column
    val_start : str
        Validation start date (ISO format: 'YYYY-MM-DD')
    val_end : str
        Validation end date (exclusive)
        
    Returns
    -------
    train_df : pd.DataFrame
        All data with timestamp < val_start
    val_df : pd.DataFrame
        Data in [val_start, val_end)
        
    Notes
    -----
    - Train always comes from the past (respects causality)
    - No temporal overlap between train and val
    - Some plants may have less data in later folds (ended Apr 2024)
    """
    val_start_dt = pd.to_datetime(val_start, utc=True)
    val_end_dt = pd.to_datetime(val_end, utc=True)
    
    # Train: everything before validation start
    train_mask = df[TIME_COL] < val_start_dt
    train_df = df[train_mask].copy()
    
    # Val: data in [val_start, val_end)
    val_mask = (df[TIME_COL] >= val_start_dt) & (df[TIME_COL] < val_end_dt)
    val_df = df[val_mask].copy()
    
    # Validation: check for temporal overlap
    if not train_df.empty and not val_df.empty:
        train_max = train_df[TIME_COL].max()
        val_min = val_df[TIME_COL].min()
        assert train_max < val_min, f"Temporal overlap detected! Train max: {train_max}, Val min: {val_min}"
    
    return train_df, val_df


def normalize_fold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    Z-score normalize LSTM features (fit on train, apply to both).
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Training data (used to fit scaler)
    val_df : pd.DataFrame
        Validation data (scaler applied but not fitted)
        
    Returns
    -------
    train_norm : pd.DataFrame
        Normalized training data
    val_norm : pd.DataFrame
        Normalized validation data  
    scaler_stats : dict
        {feature_name: {'mean': float, 'std': float}} for all 15 LSTM features
        
    Notes
    -----
    - Only normalizes original 15 LSTM features
    - One-hot plant_id columns left unchanged (already binary)
    - Scaler stats saved to fold_X_scaler.json for inference
    - Handles std=0 by setting std=1.0 (prevents division by zero)
    """
    # Features to normalize (15 original LSTM features)
    norm_features = LSTM_INPUT_FEATURES
    
    # Fit scaler on train set only
    train_means = train_df[norm_features].mean()
    train_stds = train_df[norm_features].std()
    
    # Handle zero std (constant features) - set to 1.0 to avoid division by zero
    train_stds = train_stds.replace(0.0, 1.0)
    
    # Apply normalization to both sets
    train_norm = train_df.copy()
    val_norm = val_df.copy()
    
    for feat in norm_features:
        mean = train_means[feat]
        std = train_stds[feat]
        
        train_norm[feat] = (train_df[feat] - mean) / std
        val_norm[feat] = (val_df[feat] - mean) / std
    
    # Verify one-hot columns unchanged (should still be 0 or 1)
    for onehot_col in PLANT_ONEHOT_COLS:
        assert train_norm[onehot_col].isin([0.0, 1.0]).all(), f"One-hot column {onehot_col} corrupted in train!"
        assert val_norm[onehot_col].isin([0.0, 1.0]).all(), f"One-hot column {onehot_col} corrupted in val!"
    
    # Build scaler stats dict for saving
    scaler_stats = {}
    for feat in norm_features:
        scaler_stats[feat] = {
            'mean': float(train_means[feat]),
            'std': float(train_stds[feat]),
        }
    
    return train_norm, val_norm, scaler_stats


def process_fold(
    df: pd.DataFrame,
    fold_config: dict,
    output_dir: Path,
) -> None:
    """
    Process one fold: split, normalize, save.
    
    Parameters
    ----------
    df : pd.DataFrame
        Super Matrix
    fold_config : dict
        Fold configuration with keys: fold_id, name, val_start, val_end
    output_dir : Path
        Output directory for fold files
        
    Outputs
    -------
    Creates 3 files in output_dir:
        - fold_X_train.parquet
        - fold_X_val.parquet
        - fold_X_scaler.json
    """
    fold_id = fold_config['fold_id']
    fold_name = fold_config['name']
    val_start = fold_config['val_start']
    val_end = fold_config['val_end']
    description = fold_config['description']
    
    print(f"\n{'='*80}")
    print(f"[FOLD {fold_id}/4] {fold_name.upper()} - {description}")
    print(f"{'='*80}")
    print(f"Validation period: {val_start} to {val_end}")
    
    # Step 1: Split
    print("\n[STEP 1/3] Splitting data...")
    train_df, val_df = split_fold(df, val_start, val_end)
    
    print(f"  Train: {len(train_df):,} rows (dates < {val_start})")
    print(f"  Val:   {len(val_df):,} rows (dates in [{val_start}, {val_end}))")
    
    if len(train_df) == 0:
        raise ValueError(f"Fold {fold_id} has ZERO training samples! Check date ranges.")
    if len(val_df) == 0:
        print(f"[WARNING] Fold {fold_id} has ZERO validation samples! (Some plants may have ended before {val_start})")
        print("[WARNING] Skipping this fold...")
        return
    
    # Check plant distribution
    print("\n  Plant distribution:")
    for split_name, split_df in [("Train", train_df), ("Val", val_df)]:
        print(f"    {split_name}:")
        for plant_id in sorted(split_df[PLANT_ID_COL].unique()):
            count = (split_df[PLANT_ID_COL] == plant_id).sum()
            pct = (count / len(split_df)) * 100
            print(f"      {plant_id}: {count:>6,} rows ({pct:>5.2f}%)")
    
    # Step 2: Normalize
    print("\n[STEP 2/3] Normalizing features...")
    train_norm, val_norm, scaler_stats = normalize_fold(train_df, val_df)
    
    # Check for NaN after normalization
    train_nans = train_norm[LSTM_INPUT_FEATURES].isna().sum().sum()
    val_nans = val_norm[LSTM_INPUT_FEATURES].isna().sum().sum()
    if train_nans > 0 or val_nans > 0:
        print(f"[WARNING] Found NaNs after normalization: Train={train_nans}, Val={val_nans}")
        print("[WARNING] Dropping rows with NaN...")
        train_norm = train_norm.dropna(subset=LSTM_INPUT_FEATURES)
        val_norm = val_norm.dropna(subset=LSTM_INPUT_FEATURES)
    
    print(f"  Normalized {len(LSTM_INPUT_FEATURES)} features (one-hot cols unchanged)")
    
    # Step 3: Save
    print("\n[STEP 3/3] Saving fold data...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_file = output_dir / f"fold_{fold_id}_train.parquet"
    val_file = output_dir / f"fold_{fold_id}_val.parquet"
    scaler_file = output_dir / f"fold_{fold_id}_scaler.json"
    
    train_norm.to_parquet(train_file, index=False)
    val_norm.to_parquet(val_file, index=False)
    
    with open(scaler_file, 'w') as f:
        json.dump(scaler_stats, f, indent=2)
    
    train_size_mb = train_file.stat().st_size / (1024 * 1024)
    val_size_mb = val_file.stat().st_size / (1024 * 1024)
    
    print(f"  ✓ {train_file.name} ({train_size_mb:.2f} MB)")
    print(f"  ✓ {val_file.name} ({val_size_mb:.2f} MB)")
    print(f"  ✓ {scaler_file.name}")
    
    print(f"\n[SUCCESS] Fold {fold_id} ({fold_name}) complete!")


def main():
    """
    Main execution function.
    
    Creates 4 rolling origin folds from Super Matrix:
        - Fold 1: Spring validation
        - Fold 2: Summer validation
        - Fold 3: Fall validation
        - Fold 4: Winter validation
    
    Each fold has train/val split + scaler stats (12 files total).
    """
    # Paths
    data_dir = REPO_ROOT / "data" / "processed" / "pretraining" / "germany" / "global"
    output_dir = data_dir  # Save in same directory
    
    # Load Super Matrix
    print("\n" + "="*80)
    print("ROLLING ORIGIN CROSS-VALIDATION SPLITS (Version 3)")
    print("="*80)
    
    df = load_supermatrix(data_dir)
    
    # Process each fold
    for fold_config in FOLDS:
        process_fold(df, fold_config, output_dir)
    
    # Summary
    print("\n" + "="*80)
    print("ALL FOLDS COMPLETE")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print(f"\nCreated files:")
    for fold_id in range(1, 6):
        fold_type = "CV" if fold_id <= 4 else "TEST"
        print(f"  Fold {fold_id} ({fold_type}): fold_{fold_id}_train.parquet, fold_{fold_id}_val.parquet, fold_{fold_id}_scaler.json")
    
    print("\nFold 1-4: Cross-validation (test on each season)")
    print("Fold 5: Held-out test set (final evaluation)")
    print("\n✅ Done! Ready for training with train_global_lstm_v3.py")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
