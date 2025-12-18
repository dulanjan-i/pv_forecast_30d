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

Author: PV Forecast Team
Date: December 2024
Version: 3.0 (Global Forecasting Model)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add repo root to path for imports
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.data.schema import (
    LSTM_INPUT_FEATURES,
    PLANT_IDS,
    PLANT_ID_COL,
    PLANT_ONEHOT_COLS,
    TIME_COL,
    TARGET_COL,
)


def load_plant_data(plant_id: str, base_dir: Path) -> pd.DataFrame:
    """
    Load pretrain_base.parquet for a single plant.
    
    Parameters
    ----------
    plant_id : str
        Plant identifier (e.g., 'plant_01')
    base_dir : Path
        Base directory containing plant folders
        
    Returns
    -------
    pd.DataFrame
        Plant data with columns [timestamp_utc, 15 LSTM features]
        
    Raises
    ------
    FileNotFoundError
        If pretrain_base.parquet doesn't exist for this plant
    """
    parquet_file = base_dir / f"{plant_id}_pretrain_base.parquet"
    
    if not parquet_file.exists():
        raise FileNotFoundError(
            f"Missing pretrain_base for {plant_id}: {parquet_file}\n"
            f"Run germany_build_pretrain_base.py first!"
        )
    
    df = pd.read_parquet(parquet_file)
    
    # Verify required columns
    required_cols = [TIME_COL] + LSTM_INPUT_FEATURES
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"{plant_id} missing columns: {missing}")
    
    # Add plant_id column
    df[PLANT_ID_COL] = plant_id
    
    print(f"[INFO] Loaded {plant_id}: {len(df):,} rows")
    
    return df


def create_supermatrix(base_dir: Path, output_dir: Path) -> Path:
    """
    Build Super Matrix by concatenating all 5 plants with one-hot plant IDs.
    
    Parameters
    ----------
    base_dir : Path
        Directory containing plant_XX subdirectories
    output_dir : Path
        Output directory for supermatrix_base.parquet
        
    Returns
    -------
    Path
        Path to saved supermatrix_base.parquet
        
    Process
    -------
    1. Load 5 plant pretrain_base files
    2. Add plant_id string column
    3. Concatenate vertically
    4. Create one-hot encoding (5 binary columns)
    5. Sort by timestamp (temporal ordering)
    6. Save to output_dir/supermatrix_base.parquet
    """
    print("\n" + "="*80)
    print("BUILDING GLOBAL SUPER MATRIX (Version 3)")
    print("="*80)
    
    # Step 1: Load all plants
    print("\n[STEP 1/5] Loading plant data...")
    plant_dfs = []
    
    for plant_id in PLANT_IDS:
        df = load_plant_data(plant_id, base_dir)
        plant_dfs.append(df)
    
    # Step 2: Concatenate
    print("\n[STEP 2/5] Concatenating all plants...")
    supermatrix = pd.concat(plant_dfs, axis=0, ignore_index=True)
    print(f"[INFO] Total rows after concatenation: {len(supermatrix):,}")
    
    # Step 2b: Drop NaNs (safety check - should be clean already from Version 02)
    n_before = len(supermatrix)
    supermatrix = supermatrix.dropna(subset=LSTM_INPUT_FEATURES + [TARGET_COL])
    n_after = len(supermatrix)
    n_dropped = n_before - n_after
    if n_dropped > 0:
        print(f"[WARNING] Dropped {n_dropped:,} rows with NaNs ({n_dropped/n_before*100:.2f}%)")
    else:
        print(f"[INFO] No NaNs found ✓")
    
    # Step 3: Sort by timestamp (critical for rolling origin splits)
    print("\n[STEP 3/5] Sorting by timestamp...")
    supermatrix = supermatrix.sort_values(TIME_COL).reset_index(drop=True)
    print(f"[INFO] Date range: {supermatrix[TIME_COL].min()} to {supermatrix[TIME_COL].max()}")
    
    # Step 4: Create one-hot encoding
    print("\n[STEP 4/5] Creating one-hot encoding for plant_id...")
    for plant_id in PLANT_IDS:
        # Binary column: 1 if this row belongs to plant_id, else 0
        supermatrix[plant_id] = (supermatrix[PLANT_ID_COL] == plant_id).astype(np.float32)
        print(f"[INFO]   {plant_id}: {supermatrix[plant_id].sum():,.0f} rows (1.0s)")
    
    # Verify one-hot encoding (each row should have exactly one 1.0)
    onehot_sum = supermatrix[PLANT_ONEHOT_COLS].sum(axis=1)
    if not (onehot_sum == 1.0).all():
        raise ValueError(
            f"One-hot encoding error! Each row should sum to 1.0, "
            f"but found: min={onehot_sum.min()}, max={onehot_sum.max()}"
        )
    print("[INFO] One-hot validation PASSED (all rows sum to 1.0)")
    
    # Step 5: Save
    print("\n[STEP 5/5] Saving Super Matrix...")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "supermatrix_base.parquet"
    
    supermatrix.to_parquet(output_file, index=False)
    
    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"[INFO] Saved: {output_file}")
    print(f"[INFO] File size: {file_size_mb:.2f} MB")
    print(f"[INFO] Final shape: {supermatrix.shape}")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUPER MATRIX SUMMARY")
    print("="*80)
    print(f"Total samples: {len(supermatrix):,}")
    print(f"Features: {len(LSTM_INPUT_FEATURES)} original + {len(PLANT_ONEHOT_COLS)} plant IDs = {len(LSTM_INPUT_FEATURES) + len(PLANT_ONEHOT_COLS)} total")
    print(f"\nPer-plant breakdown:")
    for plant_id in PLANT_IDS:
        count = (supermatrix[PLANT_ID_COL] == plant_id).sum()
        pct = (count / len(supermatrix)) * 100
        print(f"  {plant_id}: {count:>7,} rows ({pct:>5.2f}%)")
    
    print("\n[SUCCESS] Super Matrix ready for rolling origin splits!")
    print("="*80 + "\n")
    
    return output_file


def main():
    """
    Main execution function.
    
    Builds Super Matrix from 5 Germany plants and saves to:
        data/processed/pretraining/germany/global/supermatrix_base.parquet
    """
    # Paths
    base_dir = REPO_ROOT / "data" / "processed" / "pretraining" / "germany"
    output_dir = base_dir / "global"
    
    # Validate input directory exists
    if not base_dir.exists():
        raise FileNotFoundError(
            f"Base directory not found: {base_dir}\n"
            f"Run preprocessing pipeline first!"
        )
    
    # Build Super Matrix
    output_file = create_supermatrix(base_dir, output_dir)
    
    print(f"\n✅ Done! Super Matrix saved to:\n   {output_file}\n")
    print("Next step: Run germany_global_rolling_origin_split.py")


if __name__ == "__main__":
    main()
