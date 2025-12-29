# src/data/make_15min_singleplant_parquets.py
"""
Create plant-level 15-min train/val/test parquets for fine-tuning.

Selected target plant:
- plant_id: 3
- sample count (global source): 31982
- capacity: 7358.94 kW
- metadata quality: clean tilt and azimuth (not averaged / not reconstructed)

Why this script exists:
- On calc we already created plant-level splits, but on HPC we only SCP-copied the TFT input parquets.
- We recreate the plant-level splits on HPC for reproducibility and to keep HPC runs self-contained.
- We do NOT modify the source parquets. We write new outputs into a plant-level directory.

Split strategy:
- Filter to one plant_id
- Sort by timestamp_utc
- Time-based split (chronological):
  - train: first 70%
  - val: next 15%
  - test: last 15%

Expected columns:
- timestamp_utc (required)
- plant_id (required)
- power_norm (required)
Other features can be whatever exists in your TFT input parquet.

Usage:
  python3 -m src.data.make_15min_singleplant_parquets \
    --src_parquet data/processed/pretraining/germany/global/tft_inputs_pca32_ablations/train_tft_pvlib.parquet \
    --plant_id plant_03 \
    --out_dir data/processed/plant_level/plant_3/15min_pca32 \
    --train_frac 0.70 --val_frac 0.15 --test_frac 0.15
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


REQ_COLS = ["timestamp_utc", "plant_id", "power_norm"]


def load_and_filter(src_parquet: Path, plant_id: int) -> pd.DataFrame:
    df = pd.read_parquet(src_parquet)

    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in {src_parquet}: {missing}")

    df["plant_id"] = df["plant_id"].astype(str).str.strip()
    df = df[df["plant_id"] == str(plant_id).strip()].copy()


    # Ensure timestamp is parsed and sortable
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp_utc"]).copy()

    df = df.sort_values(["timestamp_utc"]).reset_index(drop=True)
    return df


def time_split(
    df: pd.DataFrame, train_frac: float, val_frac: float, test_frac: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    if n < 1000:
        raise ValueError(f"Too few rows after filtering: n={n}")

    if not (0 < train_frac < 1 and 0 < val_frac < 1 and 0 < test_frac < 1):
        raise ValueError("Fractions must be within (0,1)")

    s = train_frac + val_frac + test_frac
    if abs(s - 1.0) > 1e-6:
        raise ValueError(f"Fractions must sum to 1.0, got {s}")

    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    n_test = n - n_train - n_val

    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError(f"Invalid split sizes: train={n_train}, val={n_val}, test={n_test}")

    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train : n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val :].copy()

    return train_df, val_df, test_df


def summarize(df: pd.DataFrame) -> Dict:
    return {
        "rows": int(len(df)),
        "start": str(df["timestamp_utc"].min()),
        "end": str(df["timestamp_utc"].max()),
        "cols": int(df.shape[1]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_parquet", required=True)
    ap.add_argument("--plant_id", type=str, required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--train_frac", type=float, default=0.70)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--test_frac", type=float, default=0.15)
    args = ap.parse_args()

    src_parquet = Path(args.src_parquet)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_filter(src_parquet, args.plant_id)

    train_df, val_df, test_df = time_split(df, args.train_frac, args.val_frac, args.test_frac)

    train_path = out_dir / "train.parquet"
    val_path = out_dir / "val.parquet"
    test_path = out_dir / "test.parquet"
    manifest_path = out_dir / "manifest.json"

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)

    manifest = {
        "plant_id": args.plant_id,
        "capacity_kw": 7358.94,
        "global_sample_count_reference": 31982,
        "source_parquet": str(src_parquet),
        "splits": {
            "train_frac": args.train_frac,
            "val_frac": args.val_frac,
            "test_frac": args.test_frac,
        },
        "train": summarize(train_df),
        "val": summarize(val_df),
        "test": summarize(test_df),
        "note": "Plant 3 chosen due to highest samples and clean tilt/azimuth metadata.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print("[DONE] Wrote:")
    print(" ", train_path)
    print(" ", val_path)
    print(" ", test_path)
    print(" ", manifest_path)
    print("[INFO] Counts:", len(train_df), len(val_df), len(test_df))


if __name__ == "__main__":
    main()
