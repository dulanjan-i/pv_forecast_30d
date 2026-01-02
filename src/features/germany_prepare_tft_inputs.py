"""
src/features/germany_prepare_tft_inputs.py

Stage 3.7: Prepare TFT inputs from Stage 3.6 base parquets.

What this does
1) Adds reconstructed RAW columns for every variable that was z-scored during preprocessing,
   using the fold/regional scaler json. This is critical for:
   - PVLib feature building (needs physical units)
   - "day mask" or irradiance thresholding in evaluations (must use RAW, not z-scored)
2) Adds calendar / time features from timestamp_utc (safe, no leakage):
   - hour_sin/cos, doy_sin/cos, month, weekend flag
3) (Optional) Merges static plant metadata (tilt, azimuth, capacity, lat/lon...) if provided.

Inputs
- --input_parquet: Stage 3.6 parquet (e.g., regional_train_tft_base.parquet)
- --scaler_json: scaler json that contains { "stats": {col: {"mean","std"}} ... } (regional_scaler.json)
- --plant_meta_csv (optional): CSV with at least 'plant_id' and any static columns you want

Output
- --output_parquet: enriched parquet to feed PVLib builder and TFT
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

from src.data.schema import TIME_COL, PLANT_ID_COL


def _find_stats_node(obj: Any, max_depth: int = 6) -> Optional[Dict[str, Any]]:
    """Find a dict that looks like scaler stats: {col: {"mean":..., "std":...}, ...}."""
    if isinstance(obj, dict):
        # Direct match
        if obj and all(
            isinstance(v, dict) and ("mean" in v) and ("std" in v)
            for v in obj.values()
        ):
            return obj

        if max_depth <= 0:
            return None

        for v in obj.values():
            found = _find_stats_node(v, max_depth=max_depth - 1)
            if found is not None:
                return found
    elif isinstance(obj, list) and max_depth > 0:
        for v in obj:
            found = _find_stats_node(v, max_depth=max_depth - 1)
            if found is not None:
                return found
    return None


def load_scaler_stats(path: Path) -> Dict[str, Dict[str, float]]:
    """Load scaler stats as {col: {"mean": float, "std": float}} from a flexible json structure."""
    data = json.loads(path.read_text())
    stats_node = _find_stats_node(data, max_depth=8)
    if stats_node is None:
        raise ValueError(f"Unknown scaler json format: {path}")

    out: Dict[str, Dict[str, float]] = {}
    for col, v in stats_node.items():
        out[col] = {"mean": float(v["mean"]), "std": float(v["std"])}
    return out


def inverse_zscore(z: np.ndarray, mean: float, std: float) -> np.ndarray:
    return z * std + mean


def add_raw_recon_columns(df: pd.DataFrame, stats: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    For each column present in stats AND in df, create <col>_raw = z*std + mean.

    This assumes df stores the z-scored values for those columns.
    We do not touch columns not present in stats (e.g., power_norm, weather_code often unscaled).
    """
    d = df.copy()
    created = 0
    for col, ms in stats.items():
        if col not in d.columns:
            continue
        raw_col = f"{col}_raw"
        if raw_col in d.columns:
            continue
        z = d[col].astype(float).to_numpy()
        d[raw_col] = inverse_zscore(z, ms["mean"], ms["std"]).astype(np.float32)
        created += 1

    print(f"[INFO] Added {created} raw recon columns.")
    return d


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add safe calendar features derived from TIME_COL."""
    d = df.copy()
    ts = pd.to_datetime(d[TIME_COL], utc=True)

    hour = ts.dt.hour + ts.dt.minute / 60.0
    doy = ts.dt.dayofyear.astype(float)
    month = ts.dt.month.astype(int)
    weekend = (ts.dt.dayofweek >= 5).astype(int)

    d["hour_sin"] = np.sin(2 * np.pi * hour / 24.0).astype(np.float32)
    d["hour_cos"] = np.cos(2 * np.pi * hour / 24.0).astype(np.float32)

    d["doy_sin"] = np.sin(2 * np.pi * doy / 365.25).astype(np.float32)
    d["doy_cos"] = np.cos(2 * np.pi * doy / 365.25).astype(np.float32)

    d["month"] = month
    d["is_weekend"] = weekend

    return d


def maybe_merge_plant_meta(df: pd.DataFrame, plant_meta_csv: Optional[Path]) -> pd.DataFrame:
    """Left-join plant static metadata by plant_id if a CSV path is provided."""
    if plant_meta_csv is None:
        return df

    meta = pd.read_csv(plant_meta_csv)
    if PLANT_ID_COL not in meta.columns:
        raise ValueError(f"plant_meta_csv missing '{PLANT_ID_COL}': {plant_meta_csv}")

    # Drop duplicates to avoid exploding rows
    meta = meta.drop_duplicates(subset=[PLANT_ID_COL]).copy()

    out = df.merge(meta, on=PLANT_ID_COL, how="left")
    missing = int(out[meta.columns.difference([PLANT_ID_COL])].isna().all(axis=1).sum())
    if missing:
        print(f"[WARN] {missing} rows have no plant metadata after merge (check plant_id mapping).")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input_parquet", type=str, required=True)
    p.add_argument("--scaler_json", type=str, required=True)
    p.add_argument("--output_parquet", type=str, required=True)
    p.add_argument("--plant_meta_csv", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    inp = Path(args.input_parquet)
    outp = Path(args.output_parquet)
    sc = Path(args.scaler_json)
    meta = Path(args.plant_meta_csv) if args.plant_meta_csv else None

    df = pd.read_parquet(inp)
    stats = load_scaler_stats(sc)

    df = add_raw_recon_columns(df, stats)
    df = add_time_features(df)
    df = maybe_merge_plant_meta(df, meta)

    outp.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(outp, index=False)
    print(f"[SUCCESS] Wrote: {outp} rows={len(df):,} cols={df.shape[1]:,}")


if __name__ == "__main__":
    main()
