"""
src/features/germany_merge_lstm_encodings_for_tft.py

Stage 3.6: Merge regional base table (LSTM-ready inputs) with LSTM encoder outputs
(encodings) to create the TFT base training tables.

What this does
- Loads:
    1) base_parquet: regional_train.parquet or regional_val.parquet
       Contains power_norm + weather features + plant_id (+ one-hot columns).
       Weather columns are typically z-scored at this stage.
    2) enc_parquet: regional_*_lstm_encodings.parquet
       Contains timestamp_utc, plant_id, power_norm, and lstm_enc_* columns.

- Performs an inner join on (timestamp_utc, plant_id).
  This is intentional because:
  - The encodings table is indexed by valid windows only.
  - The first window_size rows per plant and irregular gaps get dropped in the
    windowing stage, so base has extra rows that should not be used.

- Validates:
  - No duplicate keys
  - No NaNs in output
  - power_norm matches between base and encodings on joined keys

Optional
- If --scaler_json is provided, selected columns are inverse-zscored and written
  as additional *_raw columns. This helps later PVLib and interpretability.

Outputs
- A single parquet suitable as the “TFT base input table” for later enrichment:
  weather + plant identifiers + LSTM encodings (+ optional raw weather columns).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


KEY_TIME = "timestamp_utc"
KEY_PLANT = "plant_id"
TARGET = "power_norm"


def _load_scaler_stats(path: Path) -> Dict[str, Dict[str, float]]:
    """
    Load scaler stats from various JSON layouts.

    Supported patterns (observed in your pipeline):
    - {"normalized_columns": [...], "stats": {...}}
    - nested objects where leaf nodes look like {"mean": x, "std": y}
    - lists/tuples like [mean, std] or {"mean": [...], "std": [...]}

    Returns
    -------
    dict: col -> {"mean": float, "std": float}
    """
    raw = json.loads(path.read_text())

    # If it has the expected top-level keys
    if isinstance(raw, dict) and "stats" in raw:
        raw = raw["stats"]

    out: Dict[str, Dict[str, float]] = {}

    def visit(obj) -> None:
        if isinstance(obj, dict):
            # Leaf node: {"mean": x, "std": y}
            if "mean" in obj and "std" in obj and isinstance(obj["mean"], (int, float)) and isinstance(obj["std"], (int, float)):
                # This leaf has no column name context, so ignore here.
                return
            for k, v in obj.items():
                # Column leaf: {"mean": x, "std": y}
                if isinstance(v, dict) and "mean" in v and "std" in v:
                    m, s = v["mean"], v["std"]
                    if isinstance(m, (int, float)) and isinstance(s, (int, float)):
                        out[str(k)] = {"mean": float(m), "std": float(s)}
                        continue
                # Column leaf: [mean, std]
                if isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
                    out[str(k)] = {"mean": float(v[0]), "std": float(v[1])}
                    continue
                visit(v)
        elif isinstance(obj, list):
            for v in obj:
                visit(v)

    visit(raw)
    if not out:
        raise ValueError(f"Unknown scaler json format: {path}")

    return out


def inverse_zscore(z: np.ndarray, mean: float, std: float) -> np.ndarray:
    if std == 0:
        return z * 0.0 + mean
    return z * std + mean


def read_parquet(p: Path) -> pd.DataFrame:
    if not p.exists():
        raise FileNotFoundError(str(p))
    df = pd.read_parquet(p)
    # Normalize timestamp dtype
    df[KEY_TIME] = pd.to_datetime(df[KEY_TIME], utc=True)
    return df


def assert_no_dup_keys(df: pd.DataFrame, name: str) -> None:
    dup = df.duplicated([KEY_TIME, KEY_PLANT]).sum()
    if dup:
        raise ValueError(f"{name}: found {dup} duplicate (timestamp_utc, plant_id) keys")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_parquet", type=str, required=True)
    ap.add_argument("--enc_parquet", type=str, required=True)
    ap.add_argument("--output_parquet", type=str, required=True)
    ap.add_argument("--scaler_json", type=str, default=None)
    ap.add_argument(
        "--raw_cols",
        type=str,
        default="global_tilted_irradiance_instant,direct_normal_irradiance_instant,shortwave_radiation_instant",
        help="Comma-separated columns to inverse-scale into *_raw (only used if scaler_json provided).",
    )
    args = ap.parse_args()

    base_p = Path(args.base_parquet)
    enc_p = Path(args.enc_parquet)
    out_p = Path(args.output_parquet)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    base = read_parquet(base_p)
    enc = read_parquet(enc_p)

    for df, nm in [(base, "base"), (enc, "enc")]:
        for col in [KEY_TIME, KEY_PLANT, TARGET]:
            if col not in df.columns:
                raise ValueError(f"{nm}: missing required column '{col}'")

    assert_no_dup_keys(base, "base")
    assert_no_dup_keys(enc, "enc")

    # Rename power_norm in enc to avoid duplicate column name on merge
    enc = enc.rename(columns={TARGET: f"{TARGET}__enc"})

    merged = base.merge(enc, on=[KEY_TIME, KEY_PLANT], how="inner")

    # Validate power_norm consistency
    if f"{TARGET}__enc" in merged.columns:
        diff = (merged[TARGET].astype(float) - merged[f"{TARGET}__enc"].astype(float)).abs()
        max_diff = float(diff.max()) if len(diff) else 0.0
        if max_diff > 1e-4:
            raise ValueError(f"power_norm mismatch too large after merge, max_abs_diff={max_diff}")
        merged = merged.drop(columns=[f"{TARGET}__enc"])

    # Optional inverse scaling into *_raw columns
    if args.scaler_json:
        stats = _load_scaler_stats(Path(args.scaler_json))
        raw_cols = [c.strip() for c in args.raw_cols.split(",") if c.strip()]
        for c in raw_cols:
            if c in merged.columns and c in stats:
                mu = stats[c]["mean"]
                sd = stats[c]["std"]
                merged[f"{c}_raw"] = inverse_zscore(merged[c].astype(float).to_numpy(), mu, sd)
            else:
                print(f"[WARN] Cannot inverse-scale '{c}': present_in_df={c in merged.columns}, present_in_scaler={c in stats}")

    # Final sanity checks
    assert_no_dup_keys(merged, "merged")
    nan_count = int(merged.isna().sum().sum())
    if nan_count:
        raise ValueError(f"merged: contains NaNs total={nan_count}")

    merged.to_parquet(out_p, index=False)
    print(f"[SUCCESS] Wrote merged TFT base: {out_p} rows={len(merged):,} cols={merged.shape[1]}")


if __name__ == "__main__":
    main()
