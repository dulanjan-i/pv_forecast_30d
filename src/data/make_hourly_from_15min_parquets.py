"""
make_hourly_from_15min_parquets.py

Purpose
- Convert existing 15-min TFT input parquets into hourly parquets for long-horizon (720-step) training.
- Adds time_idx (integer) per plant_id.
- Designed for PVLib feature parquets (tft_pvlib style) but works generically.

Aggregation rules (practical, robust):
- numeric columns: mean over the hour
- plant one-hot columns: mean (stays 0/1 if consistent)
- plant_id: first
- timestamp_utc: hour floor (UTC)

Output
- out_dir/train.parquet, out_dir/val.parquet
- out_dir/manifest.json with basic stats

Notes
- This does NOT do time splitting. It preserves the provided train/val separation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


KEY_T = "timestamp_utc"
KEY_GROUP = "plant_id"
KEY_TIME_IDX = "time_idx"


def _to_utc(df: pd.DataFrame, col: str) -> pd.Series:
    s = pd.to_datetime(df[col], utc=True, errors="coerce")
    return s


def _infer_agg_map(df: pd.DataFrame) -> dict:
    agg = {}
    for c in df.columns:
        if c in (KEY_T, KEY_GROUP):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            agg[c] = "mean"
        else:
            # best-effort for non-numeric: take first
            agg[c] = "first"
    return agg


def _resample_one_plant(df_plant: pd.DataFrame, freq: str) -> pd.DataFrame:
    df_plant = df_plant.copy()
    df_plant[KEY_T] = _to_utc(df_plant, KEY_T)
    df_plant = df_plant.dropna(subset=[KEY_T])
    df_plant = df_plant.sort_values(KEY_T)

    # floor timestamps to hour and aggregate
    df_plant["ts_bucket"] = df_plant[KEY_T].dt.floor(freq)
    agg_map = _infer_agg_map(df_plant)

    g = df_plant.groupby("ts_bucket", as_index=False).agg(agg_map)
    g[KEY_T] = g["ts_bucket"]
    g.drop(columns=["ts_bucket"], inplace=True)

    # restore group id reliably
    g[KEY_GROUP] = df_plant[KEY_GROUP].iloc[0]

    # per-plant time_idx starting at 0
    t0 = g[KEY_T].min()
    g[KEY_TIME_IDX] = ((g[KEY_T] - t0) / pd.Timedelta(freq)).astype(np.int64)

    return g


def resample_hourly(df: pd.DataFrame, freq: str = "1H") -> pd.DataFrame:
    if KEY_GROUP not in df.columns:
        raise KeyError(f"Missing {KEY_GROUP} column")
    if KEY_T not in df.columns:
        raise KeyError(f"Missing {KEY_T} column")

    out = []
    for pid, dfp in df.groupby(KEY_GROUP):
        r = _resample_one_plant(dfp, freq=freq)
        out.append(r)
    out_df = pd.concat(out, ignore_index=True)
    out_df = out_df.sort_values([KEY_GROUP, KEY_T]).reset_index(drop=True)
    return out_df


def summarize(df: pd.DataFrame) -> dict:
    df = df.copy()
    df[KEY_T] = _to_utc(df, KEY_T)
    return {
        "rows": int(len(df)),
        "plants": sorted(df[KEY_GROUP].astype(str).unique().tolist()),
        "t_start": str(df[KEY_T].min()),
        "t_end": str(df[KEY_T].max()),
        "cols": int(df.shape[1]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_train", type=str, required=True)
    ap.add_argument("--src_val", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--freq", type=str, default="1H")
    args = ap.parse_args()

    src_train = Path(args.src_train)
    src_val = Path(args.src_val)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_parquet(src_train)
    val_df = pd.read_parquet(src_val)

    train_h = resample_hourly(train_df, freq=args.freq)
    val_h = resample_hourly(val_df, freq=args.freq)

    train_p = out_dir / "train.parquet"
    val_p = out_dir / "val.parquet"
    train_h.to_parquet(train_p, index=False)
    val_h.to_parquet(val_p, index=False)

    manifest = {
        "src_train": str(src_train),
        "src_val": str(src_val),
        "freq": args.freq,
        "train": summarize(train_h),
        "val": summarize(val_h),
        "notes": "Hourly aggregation for long-head TFT training. time_idx added per plant.",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print("[DONE] wrote:", train_p)
    print("[DONE] wrote:", val_p)
    print("[DONE] wrote:", out_dir / "manifest.json")


if __name__ == "__main__":
    main()
