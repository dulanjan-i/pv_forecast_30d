from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, type=str)
    ap.add_argument("--out", dest="out", required=True, type=str)
    ap.add_argument("--lag", type=int, default=96)
    ap.add_argument("--time-col", type=str, default="timestamp_utc")
    ap.add_argument("--group-col", type=str, default="plant_id")
    ap.add_argument("--pca-prefix", type=str, default="lstm_enc_pca_")
    ap.add_argument("--drop-na", action="store_true", help="Drop rows where any lag col is NaN")
    args = ap.parse_args()

    inp = Path(args.inp)
    out = Path(args.out)
    lag = int(args.lag)

    df = pd.read_parquet(inp)

    if args.time_col not in df.columns:
        raise RuntimeError(f"Missing time col '{args.time_col}' in {inp}. cols={len(df.columns)}")

    pca_cols = [c for c in df.columns if c.startswith(args.pca_prefix) and ("_lag" not in c)]
    if not pca_cols:
        raise RuntimeError(
            f"No {args.pca_prefix}### columns found in {inp}. Found {len(df.columns)} cols total."
        )

    # Make sure time is sorted within group
    df[args.time_col] = pd.to_datetime(df[args.time_col], utc=True)

    if args.group_col in df.columns:
        df = df.sort_values([args.group_col, args.time_col]).copy()
        g = df.groupby(args.group_col, sort=False)
        for c in pca_cols:
            df[f"{c}_lag{lag}"] = g[c].shift(lag)
    else:
        # Single-series fallback
        df = df.sort_values([args.time_col]).copy()
        for c in pca_cols:
            df[f"{c}_lag{lag}"] = df[c].shift(lag)

    lag_cols = [f"{c}_lag{lag}" for c in pca_cols]
    if args.drop_na:
        before = len(df)
        df = df.dropna(subset=lag_cols)
        print(f"[INFO] drop-na: {before} -> {len(df)}")

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    # quick sanity
    nn = float(df[lag_cols[0]].notna().mean()) if len(df) else 0.0
    print(f"[OK] wrote {out}")
    print(f"[INFO] pca cols={len(pca_cols)} lag cols={len(lag_cols)} example={pca_cols[:3]}")
    print(f"[INFO] coverage {lag_cols[0]}: {nn:.4f}")


if __name__ == "__main__":
    main()
