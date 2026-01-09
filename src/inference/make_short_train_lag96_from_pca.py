from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def _find_pca_cols(cols: List[str]) -> List[str]:
    return [c for c in cols if c.startswith("lstm_enc_pca_") and not c.endswith("_lag96")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, type=str, help="input short_train parquet")
    ap.add_argument("--out", dest="out", required=True, type=str, help="output parquet with *_lag96 cols")
    ap.add_argument("--lag", type=int, default=96)
    ap.add_argument("--ts-col", type=str, default="timestamp_utc")
    ap.add_argument("--group-col", type=str, default="plant_id", help="optional grouping col if present")
    args = ap.parse_args()

    inp = Path(args.inp)
    out = Path(args.out)
    lag = int(args.lag)

    df = pd.read_parquet(inp, engine="pyarrow")
    cols = list(df.columns)

    pca_cols = _find_pca_cols(cols)
    if not pca_cols:
        raise RuntimeError(f"No lstm_enc_pca_### columns found in {inp}. Found {len(cols)} cols total.")

    ts_col = args.ts_col
    if ts_col in df.columns:
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        df = df.sort_values(ts_col)

    group_col = args.group_col
    use_group = group_col in df.columns

    # Make lagged cols
    if use_group:
        g = df.groupby(group_col, sort=False)
        for c in pca_cols:
            df[f"{c}_lag96"] = g[c].shift(lag)
    else:
        for c in pca_cols:
            df[f"{c}_lag96"] = df[c].shift(lag)

    lag_cols = [f"{c}_lag96" for c in pca_cols]

    # Drop rows where lagged cols are NaN, otherwise StandardScaler.fit will choke
    before = len(df)
    df = df.dropna(subset=lag_cols).copy()
    after = len(df)

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    print("[OK] wrote:", out)
    print("rows before:", before, "after:", after, "dropped:", before - after)
    print("pca cols:", len(pca_cols), "lag cols:", len(lag_cols))
    print("example lag cols:", lag_cols[:3])


if __name__ == "__main__":
    main()
