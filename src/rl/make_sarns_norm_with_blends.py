# src/rl/make_sarns_norm_with_blends.py
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Input SARNS parquet (may have many rows per forecast_start)")
    ap.add_argument("--out", dest="out_path", required=True, help="Output parquet (one row per forecast_start)")
    ap.add_argument("--time-col", default="forecast_start")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(in_path)

    tcol = args.time_col
    if tcol not in df.columns:
        raise ValueError(f"Missing {tcol}. Have: {list(df.columns)[:40]}")

    # Keep deterministic ordering
    df = df.sort_values([tcol, "action"] if "action" in df.columns else [tcol]).reset_index(drop=True)

    # One row per forecast_start. States are identical across actions in MINENV,
    # so taking the last row is fine and stable.
    daily = df.groupby(tcol, as_index=False).tail(1).reset_index(drop=True)

    daily.to_parquet(out_path, index=False)
    print(f"[OK] wrote: {out_path} rows={len(daily)} cols={len(daily.columns)}")


if __name__ == "__main__":
    main()
