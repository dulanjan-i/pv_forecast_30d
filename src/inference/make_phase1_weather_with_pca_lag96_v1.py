from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-weather", required=True, help="2024 weather_with_pvlib_15min.parquet (no PCA lag)")
    ap.add_argument("--pca-weather", required=True, help="weather_with_pvlib_15min.parquet that contains lstm_enc_pca_* and *_lag96")
    ap.add_argument("--out-dir", required=True, help="New phase dir to write into")
    ap.add_argument("--time-col", default="timestamp_utc")
    ap.add_argument("--id-col", default="plant_id")
    args = ap.parse_args()

    base_weather = Path(args.base_weather)
    pca_weather = Path(args.pca_weather)
    out_dir = Path(args.out_dir)

    base = pd.read_parquet(base_weather)
    pca = pd.read_parquet(pca_weather)

    # normalize dtypes
    base[args.time_col] = pd.to_datetime(base[args.time_col], utc=True)
    pca[args.time_col] = pd.to_datetime(pca[args.time_col], utc=True)

    if args.id_col not in base.columns:
        raise RuntimeError(f"base missing {args.id_col}")
    if args.id_col not in pca.columns:
        raise RuntimeError(f"pca missing {args.id_col}")

    pca_cols = [c for c in pca.columns if c.startswith("lstm_enc_pca_") and ("_lag" not in c)]
    lag_cols = [c for c in pca.columns if c.startswith("lstm_enc_pca_") and ("_lag96" in c)]
    keep_cols = [args.time_col, args.id_col] + sorted(pca_cols) + sorted(lag_cols)

    if len(pca_cols) == 0 or len(lag_cols) == 0:
        raise RuntimeError(f"pca file does not contain expected cols. pca_cols={len(pca_cols)} lag_cols={len(lag_cols)}")

    pca_small = pca[keep_cols].copy()

    key = [args.time_col, args.id_col]
    merged = base.merge(pca_small, on=key, how="left")

    # coverage sanity
    cov = float(merged["lstm_enc_pca_000_lag96"].notna().mean()) if "lstm_enc_pca_000_lag96" in merged.columns else 0.0
    print(f"[INFO] merged rows={len(merged)}  coverage lstm_enc_pca_000_lag96={cov:.4f}")
    print(f"[INFO] base min/max: {merged[args.time_col].min()}  {merged[args.time_col].max()}")

    # write both root and processed, to satisfy whichever path your pipeline expects
    (out_dir / "processed").mkdir(parents=True, exist_ok=True)
    out1 = out_dir / "weather_with_pvlib_15min.parquet"
    out2 = out_dir / "processed" / "weather_with_pvlib_15min.parquet"
    merged.to_parquet(out1, index=False)
    merged.to_parquet(out2, index=False)
    print(f"[OK] wrote:\n  {out1}\n  {out2}")


if __name__ == "__main__":
    main()
