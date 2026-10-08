"""
Make TFT ablation parquets from an existing TFT input parquet.

Modes:
- full: keep all columns
- tft_only: drop lstm_enc* and pvlib_*
- tft_lstm: drop pvlib_* only
- tft_pvlib: drop lstm_enc* only

Writes new parquets to out_dir:
- train_<mode>.parquet
- val_<mode>.parquet

Leaves originals untouched.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


KEYS = {"timestamp_utc", "plant_id", "power_norm", "time_idx"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_parquet", required=True)
    p.add_argument("--val_parquet", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--mode", required=True, choices=["full", "tft_only", "tft_lstm", "tft_pvlib"])
    return p.parse_args()


def drop_cols(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    cols = list(df.columns)

    lstm_cols = [c for c in cols if c.startswith("lstm_enc_")]  # catches lstm_enc_pca_ too
    pvlib_cols = [c for c in cols if c.startswith("pvlib_")]

    drop = set()

    if mode == "tft_only":
        drop |= set(lstm_cols)
        drop |= set(pvlib_cols)
    elif mode == "tft_lstm":
        drop |= set(pvlib_cols)
    elif mode == "tft_pvlib":
        drop |= set(lstm_cols)
    elif mode == "full":
        pass

    # Never drop required keys if present
    drop -= KEYS

    if drop:
        df = df.drop(columns=sorted(drop))

    return df


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(args.train_parquet)
    val = pd.read_parquet(args.val_parquet)

    train2 = drop_cols(train, args.mode)
    val2 = drop_cols(val, args.mode)

    train_out = out_dir / f"train_{args.mode}.parquet"
    val_out = out_dir / f"val_{args.mode}.parquet"

    train2.to_parquet(train_out, index=False)
    val2.to_parquet(val_out, index=False)

    print(f"[DONE] mode={args.mode}")
    print(f"  train_out: {train_out} cols={len(train2.columns)} rows={len(train2)}")
    print(f"  val_out:   {val_out} cols={len(val2.columns)} rows={len(val2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
