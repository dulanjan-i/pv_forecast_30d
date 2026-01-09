from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-weather", required=True, type=str)
    ap.add_argument("--out-weather", required=True, type=str)
    ap.add_argument("--lag", type=int, default=96, help="96 = 24h at 15-min resolution")
    args = ap.parse_args()

    in_p = Path(args.in_weather)
    out_p = Path(args.out_weather)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(in_p)

    if "timestamp_utc" not in df.columns:
        raise KeyError("weather parquet missing timestamp_utc")

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    # Match columns like lstm_enc_pca_000 ... lstm_enc_pca_031
    pat = re.compile(r"^lstm_enc_pca_\d{3}$")
    base_cols = [c for c in df.columns if pat.match(c)]

    if not base_cols:
        raise RuntimeError(
            "No lstm_enc_pca_### columns found in weather parquet. "
            "If your PCA encodings use a different naming scheme, adjust the regex."
        )

    lag = int(args.lag)
    for c in base_cols:
        lagged = f"{c}_lag{lag}"
        if lagged not in df.columns:
            df[lagged] = df[c].shift(lag)

    df.to_parquet(out_p, index=False)
    print(f"[OK] wrote {out_p}")
    print(f"     base_cols={len(base_cols)} lag={lag}")


if __name__ == "__main__":
    main()
