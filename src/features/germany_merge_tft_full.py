"""
src/features/germany_merge_tft_full.py

Stage 3.9: Merge TFT base + raw weather + PVLib outputs into final TFT tables.

Inputs (regional):
- tft_inputs/regional_{split}_tft_base.parquet
- weather_tft/regional_{split}_weather_tft.parquet
- pvlib_tft/regional_{split}_pvlib_tft.parquet

Output:
- tft_inputs/regional_{split}_tft_full.parquet

Join keys:
- plant_id, timestamp_utc

Rules:
- Keep target/power + LSTM encodings from BASE.
- Add weather columns from WEATHER (prefer WEATHER if duplicates exist).
- Add pvlib columns from PVLIB (prefer PVLIB if duplicates exist).
- Hard fail if the join is not 1:1 on keys.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import pandas as pd

KEYS = ["plant_id", "timestamp_utc"]


def _assert_no_dups(df: pd.DataFrame, name: str) -> None:
    d = df.duplicated(KEYS).sum()
    if d:
        raise ValueError(
            f"{name}: duplicated ({', '.join(KEYS)}) combinations={int(d)}"
        )


def _merge_one(split: str, base_p: Path, weather_p: Path, pvlib_p: Path, out_p: Path) -> None:
    base = pd.read_parquet(base_p)
    weather = pd.read_parquet(weather_p)
    pvlib = pd.read_parquet(pvlib_p)

    for name, df in [("base", base), ("weather", weather), ("pvlib", pvlib)]:
        missing = [c for c in KEYS if c not in df.columns]
        if missing:
            raise ValueError(f"{name}: missing key cols {missing}")
        _assert_no_dups(df, name)

    # Avoid duplicated non-key columns by dropping overlaps before merge.
    base_cols = set(base.columns) - set(KEYS)
    weather_cols = set(weather.columns) - set(KEYS)
    pvlib_cols = set(pvlib.columns) - set(KEYS)

    # If base already contains some weather-like cols, prefer WEATHER table.
    overlap_bw = sorted(base_cols & weather_cols)
    if overlap_bw:
        base = base.drop(columns=overlap_bw)

    # If base already contains some pvlib cols, prefer PVLIB table.
    overlap_bp = sorted(base_cols & pvlib_cols)
    if overlap_bp:
        base = base.drop(columns=overlap_bp)

    # If weather and pvlib overlap (rare), prefer PVLIB.
    overlap_wp = sorted(weather_cols & pvlib_cols)
    if overlap_wp:
        weather = weather.drop(columns=overlap_wp)

    merged = base.merge(weather, on=KEYS, how="inner").merge(pvlib, on=KEYS, how="inner")

    # Strict 1:1 key coverage
    if len(merged) != len(base):
        raise ValueError(
            f"{split}: merge row mismatch. base={len(base):,} merged={len(merged):,}. "
            "This indicates missing keys in weather/pvlib."
        )

    _assert_no_dups(merged, "merged")

    out_p.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_p, index=False)
    print(f"[SUCCESS] {split}: wrote {out_p} rows={len(merged):,} cols={merged.shape[1]}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo_root", type=str, default=None)
    p.add_argument("--base_dir", type=str, required=True)
    p.add_argument("--weather_dir", type=str, required=True)
    p.add_argument("--pvlib_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    base_dir = Path(args.base_dir)
    weather_dir = Path(args.weather_dir)
    pvlib_dir = Path(args.pvlib_dir)
    out_dir = Path(args.out_dir)

    for split in ["train", "val"]:
        base_p = base_dir / f"regional_{split}_tft_base.parquet"
        weather_p = weather_dir / f"regional_{split}_weather_tft.parquet"
        pvlib_p = pvlib_dir / f"regional_{split}_pvlib_tft.parquet"
        out_p = out_dir / f"regional_{split}_tft_full.parquet"

        for pth in [base_p, weather_p, pvlib_p]:
            if not pth.exists():
                raise FileNotFoundError(f"Missing input: {pth}")

        _merge_one(split, base_p, weather_p, pvlib_p, out_p)


if __name__ == "__main__":
    main()