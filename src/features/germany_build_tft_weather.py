"""
src/features/germany_build_tft_weather.py

Stage 3.7: Build TFT weather tables aligned to TFT base inputs.

Problem this solves
- The interim OpenMeteo weather files do NOT contain `plant_id` as a column.
- They also do NOT contain `poa_irradiance`. POA is derived later (PVLib).
- The TFT base tables already define the exact row universe (timestamp_utc, plant_id) that TFT will see.
  So the correct approach is:
    1) load the TFT base split (train or val)
    2) for each plant, load its weather parquet
    3) add plant_id derived from filename
    4) inner-join on (plant_id, timestamp_utc) to produce aligned weather rows

Key decisions
- Irradiance proxy for this stage: `global_tilted_irradiance_instant` (GTI).
- We never require `poa_irradiance` here.
- Output is two parquets, one for train split and one for val split.

Outputs
- <out_dir>/regional_train_weather_tft.parquet
- <out_dir>/regional_val_weather_tft.parquet
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from src.data.schema import TIME_COL, PLANT_ID_COL

# Weather columns expected in interim weather_15min files.
# Keep this list aligned with what you actually have from OpenMeteo processing.
# IMPORTANT: no poa_irradiance here.
WEATHER_COLS: List[str] = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "weather_code",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "shortwave_radiation_instant",
    "direct_radiation_instant",
    "diffuse_radiation_instant",
    "direct_normal_irradiance_instant",
    "global_tilted_irradiance_instant",
    "surface_pressure",
]

PLANT_FILE_RE = re.compile(r"^(plant_\d+)_weather_15min\.parquet$")


def _load_base(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing TFT base parquet: {path}")
    df = pd.read_parquet(path)
    if TIME_COL not in df.columns or PLANT_ID_COL not in df.columns:
        raise ValueError(f"{path.name}: base parquet must contain {TIME_COL} and {PLANT_ID_COL}")
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], utc=True)
    return df


def _infer_plant_id_from_filename(path: Path) -> str:
    m = PLANT_FILE_RE.match(path.name)
    if not m:
        raise ValueError(
            f"Cannot infer plant_id from filename: {path.name}. "
            f"Expected pattern: plant_XX_weather_15min.parquet"
        )
    return m.group(1)


def _load_weather_one(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing weather parquet: {path}")
    df = pd.read_parquet(path)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], utc=True)

    # Add plant_id derived from filename if missing
    if PLANT_ID_COL not in df.columns:
        pid = _infer_plant_id_from_filename(path)
        df[PLANT_ID_COL] = pid

    # Validate expected cols
    missing = [c for c in WEATHER_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{df[PLANT_ID_COL].iloc[0]}: weather parquet missing columns: {missing} in {path}")

    # Keep only the required set plus keys
    keep = [TIME_COL, PLANT_ID_COL] + WEATHER_COLS
    df = df[keep].copy()

    # Basic sanity: no duplicated keys
    dups = df.duplicated([PLANT_ID_COL, TIME_COL]).sum()
    if dups:
        raise ValueError(f"{path.name}: duplicated (plant_id, timestamp) rows: {int(dups)}")

    return df


def _build_weather_for_split(base: pd.DataFrame, weather_dir: Path) -> pd.DataFrame:
    plants = sorted(base[PLANT_ID_COL].unique().tolist())
    print(f"Plants: {plants}")
    print(f"Weather dir: {weather_dir}")

    base_keys = base[[PLANT_ID_COL, TIME_COL]].copy()

    parts: List[pd.DataFrame] = []
    for pid in plants:
        w_path = weather_dir / f"{pid}_weather_15min.parquet"
        w = _load_weather_one(w_path)

        # Filter to only keys present in base for that plant (fast and correct)
        k = base_keys[base_keys[PLANT_ID_COL] == pid]
        merged = k.merge(w, on=[PLANT_ID_COL, TIME_COL], how="inner", validate="one_to_one")

        if len(merged) != len(k):
            # This is important to see. It means you have missing weather timestamps for that base universe.
            miss = len(k) - len(merged)
            print(f"[WARN] {pid}: base rows={len(k)} weather matched={len(merged)} missing={miss}")

        parts.append(merged)

    out = pd.concat(parts, axis=0, ignore_index=True)
    out = out.sort_values([PLANT_ID_COL, TIME_COL]).reset_index(drop=True)

    # Final sanity
    if out.isna().any().any():
        n = int(out.isna().sum().sum())
        raise ValueError(f"Weather output contains NaNs (count={n}). Fix preprocessing before TFT.")

    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train_base", type=str, required=True)
    p.add_argument("--val_base", type=str, required=True)
    p.add_argument("--weather_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    train_base = Path(args.train_base)
    val_base = Path(args.val_base)
    weather_dir = Path(args.weather_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tr = _load_base(train_base)
    va = _load_base(val_base)

    print("=" * 79)
    print(f"Train base rows: {len(tr)} Val base rows: {len(va)}")
    print("=" * 79)
    print(f"Weather cols: {len(WEATHER_COLS)}")

    train_weather = _build_weather_for_split(tr, weather_dir)
    val_weather = _build_weather_for_split(va, weather_dir)

    out_tr = out_dir / "regional_train_weather_tft.parquet"
    out_va = out_dir / "regional_val_weather_tft.parquet"

    train_weather.to_parquet(out_tr, index=False)
    val_weather.to_parquet(out_va, index=False)

    print(f"[SUCCESS] Wrote train weather: {out_tr} rows={len(train_weather)} cols={train_weather.shape[1]}")
    print(f"[SUCCESS] Wrote val weather:   {out_va} rows={len(val_weather)} cols={val_weather.shape[1]}")


if __name__ == "__main__":
    main()
