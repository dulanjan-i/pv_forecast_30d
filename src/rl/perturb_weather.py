#!/usr/bin/env python3
"""
Create perturbed 15-min weather parquet variants for counterfactual RL evaluation.

Saves one parquet per (date, magnitude) variant and writes a JSON manifest
listing all generated files and the perturbation details.

Usage examples:
python src/rl/perturb_weather.py \
  --weather_in data/processed/plant_level/plant_03/weather_15min.parquet \
  --out_dir experiments/rl/counterfactuals/plant_03/weather_variants \
  --dates 2024-06-15,2024-12-10 \
  --magnitudes 0.8,0.6,0.4 \
  --plant_id plant_03

Or sample dates by season:
python src/rl/perturb_weather.py --weather_in ... --out_dir ... --sample_per_season 2 --magnitudes 0.8,0.6

The script will NOT modify ground truth files.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

logger = logging.getLogger("perturb_weather")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weather_in", required=True, help="Input 15-min weather parquet")
    ap.add_argument("--out_dir", required=True, help="Directory to write perturbed parquets + manifest")
    ap.add_argument("--plant_id", required=True, help="Plant id (used for filtering)")
    ap.add_argument("--dates", default=None, help="Comma-separated YYYY-MM-DD dates to perturb")
    ap.add_argument("--sample_per_season", type=int, default=0, help="If >0, sample this many dates per season")
    ap.add_argument("--magnitudes", required=True, help="Comma-separated multiplicative factors to apply (e.g., 0.8,0.6)")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def month_to_season(m: int) -> str:
    if m in (12, 1, 2):
        return "winter"
    if m in (3, 4, 5):
        return "spring"
    if m in (6, 7, 8):
        return "summer"
    return "autumn"


def sample_dates_by_season(dates: pd.Series, per_season: int, seed: int):
    rng = np.random.default_rng(seed)
    seasons = {"winter": [], "spring": [], "summer": [], "autumn": []}
    for d in dates:
        s = month_to_season(d.month)
        seasons[s].append(d)

    chosen = []
    for s, vals in seasons.items():
        if not vals:
            continue
        k = min(per_season, len(vals))
        picks = list(rng.choice(vals, size=k, replace=False))
        chosen.extend(picks)

    chosen = sorted(list(set(chosen)))
    return chosen


def apply_perturbation(df: pd.DataFrame, date: pd.Timestamp, magnitude: float) -> pd.DataFrame:
    """Apply multiplicative perturbation to daylight values on given date.

    magnitude: multiply irradiance and pvlib outputs by this factor (0-1 for reduction)
    """
    out = df.copy()
    out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True, errors="coerce")

    day_start = pd.Timestamp(date.date(), tz="UTC")
    day_end = day_start + pd.Timedelta(days=1)

    mask = (out["timestamp_utc"] >= day_start) & (out["timestamp_utc"] < day_end)

    if mask.sum() == 0:
        raise ValueError(f"No rows found for date {date.date()}")

    # Columns to scale if present
    scale_cols = [c for c in ["ghi", "dni", "pvlib_ac_kw", "poa_global"] if c in out.columns]

    for c in scale_cols:
        out.loc[mask, c] = out.loc[mask, c].astype(float) * float(magnitude)

    # Adjust cloud_cover inversely (more clouds -> higher cover)
    if "cloud_cover" in out.columns:
        inv = 1.0 / max(magnitude, 1e-6)
        out.loc[mask, "cloud_cover"] = np.minimum(100.0, out.loc[mask, "cloud_cover"].astype(float) * inv)

    # Optionally nudge temperature slightly (small decrease for cloudier)
    if "temperature" in out.columns:
        out.loc[mask, "temperature"] = out.loc[mask, "temperature"].astype(float) - (1.0 - magnitude) * 5.0

    # Defensive clipping
    if "pvlib_ac_kw" in out.columns:
        out["pvlib_ac_kw"] = out["pvlib_ac_kw"].clip(lower=0.0)

    return out


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading weather parquet: %s", args.weather_in)
    df = pd.read_parquet(args.weather_in)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")

    # Filter by plant_id if present
    if "plant_id" in df.columns:
        df = df[df["plant_id"].astype(str) == args.plant_id].copy()

    # Build list of candidate dates
    unique_dates = pd.Series(sorted(df["timestamp_utc"].dt.normalize().unique()))

    dates_to_use = []
    if args.dates:
        for d in args.dates.split(","):
            dates_to_use.append(pd.to_datetime(d).tz_localize("UTC"))
    elif args.sample_per_season > 0:
        dates_to_use = sample_dates_by_season(unique_dates, args.sample_per_season, args.seed)
    else:
        raise ValueError("Either --dates or --sample_per_season must be provided")

    magnitudes = [float(x) for x in args.magnitudes.split(",")]

    manifest = []

    for date in dates_to_use:
        for mag in magnitudes:
            fname = f"weather_perturb_{date.date().isoformat()}_mag{mag:.3f}.parquet"
            out_path = out_dir / fname

            logger.info("Creating variant: date=%s mag=%.3f -> %s", date.date(), mag, out_path)
            try:
                dfp = apply_perturbation(df, date, mag)
                dfp.to_parquet(out_path, index=False)

                manifest.append({
                    "date": str(date.date()),
                    "magnitude": mag,
                    "file": str(out_path),
                    "generated_at": datetime.utcnow().isoformat() + "Z",
                })
            except Exception as e:
                logger.warning("Skipping date %s mag %.3f: %s", date.date(), mag, str(e))

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as fh:
        json.dump({"source": str(args.weather_in), "plant_id": args.plant_id, "variants": manifest}, fh, indent=2)

    logger.info("Wrote manifest: %s", manifest_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
