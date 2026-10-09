#!/usr/bin/env python3
"""
Check that hourly weather is aligned to UTC (DST-safe).

For every month, compares the hour of the daily GHI peak against the hour of the daily peak of a
pvlib clear-sky curve at the plant location. A correct pipeline gives offsets of about 0 in all
months. A DST bug shows up as a constant -1 h (or +1 h) offset for the summer months.

Two modes:
  --synthetic   (default) builds an Open-Meteo-like uniform-UTC series from clear-sky GHI, runs it
                through src/data/preprocess_germany_weather.py and reports the offsets.
                Needs no data and no network.
  --csv PATH    runs the same comparison on your own hourly weather CSV (columns `timestamp_utc`, or
                legacy `date`, plus `shortwave_radiation_instant`). Stays on your machine.

Usage:
  python scripts/check_weather_alignment.py
  python scripts/check_weather_alignment.py --csv data/raw/germany/plant_03/historical_weather_hourly.csv \
      --lat <lat> --lon <lon>
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import warnings
from pathlib import Path

import pandas as pd
import pvlib

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import src.data.preprocess_germany_weather as pw  # noqa: E402
from src.data.time_utils import LOCAL_TZ  # noqa: E402

WEATHER_COLS = [
    "temperature_2m", "relative_humidity_2m", "precipitation", "weather_code", "cloud_cover",
    "wind_speed_10m", "wind_direction_10m", "direct_radiation_instant", "diffuse_radiation_instant",
    "direct_normal_irradiance_instant", "global_tilted_irradiance_instant", "surface_pressure",
]


def _daily_peak_hour(s: pd.Series) -> pd.Series:
    s = s.dropna()
    idx = s.groupby(s.index.date).idxmax()
    return pd.Series(
        [pd.Timestamp(t).hour + pd.Timestamp(t).minute / 60 for t in idx.values],
        index=pd.to_datetime(list(idx.index)),
    )


def monthly_peak_offset(ghi: pd.Series, lat: float, lon: float) -> pd.Series:
    """Median per-month offset (hours) between observed daily GHI peak and clear-sky peak (UTC)."""
    if ghi.index.tz is None:
        raise ValueError("ghi must have a tz-aware UTC index")
    ghi = ghi.tz_convert("UTC")
    clear = pvlib.location.Location(lat, lon).get_clearsky(ghi.index, model="ineichen")["ghi"]
    df = pd.DataFrame({"obs": _daily_peak_hour(ghi), "ref": _daily_peak_hour(clear)}).dropna()
    return (df.obs - df.ref).groupby(df.index.month).median().round(2)


def run_pipeline_on_hourly(raw: pd.DataFrame) -> pd.DataFrame:
    """Run the real preprocess_germany_weather code on a hourly frame; return the 15-min frame."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "plant_x").mkdir()
        raw.to_csv(tmp / "plant_x" / "historical_weather_hourly.csv", index=False)
        old = pw.RAW_GER, pw.INTERIM_GER
        pw.RAW_GER = pw.INTERIM_GER = tmp
        try:
            out = pw.preprocess_plant_weather("plant_x")
            return pd.read_parquet(out).set_index("timestamp_utc")
        finally:
            pw.RAW_GER, pw.INTERIM_GER = old


def synthetic_hourly(lat: float, lon: float, year: int = 2023, legacy: bool = False) -> pd.DataFrame:
    """Open-Meteo-like hourly frame: uniform UTC series of clear-sky GHI."""
    start = pd.Timestamp(f"{year}-01-01", tz=LOCAL_TZ).tz_convert("UTC")
    end = pd.Timestamp(f"{year + 1}-01-01", tz=LOCAL_TZ).tz_convert("UTC")
    utc = pd.date_range(start, end, freq="1h", inclusive="left")
    raw = pd.DataFrame({c: 0.0 for c in WEATHER_COLS}, index=range(len(utc)))
    raw["shortwave_radiation_instant"] = pvlib.location.Location(lat, lon).get_clearsky(utc)["ghi"].values
    if legacy:
        # what the old fetcher wrote: uniform naive labels starting at the first local wall clock
        first_local = utc[0].tz_convert(LOCAL_TZ).tz_localize(None)
        raw["date"] = pd.date_range(first_local, periods=len(utc), freq="1h")
    else:
        raw["timestamp_utc"] = utc
    return raw


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, help="hourly weather CSV to check instead of synthetic data")
    ap.add_argument("--lat", type=float, default=48.14)
    ap.add_argument("--lon", type=float, default=11.58)
    ap.add_argument("--tolerance", type=float, default=0.5, help="max |offset| in hours")
    args = ap.parse_args()

    warnings.simplefilter("ignore")
    if args.csv:
        raw = pd.read_csv(args.csv)
        label = f"csv: {args.csv}"
    else:
        raw = synthetic_hourly(args.lat, args.lon, legacy=True)
        label = "synthetic legacy-format hourly series (repair path)"
    w = run_pipeline_on_hourly(raw)
    offsets = monthly_peak_offset(w["shortwave_radiation_instant"], args.lat, args.lon)
    print(f"\nDaily GHI peak vs clear-sky peak, median offset in hours ({label})")
    print(offsets.to_string())
    bad = offsets[offsets.abs() > args.tolerance]
    if len(bad):
        print(f"\nFAIL: months with |offset| > {args.tolerance} h: {list(bad.index)}")
        return 1
    print("\nOK: weather is aligned to UTC in all months")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
