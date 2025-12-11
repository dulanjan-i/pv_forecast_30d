"""
preprocess_germany_weather.py

Preprocess Open-Meteo ERA5 weather for German plants to a 15 minute UTC grid.

INPUT
- For each plant_01 .. plant_06, this script expects hourly weather from:
    data/interim/germany/{plant_id}/historical_weather_hourly.parquet

  Each hourly parquet must contain columns at least:
    - date  (naive datetime, Europe/Berlin local time)
    - temperature_2m
    - relative_humidity_2m
    - precipitation
    - weather_code
    - cloud_cover
    - wind_speed_10m
    - wind_direction_10m
    - shortwave_radiation_instant
    - direct_radiation_instant
    - diffuse_radiation_instant
    - direct_normal_irradiance_instant
    - global_tilted_irradiance_instant
    - surface_pressure

OUTPUT
- For each plant, writes a 15 minute UTC weather time series to:
    data/interim/germany/{plant_id}_weather_15min.parquet

  With columns:
    - timestamp_utc (datetime64[ns, UTC], regular 15 minute grid)
    - same weather variables as input, resampled to 15 minutes

WHAT THIS SCRIPT DOES
- Converts Open-Meteo timestamps from Europe/Berlin local time to UTC.
- Sets a regular 1 hour index on weather data and resamples to a 15 minute grid.
- Uses a HYBRID interpolation strategy that matches the Farm2107 pipeline:

  - Radiation fields:
        shortwave_radiation_instant
        direct_radiation_instant
        diffuse_radiation_instant
        direct_normal_irradiance_instant
        global_tilted_irradiance_instant

    are interpolated linearly when going from hourly to 15 minute resolution.
    This preserves the physical smoothness of the solar curve and avoids
    discontinuities around sunrise and sunset.

  - Meteorological state variables:
        temperature_2m
        relative_humidity_2m
        precipitation
        cloud_cover
        wind_speed_10m
        wind_direction_10m
        surface_pressure

    are forward filled, with an initial backfill. This matches the practical
    assumption that these variables evolve smoothly but do not need linear
    interpolation within each hour.

  - weather_code is treated as categorical and is only forward filled with an
    initial backfill. It is never interpolated linearly.

- Enforces a strict 15 minute UTC grid between the first and last available
  hourly weather timestamps. Any gaps from the API become NaNs before the
  fill/interpolation logic is applied.

WHAT THIS SCRIPT DOES NOT DO
- Does not touch PV parquets.
- Does not merge PV and weather. That will be handled by a separate script.
- Does not use daily Open-Meteo data. Daily files remain available for optional
  feature engineering but are not processed here.
"""

from pathlib import Path
from typing import List

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_INTERIM = REPO_ROOT / "data" / "interim" / "germany"

PLANT_IDS: List[str] = [
    "plant_01",
    "plant_02",
    "plant_03",
    "plant_04",
    "plant_05",
    "plant_06",
]

TIMEZONE_LOCAL = "Europe/Berlin"
TIMEZONE_UTC = "UTC"


def preprocess_plant_weather(plant_id: str) -> Path:
    """
    Load hourly Open-Meteo weather for a plant, convert to UTC,
    resample to 15 minute grid, apply hybrid interpolation,
    and save to a single 15 minute parquet.

    Input:  data/interim/germany/{plant_id}/historical_weather_hourly.parquet
    Output: data/interim/germany/{plant_id}_weather_15min.parquet
    """
    hourly_path = BASE_INTERIM / plant_id / "historical_weather_hourly.parquet"
    if not hourly_path.exists():
        raise FileNotFoundError(f"Hourly weather file not found for {plant_id}: {hourly_path}")

    df = pd.read_parquet(hourly_path)

    if "date" not in df.columns:
        raise ValueError(f"{plant_id}: expected 'date' column in hourly weather parquet")

    # 1. Convert naive local time to UTC
    #
    # Open-Meteo archive returns timestamps in local Europe/Berlin time,
    # without timezone info, already adjusted for DST.
    # We localize to Europe/Berlin then convert to UTC to match PV data.
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    df["timestamp_utc"] = (
        df["date"]
        .dt.tz_localize(TIMEZONE_LOCAL, ambiguous="infer", nonexistent="shift_forward")
        .dt.tz_convert(TIMEZONE_UTC)
    )

    # 2. Enforce a regular 1 hour index before resampling
    df = df.set_index("timestamp_utc").sort_index()

    # For safety, coerce all numeric columns to numeric
    for col in df.columns:
        if col == "weather_code":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 3. Resample to 15 minute grid
    #
    # Use asfreq first to place existing hourly values on the new grid,
    # then apply filling / interpolation per variable group.
    df_15 = df.asfreq("15min")

    # Identify column groups
    radiation_cols = [
        "shortwave_radiation_instant",
        "direct_radiation_instant",
        "diffuse_radiation_instant",
        "direct_normal_irradiance_instant",
        "global_tilted_irradiance_instant",
    ]
    meteo_cols = [
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "cloud_cover",
        "wind_speed_10m",
        "wind_direction_10m",
        "surface_pressure",
    ]

    # Keep only columns that are actually present
    radiation_cols = [c for c in radiation_cols if c in df_15.columns]
    meteo_cols = [c for c in meteo_cols if c in df_15.columns]

    # 4. Apply interpolation and filling
    #
    # Radiation: linear interpolation over time
    if radiation_cols:
        df_15[radiation_cols] = df_15[radiation_cols].interpolate(
            method="time", limit_direction="both"
        )

    # Meteo: forward fill, then backfill at the start
    if meteo_cols:
        df_15[meteo_cols] = df_15[meteo_cols].ffill().bfill()

    # Weather code: treat as categorical, only ffill/bfill
    if "weather_code" in df_15.columns:
        df_15["weather_code"] = df_15["weather_code"].ffill().bfill()

    # 5. Finalize output
    df_15 = df_15.reset_index().rename(columns={"timestamp_utc": "timestamp_utc"})

    # Ensure timestamp_utc is timezone aware UTC
    df_15["timestamp_utc"] = pd.to_datetime(df_15["timestamp_utc"]).dt.tz_convert(TIMEZONE_UTC)

    out_path = BASE_INTERIM / f"{plant_id}_weather_15min.parquet"
    df_15.to_parquet(out_path, index=False)
    print(f"[INFO] {plant_id}: wrote {out_path} with {len(df_15)} rows")

    return out_path


def main() -> None:
    for pid in PLANT_IDS:
        print(f"\n===== {pid} =====")
        preprocess_plant_weather(pid)

    print("\n[INFO] All German plant weather preprocessed to 15 minute UTC.")


if __name__ == "__main__":
    main()
