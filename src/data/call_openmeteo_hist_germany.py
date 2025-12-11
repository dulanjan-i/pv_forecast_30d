"""
call_openmeteo_hist_germany.py

Fetch historical Open-Meteo weather for the six German Syneco plants.

Logic:
- For each plant_01 … plant_06:
    * load metadata from data/metadata/germany/plant_XX.json
    * load PV time series from data/interim/germany/plant_XX_pv_15min.parquet
    * derive start_date and end_date from PV timestamps
    * call Open-Meteo archive API with:
        - latitude, longitude from metadata
        - tilt, azimuth from metadata
        - timezone = Europe/Berlin
        - model = era5_seamless
    * save hourly and daily data into:
        data/raw/germany/plant_XX/   (CSV)
        data/interim/germany/plant_XX/  (Parquet)

Notes:
- This mirrors the Farm2107 Open-Meteo script, but parameterised by plant metadata. 
- Weather is saved in local Europe/Berlin time, same as Open-Meteo response.
- No resampling to 15-minute here, that stays in a separate preprocessing step.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

PLANTS: List[str] = [
    "plant_01",
    "plant_02",
    "plant_03",
    "plant_04",
    "plant_05",
    "plant_06",
]

METADATA_DIR = Path("data/metadata/germany")
INTERIM_PV_DIR = Path("data/interim/germany")

OPENMETEO_URL = "https://archive-api.open-meteo.com/v1/archive"
TIMEZONE = "Europe/Berlin"
MODEL = "era5_seamless"

DAILY_VARS = [
    "sunrise",
    "sunset",
    "daylight_duration",
    "sunshine_duration",
    "precipitation_sum",
    "rain_sum",
    "snowfall_sum",
    "precipitation_hours",
    "temperature_2m_max",
    "temperature_2m_min",
    "apparent_temperature_max",
]

HOURLY_VARS = [
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


# ---------------------------------------------------------------------
# Open-Meteo client
# ---------------------------------------------------------------------

cache_session = requests_cache.CachedSession(".cache", expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
# Disable SSL verification for corporate proxy/firewall
retry_session.verify = False
openmeteo = openmeteo_requests.Client(session=retry_session)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def load_metadata(plant_id: str) -> Dict:
    path = METADATA_DIR / f"{plant_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Metadata JSON not found for {plant_id}: {path}")
    with path.open("r") as f:
        return json.load(f)


def infer_date_range_from_pv(plant_id: str) -> Dict[str, str]:
    """
    Read PV parquet for this plant and return start/end dates (YYYY-MM-DD)
    in UTC, which is what the archive API expects.
    """
    pv_path = INTERIM_PV_DIR / f"{plant_id}_pv_15min.parquet"
    if not pv_path.exists():
        raise FileNotFoundError(f"PV parquet not found for {plant_id}: {pv_path}")

    df = pd.read_parquet(pv_path)
    if "timestamp_utc" not in df.columns:
        raise ValueError(
            f"'timestamp_utc' column missing in PV parquet for {plant_id}. "
            f"Columns: {df.columns.tolist()}"
        )

    ts = pd.to_datetime(df["timestamp_utc"])
    start_date = ts.min().date().strftime("%Y-%m-%d")
    end_date = ts.max().date().strftime("%Y-%m-%d")

    return {"start_date": start_date, "end_date": end_date}


def build_params(meta: Dict, start_date: str, end_date: str) -> Dict:
    # Convert azimuth from 0-360° (PVLib) to -180-180° (Open-Meteo)
    azimuth = meta["azimuth_deg"]
    if azimuth > 180:
        azimuth = azimuth - 360
    
    return {
        "latitude": meta["latitude"],
        "longitude": meta["longitude"],
        "start_date": start_date,
        "end_date": end_date,
        "daily": DAILY_VARS,
        "hourly": HOURLY_VARS,
        "models": MODEL,
        "timezone": TIMEZONE,
        # PV geometry
        "tilt": meta["tilt_deg"],
        "azimuth": azimuth,
    }


def response_to_dataframes(response) -> Dict[str, pd.DataFrame]:
    """Convert one Open-Meteo response into hourly and daily DataFrames."""
    # Hourly
    hourly = response.Hourly()
    
    # Build data dict first to get actual array lengths
    hourly_data = {}
    for i, var_name in enumerate(HOURLY_VARS):
        hourly_data[var_name] = hourly.Variables(i).ValuesAsNumpy()
    
    # Create date_range matching the actual data length
    start_dt = pd.to_datetime(hourly.Time(), unit="s", utc=True).tz_convert(TIMEZONE).tz_localize(None)
    freq = pd.Timedelta(seconds=hourly.Interval())
    n_points = len(hourly_data[HOURLY_VARS[0]])
    hourly_data["date"] = pd.date_range(start=start_dt, periods=n_points, freq=freq)
    
    hourly_df = pd.DataFrame(hourly_data)

    # Daily
    daily = response.Daily()
    
    # Build data dict first to get actual array lengths
    daily_data = {}
    for i, var_name in enumerate(DAILY_VARS):
        # Sunrise and sunset are returned as int64 timestamps
        if var_name in ("sunrise", "sunset"):
            daily_data[var_name] = daily.Variables(i).ValuesInt64AsNumpy()
        else:
            daily_data[var_name] = daily.Variables(i).ValuesAsNumpy()
    
    # Create date_range matching the actual data length
    start_dt = pd.to_datetime(daily.Time(), unit="s", utc=True).tz_convert(TIMEZONE).tz_localize(None)
    freq = pd.Timedelta(seconds=daily.Interval())
    n_points = len(daily_data[DAILY_VARS[0]])
    daily_data["date"] = pd.date_range(start=start_dt, periods=n_points, freq=freq)
    
    daily_df = pd.DataFrame(daily_data)

    return {"hourly": hourly_df, "daily": daily_df}


def save_plant_weather(plant_id: str, dfs: Dict[str, pd.DataFrame]) -> None:
    raw_dir = Path("data/raw/germany") / plant_id
    interim_dir = Path("data/interim/germany") / plant_id
    raw_dir.mkdir(parents=True, exist_ok=True)
    interim_dir.mkdir(parents=True, exist_ok=True)

    hourly = dfs["hourly"]
    daily = dfs["daily"]

    # CSV (raw)
    hourly_csv = raw_dir / "historical_weather_hourly.csv"
    daily_csv = raw_dir / "historical_weather_daily.csv"
    hourly.to_csv(hourly_csv, index=False)
    daily.to_csv(daily_csv, index=False)

    # Parquet (interim)
    hourly_parquet = interim_dir / "historical_weather_hourly.parquet"
    daily_parquet = interim_dir / "historical_weather_daily.parquet"
    hourly.to_parquet(hourly_parquet, index=False, engine="pyarrow", compression="snappy")
    daily.to_parquet(daily_parquet, index=False, engine="pyarrow", compression="snappy")

    print(
        f"[OK] {plant_id}: "
        f"hourly rows={len(hourly):,}, daily rows={len(daily):,} → "
        f"{hourly_parquet}, {daily_parquet}"
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    for plant_id in PLANTS:
        print(f"\n========== {plant_id} ==========")
        meta = load_metadata(plant_id)
        dates = infer_date_range_from_pv(plant_id)

        params = build_params(meta, dates["start_date"], dates["end_date"])
        print(f"[INFO] Requesting Open-Meteo for {plant_id} with params:")
        print(
            f"  lat={params['latitude']}, lon={params['longitude']}, "
            f"start={params['start_date']}, end={params['end_date']}, "
            f"tilt={params['tilt']}, azimuth={params['azimuth']}"
        )

        responses = openmeteo.weather_api(OPENMETEO_URL, params=params)
        response = responses[0]

        dfs = response_to_dataframes(response)
        save_plant_weather(plant_id, dfs)

    print("\n[INFO] Done: weather downloaded for all German plants.")


if __name__ == "__main__":
    main()
