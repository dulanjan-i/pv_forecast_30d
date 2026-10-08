"""
openmeteo_preprocessing.py

Convert interim Open-Meteo weather data for PVDAQ System 2107 (Farm Solar Array)
from hourly resolution to 15-minute resolution, and export processed weather
tables for downstream LSTM pretraining.

Inputs (INTERIM):
    data/interim/farm_2107/historical_weather_hourly.parquet
    data/interim/farm_2107/historical_weather_daily.parquet

Outputs (PROCESSED):
    data/processed/pretraining/farm2107_weather_15min.parquet   (time-varying features)
    data/processed/pretraining/farm2107_weather_daily.parquet   (daily aggregates, unchanged)
    
Important: This does not clip the output to the PVDAQ data range; that is done
later during dataset assembly.
    
Run from repo root:
    python src/preprocessing/openmeteo_preprocessing.py
"""

from pathlib import Path
import pandas as pd


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

INTERIM_DIR = Path("data/interim/farm_2107")
PROCESSED_DIR = Path("data/processed/pretraining")

HOURLY_IN = INTERIM_DIR / "historical_weather_hourly.parquet"
DAILY_IN = INTERIM_DIR / "historical_weather_daily.parquet"

HOURLY_OUT_15 = PROCESSED_DIR / "farm2107_weather_15min.parquet"
DAILY_OUT = PROCESSED_DIR / "farm2107_weather_daily.parquet"


# ---------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------

def load_hourly_weather(path: Path) -> pd.DataFrame:
    """Load hourly Open-Meteo weather (interim) and set index = date."""
    if not path.exists():
        raise FileNotFoundError(f"Hourly weather file not found: {path}")

    df = pd.read_parquet(path)
    if "date" not in df.columns:
        raise ValueError(f"Expected a 'date' column in {path}, got: {df.columns.tolist()}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    print("[INFO] Loaded hourly weather:")
    print(df.head())
    print(df.tail())
    print(df.info())

    return df


def resample_to_15min(df_hourly: pd.DataFrame) -> pd.DataFrame:
    """
    Resample hourly weather to 15-minute resolution using time-based interpolation.

    - Index must be a DateTimeIndex.
    - All numeric columns are interpolated.
    """
    if not isinstance(df_hourly.index, pd.DatetimeIndex):
        raise TypeError("Hourly weather DataFrame index must be a DatetimeIndex.")

    print("[INFO] Resampling hourly weather to 15-min resolution...")
    df_15 = df_hourly.resample("15min").interpolate(method="time")

    print("[INFO] 15-min weather summary:")
    print("  Index range:", df_15.index.min(), "→", df_15.index.max())
    print("  Inferred frequency:", df_15.index.inferred_freq)
    print(df_15.head())
    print(df_15.tail())
    print(df_15.info())

    return df_15


def load_daily_weather(path: Path) -> pd.DataFrame:
    """Load daily Open-Meteo weather (interim) and set index = date."""
    if not path.exists():
        raise FileNotFoundError(f"Daily weather file not found: {path}")

    df = pd.read_parquet(path)
    if "date" not in df.columns:
        raise ValueError(f"Expected a 'date' column in {path}, got: {df.columns.tolist()}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    print("[INFO] Loaded daily weather:")
    print(df.head())
    print(df.tail())
    print(df.info())

    return df


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Hourly → 15 min
    hourly_df = load_hourly_weather(HOURLY_IN)
    weather_15 = resample_to_15min(hourly_df)

    print(f"[INFO] Saving 15-min weather to {HOURLY_OUT_15}")
    weather_15.to_parquet(HOURLY_OUT_15)

    # 2) Daily (just cleaned, moved to processed)
    daily_df = load_daily_weather(DAILY_IN)
    print(f"[INFO] Saving daily weather to {DAILY_OUT}")
    daily_df.to_parquet(DAILY_OUT)

    print("[INFO] Done: processed weather written to data/processed/pretraining/")


if __name__ == "__main__":
    main()