from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


REQ_WEATHER = [
    "poa_irradiance",
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


def to_utc(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1-dir", required=True, type=str)
    ap.add_argument("--truth", required=True, type=str, help="ground_truth_15min_utc_capnorm.parquet")
    ap.add_argument("--out", required=True, type=str)
    ap.add_argument("--time-col", default="timestamp_utc", type=str)
    ap.add_argument("--plant-id", default="plant_03", type=str)
    ap.add_argument("--plant-col", default="plant_id", type=str)
    args = ap.parse_args()

    phase1_dir = Path(args.phase1_dir)
    weather_p = phase1_dir / "weather_with_pvlib_15min.parquet"
    if not weather_p.exists():
        raise FileNotFoundError(f"Missing: {weather_p}")

    truth = pd.read_parquet(args.truth, engine="pyarrow")
    weather = pd.read_parquet(weather_p, engine="pyarrow")

    truth[args.time_col] = to_utc(truth[args.time_col])
    weather[args.time_col] = to_utc(weather[args.time_col])

    if args.plant_col in weather.columns:
        weather = weather[weather[args.plant_col].astype(str) == str(args.plant_id)].copy()

    if "power_norm" not in truth.columns:
        raise KeyError("truth parquet missing power_norm")
    missing_weather = [c for c in REQ_WEATHER if c not in weather.columns]
    if missing_weather:
        raise KeyError("weather parquet missing required cols:\n" + "\n".join(missing_weather))

    df = weather[[args.time_col] + REQ_WEATHER].merge(
        truth[[args.time_col, "power_norm"]],
        on=args.time_col,
        how="inner",
    ).sort_values(args.time_col)

    # Final schema for LSTM encoder (exact order expected by your YAML)
    ordered = ["power_norm"] + REQ_WEATHER
    df = df[[args.time_col] + ordered]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"[OK] wrote hist_stream: {out} rows={len(df)} cols={len(df.columns)}")
    print("[OK] time range:", df[args.time_col].min(), "->", df[args.time_col].max())


if __name__ == "__main__":
    main()
