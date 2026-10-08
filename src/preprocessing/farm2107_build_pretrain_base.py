"""
farm2107_build_pretrain_base.py

Merge PVDAQ System 2107 PV data (15-min) with Open-Meteo weather (15-min)
into a unified base table for LSTM pretraining.

Inputs:
    data/interim/farm_2107/farm2107_elec_irradiance_15min.parquet
    data/processed/pretraining/farm2107_weather_15min.parquet

Outputs:
    data/processed/pretraining/farm2107_pretrain_base.parquet

Note:
    - This script aligns by timestamps and clips weather to the PV time span.
    - No normalization beyond pv_power_norm (already in the PV parquet) is done here.
      Feature scaling for the LSTM can be handled in the training pipeline.
"""

from pathlib import Path
import pandas as pd

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PV_INTERIM = Path("data/interim/farm_2107/farm2107_elec_irradiance_15min.parquet")
WEATHER_15 = Path("data/processed/pretraining/farm2107_weather_15min.parquet")
OUT_BASE   = Path("data/processed/pretraining/farm2107_pretrain_base.parquet")


def load_pv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"PV file not found: {path}")

    df = pd.read_parquet(path)
    if "measured_on" not in df.columns:
        raise ValueError(f"'measured_on' column missing in {path}")

    df = df.copy()
    df["measured_on"] = pd.to_datetime(df["measured_on"])
    df = df.sort_values("measured_on").reset_index(drop=True)

    print("[INFO] Loaded PV 15-min data:")
    print(df.head())
    print(df.tail())
    print(df.info())

    return df


def load_weather_15(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Weather 15-min file not found: {path}")

    df = pd.read_parquet(path)

    # Depending on how you saved it, 'date' might be a column or the index.
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    elif isinstance(df.index, pd.DatetimeIndex):
        pass
    else:
        raise ValueError("Expected a 'date' column or DatetimeIndex in weather data.")

    df = df.sort_index()

    print("[INFO] Loaded weather 15-min data:")
    print(df.head())
    print(df.tail())
    print(df.info())

    return df


def clip_weather_to_pv(weather: pd.DataFrame, pv: pd.DataFrame) -> pd.DataFrame:
    pv_start = pv["measured_on"].min()
    pv_end   = pv["measured_on"].max()

    print(f"[INFO] PV time span:      {pv_start} → {pv_end}")
    print(f"[INFO] Weather time span: {weather.index.min()} → {weather.index.max()}")

    w_clip = weather.loc[pv_start:pv_end]

    print("[INFO] Clipped weather 15-min data:")
    print(w_clip.head())
    print(w_clip.tail())
    print("  New span:", w_clip.index.min(), "→", w_clip.index.max())
    print("  Length:", len(w_clip))

    return w_clip


def merge_pv_weather(pv: pd.DataFrame, weather_15: pd.DataFrame) -> pd.DataFrame:
    w = weather_15.copy()
    w.index.name = "measured_on"

    merged = pv.merge(w, on="measured_on", how="left", validate="1:1")

    print("[INFO] Merged PV + weather:")
    print(merged.head())
    print(merged.tail())
    print(merged.info())

    return merged


def main():
    OUT_BASE.parent.mkdir(parents=True, exist_ok=True)

    pv_df = load_pv(PV_INTERIM)
    w15_df = load_weather_15(WEATHER_15)
    w15_clip = clip_weather_to_pv(w15_df, pv_df)

    base_df = merge_pv_weather(pv_df, w15_clip)

    print(f"[INFO] Saving unified pretraining base to {OUT_BASE}")
    base_df.to_parquet(OUT_BASE)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()