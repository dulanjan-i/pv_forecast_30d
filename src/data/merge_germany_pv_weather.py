"""
merge_germany_pv_weather.py - Stage 2 Transfer Learning (Version 02)

Merge German PV (15-min, UTC) with German weather (15-min, UTC).

INPUT (flat interim files)
- data/interim/germany/{plant_id}_pv_15min.parquet
- data/interim/germany/{plant_id}_weather_15min.parquet

OUTPUT (flat processed files)
- data/processed/germany/{plant_id}_pv_weather_15min.parquet

Rules:
- Inner join on timestamp_utc (intersection of available timestamps).
  This avoids edge misalignment due to timezone/DST handling and ensures
  every row has both PV and weather.
- No feature engineering here, only clean merge.

Version 02 Changes (Dec 2025):
- Excluded plant_04 due to data quality issues (100% zeros during Mar-Jun 2024)
- See reports/stage2_version01_failed_chronological_split.md for details
"""

from pathlib import Path
from typing import List
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
INTERIM = REPO_ROOT / "data" / "interim" / "germany"
PROCESSED = REPO_ROOT / "data" / "processed" / "germany"

# Version 02: Excluded plant_04 (data quality issue - 100% zeros in Mar-Jun 2024)
PLANT_IDS: List[str] = ["plant_01","plant_02","plant_03","plant_05","plant_06"]


def merge_one(plant_id: str) -> Path:
    pv_path = INTERIM / f"{plant_id}_pv_15min.parquet"
    wx_path = INTERIM / f"{plant_id}_weather_15min.parquet"

    if not pv_path.exists():
        raise FileNotFoundError(f"Missing PV parquet: {pv_path}")
    if not wx_path.exists():
        raise FileNotFoundError(f"Missing weather parquet: {wx_path}")

    df_pv = pd.read_parquet(pv_path)
    df_wx = pd.read_parquet(wx_path)

    # Ensure timestamp column exists and is datetime
    for df, name in [(df_pv, "pv"), (df_wx, "weather")]:
        if "timestamp_utc" not in df.columns:
            raise ValueError(f"{plant_id}: {name} missing timestamp_utc")
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)

    # Drop redundant 'date' column from weather if present
    if "date" in df_wx.columns:
        df_wx = df_wx.drop(columns=["date"])

    # Inner join = intersection of timestamps
    df = df_pv.merge(df_wx, on="timestamp_utc", how="inner", validate="one_to_one")

    # Sort and write
    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED / f"{plant_id}_pv_weather_15min.parquet"
    df.to_parquet(out_path, index=False)

    print(
        f"[OK] {plant_id}: merged rows={len(df):,} | "
        f"pv_rows={len(df_pv):,} wx_rows={len(df_wx):,} -> {out_path}"
    )
    return out_path


def main():
    for pid in PLANT_IDS:
        merge_one(pid)


if __name__ == "__main__":
    main()
