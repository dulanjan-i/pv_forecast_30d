"""
Enforce a strict 15 minute time grid for German PV interim parquets.

INPUT:
- Reads per plant PV time series from:
    data/interim/germany/plant_XX_pv_15min.parquet

ASSUMPTIONS:
- Each parquet has the columns:
    - timestamp_utc  (datetime64[ns, UTC])
    - power_kw
    - power_w
    - power_norm
- These files have already passed basic cleaning and (if needed) scaling fixes.

WHAT THIS SCRIPT DOES:
- For each plant:
    - Sort by timestamp_utc.
    - Build a complete 15 minute DateTime index from min(timestamp_utc) to max(timestamp_utc).
    - Reindex the data onto this regular 15 minute grid.
    - Existing values stay unchanged at their timestamps.
    - Missing timestamps are created explicitly and contain NaNs in power columns.
- Overwrites the same parquet files in place.

WHAT THIS SCRIPT DOES NOT DO:
- Does not touch raw CSVs.
- Does not merge any weather data.
- Does not perform scaling or unit corrections.
- Does not fill gaps or interpolate missing values.
  It only makes gaps explicit as NaNs on a regular grid.
"""

from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
PV_DIR = REPO_ROOT / "data" / "interim" / "germany"

PLANT_IDS = ["plant_01", "plant_02", "plant_03", "plant_04", "plant_05", "plant_06"]


def enforce_15min_grid_for_plant(pid: str) -> None:
    parquet_path = PV_DIR / f"{pid}_pv_15min.parquet"
    if not parquet_path.exists():
        print(f"[WARN] parquet not found for {pid}, skipping")
        return

    df = pd.read_parquet(parquet_path)

    if "timestamp_utc" not in df.columns:
        print(f"[WARN] {pid}: timestamp_utc missing, skipping")
        return

    df = df.sort_values("timestamp_utc").drop_duplicates(subset=["timestamp_utc"]).copy()

    ts = df["timestamp_utc"]
    tz = ts.dt.tz

    full_index = pd.date_range(
        start=ts.min(),
        end=ts.max(),
        freq="15min",
        tz=tz,
    )

    # Reindex on strict 15 minute grid
    df = df.set_index("timestamp_utc").reindex(full_index)
    df.index.name = "timestamp_utc"
    df = df.reset_index()

    df.to_parquet(parquet_path, index=False)
    print(f"[INFO] {pid}: enforced 15 min grid, wrote {parquet_path}")


def main() -> None:
    for pid in PLANT_IDS:
        enforce_15min_grid_for_plant(pid)


if __name__ == "__main__":
    main()
    
