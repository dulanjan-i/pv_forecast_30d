"""
German PV plant preprocessing

Goal:
    Build a clean 15-min PV time series for each German plant, with:
        - timestamp_utc
        - power_kw        (plant-level AC power or production signal)
        - power_w
        - power_norm      (normalized by installed_capacity_kw)

This is an INTERIM product, analogous to:
    data/interim/farm_2107/farm2107_elec_irradiance_15min.parquet

Later scripts will:
    - merge PV with weather
    - build final feature tables under data/processed/
"""

from pathlib import Path
import json

import pandas as pd


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = REPO_ROOT / "data" / "raw" / "germany"
META_DIR = REPO_ROOT / "data" / "metadata" / "germany"
INTERIM_DIR = REPO_ROOT / "data" / "interim" / "germany"

INTERIM_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def load_metadata(plant_id: str) -> dict:
    meta_path = META_DIR / f"{plant_id}.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found for {plant_id}: {meta_path}")
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_raw_plant_csvs(plant_id: str) -> pd.DataFrame:
    """
    Load all CSVs for a plant and stack them into a single DataFrame.

    CSV format (German decimal):
        ;Ist-Erzeugung
        2022-12-31 23:00:00+00:00;0,252
    """
    plant_dir = RAW_DIR / plant_id
    csv_files = sorted(plant_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found for {plant_id} in {plant_dir}")

    dfs = []
    for path in csv_files:
        df = pd.read_csv(
            path,
            sep=";",
            header=0,
            names=["timestamp", "power_raw"],
            decimal=",",   # 0,252 -> 0.252
        )
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)

    # Parse timestamps as UTC
    df_all["timestamp_utc"] = pd.to_datetime(df_all["timestamp"], utc=True)

    # Drop the original text column
    df_all = df_all.drop(columns=["timestamp"])

    return df_all


def add_power_columns(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """
    Add power_kw, power_w, power_norm.

    Assumption for now:
        power_raw is already in kW (trading export).
    If later we discover it is kWh/15min, this is the single place to fix.
    """
    cap_kw = float(meta["installed_capacity_kw"])

    df = df.copy()

    # Robust conversion to float
    df["power_kw"] = pd.to_numeric(df["power_raw"], errors="coerce")

    # Power in W
    df["power_w"] = df["power_kw"] * 1000.0

    # Normalized power
    if cap_kw > 0:
        df["power_norm"] = df["power_kw"] / cap_kw
    else:
        df["power_norm"] = pd.NA

    # Keep only canonical columns
    df = df[["timestamp_utc", "power_kw", "power_w", "power_norm"]]

    # Sort and deduplicate
    df = df.sort_values("timestamp_utc").drop_duplicates(subset=["timestamp_utc"])

    return df


def preprocess_plant(plant_id: str) -> Path:
    meta = load_metadata(plant_id)
    df_raw = load_raw_plant_csvs(plant_id)
    df_proc = add_power_columns(df_raw, meta)

    out_path = INTERIM_DIR / f"{plant_id}_pv_15min.parquet"
    df_proc.to_parquet(out_path, index=False)
    print(f"[INFO] Wrote {out_path}")
    return out_path


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    plant_ids = [
        "plant_01",
        "plant_02",
        "plant_03",
        "plant_04",
        "plant_05",
        "plant_06",
    ]
    for pid in plant_ids:
        preprocess_plant(pid)


if __name__ == "__main__":
    main()
