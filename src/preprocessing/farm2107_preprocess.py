"""
Preprocessing for Farm Solar Array (System 2107)

Goal:
- Build a clean, 15-min time series with:
    * plant-level AC power (kW)
    * normalized PV power (pv_power_norm)
    * POA irradiance (W/m^2), aligned on the same timestamps

Inputs (expected in data/raw/farm_2107/):
    - 2107_electrical_data_v1.csv
    - 2107_electrical_data_2024.csv
    - 2107_irradiance_data.csv
    - 2107_irradiance_data_2024.csv

Outputs:
    - data/interim/farm_2107/farm2107_electrical_15min.parquet
    - data/interim/farm_2107/farm2107_elec_irradiance_15min.parquet
"""

import os
import pandas as pd

# ---------- CONFIG ----------

RAW_DIR = "data/raw/farm_2107"
INTERIM_DIR = "data/interim/farm_2107"
PROCESSED_DIR = "data/processed/farm_2107"

# DC capacity from metadata (kW)
DC_CAPACITY_KW = 893.0

# Canonical time resolution for pretraining
RESAMPLE_FREQ = "15T"  # 15 minutes

os.makedirs(INTERIM_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)


# ---------- ELECTRICAL (AC POWER) ----------

def load_and_merge_electrical() -> pd.DataFrame:
    """
    Load and merge all relevant electrical CSVs, then:
    - parse timestamps,
    - drop rows with no AC power info at all.

    Returns
    -------
    df : pd.DataFrame
        Columns include:
        - measured_on (datetime64[ns])
        - one column per inverter AC power (inv_XX_ac_power_...)
    """
    files = [
        "2107_electrical_data_v1.csv",
        "2107_electrical_data_2024.csv",
    ]

    dfs = []
    for fname in files:
        path = os.path.join(RAW_DIR, fname)
        if not os.path.exists(path):
            # If a file is missing, just skip it (but warn via print)
            print(f"[WARN] Electrical file not found, skipping: {path}")
            continue

        print(f"[INFO] Loading electrical file: {path}")
        df = pd.read_csv(path)

        if "measured_on" not in df.columns:
            raise ValueError(f"'measured_on' column missing in {path}")

        # Parse timestamps
        df["measured_on"] = pd.to_datetime(df["measured_on"])

        dfs.append(df)

    if not dfs:
        raise RuntimeError("No electrical data files loaded. Check paths/names.")

    # Vertical concat: stack all time rows
    df = pd.concat(dfs, axis=0, ignore_index=True)

    # Sort by time to get a proper chronological series
    df = df.sort_values("measured_on")

    # Identify AC power columns (per inverter)
    ac_cols = [c for c in df.columns if "ac_power" in c.lower()]

    if not ac_cols:
        raise RuntimeError("No AC power columns found in electrical data.")

    # Drop rows where all AC power values are NaN
    df = df.dropna(subset=ac_cols, how="all")

    print(f"[INFO] Electrical data rows after cleaning: {len(df)}")
    print(f"[INFO] Number of inverter AC power columns: {len(ac_cols)}")

    return df


def build_plant_level_power(df: pd.DataFrame) -> pd.DataFrame:
    """
    From merged electrical data, compute:
    - plant-level AC power by summing all inverter AC power,
    - normalized PV power.

    Parameters
    ----------
    df : pd.DataFrame
        Electrical data with:
        - measured_on
        - inv_XX_ac_power_* columns

    Returns
    -------
    df_out : pd.DataFrame
        Columns:
        - measured_on
        - p_ac_plant_kw
        - pv_power_norm
    """
    ac_cols = [c for c in df.columns if "ac_power" in c.lower()]

    # Sum AC power across all inverters to get plant-level AC power (kW)
    df["p_ac_plant_kw"] = df[ac_cols].sum(axis=1)

    # Normalize by DC capacity to get a [~0, 1+epsilon] target
    df["pv_power_norm"] = df["p_ac_plant_kw"] / DC_CAPACITY_KW

    # Keep only the core columns for now
    df_out = df[["measured_on", "p_ac_plant_kw", "pv_power_norm"]].copy()

    # Sort just in case
    df_out = df_out.sort_values("measured_on")

    print("[INFO] Example rows of plant-level AC power:")
    print(df_out.head())

    return df_out


def resample_to_15min(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample any irregular / 5-min electrical data to 15-min resolution.

    We use mean for power because it's an instantaneous-like signal,
    and for LSTM training the 15-min "average" power is a reasonable representation.

    Parameters
    ----------
    df : pd.DataFrame
        Columns:
        - measured_on
        - p_ac_plant_kw
        - pv_power_norm

    Returns
    -------
    df_15 : pd.DataFrame
        Same columns, 15-min frequency.
    """
    df = df.set_index("measured_on").sort_index()

    # Resample to canonical 15-min grid
    df_15 = df.resample(RESAMPLE_FREQ).mean()

    df_15 = df_15.reset_index()

    print("[INFO] Electrical data after 15-min resampling:")
    print(df_15.head())

    return df_15


# ---------- IRRADIANCE (POA) ----------

def load_and_merge_irradiance() -> pd.DataFrame:
    """
    Load and merge irradiance CSVs (POA), then parse timestamps.

    Returns
    -------
    df : pd.DataFrame
        Columns:
        - measured_on
        - poa_irradiance (W/m^2)
    """
    files = [
        "2107_irradiance_data.csv",
        "2107_irradiance_data_2024.csv",
    ]

    dfs = []
    for fname in files:
        path = os.path.join(RAW_DIR, fname)
        if not os.path.exists(path):
            print(f"[WARN] Irradiance file not found, skipping: {path}")
            continue

        print(f"[INFO] Loading irradiance file: {path}")
        df = pd.read_csv(path)

        if "measured_on" not in df.columns:
            raise ValueError(f"'measured_on' column missing in {path}")

        df["measured_on"] = pd.to_datetime(df["measured_on"])

        # Find POA irradiance column (name may have ID suffix)
        poa_cols = [c for c in df.columns if "poa_irradiance" in c.lower()]
        if not poa_cols:
            raise RuntimeError(f"No POA irradiance column found in {path}")

        # For now assume a single POA column
        poa_col = poa_cols[0]

        # Rename to a clean standard name
        df = df[["measured_on", poa_col]].rename(columns={poa_col: "poa_irradiance"})

        dfs.append(df)

    if not dfs:
        raise RuntimeError("No irradiance data files loaded. Check paths/names.")

    df = pd.concat(dfs, axis=0, ignore_index=True)
    df = df.sort_values("measured_on")

    print(f"[INFO] Irradiance rows after merging: {len(df)}")
    print("[INFO] Example irradiance rows:")
    print(df.head())

    return df


def resample_irradiance_to_15min(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample irradiance to the same 15-min grid.

    For POA irradiance, mean over the 15-min window is acceptable.

    Parameters
    ----------
    df : pd.DataFrame
        Columns:
        - measured_on
        - poa_irradiance

    Returns
    -------
    df_15 : pd.DataFrame
        15-min POA irradiance time series.
    """
    df = df.set_index("measured_on").sort_index()

    df_15 = df.resample(RESAMPLE_FREQ).mean()

    df_15 = df_15.reset_index()

    print("[INFO] Irradiance data after 15-min resampling:")
    print(df_15.head())

    return df_15


# ---------- MERGE ELECTRICAL + IRRADIANCE ----------

def merge_elec_and_irradiance(
    df_elec_15: pd.DataFrame, df_irr_15: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge 15-min electrical (AC power) and 15-min irradiance on timestamp.

    Parameters
    ----------
    df_elec_15 : pd.DataFrame
        Columns:
        - measured_on
        - p_ac_plant_kw
        - pv_power_norm

    df_irr_15 : pd.DataFrame
        Columns:
        - measured_on
        - poa_irradiance

    Returns
    -------
    df_merged : pd.DataFrame
        Columns:
        - measured_on
        - p_ac_plant_kw
        - pv_power_norm
        - poa_irradiance
    """
    df_merged = pd.merge(
        df_elec_15,
        df_irr_15,
        on="measured_on",
        how="inner",  # only keep timestamps where both exist
    )

    print("[INFO] Merged electrical + irradiance (15-min):")
    print(df_merged.head())

    return df_merged


# ---------- MAIN PIPELINE ----------

def main():
    # 1) Electrical: load, merge, build plant-level AC, normalize, resample to 15-min
    df_elec_raw = load_and_merge_electrical()
    df_elec_core = build_plant_level_power(df_elec_raw)
    df_elec_15 = resample_to_15min(df_elec_core)

    # Save intermediate electrical-only 15-min dataset
    elec_out_path = os.path.join(
        INTERIM_DIR, "farm2107_electrical_15min.parquet"
    )
    df_elec_15.to_parquet(elec_out_path, index=False)
    print(f"[INFO] Saved electrical 15-min data to: {elec_out_path}")

    # 2) Irradiance: load, merge, resample to 15-min
    df_irr_raw = load_and_merge_irradiance()
    df_irr_15 = resample_irradiance_to_15min(df_irr_raw)

    # 3) Merge electrical + irradiance on 15-min timestamps
    df_core = merge_elec_and_irradiance(df_elec_15, df_irr_15)

    # Save core interim dataset (this will be the base for LSTM pretraining table)
    core_out_path = os.path.join(
        INTERIM_DIR, "farm2107_elec_irradiance_15min.parquet"
    )
    df_core.to_parquet(core_out_path, index=False)
    print(f"[INFO] Saved electrical + irradiance 15-min data to: {core_out_path}")


if __name__ == "__main__":
    main()