"""
src/features/germany_build_pvlib_for_tft.py

Stage 3.8: Build PVLib-derived features for TFT (Germany regional pipeline).

What this script does
- Reads the Stage 3.7 weather tables (raw Open-Meteo ERA5-like variables at 15-min).
- Loads plant metadata JSON per plant_id (lat/lon/tilt/azimuth/capacity).
- Uses pvlib to compute:
    - solar position (zenith, azimuth)
    - plane-of-array irradiance components (POA global, direct, diffuse, ground)
    - PVWatts-style DC and AC power proxies (scaled by plant capacity)
- Writes per-split PVLib feature tables (train and val) with keys:
    (plant_id, timestamp_utc)

Why these fixes exist (based on your errors)
1) dni_extra requirement:
   pvlib.irradiance.get_total_irradiance(model="haydavies") requires dni_extra.
   We provide dni_extra via pvlib.irradiance.get_extra_radiation(times).

2) "All arrays must be of the same length":
   This usually happens if timestamps are duplicated, unsorted, or indices misalign.
   We enforce:
   - per-plant strict sorting by timestamp
   - drop duplicate timestamps per plant (keep last)
   - set a consistent DatetimeIndex and use aligned numpy arrays

3) Capacity key mismatch:
   Your metadata uses installed_capacity_kw.
   We now accept multiple key variants:
   - installed_capacity_kw (preferred from your JSON)
   - capacity_kw, pdc0_kw
   - installed_capacity_mw, capacity_mw, pdc0_mw
   - pdc0_w (as watts)

Azimuth convention
- pvlib expects azimuth degrees clockwise from North:
  N=0, E=90, S=180, W=270.
- If metadata provides `azimuth_deg`, we use it directly.
- If metadata only provides `azimuth_deg_sy` (South=0 style), we convert:
    azimuth_deg = (180 + azimuth_deg_sy) % 360

Inputs
- train_weather parquet: produced by Stage 3.7
- val_weather parquet:   produced by Stage 3.7
- meta_dir: data/metadata/germany (expects plant_XX.json files)

Outputs
- out_dir/train_pvlib_tft.parquet
- out_dir/val_pvlib_tft.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import pvlib
import inspect

# Configure logging
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Canonical column names (keep consistent with your schema)
# ---------------------------------------------------------------------
TIME_COL = "timestamp_utc"
PLANT_ID_COL = "plant_id"

# Weather columns expected in Stage 3.7 outputs
REQ_WEATHER_COLS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "weather_code",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "shortwave_radiation_instant",          # GHI proxy
    "direct_radiation_instant",             # often DNI-projected, but we rely on DNI below
    "diffuse_radiation_instant",            # DHI
    "direct_normal_irradiance_instant",     # DNI
    "global_tilted_irradiance_instant",     # GTI (not used for POA calc here)
    "surface_pressure",
]

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _read_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def compute_cell_temperature(poa_global: np.ndarray, temp_air: np.ndarray, wind: np.ndarray) -> np.ndarray:
    """
    Compute cell temperature with a robust fallback across pvlib versions.

    Preferred: SAPM cell temperature
    - Newer pvlib requires (a, b, deltaT)
    - We use pvlib built-in params for a common mounting configuration.

    Fallback: PVsyst cell temperature
    - Uses (u_c, u_v), also available in pvlib defaults.

    Returns
    -------
    np.ndarray
        Cell temperature in degC, same length as inputs.
    """
    poa_global = np.asarray(poa_global, dtype=float)
    temp_air = np.asarray(temp_air, dtype=float)
    wind = np.asarray(wind, dtype=float)

    # Try SAPM
    try:
        sig = inspect.signature(pvlib.temperature.sapm_cell)
        params = sig.parameters

        if {"a", "b", "deltaT"}.issubset(set(params.keys())):
            # Newer pvlib API, needs a, b, deltaT
            sapm_params = None
            try:
                sapm_params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"]["open_rack_glass_glass"]
            except Exception:
                # Reasonable fallback config name
                sapm_params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"]["open_rack_glass_polymerback"]

            a = float(sapm_params["a"])
            b = float(sapm_params["b"])
            deltaT = float(sapm_params["deltaT"])
            return pvlib.temperature.sapm_cell(poa_global, temp_air, wind, a=a, b=b, deltaT=deltaT).to_numpy()

        # Older pvlib API (no a,b,deltaT required)
        out = pvlib.temperature.sapm_cell(poa_global, temp_air, wind)
        return np.asarray(out, dtype=float)

    except Exception as e:
        logger.warning(
            f"SAPM cell temperature computation failed, falling back to PVsyst method. "
            f"Error: {type(e).__name__}: {e}"
        )

    # Fallback: PVsyst
    try:
        pvsyst_params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["pvsyst"]["freestanding"]
        u_c = float(pvsyst_params["u_c"])
        u_v = float(pvsyst_params["u_v"])
        out = pvlib.temperature.pvsyst_cell(poa_global, temp_air, wind, u_c=u_c, u_v=u_v)
        return np.asarray(out, dtype=float)
    except Exception as e:
        raise RuntimeError(f"Failed to compute cell temperature with sapm_cell and pvsyst_cell. Last error: {e}")

def _get_capacity_kw(meta: dict) -> float:
    """
    Return plant capacity in kW from metadata.
    Accepts common key variants.
    """
    # Most likely in your JSON
    if isinstance(meta.get("installed_capacity_kw", None), (int, float)):
        return float(meta["installed_capacity_kw"])

    # Other possible variants
    for k in ["capacity_kw", "pdc0_kw"]:
        if isinstance(meta.get(k, None), (int, float)):
            return float(meta[k])

    # MW variants
    for k in ["installed_capacity_mw", "capacity_mw", "pdc0_mw"]:
        if isinstance(meta.get(k, None), (int, float)):
            return float(meta[k]) * 1000.0

    # Watts variant
    if isinstance(meta.get("pdc0_w", None), (int, float)):
        return float(meta["pdc0_w"]) / 1000.0

    raise ValueError("Plant metadata missing a valid capacity field (installed_capacity_kw/capacity_kw/pdc0_kw/...).")


def _get_surface_azimuth(meta: dict) -> float:
    """
    pvlib expects azimuth degrees clockwise from North:
    N=0, E=90, S=180, W=270
    """
    if isinstance(meta.get("azimuth_deg", None), (int, float)):
        return float(meta["azimuth_deg"])

    # Convert from South=0 style if present
    if isinstance(meta.get("azimuth_deg_sy", None), (int, float)):
        return (180.0 + float(meta["azimuth_deg_sy"])) % 360.0

    raise ValueError("Plant metadata missing azimuth (expected azimuth_deg or azimuth_deg_sy).")


def _pressure_to_pa(p: pd.Series) -> pd.Series:
    """
    Open-Meteo surface_pressure can be hPa or Pa depending on pipeline.
    Heuristic:
    - If median < 2000 -> assume hPa and convert to Pa
    - Else assume already Pa
    """
    s = p.astype(float)
    med = float(np.nanmedian(s.to_numpy()))
    if med < 2000.0:
        return s * 100.0
    return s


def _ensure_clean_time_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforce:
    - timestamp_utc parsed
    - sorted
    - duplicates removed per plant
    """
    d = df.copy()
    d[TIME_COL] = pd.to_datetime(d[TIME_COL], utc=True)
    d = d.sort_values([PLANT_ID_COL, TIME_COL]).reset_index(drop=True)
    # Drop duplicate timestamps within each plant (keep last to match most recent row)
    d = d.drop_duplicates(subset=[PLANT_ID_COL, TIME_COL], keep="last")
    return d


def _compute_pvlib_for_one_plant(wp: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """
    Compute pvlib features for a single plant weather table.

    Returns DataFrame with columns:
    - plant_id, timestamp_utc
    - pvlib_solar_zenith, pvlib_solar_azimuth
    - pvlib_poa_global, pvlib_poa_direct, pvlib_poa_diffuse, pvlib_poa_ground_diffuse
    - pvlib_dc_kw, pvlib_ac_kw
    """
    pid = str(wp[PLANT_ID_COL].iloc[0])

    lat = float(meta["latitude"])
    lon = float(meta["longitude"])
    tilt = float(meta["tilt_deg"])
    azm = float(_get_surface_azimuth(meta))
    cap_kw = float(_get_capacity_kw(meta))

    # Sort and set index
    wp = wp.copy()
    wp[TIME_COL] = pd.to_datetime(wp[TIME_COL], utc=True)
    wp = wp.sort_values(TIME_COL).reset_index(drop=True)

    times = wp[TIME_COL]
    idx = pd.DatetimeIndex(times)

    # Irradiance: clamp negatives to 0 (interpolation artifacts can create negatives)
    dni = wp["direct_normal_irradiance_instant"].astype(float).clip(lower=0.0).to_numpy()
    ghi = wp["shortwave_radiation_instant"].astype(float).clip(lower=0.0).to_numpy()
    dhi = wp["diffuse_radiation_instant"].astype(float).clip(lower=0.0).to_numpy()

    temp_air = wp["temperature_2m"].astype(float).to_numpy()
    wind = wp["wind_speed_10m"].astype(float).clip(lower=0.0).to_numpy()

    # Solar position
    solpos = pvlib.solarposition.get_solarposition(time=idx, latitude=lat, longitude=lon)
    solar_zenith = solpos["zenith"].to_numpy()
    solar_azimuth = solpos["azimuth"].to_numpy()

    # dni_extra required for haydavies
    dni_extra = pvlib.irradiance.get_extra_radiation(idx).to_numpy()

    # POA irradiance via Hay-Davies
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt,
        surface_azimuth=azm,
        solar_zenith=solar_zenith,
        solar_azimuth=solar_azimuth,
        dni=dni,
        ghi=ghi,
        dhi=dhi,
        dni_extra=dni_extra,
        model="haydavies",
        albedo=0.2,
    )

    # Ensure numpy arrays
    poa_global = np.asarray(poa["poa_global"]).clip(min=0.0)
    poa_direct = np.asarray(poa.get("poa_direct", np.full_like(poa_global, np.nan))).clip(min=0.0)
    poa_diffuse = np.asarray(poa.get("poa_diffuse", np.full_like(poa_global, np.nan))).clip(min=0.0)
    poa_ground_diffuse = np.asarray(poa.get("poa_ground_diffuse", np.full_like(poa_global, np.nan))).clip(min=0.0)

    # Cell temperature (simple SAPM model)
    temp_cell = compute_cell_temperature(poa_global, temp_air, wind)

    # PVWatts DC/AC proxies
    pdc0_w = cap_kw * 1000.0
    gamma_pdc = -0.003  # typical
    pdc_w = pvlib.pvsystem.pvwatts_dc(poa_global, temp_cell, pdc0=pdc0_w, gamma_pdc=gamma_pdc)
    pac_w = pvlib.inverter.pvwatts(pdc_w, pdc0=pdc0_w)

    out = pd.DataFrame(
        {
            PLANT_ID_COL: pid,
            TIME_COL: idx,
            "pvlib_solar_zenith": solar_zenith.astype(np.float32),
            "pvlib_solar_azimuth": solar_azimuth.astype(np.float32),
            "pvlib_poa_global": poa_global.astype(np.float32),
            "pvlib_poa_direct": poa_direct.astype(np.float32),
            "pvlib_poa_diffuse": poa_diffuse.astype(np.float32),
            "pvlib_poa_ground_diffuse": poa_ground_diffuse.astype(np.float32),
            "pvlib_dc_kw": (np.asarray(pdc_w) / 1000.0).astype(np.float32),
            "pvlib_ac_kw": (np.asarray(pac_w) / 1000.0).astype(np.float32),
        }
    )
    return out


def build_pvlib_table(weather_df: pd.DataFrame, meta_dir: Path) -> pd.DataFrame:
    """
    Build pvlib features for all plants present in weather_df.
    """
    weather_df = _ensure_clean_time_index(weather_df)

    # Basic column validation
    need = set([PLANT_ID_COL, TIME_COL]) | set(REQ_WEATHER_COLS)
    missing = sorted(need - set(weather_df.columns))
    if missing:
        raise ValueError(f"Weather table missing required columns: {missing}")

    out_parts: List[pd.DataFrame] = []
    plants = sorted(weather_df[PLANT_ID_COL].unique().tolist())

    for pid in plants:
        w_path = meta_dir / f"{pid}.json"
        if not w_path.exists():
            raise FileNotFoundError(f"Missing plant metadata JSON: {w_path}")

        meta = _read_json(w_path)

        wp = weather_df.loc[weather_df[PLANT_ID_COL] == pid].copy()
        if len(wp) == 0:
            continue

        outp = _compute_pvlib_for_one_plant(wp, meta)
        out_parts.append(outp)

    if not out_parts:
        raise ValueError("No PVLib outputs produced. Check plant_ids and metadata paths.")

    out = pd.concat(out_parts, axis=0, ignore_index=True)
    out[TIME_COL] = pd.to_datetime(out[TIME_COL], utc=True)
    out = out.sort_values([PLANT_ID_COL, TIME_COL]).reset_index(drop=True)
    out = out.drop_duplicates(subset=[PLANT_ID_COL, TIME_COL], keep="last")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train_weather", type=str, required=True)
    p.add_argument("--val_weather", type=str, required=True)
    p.add_argument("--meta_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    train_weather = Path(args.train_weather)
    val_weather = Path(args.val_weather)
    meta_dir = Path(args.meta_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("STAGE 3.8: PVLib feature build")
    print(f"train_weather: {train_weather}")
    print(f"val_weather:   {val_weather}")
    print(f"meta_dir:      {meta_dir}")
    print(f"out_dir:       {out_dir}")
    print("=" * 80)

    tw = pd.read_parquet(train_weather)
    vw = pd.read_parquet(val_weather)

    print(f"[INFO] {train_weather.name}: rows={len(tw):,} plants={sorted(tw[PLANT_ID_COL].unique().tolist())}")
    train_tbl = build_pvlib_table(tw, meta_dir)

    print(f"[INFO] {val_weather.name}: rows={len(vw):,} plants={sorted(vw[PLANT_ID_COL].unique().tolist())}")
    val_tbl = build_pvlib_table(vw, meta_dir)

    out_train = out_dir / "regional_train_pvlib_tft.parquet"
    out_val = out_dir / "regional_val_pvlib_tft.parquet"

    train_tbl.to_parquet(out_train, index=False)
    val_tbl.to_parquet(out_val, index=False)

    print(f"[SUCCESS] Wrote train pvlib: {out_train} rows={len(train_tbl):,} cols={train_tbl.shape[1]}")
    print(f"[SUCCESS] Wrote val pvlib:   {out_val} rows={len(val_tbl):,} cols={val_tbl.shape[1]}")


if __name__ == "__main__":
    main()
