# src/evaluation/add_solar_daylight_flag.py
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def compute_solar_elevation_deg(times_utc: pd.DatetimeIndex, lat: float, lon: float) -> pd.Series:
    """
    Returns a Series indexed by times_utc with solar elevation in degrees.
    Uses pvlib if available, falls back to astral if pvlib is missing.
    times_utc must be tz-aware (UTC).
    """
    if times_utc.tz is None:
        raise ValueError("times_utc must be timezone-aware (UTC).")

    # Compute only for unique timestamps (fast)
    uniq = pd.DatetimeIndex(pd.unique(times_utc)).sort_values()

    try:
        import pvlib  # type: ignore

        solpos = pvlib.solarposition.get_solarposition(
            time=uniq,
            latitude=lat,
            longitude=lon,
        )
        # Prefer apparent_elevation when present
        col = "apparent_elevation" if "apparent_elevation" in solpos.columns else "elevation"
        elev = solpos[col].astype("float32")
        return pd.Series(elev.values, index=uniq, name="solar_elevation_deg")

    except Exception as e:
        # Fallback: astral (slower, but works)
        try:
            from astral import Observer  # type: ignore
            from astral.sun import elevation as astral_elevation  # type: ignore
        except Exception:
            raise RuntimeError(
                "Could not compute solar position. Install pvlib (preferred) or astral.\n"
                f"pvlib error was: {repr(e)}"
            )

        obs = Observer(latitude=lat, longitude=lon)
        vals = np.empty(len(uniq), dtype=np.float32)
        for i, t in enumerate(uniq):
            # astral expects a python datetime; keep tz info
            vals[i] = float(astral_elevation(observer=obs, dateandtime=t.to_pydatetime()))
        return pd.Series(vals, index=uniq, name="solar_elevation_deg")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-parquet", required=True, type=str, help="Input parquet path")
    ap.add_argument("--out-parquet", required=True, type=str, help="Output parquet path")
    ap.add_argument("--timestamp-col", default="timestamp_utc", type=str, help="Timestamp column (tz-aware UTC)")
    ap.add_argument("--lat", required=True, type=float, help="Plant latitude in decimal degrees")
    ap.add_argument("--lon", required=True, type=float, help="Plant longitude in decimal degrees")
    ap.add_argument(
        "--elev-threshold-deg",
        default=0.0,
        type=float,
        help="Daylight threshold in degrees (0.0 is sun above horizon, 5.0 is stricter)",
    )
    args = ap.parse_args()

    in_path = Path(args.in_parquet)
    out_path = Path(args.out_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Read via pyarrow, convert to pandas (keeps it simple and reliable)
    table = pq.read_table(in_path)
    df = table.to_pandas()

    ts_col = args.timestamp_col
    if ts_col not in df.columns:
        raise KeyError(f"Missing timestamp column '{ts_col}'. Available: {list(df.columns)}")

    ts = pd.to_datetime(df[ts_col], utc=True)
    if ts.dt.tz is None:
        # force UTC tz-aware
        ts = ts.dt.tz_localize("UTC")

    elev_map = compute_solar_elevation_deg(pd.DatetimeIndex(ts), lat=args.lat, lon=args.lon)

    # Map back to all rows
    df["solar_elevation_deg"] = ts.map(elev_map).astype("float32")
    df["is_daylight"] = (df["solar_elevation_deg"] > float(args.elev_threshold_deg))

    # Quick sanity prints
    daylight_frac = float(df["is_daylight"].mean()) if len(df) else 0.0
    print(f"[OK] Wrote solar elevation, daylight fraction = {daylight_frac:.4f}")
    print(
        "[INFO] solar_elevation_deg min/median/max:",
        float(df["solar_elevation_deg"].min()),
        float(df["solar_elevation_deg"].median()),
        float(df["solar_elevation_deg"].max()),
    )

    out_table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(out_table, out_path)
    print(f"[SUCCESS] WROTE: {out_path}")


if __name__ == "__main__":
    main()
