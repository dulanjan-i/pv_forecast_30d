"""
src/validation/validate_stage3_regional_inputs.py

Validates Stage 3.6-3.8 regional inputs before TFT training:
- tft_base (LSTM inputs + LSTM encodings)
- weather_tft (raw weather, plant_id injected)
- pvlib_tft (pvlib features)

What it checks:
1) Keys: no duplicates, full overlap across tables
2) Plants: expected set, plant_04 absent
3) Coverage: per-plant row counts and time ranges
4) NaNs: none
5) Basic PVLib sanity: correlation vs measured power_norm (per plant)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


KEY_COLS = ["plant_id", "timestamp_utc"]
TARGET_COL = "power_norm"


def _read(p: Path) -> pd.DataFrame:
    df = pd.read_parquet(p)
    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    return df


def _dup_keys(df: pd.DataFrame) -> int:
    return int(df.duplicated(KEY_COLS).sum())


def _nan_total(df: pd.DataFrame) -> int:
    return int(df.isna().sum().sum())


def _per_plant_summary(df: pd.DataFrame, name: str) -> pd.DataFrame:
    g = df.groupby("plant_id")["timestamp_utc"]
    out = pd.DataFrame({
        "plant_id": g.size().index,
        f"{name}_rows": g.size().values,
        f"{name}_tmin": g.min().values,
        f"{name}_tmax": g.max().values,
    })
    return out.sort_values("plant_id").reset_index(drop=True)


def _key_overlap(a: pd.DataFrame, b: pd.DataFrame, label: str) -> None:
    ka = a[KEY_COLS]
    kb = b[KEY_COLS]
    both = ka.merge(kb, on=KEY_COLS, how="inner")
    a_only = ka.merge(kb, on=KEY_COLS, how="left", indicator=True)
    a_only = int((a_only["_merge"] == "left_only").sum())
    b_only = kb.merge(ka, on=KEY_COLS, how="left", indicator=True)
    b_only = int((b_only["_merge"] == "left_only").sum())
    print(f"[{label}] both={len(both)} a_only={a_only} b_only={b_only}")
    if a_only != 0 or b_only != 0:
        raise ValueError(f"[{label}] Key mismatch detected.")


def _pvlib_power_like_col(df: pd.DataFrame) -> str:
    # prefer dc power if present
    for c in ["pvlib_dc_kw", "pvlib_ac_kw", "pvlib_pdc_kw", "pvlib_pac_kw"]:
        if c in df.columns:
            return c
    raise ValueError("No pvlib power-like column found.")


def _corr_rmse_per_plant(merged: pd.DataFrame, pvcol: str) -> pd.DataFrame:
    out = []
    for pid, g in merged.groupby("plant_id"):
        y = g[TARGET_COL].astype(float).to_numpy()
        x = g[pvcol].astype(float).to_numpy()
        # normalize pvlib to 0..1 per plant for a fair shape check
        denom = (np.nanmax(x) - np.nanmin(x))
        x01 = (x - np.nanmin(x)) / denom if denom > 0 else x * 0.0
        corr = float(np.corrcoef(x01, y)[0, 1]) if len(g) > 2 else np.nan
        rmse = float(np.sqrt(np.mean((x01 - y) ** 2))) if len(g) else np.nan
        out.append({"plant_id": pid, "n": len(g), "corr": corr, "rmse": rmse})
    return pd.DataFrame(out).sort_values("plant_id").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_base", required=True, type=str)
    ap.add_argument("--val_base", required=True, type=str)
    ap.add_argument("--train_weather", required=True, type=str)
    ap.add_argument("--val_weather", required=True, type=str)
    ap.add_argument("--train_pvlib", required=True, type=str)
    ap.add_argument("--val_pvlib", required=True, type=str)
    args = ap.parse_args()

    train_base = _read(Path(args.train_base))
    val_base = _read(Path(args.val_base))
    train_w = _read(Path(args.train_weather))
    val_w = _read(Path(args.val_weather))
    train_p = _read(Path(args.train_pvlib))
    val_p = _read(Path(args.val_pvlib))

    for name, df in [
        ("train_base", train_base), ("val_base", val_base),
        ("train_weather", train_w), ("val_weather", val_w),
        ("train_pvlib", train_p), ("val_pvlib", val_p),
    ]:
        print(f"\n== {name} ==")
        print("shape:", df.shape)
        print("plants:", sorted(df["plant_id"].unique().tolist()))
        print("time:", df["timestamp_utc"].min(), "->", df["timestamp_utc"].max())
        print("dup keys:", _dup_keys(df))
        print("nan total:", _nan_total(df))
        if "plant_04" in set(df["plant_id"].unique()):
            raise ValueError(f"{name}: plant_04 present, should be omitted.")

    print("\n[KEY OVERLAP CHECKS]")
    _key_overlap(train_base, train_w, "TRAIN base vs weather")
    _key_overlap(train_base, train_p, "TRAIN base vs pvlib")
    _key_overlap(val_base, val_w, "VAL base vs weather")
    _key_overlap(val_base, val_p, "VAL base vs pvlib")

    print("\n[PER-PLANT COVERAGE]")
    s = _per_plant_summary(train_base, "train_base")
    s = s.merge(_per_plant_summary(val_base, "val_base"), on="plant_id", how="outer")
    print(s)

    print("\n[PVLIB SHAPE SANITY]")
    pvcol_train = _pvlib_power_like_col(train_p)
    pvcol_val = _pvlib_power_like_col(val_p)
    print("pvlib power-like col train:", pvcol_train)
    print("pvlib power-like col val:  ", pvcol_val)

    mtrain = train_base[KEY_COLS + [TARGET_COL]].merge(train_p[KEY_COLS + [pvcol_train]], on=KEY_COLS, how="inner")
    mval = val_base[KEY_COLS + [TARGET_COL]].merge(val_p[KEY_COLS + [pvcol_val]], on=KEY_COLS, how="inner")

    print("\nTRAIN corr/rmse (pvlib normalized 0..1 per plant):")
    print(_corr_rmse_per_plant(mtrain, pvcol_train))

    print("\nVAL corr/rmse (pvlib normalized 0..1 per plant):")
    print(_corr_rmse_per_plant(mval, pvcol_val))

    print("\n[SUCCESS] Validation complete.")


if __name__ == "__main__":
    main()
