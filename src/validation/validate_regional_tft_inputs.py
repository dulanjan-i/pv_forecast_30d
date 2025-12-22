"""
src/validation/validate_regional_tft_inputs.py

One-shot validator for the regional TFT training inputs.

What it checks (train and val):
- Files exist
- No duplicate keys on (plant_id, timestamp_utc)
- No NaNs
- Plant set matches and plant_04 is absent
- Time ranges are sane
- Merge completeness: base ⋈ weather ⋈ pvlib has 100% key match
- Column collisions: no duplicated columns after merge, and expected feature groups exist
- Basic PVLib sanity: pvlib_dc_kw is non-negative and correlates with measured power_norm per plant

Usage:
  python src/validation/validate_regional_tft_inputs.py
"""

from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

TIME_COL = "timestamp_utc"
PLANT_COL = "plant_id"
KEY = [PLANT_COL, TIME_COL]

BASE_DIR = REPO_ROOT / "data" / "processed" / "pretraining" / "germany" / "global"
BASE_IN = BASE_DIR / "tft_inputs"
WEATHER_IN = BASE_DIR / "weather_tft"
PVLIB_IN = BASE_DIR / "pvlib_tft"
OUT_DIR = BASE_DIR / "validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = {
    "train": {
        "base": BASE_IN / "regional_train_tft_base.parquet",
        "weather": WEATHER_IN / "regional_train_weather_tft.parquet",
        "pvlib": PVLIB_IN / "regional_train_pvlib_tft.parquet",
    },
    "val": {
        "base": BASE_IN / "regional_val_tft_base.parquet",
        "weather": WEATHER_IN / "regional_val_weather_tft.parquet",
        "pvlib": PVLIB_IN / "regional_val_pvlib_tft.parquet",
    },
}

PVLIB_POWER_COL = "pvlib_dc_kw"  # adjust if you renamed


def must_exist(p: Path) -> None:
    if not p.exists():
        raise FileNotFoundError(str(p))


def basic_df_checks(name: str, df: pd.DataFrame) -> dict:
    out = {}
    out["rows"] = int(len(df))
    out["cols"] = int(df.shape[1])
    out["time_min"] = str(pd.to_datetime(df[TIME_COL], utc=True).min())
    out["time_max"] = str(pd.to_datetime(df[TIME_COL], utc=True).max())
    plants = sorted(df[PLANT_COL].astype(str).unique().tolist())
    out["plants"] = plants
    out["has_plant_04"] = "plant_04" in plants

    dup = int(df.duplicated(KEY).sum())
    out["dup_keys"] = dup

    nan = int(df.isna().sum().sum())
    out["nan_total"] = nan
    return out


def merge_and_check(split: str, base: pd.DataFrame, w: pd.DataFrame, p: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report = {}

    # Ensure key dtypes consistent
    for d in (base, w, p):
        d[TIME_COL] = pd.to_datetime(d[TIME_COL], utc=True)
        d[PLANT_COL] = d[PLANT_COL].astype(str)

    # Check collisions before merge
    base_cols = set(base.columns)
    w_cols = set(w.columns) - set(KEY)
    p_cols = set(p.columns) - set(KEY)

    overlap_bw = sorted((base_cols & w_cols) - set(KEY))
    overlap_bp = sorted((base_cols & p_cols) - set(KEY))
    overlap_wp = sorted((w_cols & p_cols) - set(KEY))

    report["overlap_base_weather"] = overlap_bw
    report["overlap_base_pvlib"] = overlap_bp
    report["overlap_weather_pvlib"] = overlap_wp

    # Merge (suffixes should never be used if overlaps are empty)
    m = base.merge(w, on=KEY, how="inner", validate="one_to_one")
    m = m.merge(p, on=KEY, how="inner", validate="one_to_one")

    report["merged_rows"] = int(len(m))
    report["merged_cols"] = int(m.shape[1])

    # Completeness checks
    base_keys = base[KEY].copy()
    w_keys = w[KEY].copy()
    p_keys = p[KEY].copy()
    base_set = set(map(tuple, base_keys.to_numpy()))
    w_set = set(map(tuple, w_keys.to_numpy()))
    p_set = set(map(tuple, p_keys.to_numpy()))
    m_set = set(map(tuple, m[KEY].to_numpy()))

    report["keys_base_only"] = int(len(base_set - m_set))
    report["keys_weather_only"] = int(len(w_set - m_set))
    report["keys_pvlib_only"] = int(len(p_set - m_set))

    report["merged_dup_keys"] = int(m.duplicated(KEY).sum())
    report["merged_nan_total"] = int(m.isna().sum().sum())

    # PVLib sanity
    if PVLIB_POWER_COL in m.columns:
        x = m[PVLIB_POWER_COL].astype(float).to_numpy()
        report["pvlib_min"] = float(np.nanmin(x))
        report["pvlib_max"] = float(np.nanmax(x))
        report["pvlib_negative_rows"] = int((x < -1e-6).sum())
    else:
        report["pvlib_power_col_missing"] = PVLIB_POWER_COL

    return m, report


def pvlib_corr_per_plant(m: pd.DataFrame) -> pd.DataFrame:
    if PVLIB_POWER_COL not in m.columns or "power_norm" not in m.columns:
        return pd.DataFrame()

    out = []
    for pid, g in m.groupby(PLANT_COL):
        a = g["power_norm"].astype(float).to_numpy()
        b = g[PVLIB_POWER_COL].astype(float).to_numpy()

        # normalize pvlib per plant to 0..1 for correlation sanity
        bmax = np.nanmax(b) if len(b) else np.nan
        bn = b / bmax if np.isfinite(bmax) and bmax > 0 else np.zeros_like(b)

        corr = np.corrcoef(a, bn)[0, 1] if len(a) > 10 else np.nan
        rmse = float(np.sqrt(np.mean((a - bn) ** 2))) if len(a) else np.nan

        out.append({"plant_id": pid, "n": int(len(g)), "corr": corr, "rmse": rmse})

    return pd.DataFrame(out).sort_values("plant_id")


def main() -> None:
    all_reports = {}

    for split, paths in SPLITS.items():
        for p in paths.values():
            must_exist(p)

        base = pd.read_parquet(paths["base"])
        w = pd.read_parquet(paths["weather"])
        p = pd.read_parquet(paths["pvlib"])

        rep = {
            "base": basic_df_checks(f"{split}_base", base),
            "weather": basic_df_checks(f"{split}_weather", w),
            "pvlib": basic_df_checks(f"{split}_pvlib", p),
        }

        merged, mrep = merge_and_check(split, base, w, p)
        rep["merge"] = mrep

        # Hard assertions that should be true
        assert rep["base"]["dup_keys"] == 0
        assert rep["weather"]["dup_keys"] == 0
        assert rep["pvlib"]["dup_keys"] == 0
        assert rep["merge"]["merged_dup_keys"] == 0
        assert rep["merge"]["merged_nan_total"] == 0
        assert rep["base"]["has_plant_04"] is False
        assert rep["weather"]["has_plant_04"] is False
        assert rep["pvlib"]["has_plant_04"] is False
        assert rep["merge"]["keys_base_only"] == 0
        assert rep["merge"]["keys_weather_only"] == 0
        assert rep["merge"]["keys_pvlib_only"] == 0

        corr_df = pvlib_corr_per_plant(merged)
        corr_path = OUT_DIR / f"{split}_pvlib_corr.csv"
        corr_df.to_csv(corr_path, index=False)

        all_reports[split] = rep
        print(f"[OK] {split}: merged rows={len(merged):,} cols={merged.shape[1]} corr_csv={corr_path}")

    report_path = OUT_DIR / "regional_tft_inputs_report.json"
    import json
    with open(report_path, "w") as f:
        json.dump(all_reports, f, indent=2)
    print(f"[DONE] Wrote report: {report_path}")


if __name__ == "__main__":
    main()
