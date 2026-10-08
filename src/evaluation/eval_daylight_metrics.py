# src/evaluation/eval_daylight_metrics.py
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def _must_have(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"{name} is missing columns: {missing}\nColumns present: {list(df.columns)}")


def _pick_gt_power_col(df: pd.DataFrame) -> str:
    # Prefer user's guess, then common alternatives
    for c in ["power_norm", "measured_power_norm", "gt_power_norm", "target_power_norm", "power"]:
        if c in df.columns:
            return c
    raise SystemExit(
        "Could not find GT power column. Expected one of: "
        "power_norm, measured_power_norm, gt_power_norm, target_power_norm, power\n"
        f"Columns present: {list(df.columns)}"
    )


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = (y_pred - y_true).astype(np.float64)
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    mbe = float(np.mean(err))
    return {"MAE": mae, "RMSE": rmse, "MBE": mbe}


def _bucket(hours_ahead: pd.Series) -> pd.Series:
    # Match what you printed earlier
    h = hours_ahead.astype(np.float64)
    out = np.full(len(h), "8-30d", dtype=object)
    out[h <= 24.0] = "0-24h"
    out[(h > 24.0) & (h <= 7.0 * 24.0)] = "2-7d"
    return pd.Series(out, index=hours_ahead.index)


def score_one(
    name: str,
    pred_df: pd.DataFrame,
    gt_sun_df: pd.DataFrame,
    pred_col: str,
    gt_col: str,
    daylight_only: bool,
    capacity_kw: float | None,
    out_dir: Path,
) -> pd.DataFrame:
    _must_have(pred_df, ["timestamp_utc", pred_col], f"{name} predictions")
    _must_have(gt_sun_df, ["timestamp_utc", gt_col, "solar_elevation_deg"], "GT-with-sun")

    df = pred_df.merge(
        gt_sun_df[["timestamp_utc", gt_col, "solar_elevation_deg"]],
        on="timestamp_utc",
        how="inner",
    )

    df["is_daylight"] = df["solar_elevation_deg"] > 0.0
    if "hours_ahead" in df.columns:
        df["bucket"] = _bucket(df["hours_ahead"])
    else:
        df["bucket"] = "all"

    if daylight_only:
        df = df[df["is_daylight"]].copy()

    if len(df) == 0:
        raise SystemExit(f"{name}: zero rows after merge/filter. Something is off.")

    y_true = df[gt_col].to_numpy(dtype=np.float64)
    y_pred = df[pred_col].to_numpy(dtype=np.float64)

    # Overall
    rows = []
    m = _compute_metrics(y_true, y_pred)
    rows.append({"model": name, "scope": "daylight" if daylight_only else "all", "group": "overall", **m, "n": len(df)})

    # Buckets
    if "hours_ahead" in df.columns:
        for b, g in df.groupby("bucket", sort=False):
            mb = _compute_metrics(g[gt_col].to_numpy(dtype=np.float64), g[pred_col].to_numpy(dtype=np.float64))
            rows.append({"model": name, "scope": "daylight" if daylight_only else "all", "group": f"bucket:{b}", **mb, "n": len(g)})

    # Monthly RMSE
    # Keep timezone safe by using .dt.tz_convert(None) for Period conversion
    ts = pd.to_datetime(df["timestamp_utc"], utc=True)
    month = ts.dt.tz_convert(None).dt.to_period("M").astype(str)
    df["_month"] = month
    for mo, g in df.groupby("_month", sort=True):
        mm = _compute_metrics(g[gt_col].to_numpy(dtype=np.float64), g[pred_col].to_numpy(dtype=np.float64))
        rows.append({"model": name, "scope": "daylight" if daylight_only else "all", "group": f"month:{mo}", **mm, "n": len(g)})

    out = pd.DataFrame(rows)

    # Convert to kW and annual kWh-style error totals if capacity is known
    if capacity_kw is not None:
        out["MAE_kW"] = out["MAE"] * capacity_kw
        out["RMSE_kW"] = out["RMSE"] * capacity_kw
        out["MBE_kW"] = out["MBE"] * capacity_kw

        # For overall only, also compute total absolute energy error across all samples (15-min => 0.25h)
        # This is a useful "scale" number, not a perfect operational KPI.
        abs_err_kwh_total = float(np.sum(np.abs(y_pred - y_true) * capacity_kw * 0.25))
        err_kwh_total = float(np.sum((y_pred - y_true) * capacity_kw * 0.25))
        out.loc[out["group"] == "overall", "abs_err_total_kWh"] = abs_err_kwh_total
        out.loc[out["group"] == "overall", "signed_err_total_kWh"] = err_kwh_total

    # Save a per-model scored table for traceability
    tag = "daylight" if daylight_only else "all"
    out_path = out_dir / f"metrics_{name.lower()}_{tag}.csv"
    out.to_csv(out_path, index=False)
    print(f"[OK] WROTE {out_path}  (rows={len(out)})")

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-sun", required=True, help="GT parquet that includes solar_elevation_deg (and power_norm).")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--pred-col", default="predicted_power_norm")
    ap.add_argument("--gt-col", default=None, help="If omitted, auto-detect (power_norm etc).")
    ap.add_argument("--capacity-kw", type=float, default=None)
    ap.add_argument("--out-dir", default="freeze/final_thesis_v1/phase1_2024daily_final/processed/eval_daylight")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Reading parquets...")
    gt = pd.read_parquet(args.gt_sun, engine="pyarrow")
    base = pd.read_parquet(args.baseline, engine="pyarrow")
    pol = pd.read_parquet(args.policy, engine="pyarrow")

    gt_col = args.gt_col or _pick_gt_power_col(gt)
    print(f"[INFO] Using GT col: {gt_col}")
    print(f"[INFO] Using pred col: {args.pred_col}")

    # Score: all samples
    a_base = score_one("BASELINE", base, gt, args.pred_col, gt_col, False, args.capacity_kw, out_dir)
    a_pol = score_one("POLICY", pol, gt, args.pred_col, gt_col, False, args.capacity_kw, out_dir)

    # Score: daylight only
    d_base = score_one("BASELINE", base, gt, args.pred_col, gt_col, True, args.capacity_kw, out_dir)
    d_pol = score_one("POLICY", pol, gt, args.pred_col, gt_col, True, args.capacity_kw, out_dir)

    # Quick policy-minus-baseline deltas for the OVERALL row
    def overall(df: pd.DataFrame) -> pd.Series:
        return df[df["group"] == "overall"].iloc[0]

    print("\n==================== POLICY - BASELINE (overall) ====================")

    fields = [
        "MAE", "RMSE", "MBE",
        "MAE_kW", "RMSE_kW", "MBE_kW",
        "abs_err_total_kWh", "signed_err_total_kWh",
    ]

    def delta_overall(pol_df: pd.DataFrame, base_df: pd.DataFrame) -> pd.Series:
        p = overall(pol_df)
        b = overall(base_df)

        out = {}
        for f in fields:
            if f in p.index and f in b.index and pd.notna(p[f]) and pd.notna(b[f]):
                out[f] = float(p[f]) - float(b[f])
        return pd.Series(out)

    oa = delta_overall(a_pol, a_base)
    od = delta_overall(d_pol, d_base)

    for scope, s in [("ALL", oa), ("DAYLIGHT", od)]:
        print(f"\n[{scope}]")
        for f in fields:
            if f in s.index and pd.notna(s[f]):
                print(f"{f:>18s}: {float(s[f]): .6f}")

if __name__ == "__main__":
    main()
