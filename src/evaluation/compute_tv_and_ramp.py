#!/usr/bin/env python3
# scripts/compute_tv_and_ramp.py
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path.cwd()
PHASE_PROC = ROOT / "freeze" / "final_thesis_v1" / "phase1_2024daily_final" / "processed"
OUT = PHASE_PROC.parent / "metrics"
OUT.mkdir(parents=True, exist_ok=True)

TRUTH_P = PHASE_PROC / "ground_truth_15min_utc_capnorm_with_sun.parquet"
PRED_BASE = PHASE_PROC / "predictions_phase1_baseline_rerun.parquet"
PRED_V1 = PHASE_PROC / "predictions_phase1_policy_minenv_v1.parquet"
PRED_V2 = PHASE_PROC / "predictions_phase1_policy_minenv_v2.parquet"

def read_parquet(p):
    return pq.read_table(p.as_posix()).to_pandas()

def safe_rmse(a,b):
    a,b = np.asarray(a), np.asarray(b)
    m = np.isfinite(a)&np.isfinite(b)
    if not m.any(): return float("nan")
    return float(np.sqrt(np.mean((a[m]-b[m])**2)))

def tv(a):
    a = np.asarray(a, dtype=float)
    if a.size<2: return 0.0
    return float(np.mean(np.abs(np.diff(a))))

def per_forecast_shape(pred_df, truth_df, pred_col="predicted_power_norm", truth_col="power_norm"):
    for df in (pred_df, truth_df):
        if "timestamp_utc" in df.columns:
            df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
        if "forecast_start" in df.columns:
            df["forecast_start"] = pd.to_datetime(df["forecast_start"], utc=True)
    merged = pred_df.merge(truth_df, on="timestamp_utc", how="inner", suffixes=("_pred","_truth"))
    if "is_daylight" in truth_df.columns:
        merged["is_daylight"] = merged["is_daylight"]
    elif "sun_elev" in truth_df.columns:
        merged["is_daylight"] = merged["sun_elev"] > 0
    else:
        merged["is_daylight"] = merged[truth_col] > 0

    rows=[]
    for fs, grp in merged.groupby("forecast_start"):
        y = grp[truth_col].to_numpy(dtype=float)
        yhat = grp[pred_col].to_numpy(dtype=float)
        mask = grp["is_daylight"].to_numpy(bool)
        if mask.size != y.size:
            mask = np.ones_like(y, dtype=bool)
        y_dl = y[mask]; yhat_dl = yhat[mask]
        tv_y = tv(y_dl); tv_yhat = tv(yhat_dl)
        tv_deficit = max(0.0, tv_y - tv_yhat)
        ramp_rmse = safe_rmse(np.diff(yhat_dl) if yhat_dl.size>=2 else [], np.diff(y_dl) if y_dl.size>=2 else [])
        rows.append({
            "forecast_start": fs, "tv_true": tv_y, "tv_pred": tv_yhat,
            "tv_deficit": tv_deficit, "ramp_rmse": ramp_rmse,
            "n_timesteps": int(y.size), "n_daylight": int(y_dl.size)
        })
    return pd.DataFrame(rows)

def aggregate(df):
    return {
        "count_forecasts": int(len(df)),
        "mean_tv_deficit": float(df["tv_deficit"].mean()),
        "median_tv_deficit": float(df["tv_deficit"].median()),
        "mean_ramp_rmse": float(df["ramp_rmse"].mean()),
        "median_ramp_rmse": float(df["ramp_rmse"].median()),
    }

def main():
    truth = read_parquet(TRUTH_P)
    base = read_parquet(PRED_BASE)
    v1 = read_parquet(PRED_V1)
    v2 = read_parquet(PRED_V2)

    pf_base = per_forecast_shape(base, truth)
    pf_v1 = per_forecast_shape(v1, truth)
    pf_v2 = per_forecast_shape(v2, truth)

    pf_base.to_csv(OUT / "per_forecast_shape_base.csv", index=False)
    pf_v1.to_csv(OUT / "per_forecast_shape_v1.csv", index=False)
    pf_v2.to_csv(OUT / "per_forecast_shape_v2.csv", index=False)

    s_base = aggregate(pf_base); s_base["mode"]="baseline"
    s_v1 = aggregate(pf_v1); s_v1["mode"]="v1"
    s_v2 = aggregate(pf_v2); s_v2["mode"]="v2"
    pd.DataFrame([s_base,s_v1,s_v2]).to_csv(OUT / "shape_metrics_summary.csv", index=False)
    print("[OK] wrote per-forecast shape CSVs and summary to", OUT)

if __name__=="__main__":
    main()
    