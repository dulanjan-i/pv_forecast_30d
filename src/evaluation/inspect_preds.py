import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
import sys

files={
 "baseline_core":"freeze/final_thesis_v1/phase1_2024daily_final/processed/predictions_phase1_baseline_rerun.parquet",
 "tft_only":"freeze/final_thesis_v1/inference_v3_runs/derived_only/tft_only.parquet",
 "pvlib_only":"freeze/final_thesis_v1/inference_v3_runs/derived_only/pvlib_only.parquet",
}


def pred_col(df):
    for c in ["y_hat","y_pred","pred","power_pred","power_norm_pred","prediction","p_hat"]:
        if c in df.columns:
            return c
    # fallback: first numeric column
    num = [c for c in df.columns if is_numeric_dtype(df[c])]
    return num[0] if num else None


for name,p in files.items():
    try:
        df = pd.read_parquet(p)
    except Exception as e:
        print(f"{name}: failed to read {p}: {e}")
        continue
    if df is None or len(df)==0:
        print(f"{name}: empty dataframe ({p})")
        continue
    c = pred_col(df)
    if c is None:
        print(f"{name}: no numeric prediction column found in {p}")
        continue
    # coerce to numeric safely
    try:
        vals = pd.to_numeric(df[c], errors="coerce")
        if vals.dropna().empty:
            print(f"{name}: column '{c}' exists but contains no numeric values")
            continue
        mx = float(vals.max())
        mn = float(vals.min())
        print(name, "pred_col", c, "min", mn, "max", mx)
    except Exception as e:
        print(f"{name}: error processing column {c}: {e}")
