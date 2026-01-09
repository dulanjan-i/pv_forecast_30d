import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
import sys

TFT = "freeze/final_thesis_v1/inference_v3_runs/derived_only/tft_only.parquet"
SHORT = "freeze/final_thesis_v1/inference_v3_runs/derived_only/short_only.parquet"
LONG = "freeze/final_thesis_v1/inference_v3_runs/derived_only/long_only.parquet"

try:
    tft = pd.read_parquet(TFT)
    sh = pd.read_parquet(SHORT)
    lo = pd.read_parquet(LONG)
except Exception as e:
    print("Failed to read one of the parquet files:", e)
    sys.exit(2)

# try common join keys
keys = [k for k in ["timestamp_utc", "plant_id", "horizon", "lead_time", "target_time"]
        if k in tft.columns and k in sh.columns and k in lo.columns]
if not keys:
    keys = [k for k in ["timestamp_utc", "plant_id"]
            if k in tft.columns and k in sh.columns and k in lo.columns]

print("join keys:", keys)
if not keys:
    print("No common join keys found between the three tables. Exiting.")
    sys.exit(1)

# guess prediction column
def pred_col(df, keys):
    for c in ["y_hat", "y_pred", "pred", "power_pred", "power_norm_pred", "prediction", "p_hat"]:
        if c in df.columns:
            return c
    # fallback: numeric columns excluding keys (use pandas is_numeric_dtype to avoid TZ dtype issues)
    num = [c for c in df.columns if c not in keys and is_numeric_dtype(df[c])]
    return num[0] if num else None

ct = pred_col(tft, keys)
cs = pred_col(sh, keys)
cl = pred_col(lo, keys)
print("pred cols:", ct, cs, cl)

if ct is None or cs is None or cl is None:
    print("Could not auto-detect a numeric prediction column in one of the files:")
    print(f"  tft:{ct} short:{cs} long:{cl}")
    print("Please provide explicit column names.")
    sys.exit(1)

# Ensure selected columns actually exist in the dataframes (defensive)
for name, df, col in [("tft", tft, ct), ("short", sh, cs), ("long", lo, cl)]:
    if col not in df.columns:
        print(f"Expected column '{col}' not found in {name} dataframe. Exiting.")
        sys.exit(1)

# Merge safely: select only keys + pred cols from short/long to avoid duplicate columns
left = tft
right_sh = sh[keys + [cs]].copy()
right_lo = lo[keys + [cl]].copy()

m = left.merge(right_sh, on=keys, how="inner").merge(right_lo, on=keys, how="inner")

# Drop rows with NaNs in the prediction columns
m = m.dropna(subset=[ct, cs, cl])

if len(m) == 0:
    print("No rows after merging and dropping NaNs. Exiting.")
    sys.exit(1)

# Convert to numeric arrays (coerce if needed)
y = pd.to_numeric(m[ct], errors="coerce").to_numpy(dtype=float)
ys = pd.to_numeric(m[cs], errors="coerce").to_numpy(dtype=float)
yl = pd.to_numeric(m[cl], errors="coerce").to_numpy(dtype=float)

# best linear blend y ≈ a*ys + (1-a)*yl
den = np.mean((ys - yl) ** 2)
if den <= 0 or np.isnan(den):
    print("Degenerate denominator when computing optimal blend (ys == yl).")
    a = np.nan
else:
    a = np.mean((y - yl) * (ys - yl)) / den

y_blend = a * ys + (1.0 - a) * yl

rmse = np.sqrt(np.mean((y - y_blend) ** 2))
print("best a:", float(a) if not np.isnan(a) else "nan")
print("RMSE(tft_only vs best blend(short,long)):", float(rmse))
print("max abs diff:", float(np.max(np.abs(y - y_blend))))
