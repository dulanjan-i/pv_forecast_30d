#!/usr/bin/env python3
# scripts/annual_aggregates_from_monthly.py
import pandas as pd
from pathlib import Path
import numpy as np

SRC = Path("freeze/final_thesis_v1/phase1_2024daily_final/plots/core_vs_rl_v1_v2[FINAL_CANONICAL]/monthly_rmse_core_vs_rl_v1_v2.csv")
OUT = SRC.parent / "annual_aggregates.csv"

df = pd.read_csv(SRC, parse_dates=["month"])
out_rows = []
for col in ["rmse_core","rmse_v1","rmse_v2"]:
    vals = df[col].to_numpy(dtype=float)
    mean = float(vals.mean())
    rms = float(np.sqrt(np.mean(vals**2)))
    mn = float(vals.min()); mx = float(vals.max())
    out_rows.append({"series":col, "mean_rmse":mean, "rms_agg":rms, "min_rmse":mn, "max_rmse":mx})
pd.DataFrame(out_rows).to_csv(OUT, index=False)
print("[OK] wrote", OUT)