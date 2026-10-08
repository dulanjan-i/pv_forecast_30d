#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

base = Path("freeze/final_thesis_v1/rl/rl_thesis_metrics_rerun_canonical")
out_dir = Path("freeze/final_thesis_v1/rl")
out_dir.mkdir(parents=True, exist_ok=True)

a1 = base / "per_forecast_actions_v1.csv"
a2 = base / "per_forecast_actions_v2.csv"

if not a1.exists() or not a2.exists():
    raise SystemExit("per_forecast_actions_v1.csv or per_forecast_actions_v2.csv not found")

df1 = pd.read_csv(a1)
df2 = pd.read_csv(a2)

# Ensure action column exists
if "action" not in df1.columns or "action" not in df2.columns:
    raise SystemExit("Missing 'action' column in input CSVs")

# Count actions 0..7
actions = list(range(8))
counts1 = df1["action"].value_counts().reindex(actions, fill_value=0).sort_index().to_numpy()
counts2 = df2["action"].value_counts().reindex(actions, fill_value=0).sort_index().to_numpy()

# Plot
x = np.arange(1, 9)  # display as 1..8
width = 0.35
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - width/2, counts1, width, label="v1", color="#add8e6")
ax.bar(x + width/2, counts2, width, label="v2", color="#ff7f0e")

ax.set_xlabel("Action (1..8)")
ax.set_ylabel("Count (forecasts)")
ax.set_title("Per-forecast Action Distribution — v1 vs v2")
ax.set_xticks(x)
ax.set_xticklabels([str(i) for i in x])
ax.legend()
ax.grid(axis="y", alpha=0.25)

out_png = out_dir / "action_counts_hist_v1_v2.png"
fig.tight_layout()
fig.savefig(out_png, dpi=150)
print("WROTE", out_png)