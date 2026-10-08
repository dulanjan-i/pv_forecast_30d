#!/usr/bin/env python3
"""
Create a facet-style absolute-error histogram (one panel per model)
for MiRACLE Core, Heuristic RL, and Full RL using the thesis color scheme
and a tuned x-axis limit to better show tails (x_max=0.6).
"""
from pathlib import Path
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

OUT = Path('freeze_corrected/final_thesis_v1/benchmarks/thesis_ready/figures')
OUT.mkdir(parents=True, exist_ok=True)

# Paths
TRUTH = Path('freeze/final_thesis_v1/phase1_2024daily_final/processed/ground_truth_15min_utc_capnorm.parquet')
BASE = Path('freeze/final_thesis_v1/phase1_2024daily_final/processed/predictions_phase1_baseline_rerun.parquet')
HEUR = Path('freeze_corrected/final_thesis_v1/phase1_2024daily_final/processed/predictions_phase1_policy_heuristic_rl.parquet')
FULL = Path('freeze_corrected/final_thesis_v1/phase1_2024daily_final/processed/predictions_phase1_policy_full_rl.parquet')

COL_TRUTH = '#888888'
COL_CORE = '#00AA00'
COL_HEUR = '#6BA3D8'
COL_FULL = '#FAA43A'

# read
truth = pq.read_table(TRUTH.as_posix()).to_pandas()
base = pq.read_table(BASE.as_posix()).to_pandas()
heur = pq.read_table(HEUR.as_posix()).to_pandas()
full = pq.read_table(FULL.as_posix()).to_pandas()

for df in (truth, base, heur, full):
    if 'timestamp_utc' in df.columns:
        df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], utc=True)

# helper most recent per timestamp
def stitched(df, truth_df):
    # pick smallest hours_ahead per timestamp
    if 'hours_ahead' in df.columns:
        df = df.sort_values(['timestamp_utc','hours_ahead']).groupby('timestamp_utc', as_index=False).first()
    j = df.merge(truth_df[['timestamp_utc','power_norm']].rename(columns={'power_norm':'y_true'}), on='timestamp_utc', how='inner')
    return j

truth_df = truth.copy()
base_j = stitched(base, truth_df)
heur_j = stitched(heur, truth_df)
full_j = stitched(full, truth_df)

# compute abs errors
base_abs = (base_j['y_true'] - base_j['predicted_power_norm']).abs()
heur_abs = (heur_j['y_true'] - heur_j['predicted_power_norm']).abs()
full_abs = (full_j['y_true'] - full_j['predicted_power_norm']).abs()

# Determine x_max from tails but cap at 0.6 for better visibility
p95_vals = [np.quantile(arr[~np.isnan(arr)], 0.95) for arr in (base_abs.values, heur_abs.values, full_abs.values)]
raw_max = max(p95_vals) * 1.1
x_max = min(0.6, max(0.3, raw_max))

bins = np.linspace(0, x_max, 80)

# Facet: 1 row, 3 cols
fig, axes = plt.subplots(1, 3, figsize=(15,4.5), squeeze=False)
fig.suptitle('Absolute Error Histogram (Facets) — MiRACLE Core vs Heuristic RL vs Full RL', fontsize=14, fontweight='bold')

# Panel 1: Heuristic (comparison, behind)
ax = axes[0,0]
ax.hist(heur_abs.clip(0, x_max), bins=bins, color=COL_HEUR, alpha=0.6, edgecolor='black', linewidth=0.5)
ax.hist(base_abs.clip(0, x_max), bins=bins, color=COL_CORE, alpha=0.7, edgecolor='black', linewidth=0.5)
ax.set_title('Heuristic RL')
ax.set_xlabel('Abs error (clipped)')
ax.set_ylabel('Count')
ax.grid(True, alpha=0.3, linestyle=':')
ax.set_xlim(0, x_max)

# Panel 2: Full RL
ax = axes[0,1]
ax.hist(full_abs.clip(0, x_max), bins=bins, color=COL_FULL, alpha=0.6, edgecolor='black', linewidth=0.5)
ax.hist(base_abs.clip(0, x_max), bins=bins, color=COL_CORE, alpha=0.7, edgecolor='black', linewidth=0.5)
ax.set_title('Full RL')
ax.set_xlabel('Abs error (clipped)')
ax.set_ylabel('Count')
ax.grid(True, alpha=0.3, linestyle=':')
ax.set_xlim(0, x_max)

# Panel 3: MiRACLE Core (baseline vs itself for consistency)
ax = axes[0,2]
ax.hist(base_abs.clip(0, x_max), bins=bins, color=COL_CORE, alpha=0.7, edgecolor='black', linewidth=0.5)
ax.set_title('MiRACLE v1.0 Core')
ax.set_xlabel('Abs error (clipped)')
ax.set_ylabel('Count')
ax.grid(True, alpha=0.3, linestyle=':')
ax.set_xlim(0, x_max)

plt.tight_layout(rect=(0,0,1,0.95))
out = OUT / 'facets_abs_error_hist_rl_vs_core.png'
plt.savefig(out.as_posix(), dpi=300, bbox_inches='tight')
plt.close()
print('WROTE', out)
