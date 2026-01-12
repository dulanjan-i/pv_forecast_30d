#!/usr/bin/env python3
"""
Produce two thesis-styled plots using the freeze color mapping:
  - abs_error_miracle_heuristic_full_thesis.png (histogram of absolute errors)
  - summer_week_miracle_heuristic_full_timeseries.png (time series for summer week)

Uses:
  - Ground truth: freeze/final_thesis_v1/phase1_2024daily_final/processed/ground_truth_15min_utc_capnorm.parquet
  - Baseline (MiRACLE Core): freeze/.../predictions_phase1_baseline_rerun.parquet
  - Heuristic RL: freeze_corrected/.../predictions_phase1_policy_heuristic_rl.parquet
  - Full RL: freeze_corrected/.../predictions_phase1_policy_full_rl.parquet

Applies color scheme from THESIS_COLOR_SCHEME_APPLIED.md
"""
from pathlib import Path
import pyarrow.parquet as pq
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Paths
ROOT = Path('freeze_corrected/final_thesis_v1')
TRUTH = Path('freeze/final_thesis_v1/phase1_2024daily_final/processed/ground_truth_15min_utc_capnorm.parquet')
BASE = Path('freeze/final_thesis_v1/phase1_2024daily_final/processed/predictions_phase1_baseline_rerun.parquet')
HEUR = ROOT / 'phase1_2024daily_final/processed/predictions_phase1_policy_heuristic_rl.parquet'
FULL = ROOT / 'phase1_2024daily_final/processed/predictions_phase1_policy_full_rl.parquet'
OUT_DIR = ROOT / 'benchmarks/thesis_ready/figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Colors per THESIS_COLOR_SCHEME_APPLIED.md
COL_TRUTH = '#888888'      # light grey
COL_CORE = '#00AA00'       # bold green
COL_HEUR = '#6BA3D8'       # light blue (comparison 1)
COL_FULL = '#FAA43A'       # orange (comparison 2 from palette)

DPI = 300

# read data
truth = pq.read_table(TRUTH.as_posix()).to_pandas()
base = pq.read_table(BASE.as_posix()).to_pandas()
heur = pq.read_table(HEUR.as_posix()).to_pandas()
full = pq.read_table(FULL.as_posix()).to_pandas()

for df in (truth, base, heur, full):
    if 'timestamp_utc' in df.columns:
        df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], utc=True)

# helper to pick most recent per timestamp (smallest hours_ahead)
def pick_most_recent(df):
    if 'hours_ahead' in df.columns:
        df = df.sort_values(['timestamp_utc','hours_ahead'])
        df = df.groupby('timestamp_utc', as_index=False).first()
    return df

base_s = pick_most_recent(base)
heur_s = pick_most_recent(heur)
full_s = pick_most_recent(full)

# merge with truth
truth_small = truth[['timestamp_utc','power_norm']].rename(columns={'power_norm':'y_true'})
base_j = base_s.merge(truth_small, on='timestamp_utc', how='inner')
heur_j = heur_s.merge(truth_small, on='timestamp_utc', how='inner')
full_j = full_s.merge(truth_small, on='timestamp_utc', how='inner')

# ------------------- Abs error histogram (thesis style) -------------------
base_abs = np.abs(base_j['y_true'] - base_j['predicted_power_norm'])
heur_abs = np.abs(heur_j['y_true'] - heur_j['predicted_power_norm'])
full_abs = np.abs(full_j['y_true'] - full_j['predicted_power_norm'])

plt.figure(figsize=(7.5,4.5), dpi=DPI)
max_abs = 1.0
bins = np.linspace(0, max_abs, 80)
# Plot comparison models first (behind) and MiRACLE Core on top, matching thesis style
plt.hist(heur_abs.clip(0, max_abs), bins=bins, color=COL_HEUR, alpha=0.6, label='Heuristic RL',
         edgecolor='black', linewidth=0.5)
plt.hist(full_abs.clip(0, max_abs), bins=bins, color=COL_FULL, alpha=0.6, label='Full RL',
         edgecolor='black', linewidth=0.5)
plt.hist(base_abs.clip(0, max_abs), bins=bins, color=COL_CORE, alpha=0.7, label='MiRACLE v1.0 Core',
         edgecolor='black', linewidth=0.5)
plt.xlabel('Absolute Error (Normalized Power)')
plt.ylabel('Count')
plt.title('Absolute Error Distribution — MiRACLE Core vs Heuristic RL vs Full RL', fontsize=12)
plt.grid(alpha=0.3, linestyle=':', zorder=0)
plt.legend(framealpha=0.9)
plt.tight_layout()
out1 = OUT_DIR / 'abs_error_miracle_heuristic_full_thesis.png'
plt.savefig(out1.as_posix(), dpi=DPI)
plt.close()
print('WROTE', out1)

# ------------------- Summer week time series (one plot) -------------------
# Use the same summer week window as pipeline
start = pd.to_datetime('2024-07-01T00:00:00Z')
end = pd.to_datetime('2024-07-08T00:00:00Z')

# Prepare stitched series: ensure same timestamps and align
def make_series(df_j, col='predicted_power_norm'):
    s = df_j[['timestamp_utc', col, 'y_true']].copy()
    s = s.dropna(subset=['timestamp_utc'])
    s = s[(s['timestamp_utc'] >= start) & (s['timestamp_utc'] <= end)]
    return s.sort_values('timestamp_utc')

base_ss = make_series(base_j, 'predicted_power_norm')
heur_ss = make_series(heur_j, 'predicted_power_norm')
full_ss = make_series(full_j, 'predicted_power_norm')

# Find shared timestamps (inner join on timestamp_utc for plotting alignment)
common_ts = set(base_ss['timestamp_utc']).intersection(heur_ss['timestamp_utc']).intersection(full_ss['timestamp_utc']).intersection(truth_small['timestamp_utc'])
common_ts = sorted(list(common_ts))

if not common_ts:
    # Fall back to using base timestamps
    common_ts = base_ss['timestamp_utc'].tolist()

# index by timestamp
base_ss = base_ss.set_index('timestamp_utc').reindex(common_ts).reset_index()
heur_ss = heur_ss.set_index('timestamp_utc').reindex(common_ts).reset_index()
full_ss = full_ss.set_index('timestamp_utc').reindex(common_ts).reset_index()
truth_week = truth_small.set_index('timestamp_utc').reindex(common_ts).reset_index()

plt.figure(figsize=(11,3.5), dpi=DPI)
# Ground truth subtle
plt.plot(truth_week['timestamp_utc'], truth_week['y_true'], color=COL_TRUTH, linewidth=1.5, alpha=0.7, label='Ground Truth')
# MiRACLE core highlighted
plt.plot(base_ss['timestamp_utc'], base_ss['predicted_power_norm'], color=COL_CORE, linewidth=2.5, alpha=1.0, linestyle='-', label='MiRACLE v1.0 Core')
# Heuristic RL
plt.plot(heur_ss['timestamp_utc'], heur_ss['predicted_power_norm'], color=COL_HEUR, linewidth=1.5, alpha=0.95, label='Heuristic RL')
# Full RL
plt.plot(full_ss['timestamp_utc'], full_ss['predicted_power_norm'], color=COL_FULL, linewidth=1.5, alpha=0.95, label='Full RL')

plt.ylabel('Normalized Power')
plt.title('Summer Week: MiRACLE Core vs Heuristic RL vs Full RL (2024-07-01 → 2024-07-08)', fontsize=12)
plt.grid(alpha=0.3, linestyle=':')
plt.legend(framealpha=0.9, ncol=2)
plt.tight_layout()
out2 = OUT_DIR / 'summer_week_miracle_heuristic_full_timeseries_thesis.png'
plt.savefig(out2.as_posix(), dpi=DPI)
plt.close()
print('WROTE', out2)
