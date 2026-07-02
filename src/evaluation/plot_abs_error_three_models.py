#!/usr/bin/env python3
import sys
from pathlib import Path
import pyarrow.parquet as pq
import pandas as pd
import matplotlib.pyplot as plt

out_dir = Path('freeze_corrected/final_thesis_v1/benchmarks/thesis_ready/figures')
out_dir.mkdir(parents=True, exist_ok=True)

truth = pq.read_table('freeze/final_thesis_v1/phase1_2024daily_final/processed/ground_truth_15min_utc_capnorm.parquet').to_pandas()
base = pq.read_table('freeze/final_thesis_v1/phase1_2024daily_final/processed/predictions_phase1_baseline_rerun.parquet').to_pandas()
heur = pq.read_table('freeze_corrected/final_thesis_v1/phase1_2024daily_final/processed/predictions_phase1_policy_heuristic_rl.parquet').to_pandas()
full = pq.read_table('freeze_corrected/final_thesis_v1/phase1_2024daily_final/processed/predictions_phase1_policy_full_rl.parquet').to_pandas()

for df in (truth, base, heur, full):
    if 'timestamp_utc' in df.columns:
        df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], utc=True)

# join on timestamp_utc selecting most recent per timestamp if multiple hours_ahead present
base_j = base.merge(truth[['timestamp_utc','power_norm']].rename(columns={'power_norm':'y_true'}), on='timestamp_utc')
heur_j = heur.merge(truth[['timestamp_utc','power_norm']].rename(columns={'power_norm':'y_true'}), on='timestamp_utc')
full_j = full.merge(truth[['timestamp_utc','power_norm']].rename(columns={'power_norm':'y_true'}), on='timestamp_utc')

# If multiple rows per timestamp, take smallest hours_ahead
for df in (base_j, heur_j, full_j):
    if 'hours_ahead' in df.columns:
        df.sort_values(['timestamp_utc','hours_ahead'], inplace=True)
        df.drop_duplicates(subset=['timestamp_utc'], keep='first', inplace=True)

base_abs = (base_j['y_true'] - base_j['predicted_power_norm']).abs()
heur_abs = (heur_j['y_true'] - heur_j['predicted_power_norm']).abs()
full_abs = (full_j['y_true'] - full_j['predicted_power_norm']).abs()

plt.figure(figsize=(8,5))
plt.hist(base_abs, bins=80, alpha=0.6, label='MiRACLE Core (Baseline)', density=False)
plt.hist(heur_abs, bins=80, alpha=0.6, label='Heuristic RL', density=False)
plt.hist(full_abs, bins=80, alpha=0.6, label='Full RL', density=False)
plt.xlabel('Absolute Error (Normalized Power)')
plt.ylabel('Count')
plt.title('Absolute Error Distribution: MiRACLE Core vs Heuristic RL vs Full RL')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(out_dir / 'abs_error_miracle_heuristic_full.png', dpi=300)
print('WROTE', out_dir / 'abs_error_miracle_heuristic_full.png')
