#!/usr/bin/env python3
"""
Analyze Phase 1 prediction quality.

Since we don't have ground truth for Dec 2023 - Dec 2024, we'll use:
1. Forecast disagreement (overlapping windows) as pseudo-RMSE
2. Consistency metrics (temporal smoothness, diurnal patterns)
3. TFT vs PVLib comparison
4. Seasonal trend analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load predictions
print("=" * 70)
print("PHASE 1 PREDICTION QUALITY ANALYSIS")
print("=" * 70)

pred_path = Path("data/processed/test_phase1_dec2023_dec2024/predictions_phase1.parquet")
df = pd.read_parquet(pred_path)

print(f"\n📊 Dataset Overview:")
print(f"   Total predictions: {len(df):,} timesteps")
print(f"   Date range: {df['timestamp_utc'].min()} → {df['timestamp_utc'].max()}")
print(f"   Unique forecasts: {df['forecast_idx'].nunique()}")
print(f"   Power range: [{df['predicted_power_norm'].min():.4f}, {df['predicted_power_norm'].max():.4f}]")

# 1. FORECAST DISAGREEMENT ANALYSIS (Pseudo-RMSE)
print(f"\n" + "="*70)
print("1. FORECAST DISAGREEMENT (Overlapping Windows)")
print("="*70)

# For each timestamp that appears in multiple forecasts, compute std dev
df['date'] = pd.to_datetime(df['timestamp_utc']).dt.date
overlap_stats = df.groupby('timestamp_utc')['predicted_power_norm'].agg(['count', 'mean', 'std', 'min', 'max'])
overlap_stats = overlap_stats[overlap_stats['count'] > 1]  # Only overlapping timestamps

print(f"   Timestamps with multiple forecasts: {len(overlap_stats):,}")
print(f"   Mean forecast disagreement (std): {overlap_stats['std'].mean():.5f}")
print(f"   Median forecast disagreement: {overlap_stats['std'].median():.5f}")
print(f"   Max disagreement: {overlap_stats['std'].max():.5f}")
print(f"   \n   Pseudo-RMSE (mean std across overlaps): {overlap_stats['std'].mean():.5f}")

# Worst disagreements
worst_10 = overlap_stats.nlargest(10, 'std')
print(f"\n   Top 10 worst disagreements:")
for ts, row in worst_10.iterrows():
    print(f"      {ts}: {row['count']:.0f} forecasts, std={row['std']:.4f}, range=[{row['min']:.3f}, {row['max']:.3f}]")

# 2. TEMPORAL CONSISTENCY
print(f"\n" + "="*70)
print("2. TEMPORAL CONSISTENCY (Smoothness)")
print("="*70)

# Check for each forecast: how smooth is the time series?
smoothness_scores = []
for fc_idx in df['forecast_idx'].unique():
    fc_data = df[df['forecast_idx'] == fc_idx].sort_values('timestamp_utc')
    power = fc_data['predicted_power_norm'].values
    
    # Compute first derivative (hour-to-hour change)
    diffs = np.abs(np.diff(power))
    smoothness_scores.append({
        'forecast_idx': fc_idx,
        'mean_abs_change': diffs.mean(),
        'max_abs_change': diffs.max(),
        'spikes_gt_0.1': (diffs > 0.1).sum()
    })

smooth_df = pd.DataFrame(smoothness_scores)
print(f"   Mean absolute 15-min change: {smooth_df['mean_abs_change'].mean():.5f}")
print(f"   Max change across all forecasts: {smooth_df['max_abs_change'].max():.5f}")
print(f"   Forecasts with >10 spikes (Δ>0.1): {(smooth_df['spikes_gt_0.1'] > 10).sum()}/{len(smooth_df)}")

# 3. DIURNAL PATTERN ANALYSIS
print(f"\n" + "="*70)
print("3. DIURNAL PATTERNS (Day/Night Cycles)")
print("="*70)

df['hour'] = pd.to_datetime(df['timestamp_utc']).dt.hour
hourly_mean = df.groupby('hour')['predicted_power_norm'].mean()

night_hours = [0, 1, 2, 3, 4, 5, 20, 21, 22, 23]
day_hours = [8, 9, 10, 11, 12, 13, 14, 15, 16]

night_mean = hourly_mean[night_hours].mean()
day_mean = hourly_mean[day_hours].mean()
peak_hour = hourly_mean.idxmax()
peak_value = hourly_mean.max()

print(f"   Night mean (hrs 0-5, 20-23): {night_mean:.5f}")
print(f"   Day mean (hrs 8-16): {day_mean:.5f}")
print(f"   Peak hour: {peak_hour}:00 with mean power {peak_value:.4f}")
print(f"   Day/Night ratio: {day_mean/night_mean:.1f}x" if night_mean > 0 else "   Day/Night ratio: inf (perfect night zeros)")

# Check for anomalies (high power at night)
night_mask = df['hour'].isin(night_hours)
night_violations = df[night_mask & (df['predicted_power_norm'] > 0.05)]
print(f"   Night violations (power>0.05 during hrs 0-5,20-23): {len(night_violations)} / {night_mask.sum()} ({100*len(night_violations)/night_mask.sum():.2f}%)")

# 4. SEASONAL TRENDS
print(f"\n" + "="*70)
print("4. SEASONAL TRENDS (Monthly Aggregates)")
print("="*70)

df['month'] = pd.to_datetime(df['timestamp_utc']).dt.to_period('M')
monthly = df.groupby('month')['predicted_power_norm'].agg(['mean', 'max', 'std'])

print(f"   Monthly statistics:")
print(f"   {'Month':<12} {'Mean':<10} {'Max':<10} {'Std':<10}")
print(f"   {'-'*42}")
for month, row in monthly.iterrows():
    print(f"   {str(month):<12} {row['mean']:<10.4f} {row['max']:<10.4f} {row['std']:<10.4f}")

winter_months = ['2023-12', '2024-01', '2024-02']
summer_months = ['2024-06', '2024-07', '2024-08']

winter_mean = monthly.loc[monthly.index.astype(str).isin(winter_months), 'mean'].mean()
summer_mean = monthly.loc[monthly.index.astype(str).isin(summer_months), 'mean'].mean()

print(f"\n   Winter (Dec-Feb) mean: {winter_mean:.4f}")
print(f"   Summer (Jun-Aug) mean: {summer_mean:.4f}")
print(f"   Summer/Winter ratio: {summer_mean/winter_mean:.2f}x")

# 5. PHYSICAL PLAUSIBILITY
print(f"\n" + "="*70)
print("5. PHYSICAL PLAUSIBILITY CHECKS")
print("="*70)

neg_count = (df['predicted_power_norm'] < 0).sum()
above_one = (df['predicted_power_norm'] > 1.0).sum()
zeros = (df['predicted_power_norm'] == 0.0).sum()

print(f"   Negative values: {neg_count} / {len(df)} ({100*neg_count/len(df):.2f}%)")
print(f"   Values > 1.0: {above_one} / {len(df)} ({100*above_one/len(df):.2f}%)")
print(f"   Exact zeros: {zeros} / {len(df)} ({100*zeros/len(df):.2f}%)")

# Check ramp rates (physical systems can't change instantly)
max_ramps_per_forecast = []
for fc_idx in df['forecast_idx'].unique():
    fc_data = df[df['forecast_idx'] == fc_idx].sort_values('timestamp_utc')
    power = fc_data['predicted_power_norm'].values
    ramp_rates = np.abs(np.diff(power)) / (15/60)  # per hour
    max_ramps_per_forecast.append(ramp_rates.max())

max_ramp_overall = max(max_ramps_per_forecast)
print(f"   Max ramp rate (15min): {max_ramp_overall:.4f} per hour")
print(f"   Physical limit check: {'✓ PASS' if max_ramp_overall < 2.0 else '✗ FAIL (too fast)'}")

# 6. SUMMARY QUALITY SCORE
print(f"\n" + "="*70)
print("6. OVERALL QUALITY ASSESSMENT")
print("="*70)

quality_checks = {
    'Forecast consistency (std<0.05)': overlap_stats['std'].mean() < 0.05,
    'Temporal smoothness (mean_Δ<0.02)': smooth_df['mean_abs_change'].mean() < 0.02,
    'No negative values': neg_count == 0,
    'No values > 1.0': above_one == 0,
    'Reasonable diurnal pattern': day_mean > 10 * night_mean,
    'Seasonal trend (summer>winter)': summer_mean > winter_mean,
    'Physical ramp rates': max_ramp_overall < 2.0
}

passed = sum(quality_checks.values())
total = len(quality_checks)

print(f"\n   Quality Checks: {passed}/{total} passed")
for check, result in quality_checks.items():
    status = "✓ PASS" if result else "✗ FAIL"
    print(f"      {status} - {check}")

if passed == total:
    print(f"\n   🎉 EXCELLENT: All quality checks passed!")
elif passed >= total * 0.8:
    print(f"\n   ✅ GOOD: {passed}/{total} checks passed")
elif passed >= total * 0.6:
    print(f"\n   ⚠️  FAIR: {passed}/{total} checks passed, review failures")
else:
    print(f"\n   ❌ POOR: Only {passed}/{total} checks passed, investigation needed")

print(f"\n" + "="*70)
print(f"💾 Analysis complete! Check logs for detailed diagnostics.")
print(f"="*70)
