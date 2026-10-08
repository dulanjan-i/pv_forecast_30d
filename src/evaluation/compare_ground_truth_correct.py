#!/usr/bin/env python3
"""
Compare Phase 1 predictions against CORRECT ground truth data.
Using plant_03_ground_truth.csv with actual kW measurements!
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("🎯 GROUND TRUTH VALIDATION: Phase 1 vs Real Measurements (2024)")
print("=" * 80)

# Load actual measurements (pre-aligned to 15-min)
print("\n📂 Loading ground truth data (15-min aligned)...")
ground_truth_15min = pd.read_parquet('data/processed/test_phase1_dec2023_dec2024/ground_truth_15min.parquet')
ground_truth_15min = ground_truth_15min.rename(columns={'timestamp_utc': 'timestamp'})

capacity_kw = 7358.9

print(f"   Loaded: {len(ground_truth_15min):,} timesteps (15-min resolution)")
print(f"   Range: {ground_truth_15min['timestamp'].min()} → {ground_truth_15min['timestamp'].max()}")
print(f"   Max power: {ground_truth_15min['power_kw'].max():.1f} kW")
print(f"   Max normalized: {ground_truth_15min['power_norm'].max():.4f}")

# Load predictions
print("\n📂 Loading Phase 1 predictions...")
predictions = pd.read_parquet('data/processed/test_phase1_dec2023_dec2024/predictions_phase1.parquet')
predictions['timestamp'] = pd.to_datetime(predictions['timestamp_utc'])

print(f"   Loaded: {len(predictions):,} timesteps")
print(f"   Range: {predictions['timestamp'].min()} → {predictions['timestamp'].max()}")

# Merge on timestamp
print("\n🔗 Merging predictions with ground truth...")
merged = pd.merge(
    predictions[['timestamp', 'predicted_power_norm', 'forecast_idx', 'step_ahead']],
    ground_truth_15min[['timestamp', 'power_norm', 'power_kw']],
    on='timestamp',
    how='inner'
)

print(f"   ✅ Matched timesteps: {len(merged):,}")
print(f"   Coverage: {len(merged) / len(predictions) * 100:.1f}% of predictions")
print(f"   Overlap: {merged['timestamp'].min()} → {merged['timestamp'].max()}")

if len(merged) < 100:
    print(f"\n❌ ERROR: Only {len(merged)} overlapping timesteps!")
    exit(1)

# Calculate errors
print("\n" + "=" * 80)
print("📊 ERROR METRICS")
print("=" * 80)

y_true = merged['power_norm'].values
y_pred = merged['predicted_power_norm'].values

# Remove any NaN values
valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
y_true = y_true[valid_mask]
y_pred = y_pred[valid_mask]

print(f"\nValid comparisons: {len(y_true):,} timesteps")

rmse_norm = np.sqrt(mean_squared_error(y_true, y_pred))
mae_norm = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

# Absolute units
rmse_kw = rmse_norm * capacity_kw
mae_kw = mae_norm * capacity_kw

# MAPE (avoid division by zero)
mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100

print(f"\n📉 OVERALL PERFORMANCE:")
print(f"   RMSE (normalized):  {rmse_norm:.5f}  ({rmse_norm * 100:.2f}% of capacity)")
print(f"   RMSE (absolute):    {rmse_kw:.1f} kW")
print(f"   MAE (normalized):   {mae_norm:.5f}  ({mae_norm * 100:.2f}% of capacity)")
print(f"   MAE (absolute):     {mae_kw:.1f} kW")
print(f"   R² Score:           {r2:.4f}")
print(f"   MAPE:               {mape:.2f}%")

# Compare to pseudo-RMSE
pseudo_rmse = 0.00693
print(f"\n📊 COMPARISON TO PSEUDO-RMSE:")
print(f"   Pseudo-RMSE (forecast disagreement): {pseudo_rmse:.5f}")
print(f"   True RMSE (vs ground truth):         {rmse_norm:.5f}")
print(f"   Ratio (True / Pseudo):                {rmse_norm / pseudo_rmse:.2f}×")

if rmse_norm < pseudo_rmse * 2:
    print(f"   ✅ EXCELLENT: Pseudo-RMSE was accurate within 2×!")
elif rmse_norm < pseudo_rmse * 5:
    print(f"   ✓ GOOD: Pseudo-RMSE within 5× of true error")
else:
    print(f"   ⚠️  Pseudo-RMSE significantly underestimated true error")

# Bias analysis
bias = y_pred - y_true
print(f"\n📈 BIAS ANALYSIS:")
print(f"   Mean bias:         {bias.mean():+.5f} ({bias.mean() * 100:+.2f}%)")
print(f"   Median bias:       {np.median(bias):+.5f}")
print(f"   Std dev:           {bias.std():.5f}")
print(f"   Over-predictions:  {(bias > 0).sum():,} / {len(bias):,} ({(bias > 0).mean() * 100:.1f}%)")

if abs(bias.mean()) < 0.01:
    print(f"   ✅ MODEL IS WELL-CALIBRATED (minimal bias)")
elif bias.mean() > 0:
    print(f"   ⚠️  MODEL OVERESTIMATES by {abs(bias.mean() * capacity_kw):.0f} kW on average")
else:
    print(f"   ⚠️  MODEL UNDERESTIMATES by {abs(bias.mean() * capacity_kw):.0f} kW on average")

# Daylight vs nighttime
merged_valid = merged[valid_mask].copy()
merged_valid['hour'] = pd.to_datetime(merged_valid['timestamp']).dt.hour
day_mask = (merged_valid['hour'] >= 6) & (merged_valid['hour'] <= 18)

print(f"\n🌞 DAYLIGHT PERFORMANCE (06:00-18:00):")
if day_mask.sum() > 0:
    y_true_day = merged_valid[day_mask]['power_norm'].values
    y_pred_day = merged_valid[day_mask]['predicted_power_norm'].values
    rmse_day = np.sqrt(mean_squared_error(y_true_day, y_pred_day))
    mae_day = mean_absolute_error(y_true_day, y_pred_day)
    r2_day = r2_score(y_true_day, y_pred_day)
    
    print(f"   Samples: {day_mask.sum():,}")
    print(f"   RMSE: {rmse_day:.5f} ({rmse_day * capacity_kw:.1f} kW)")
    print(f"   MAE:  {mae_day:.5f} ({mae_day * capacity_kw:.1f} kW)")
    print(f"   R²:   {r2_day:.4f}")

# Monthly breakdown
print(f"\n" + "=" * 80)
print("📅 MONTHLY ERROR BREAKDOWN")
print("=" * 80)

merged_valid['month'] = pd.to_datetime(merged_valid['timestamp']).dt.to_period('M')
monthly_stats = []

for month, group in merged_valid.groupby('month'):
    if len(group) < 10:
        continue
    y_t = group['power_norm'].values
    y_p = group['predicted_power_norm'].values
    
    monthly_stats.append({
        'month': str(month),
        'count': len(group),
        'rmse': np.sqrt(mean_squared_error(y_t, y_p)),
        'mae': mean_absolute_error(y_t, y_p),
        'r2': r2_score(y_t, y_p),
        'bias': (y_p - y_t).mean()
    })

monthly_df = pd.DataFrame(monthly_stats)

print(f"\n{'Month':<10} {'Samples':<10} {'RMSE':<10} {'MAE':<10} {'R²':<8} {'Bias':<10}")
print("-" * 70)
for _, row in monthly_df.iterrows():
    print(f"{row['month']:<10} {int(row['count']):<10,} "
          f"{row['rmse']:<10.5f} {row['mae']:<10.5f} "
          f"{row['r2']:<8.4f} {row['bias']:<+10.5f}")

# Best/worst days
print(f"\n📊 DAILY ERROR ANALYSIS:")
merged_valid['date'] = pd.to_datetime(merged_valid['timestamp']).dt.date
daily_errors = []

for date, group in merged_valid.groupby('date'):
    if len(group) < 10:
        continue
    y_t = group['power_norm'].values
    y_p = group['predicted_power_norm'].values
    daily_errors.append({
        'date': date,
        'rmse': np.sqrt(mean_squared_error(y_t, y_p)),
        'samples': len(group)
    })

daily_df = pd.DataFrame(daily_errors).sort_values('rmse')

print(f"\n🏆 BEST 5 DAYS (Lowest RMSE):")
for _, row in daily_df.head(5).iterrows():
    print(f"   {row['date']}: RMSE = {row['rmse']:.5f} ({row['rmse'] * capacity_kw:.1f} kW)")

print(f"\n💥 WORST 5 DAYS (Highest RMSE):")
for _, row in daily_df.tail(5).iterrows():
    print(f"   {row['date']}: RMSE = {row['rmse']:.5f} ({row['rmse'] * capacity_kw:.1f} kW)")

# Final verdict
print(f"\n" + "=" * 80)
print("🎯 FINAL VERDICT")
print("=" * 80)

if rmse_norm < 0.05:
    grade, verdict = "A+ EXCELLENT", "Production-ready performance!"
elif rmse_norm < 0.10:
    grade, verdict = "A VERY GOOD", "Strong performance, deployment-ready"
elif rmse_norm < 0.15:
    grade, verdict = "B GOOD", "Acceptable, room for optimization"
elif rmse_norm < 0.20:
    grade, verdict = "C FAIR", "Usable but needs improvement"
else:
    grade, verdict = "D NEEDS WORK", "Significant errors detected"

print(f"\n   Grade: {grade}")
print(f"   RMSE: {rmse_norm:.5f} ({rmse_norm * 100:.2f}% of capacity) = {rmse_kw:.1f} kW")
print(f"   R²: {r2:.4f}")
print(f"   Verdict: {verdict}")

# Save comparison
output_path = Path("data/processed/test_phase1_dec2023_dec2024/validation_vs_ground_truth.parquet")
merged_valid.to_parquet(output_path, index=False)
print(f"\n✅ Saved validation data to: {output_path}")

print("\n" + "=" * 80)
print("✨ Validation complete! Real ground truth comparison successful!")
print("=" * 80)
