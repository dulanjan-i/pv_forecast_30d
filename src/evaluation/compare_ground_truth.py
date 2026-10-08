#!/usr/bin/env python3
"""
Compare Phase 1 predictions against actual ground truth data.
THE BIG REVEAL!
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("=" * 80)
print("🎯 GROUND TRUTH VALIDATION: Phase 1 Predictions vs Real Measurements")
print("=" * 80)

# Load actual measurements
print("\n📂 Loading ground truth data...")
df_2023 = pd.read_csv(
    'data/raw/germany/plant_03/Data_SP_ID_3_GRB_2023.csv',
    sep=';',
    decimal=',',
    names=['timestamp', 'power_kw'],
    skiprows=1,
    parse_dates=['timestamp']
)

df_2024 = pd.read_csv(
    'data/raw/germany/plant_03/Data_SP_ID_3_GRB_2024.csv',
    sep=';',
    decimal=',',
    names=['timestamp', 'power_kw'],
    skiprows=1,
    parse_dates=['timestamp']
)

# Combine
ground_truth = pd.concat([df_2023, df_2024], ignore_index=True)
ground_truth['timestamp'] = pd.to_datetime(ground_truth['timestamp'], utc=True)
ground_truth = ground_truth.sort_values('timestamp').reset_index(drop=True)

print(f"   Ground truth loaded: {len(ground_truth):,} timesteps")
print(f"   Date range: {ground_truth['timestamp'].min()} → {ground_truth['timestamp'].max()}")
print(f"   Power range: {ground_truth['power_kw'].min():.1f} - {ground_truth['power_kw'].max():.1f} kW")

# Load predictions
print("\n📂 Loading Phase 1 predictions...")
predictions = pd.read_parquet('data/processed/test_phase1_dec2023_dec2024/predictions_phase1.parquet')
predictions['timestamp'] = pd.to_datetime(predictions['timestamp_utc'], utc=True)

print(f"   Predictions loaded: {len(predictions):,} timesteps")
print(f"   Date range: {predictions['timestamp'].min()} → {predictions['timestamp'].max()}")

# Normalize ground truth to [0, 1]
capacity_kw = 7358.9
ground_truth['power_norm'] = ground_truth['power_kw'] / capacity_kw

print(f"   Normalized with capacity: {capacity_kw:,.1f} kW")

# Merge
print("\n🔗 Merging predictions with ground truth...")
merged = pd.merge(
    predictions[['timestamp', 'predicted_power_norm', 'forecast_idx', 'step_ahead']],
    ground_truth[['timestamp', 'power_norm', 'power_kw']],
    on='timestamp',
    how='inner'
)

print(f"   Matched timesteps: {len(merged):,}")
print(f"   Coverage: {len(merged) / len(predictions) * 100:.1f}% of predictions")
print(f"   Date range: {merged['timestamp'].min()} → {merged['timestamp'].max()}")

if len(merged) == 0:
    print("\n❌ ERROR: No overlapping timestamps found!")
    print("   Ground truth range:", ground_truth['timestamp'].min(), "→", ground_truth['timestamp'].max())
    print("   Prediction range:", predictions['timestamp'].min(), "→", predictions['timestamp'].max())
    exit(1)

# Calculate errors
print("\n" + "=" * 80)
print("📊 ERROR METRICS (Normalized Power [0, 1])")
print("=" * 80)

y_true = merged['power_norm'].values
y_pred = merged['predicted_power_norm'].values

rmse_norm = np.sqrt(mean_squared_error(y_true, y_pred))
mae_norm = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

# Absolute units
rmse_kw = rmse_norm * capacity_kw
mae_kw = mae_norm * capacity_kw

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
    print(f"   ✓ EXCELLENT: True RMSE within 2× of pseudo-RMSE estimate!")
elif rmse_norm < pseudo_rmse * 3:
    print(f"   ✓ GOOD: True RMSE within 3× of pseudo-RMSE estimate")
else:
    print(f"   ⚠️  WARNING: True RMSE significantly exceeds pseudo-RMSE estimate")

# Daylight vs nighttime performance
merged['hour'] = merged['timestamp'].dt.hour
day_mask = (merged['hour'] >= 6) & (merged['hour'] <= 18)
night_mask = ~day_mask

print(f"\n🌞 DAYLIGHT PERFORMANCE (06:00-18:00 UTC):")
if day_mask.sum() > 0:
    rmse_day = np.sqrt(mean_squared_error(merged[day_mask]['power_norm'], 
                                           merged[day_mask]['predicted_power_norm']))
    mae_day = mean_absolute_error(merged[day_mask]['power_norm'], 
                                   merged[day_mask]['predicted_power_norm'])
    r2_day = r2_score(merged[day_mask]['power_norm'], 
                      merged[day_mask]['predicted_power_norm'])
    
    print(f"   Samples: {day_mask.sum():,}")
    print(f"   RMSE: {rmse_day:.5f} ({rmse_day * 100:.2f}%)")
    print(f"   MAE:  {mae_day:.5f} ({mae_day * 100:.2f}%)")
    print(f"   R²:   {r2_day:.4f}")

print(f"\n🌙 NIGHTTIME PERFORMANCE (18:00-06:00 UTC):")
if night_mask.sum() > 0:
    rmse_night = np.sqrt(mean_squared_error(merged[night_mask]['power_norm'], 
                                             merged[night_mask]['predicted_power_norm']))
    mae_night = mean_absolute_error(merged[night_mask]['power_norm'], 
                                     merged[night_mask]['predicted_power_norm'])
    
    print(f"   Samples: {night_mask.sum():,}")
    print(f"   RMSE: {rmse_night:.5f} ({rmse_night * 100:.2f}%)")
    print(f"   MAE:  {mae_night:.5f} ({mae_night * 100:.2f}%)")

# Monthly breakdown
print(f"\n" + "=" * 80)
print("📅 MONTHLY ERROR BREAKDOWN")
print("=" * 80)

merged['month'] = merged['timestamp'].dt.to_period('M')
monthly_errors = merged.groupby('month').apply(
    lambda x: pd.Series({
        'count': len(x),
        'rmse': np.sqrt(mean_squared_error(x['power_norm'], x['predicted_power_norm'])),
        'mae': mean_absolute_error(x['power_norm'], x['predicted_power_norm']),
        'r2': r2_score(x['power_norm'], x['predicted_power_norm']),
        'mean_true': x['power_norm'].mean(),
        'mean_pred': x['predicted_power_norm'].mean(),
        'bias': (x['predicted_power_norm'] - x['power_norm']).mean()
    })
).reset_index()

print(f"\n{'Month':<10} {'Samples':<10} {'RMSE':<10} {'MAE':<10} {'R²':<8} {'Bias':<10}")
print("-" * 70)
for _, row in monthly_errors.iterrows():
    print(f"{str(row['month']):<10} {int(row['count']):<10,} "
          f"{row['rmse']:<10.5f} {row['mae']:<10.5f} "
          f"{row['r2']:<8.4f} {row['bias']:<+10.5f}")

# Seasonal summary
winter = monthly_errors[monthly_errors['month'].astype(str).str.contains('2023-12|2024-01|2024-02')]
spring = monthly_errors[monthly_errors['month'].astype(str).str.contains('2024-03|2024-04|2024-05')]
summer = monthly_errors[monthly_errors['month'].astype(str).str.contains('2024-06|2024-07')]

print(f"\n📊 SEASONAL SUMMARY:")
if len(winter) > 0:
    print(f"   Winter (Dec-Feb): RMSE={winter['rmse'].mean():.5f}, R²={winter['r2'].mean():.4f}")
if len(spring) > 0:
    print(f"   Spring (Mar-May): RMSE={spring['rmse'].mean():.5f}, R²={spring['r2'].mean():.4f}")
if len(summer) > 0:
    print(f"   Summer (Jun-Jul): RMSE={summer['rmse'].mean():.5f}, R²={summer['r2'].mean():.4f}")

# Bias analysis
print(f"\n" + "=" * 80)
print("📈 BIAS ANALYSIS")
print("=" * 80)

bias = merged['predicted_power_norm'] - merged['power_norm']
print(f"\n   Mean Bias:        {bias.mean():+.5f} ({bias.mean() * 100:+.2f}%)")
print(f"   Median Bias:      {bias.median():+.5f}")
print(f"   Std Dev:          {bias.std():.5f}")
print(f"   Over-predictions: {(bias > 0).sum():,} / {len(bias):,} ({(bias > 0).mean() * 100:.1f}%)")
print(f"   Under-predictions: {(bias < 0).sum():,} / {len(bias):,} ({(bias < 0).mean() * 100:.1f}%)")

if bias.mean() > 0.01:
    print(f"\n   ⚠️  MODEL TENDS TO OVERESTIMATE (positive bias)")
elif bias.mean() < -0.01:
    print(f"\n   ⚠️  MODEL TENDS TO UNDERESTIMATE (negative bias)")
else:
    print(f"\n   ✓ MODEL IS WELL-CALIBRATED (minimal bias)")

# Best/worst days
merged['date'] = merged['timestamp'].dt.date
daily_rmse = merged.groupby('date').apply(
    lambda x: np.sqrt(mean_squared_error(x['power_norm'], x['predicted_power_norm']))
).reset_index(name='rmse')

print(f"\n🏆 BEST DAYS (Lowest RMSE):")
for _, row in daily_rmse.nsmallest(5, 'rmse').iterrows():
    print(f"   {row['date']}: RMSE = {row['rmse']:.5f}")

print(f"\n💥 WORST DAYS (Highest RMSE):")
for _, row in daily_rmse.nlargest(5, 'rmse').iterrows():
    print(f"   {row['date']}: RMSE = {row['rmse']:.5f}")

# Final verdict
print(f"\n" + "=" * 80)
print("🎯 FINAL VERDICT")
print("=" * 80)

if rmse_norm < 0.05:
    grade = "A+ EXCELLENT"
    verdict = "Production-ready performance!"
elif rmse_norm < 0.10:
    grade = "A VERY GOOD"
    verdict = "Strong performance, suitable for most applications"
elif rmse_norm < 0.15:
    grade = "B GOOD"
    verdict = "Acceptable performance, some room for improvement"
else:
    grade = "C FAIR"
    verdict = "Significant errors, requires investigation"

print(f"\n   Grade: {grade}")
print(f"   RMSE: {rmse_norm:.5f} ({rmse_norm * 100:.2f}% of capacity)")
print(f"   R² Score: {r2:.4f}")
print(f"   Verdict: {verdict}")

print(f"\n💾 Comparison complete! Check detailed metrics above.")
print("=" * 80)

# Save comparison
output_path = Path("data/processed/test_phase1_dec2023_dec2024/validation_vs_ground_truth.parquet")
merged.to_parquet(output_path, index=False)
print(f"\n✅ Saved comparison data to: {output_path}")
