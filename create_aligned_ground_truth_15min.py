#!/usr/bin/env python3
"""
Create properly aligned 15-minute ground truth data.
Ensures timestamps match predictions exactly (00:00, 00:15, 00:30, 00:45)
"""

import pandas as pd
import numpy as np

print("Creating aligned 15-minute ground truth dataset...")
print("=" * 70)

# Load 5-minute data
df = pd.read_csv('data/raw/germany/plant_03/plant_03_ground_truth.csv')
df['datetime'] = pd.to_datetime(df['Datum'], format='mixed', dayfirst=False)
df['power_kw'] = pd.to_numeric(df['Wirkleistung'], errors='coerce')
df = df.dropna(subset=['power_kw'])

print(f"Loaded {len(df):,} 5-minute timesteps")
print(f"Date range: {df['datetime'].min()} → {df['datetime'].max()}")
print(f"Power range: {df['power_kw'].min():.1f} - {df['power_kw'].max():.1f} kW")

# Check current alignment
print(f"\nCurrent timestamp samples:")
print(df['datetime'].head(10).tolist())

# We need to align to 00:00, 00:15, 00:30, 00:45
# The data starts at 00:05, 00:10, 00:15, 00:20, 00:25, 00:30...
# For each 15-min window [00:00-00:15), take average of 00:05, 00:10
# For [00:15-00:30), take average of 00:15, 00:20, 00:25

df = df.set_index('datetime')

# Resample to 15min, but specify the offset to align with predictions
# Use 'origin' parameter to control alignment
df_15min = df['power_kw'].resample('15min', origin='start_day').mean().reset_index()
df_15min.columns = ['timestamp_utc', 'power_kw']

print(f"\n✅ Resampled to {len(df_15min):,} 15-minute timesteps")
print(f"New timestamp samples:")
print(df_15min['timestamp_utc'].head(10).tolist())

# Normalize by capacity
capacity_kw = 7358.9
df_15min['power_norm'] = df_15min['power_kw'] / capacity_kw

print(f"\nNormalized power stats:")
print(f"  Max normalized: {df_15min['power_norm'].max():.4f}")
print(f"  Mean normalized: {df_15min['power_norm'].mean():.4f}")

# Save
output_path = 'data/processed/test_phase1_dec2023_dec2024/ground_truth_15min.parquet'
df_15min.to_parquet(output_path, index=False)

print(f"\n✅ Saved to: {output_path}")

# Quick alignment check with predictions
print("\n" + "=" * 70)
print("ALIGNMENT VERIFICATION")
print("=" * 70)

pred = pd.read_parquet('data/processed/test_phase1_dec2023_dec2024/predictions_phase1.parquet')
pred['timestamp'] = pd.to_datetime(pred['timestamp_utc'])

# Merge
merged = pd.merge(
    pred[['timestamp', 'predicted_power_norm']],
    df_15min.rename(columns={'timestamp_utc': 'timestamp'}),
    on='timestamp',
    how='inner'
)

print(f"\n✓ Matched timestamps: {len(merged):,}")
print(f"✓ Coverage: {len(merged) / len(pred) * 100:.1f}% of predictions")
print(f"✓ Date range: {merged['timestamp'].min()} → {merged['timestamp'].max()}")

# Check timestamp alignment
print(f"\nFirst 10 matched timestamps:")
for ts in merged['timestamp'].head(10):
    print(f"  {ts}")

print("\n✨ Ground truth ready for comparison!")
