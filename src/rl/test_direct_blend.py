#!/usr/bin/env python3
"""
Direct test: manually blend Day 1 with different weights and measure RMSE.
Bypass RL system to verify blend weights actually affect predictions.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from src.inference.physics_aware_forecaster import PhysicsAwareForecaster

print('='*80)
print('DIRECT BLEND TEST: Do weights affect Day 1 RMSE?')
print('='*80)

# Load forecaster
print('\nLoading PhysicsAwareForecaster...')
forecaster = PhysicsAwareForecaster(
    short_ckpt=Path('V1.0_FINAL_TFT/shorthead_seed42/checkpoints/best.ckpt'),
    long_ckpt=Path('V1.0_FINAL_TFT/longhead_seed43/checkpoints/best.ckpt'),
    plant_metadata=Path('V1.0_FINAL_TFT/plant_metadata/plant_03.json'),
    short_train_parquet=Path('data/processed/plant_level/plant_03/15min_pca32/train.parquet'),
    long_train_parquet=Path('data/processed/plant_level/plant_03/hourly_longhead/train.parquet'),
    device='cuda:0'
)

# Load test data
test_df = pd.read_parquet('data/processed/plant_level/plant_03/15min_pca32/test.parquet')
test_df = test_df.sort_values('timestamp_utc').reset_index(drop=True)

# Pick one window
lookback = 672
i = lookback + 1000
historical_window = test_df.iloc[i-lookback:i].copy()
forecast_window = test_df.iloc[i:i+96].copy()  # Day 1 only
ground_truth = forecast_window['power_norm'].values
forecast_start = forecast_window['timestamp_utc'].iloc[0]

print(f'\n✅ Test window: {forecast_start}')
print(f'   Ground truth Day 1 mean: {ground_truth.mean():.4f}')

# Get raw predictions
print('\nRunning forecaster to get raw short/long/physics predictions...')
result = forecaster.predict_30d(
    forecast_start=forecast_start,
    weather_df=forecast_window,
    historical_df=historical_window
)

# Extract Day 1 predictions
short_day1 = result['forecast_short'][:96]
long_day1 = result['forecast_long'][:96]
physics_day1 = result['forecast_physics'][:96]

print(f'Short Day 1 mean: {short_day1.mean():.4f}')
print(f'Long Day 1 mean: {long_day1.mean():.4f}')
print(f'Physics Day 1 mean: {physics_day1.mean():.4f}')

# Test different blend weights
blend_scenarios = [
    {'name': 'BLEND_HIGH_PHYSICS', 'short': 0.2, 'long': 0.2, 'physics': 0.6},
    {'name': 'BLEND_HIGH_SHORT', 'short': 0.6, 'long': 0.2, 'physics': 0.2},
    {'name': 'BLEND_HIGH_LONG', 'short': 0.2, 'long': 0.6, 'physics': 0.2},
    {'name': 'BLEND_EQUAL', 'short': 0.33, 'long': 0.33, 'physics': 0.34},
]

results = []

print('\n' + '='*80)
print('Testing blend weights on Day 1:')
print('='*80)

for scenario in blend_scenarios:
    # Manual blend
    blended_day1 = (
        scenario['short'] * short_day1 +
        scenario['long'] * long_day1 +
        scenario['physics'] * physics_day1
    )
    
    # Compute Day 1 RMSE
    rmse = np.sqrt(np.mean((blended_day1 - ground_truth)**2))
    
    print(f"\n{scenario['name']}:")
    print(f"  Weights: short={scenario['short']}, long={scenario['long']}, physics={scenario['physics']}")
    print(f"  Blended mean: {blended_day1.mean():.4f}")
    print(f"  Day 1 RMSE: {rmse:.6f}")
    
    results.append({
        'scenario': scenario['name'],
        'blend_short': scenario['short'],
        'blend_long': scenario['long'],
        'blend_physics': scenario['physics'],
        'rmse_day1': rmse
    })

# Summary
print('\n' + '='*80)
print('SUMMARY:')
print('='*80)

df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))

rmse_range = df_results['rmse_day1'].max() - df_results['rmse_day1'].min()
print(f'\nRMSE range: {rmse_range:.6f}')

if rmse_range < 1e-6:
    print('❌ PROBLEM: Blend weights have NO effect')
elif rmse_range < 0.001:
    print('⚠️  WARNING: Blend weights have minimal effect (< 0.001)')
else:
    print(f'✅ GOOD: Blend weights DO affect Day 1 RMSE (range: {rmse_range:.6f})')
    best = df_results.loc[df_results['rmse_day1'].idxmin()]
    print(f'\nBest blend: {best["scenario"]} (RMSE={best["rmse_day1"]:.6f})')

print('='*80)
