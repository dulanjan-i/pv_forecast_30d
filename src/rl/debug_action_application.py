#!/usr/bin/env python3
"""
Debug: Verify that RL actions actually change predictions.

Tests:
1. Force BLEND_HIGH_PHYSICS vs BLEND_HIGH_SHORT - should give different RMSE
2. Log detailed per-step metrics to confirm actions are applied
3. Check if blend weights actually affect final predictions

Author: MiRACLE Team
Date: 2026-01-04
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
import torch

from src.rl.rl_integrated_forecaster import RLIntegratedForecaster
from src.inference.physics_aware_forecaster import PhysicsAwareForecaster


def force_action_test():
    """Test if forcing different actions gives different RMSE."""
    
    print('='*80)
    print('FORCED ACTION TEST: Do actions actually change predictions?')
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
    print('Loading test data...')
    test_df = pd.read_parquet('data/processed/plant_level/plant_03/15min_pca32/test.parquet')
    test_df = test_df.sort_values('timestamp_utc').reset_index(drop=True)
    
    # Extract ONE window for detailed inspection
    lookback = 672
    window_size = 96
    
    i = lookback + 1000  # Pick a window in the middle
    historical_window = test_df.iloc[i-lookback:i].copy()
    forecast_window = test_df.iloc[i:i+window_size].copy()
    ground_truth = forecast_window['power_norm'].values
    forecast_start = forecast_window['timestamp_utc'].iloc[0]
    
    print(f'\n✅ Using test window at index {i}')
    print(f'   Forecast start: {forecast_start}')
    print(f'   Ground truth mean: {ground_truth.mean():.4f}')
    
    # Test extreme policies
    forced_policies = [
        (6, 'BLEND_HIGH_PHYSICS', {'short': 0.2, 'long': 0.2, 'physics': 0.6}),
        (4, 'BLEND_HIGH_SHORT', {'short': 0.6, 'long': 0.2, 'physics': 0.2}),
        (5, 'BLEND_HIGH_LONG', {'short': 0.2, 'long': 0.6, 'physics': 0.2}),
    ]
    
    results = []
    
    for action_id, action_name, forced_weights in forced_policies:
        print(f'\n{"="*80}')
        print(f'Testing forced action: {action_id} ({action_name})')
        print(f'Forced blend weights: {forced_weights}')
        print("="*80)
        
        # Create RL forecaster
        rl_forecaster = RLIntegratedForecaster(
            forecaster=forecaster,
            rl_mode='heuristic',  # Use heuristic to control actions
            checkpoint_dir=Path('checkpoints/rl_meta_controller')
        )
        
        # Manually override blend weights BEFORE forecast
        rl_forecaster.blend_weights = forced_weights.copy()
        
        # Run forecast
        forecast, info = rl_forecaster.forecast_with_rl(
            weather_data=forecast_window,
            forecast_start=forecast_start,
            historical_data=historical_window,
            ground_truth=ground_truth
        )
        
        # Compute RMSE
        forecast_aligned = forecast[:len(ground_truth)]
        rmse = np.sqrt(np.mean((forecast_aligned - ground_truth)**2))
        
        # Log detailed stats
        print(f'\nResults:')
        print(f'  Final blend weights: {rl_forecaster.blend_weights}')
        print(f'  Forecast mean: {forecast_aligned.mean():.4f}')
        print(f'  Forecast std: {forecast_aligned.std():.4f}')
        print(f'  Forecast min/max: {forecast_aligned.min():.4f} / {forecast_aligned.max():.4f}')
        print(f'  RMSE: {rmse:.6f}')
        
        results.append({
            'action_id': action_id,
            'action_name': action_name,
            'blend_short': forced_weights['short'],
            'blend_long': forced_weights['long'],
            'blend_physics': forced_weights['physics'],
            'forecast_mean': forecast_aligned.mean(),
            'forecast_std': forecast_aligned.std(),
            'rmse': rmse
        })
    
    # Summary
    print('\n' + '='*80)
    print('SUMMARY: Do different actions give different RMSE?')
    print('='*80)
    
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))
    
    rmse_range = df_results['rmse'].max() - df_results['rmse'].min()
    rmse_std = df_results['rmse'].std()
    
    print(f'\nRMSE range: {rmse_range:.6f}')
    print(f'RMSE std: {rmse_std:.6f}')
    
    if rmse_range < 1e-6:
        print('\n❌ PROBLEM: RMSE is IDENTICAL across all actions!')
        print('   → Actions are NOT affecting predictions')
        print('   → Check blend_weights application in PhysicsAwareForecaster')
    elif rmse_range < 0.001:
        print('\n⚠️  WARNING: RMSE barely changes (< 0.001)')
        print('   → Blend weights have minimal impact')
        print('   → Action space may be too weak')
    else:
        print('\n✅ GOOD: Different actions give different RMSE')
        print(f'   → Blend weights are working (range: {rmse_range:.6f})')
    
    print('='*80)


def detailed_step_logging():
    """Log detailed per-step information to trace action application."""
    
    print('\n\n' + '='*80)
    print('DETAILED STEP LOGGING: Trace action → blend → prediction')
    print('='*80)
    
    # Load forecaster
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
    
    # Pick 3 windows
    lookback = 672
    window_size = 96
    
    for window_num in range(3):
        i = lookback + window_num * 500
        historical_window = test_df.iloc[i-lookback:i].copy()
        forecast_window = test_df.iloc[i:i+window_size].copy()
        ground_truth = forecast_window['power_norm'].values
        forecast_start = forecast_window['timestamp_utc'].iloc[0]
        
        print(f'\n{"="*80}')
        print(f'Window {window_num+1}: {forecast_start}')
        print("="*80)
        
        # Test learned mode
        rl_forecaster = RLIntegratedForecaster(
            forecaster=forecaster,
            rl_mode='learned',
            checkpoint_dir=Path('checkpoints/rl_meta_controller_regularized')
        )
        
        print(f'Initial blend weights: {rl_forecaster.blend_weights}')
        
        forecast, info = rl_forecaster.forecast_with_rl(
            weather_data=forecast_window,
            forecast_start=forecast_start,
            historical_data=historical_window,
            ground_truth=ground_truth
        )
        
        forecast_aligned = forecast[:len(ground_truth)]
        rmse = np.sqrt(np.mean((forecast_aligned - ground_truth)**2))
        
        print(f'Action chosen: {info["action_index"]} ({info["action_name"]})')
        print(f'Final blend weights: {rl_forecaster.blend_weights}')
        print(f'Forecast summary: mean={forecast_aligned.mean():.4f}, std={forecast_aligned.std():.4f}')
        print(f'RMSE: {rmse:.6f}')


if __name__ == '__main__':
    # Run tests
    force_action_test()
    detailed_step_logging()
