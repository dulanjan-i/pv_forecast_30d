#!/usr/bin/env python3
"""
Compare Heuristic vs Learned RL Policy on Held-Out Test Windows

Runs both heuristic and learned RL policies on the same test windows,
computes RMSE, action distributions, and statistical significance tests.

Usage:
    python src/rl/compare_heuristic_vs_learned.py \
        --checkpoint-dir freeze/final_thesis_v1/rl/ddqn_minenv_v2 \
        --num-windows 42

Author: MiRACLE Team
Date: 2026-01-04
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from scipy import stats

from src.rl.rl_integrated_forecaster import RLIntegratedForecaster
from src.inference.physics_aware_forecaster import PhysicsAwareForecaster


def main():
    parser = argparse.ArgumentParser(description='Compare heuristic vs learned RL policies')
    parser.add_argument('--checkpoint-dir', type=str, 
                        default='freeze/final_thesis_v1/rl/ddqn_minenv_v2',
                        help='Directory containing RL checkpoint (ddqn_best.pt or ddqn_final.pt)')
    parser.add_argument('--num-windows', type=int, default=42,
                        help='Number of test windows to evaluate')
    parser.add_argument('--short-ckpt', type=str,
                        default='V1.0_FINAL_TFT/shorthead_seed42/checkpoints/best.ckpt',
                        help='Short-head TFT checkpoint')
    parser.add_argument('--long-ckpt', type=str,
                        default='V1.0_FINAL_TFT/longhead_seed43/checkpoints/best.ckpt',
                        help='Long-head TFT checkpoint')
    parser.add_argument('--plant-meta', type=str,
                        default='V1.0_FINAL_TFT/plant_metadata/plant_03.json',
                        help='Plant metadata JSON')
    parser.add_argument('--short-train', type=str,
                        default='data/processed/plant_level/plant_03/15min_pca32/train.parquet',
                        help='Short-head training data')
    parser.add_argument('--long-train', type=str,
                        default='data/processed/plant_level/plant_03/hourly_longhead/train.parquet',
                        help='Long-head training data')
    parser.add_argument('--test-data', type=str,
                        default='data/processed/plant_level/plant_03/15min_pca32/test.parquet',
                        help='Test data parquet')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device for inference')
    
    args = parser.parse_args()
    
    print('='*80)
    print('HEURISTIC VS LEARNED POLICY COMPARISON')
    print('='*80)
    print(f'Checkpoint directory: {args.checkpoint_dir}')
    print(f'Number of test windows: {args.num_windows}')
    print(f'Device: {args.device}')
    print('='*80)
    
    # Load base forecaster
    print('\nLoading PhysicsAwareForecaster...')
    forecaster = PhysicsAwareForecaster(
        short_ckpt=Path(args.short_ckpt),
        long_ckpt=Path(args.long_ckpt),
        plant_metadata=Path(args.plant_meta),
        short_train_parquet=Path(args.short_train),
        long_train_parquet=Path(args.long_train),
        device=args.device
    )
    print('✅ Forecaster loaded')
    
    # Load test data
    print('\nLoading test data...')
    test_df = pd.read_parquet(args.test_data)
    test_df = test_df.sort_values('timestamp_utc').reset_index(drop=True)
    
    # Extract test windows
    lookback = 672  # 7 days @ 15min for encoder
    window_size = 96  # 1 day forecast @ 15min
    stride = 96
    num_windows = args.num_windows
    
    windows = []
    for i in range(lookback, min(len(test_df) - window_size, lookback + num_windows*stride), stride):
        historical_window = test_df.iloc[i-lookback:i].copy()
        forecast_window = test_df.iloc[i:i+window_size].copy()
        
        if 'power_norm' in forecast_window.columns:
            ground_truth = forecast_window['power_norm'].values
        elif 'power_normalized' in forecast_window.columns:
            ground_truth = forecast_window['power_normalized'].values
        else:
            ground_truth = forecast_window['target'].values
        
        windows.append((historical_window, forecast_window, ground_truth))
    
    print(f'✅ Extracted {len(windows)} held-out test windows')
    
    # Run both modes
    results = []
    
    for mode in ['heuristic', 'learned']:
        print(f'\n{"="*80}')
        print(f'Running {mode.upper()} mode')
        print("="*80)
        
        rl_forecaster = RLIntegratedForecaster(
            forecaster=forecaster,
            rl_mode=mode,
            checkpoint_dir=Path(args.checkpoint_dir)
        )
        
        for idx, (hist, weather, gt) in enumerate(tqdm(windows, desc=mode)):
            try:
                forecast_start = weather['timestamp_utc'].iloc[0]
                forecast, info = rl_forecaster.forecast_with_rl(
                    weather_data=weather,
                    forecast_start=forecast_start,
                    historical_data=hist,
                    ground_truth=gt
                )
                
                # Align and compute Day 1 RMSE only (where blend matters)
                day1_length = min(96, len(gt), len(forecast))
                forecast_day1 = forecast[:day1_length]
                gt_day1 = gt[:day1_length]
                rmse = np.sqrt(np.mean((forecast_day1 - gt_day1)**2))
                
                results.append({
                    'mode': mode,
                    'window_idx': idx,
                    'action': info['action_index'],
                    'action_name': info['action_name'],
                    'rmse': rmse,
                    'blend_short': rl_forecaster.blend_weights['short'],
                    'blend_long': rl_forecaster.blend_weights['long'],
                    'blend_physics': rl_forecaster.blend_weights['physics']
                })
            except Exception as e:
                print(f'Error window {idx}: {e}')
                continue
    
    # Save results
    df_results = pd.DataFrame(results)
    output_path = Path(args.checkpoint_dir) / 'heuristic_vs_learned_comparison.csv'
    df_results.to_csv(output_path, index=False)
    print(f'\n✅ Results saved to: {output_path}')
    
    # Analyze results
    print('\n' + '='*80)
    print('COMPARISON RESULTS')
    print('='*80)
    
    for mode in ['heuristic', 'learned']:
        subset = df_results[df_results['mode'] == mode]
        print(f'\n{mode.upper()} Mode:')
        print(f'  Mean RMSE: {subset["rmse"].mean():.6f} ± {subset["rmse"].std():.6f}')
        print(f'  Median RMSE: {subset["rmse"].median():.6f}')
        print(f'  Min/Max RMSE: {subset["rmse"].min():.6f} / {subset["rmse"].max():.6f}')
        print(f'  Action distribution:')
        for action, count in subset['action'].value_counts().items():
            action_name = subset[subset['action'] == action]['action_name'].iloc[0]
            print(f'    {action} ({action_name}): {count} ({100*count/len(subset):.1f}%)')
    
    # Statistical test
    heur_rmse = df_results[df_results['mode'] == 'heuristic']['rmse'].values
    learn_rmse = df_results[df_results['mode'] == 'learned']['rmse'].values
    
    t_stat, p_value = stats.ttest_rel(learn_rmse, heur_rmse)
    improvement = ((heur_rmse.mean() - learn_rmse.mean()) / heur_rmse.mean()) * 100
    
    print(f'\nStatistical Test (Paired t-test):')
    print(f'  t-statistic: {t_stat:.4f}')
    print(f'  p-value: {p_value:.6f}')
    print(f'  Relative improvement: {improvement:+.2f}%')
    
    if p_value < 0.05 and improvement > 0:
        print(f'  ✅ Learned policy is SIGNIFICANTLY BETTER (p < 0.05)')
    elif p_value < 0.05 and improvement < 0:
        print(f'  ❌ Learned policy is SIGNIFICANTLY WORSE (p < 0.05)')
    else:
        print(f'  ⚠️  No significant difference (p >= 0.05)')
    
    print('='*80)


if __name__ == '__main__':
    main()
