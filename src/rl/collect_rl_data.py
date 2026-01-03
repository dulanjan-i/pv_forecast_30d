#!/usr/bin/env python3
"""
RL Data Collection Script for MiRACLE

Runs PhysicsAwareForecaster with RL in HEURISTIC mode to collect training data.
Records (state, action, reward, next_state) transitions for offline DDQN training.

This script:
1. Loads real TFT models and test data
2. Runs forecaster in heuristic mode (rule-based RL)
3. Records every decision and outcome
4. Saves transitions to parquet
5. Can run overnight (1000+ samples)

Usage:
    python scripts/collect_rl_data.py --num-samples 1000 --output data/rl_transitions/run_001.parquet

    # Resume from checkpoint
    python scripts/collect_rl_data.py --num-samples 2000 --resume data/rl_transitions/run_001.parquet

    # Watch live on dashboard (separate terminal)
    streamlit run src/rl/monitoring_dashboard.py

Author: MiRACLE Team
Date: 2026-01-03
"""

import sys
from pathlib import Path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import logging
from datetime import datetime

from src.rl.rl_integrated_forecaster import RLIntegratedForecaster
from src.inference.physics_aware_forecaster import PhysicsAwareForecaster

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_test_data(test_parquet: Path, num_samples: int = None):
    """
    Load test data for rolling forecasts.
    
    Args:
        test_parquet: Path to test parquet (short or long)
        num_samples: How many forecast windows to extract
    
    Returns:
        List of (weather_df, ground_truth) tuples
    """
    logger.info(f"Loading test data from {test_parquet}")
    df = pd.read_parquet(test_parquet)
    
    # Get time column
    time_col = 'timestamp_utc' if 'timestamp_utc' in df.columns else 'time'
    
    # Sort by time
    df = df.sort_values(time_col).reset_index(drop=True)
    
    # Extract windows (every 24 hours = 96 steps @ 15min)
    windows = []
    window_size = 2880  # 30 days of 15-min data
    stride = 96  # Move forward 1 day each time
    
    max_windows = num_samples if num_samples else (len(df) - window_size) // stride
    
    for i in range(0, len(df) - window_size, stride):
        if len(windows) >= max_windows:
            break
        
        window = df.iloc[i:i+window_size].copy()
        
        # Pass full window to forecaster (it needs all columns: power_norm, poa_irradiance, plant features, weather, pvlib)
        weather_df = window.copy()
        if time_col != 'timestamp_utc':
            weather_df = weather_df.rename(columns={time_col: 'timestamp_utc'})
        
        # Ground truth (target) - power_norm is the normalized power
        if 'power_norm' in window.columns:
            ground_truth = window['power_norm'].values
        elif 'power_normalized' in window.columns:
            ground_truth = window['power_normalized'].values
        elif 'target' in window.columns:
            ground_truth = window['target'].values
        else:
            logger.warning("No ground truth column found, using zeros")
            ground_truth = np.zeros(len(window))
        
        windows.append((weather_df, ground_truth))
    
    logger.info(f"Extracted {len(windows)} forecast windows")
    return windows


def collect_transitions(
    rl_forecaster: RLIntegratedForecaster,
    test_windows: list,
    num_samples: int,
    save_path: Path,
    checkpoint_freq: int = 100
):
    """
    Collect RL transitions by running forecasts with heuristic mode.
    
    Args:
        rl_forecaster: RLIntegratedForecaster in heuristic mode
        test_windows: List of (weather, ground_truth) tuples
        num_samples: Total samples to collect
        save_path: Where to save transitions
        checkpoint_freq: Save every N samples
    
    Returns:
        DataFrame with all transitions
    """
    logger.info(f"Starting data collection: target={num_samples} samples")
    
    transitions = []
    
    # Progress bar
    pbar = tqdm(total=num_samples, desc="Collecting transitions")
    
    for sample_idx in range(num_samples):
        # Cycle through test windows
        window_idx = sample_idx % len(test_windows)
        weather_df, ground_truth = test_windows[window_idx]
        
        try:
            # Get forecast start time from weather data
            forecast_start = weather_df['timestamp_utc'].iloc[0]
            
            # Run forecast (RL picks action via heuristics, updates state)
            forecast, info = rl_forecaster.forecast_with_rl(
                weather_data=weather_df,
                forecast_start=forecast_start,
                ground_truth=ground_truth  # Pass ground truth for RMSE computation
            )
            
            # Extract state/action (already computed in forecast_with_rl)
            state = info['meta_state'].copy()
            action = info['action_index']
            
            # Compute reward from forecast error vs ground truth
            # Align forecast with ground truth (forecast is typically 1-day, ground_truth is 30-day window)
            if hasattr(forecast, 'shape'):
                forecast_len = len(forecast)
            elif hasattr(forecast, '__len__'):
                forecast_len = len(forecast)
            else:
                forecast_len = 96  # Default 1 day @ 15min
            
            # Take only the first N steps of ground truth to match forecast
            gt_aligned = ground_truth[:forecast_len]
            forecast_aligned = forecast[:forecast_len] if hasattr(forecast, '__getitem__') else np.zeros(forecast_len)
            
            if len(gt_aligned) > 0 and len(forecast_aligned) == len(gt_aligned):
                rmse = np.sqrt(np.mean((forecast_aligned - gt_aligned) ** 2))
                reward = -rmse / 0.01  # Normalize: 0.01 RMSE = -1.0 reward
            else:
                rmse = 0.05
                reward = -5.0
            
            # Compute next_state from current metrics (will be state for next step)
            metrics = info['metrics']
            
            # For first step, use current state as both state and next_state
            if sample_idx == 0:
                next_state = state.copy()
            else:
                # Next state is current state (closed loop)
                next_state = state.copy()
            
            # Extract transition
            transition = {
                'sample_idx': sample_idx,
                'timestamp': pd.Timestamp.now(tz='UTC').isoformat(),
                'forecast_start': forecast_start.isoformat() if hasattr(forecast_start, 'isoformat') else str(forecast_start),
                'action': action,
                'action_name': info['action_name'],
                'reward': reward,
                'action_success': info['action_success'],
                
                # State (35 dims) - flatten to columns
                **{f'state_{i}': state[i] for i in range(len(state))},
                
                # Next state (35 dims)
                **{f'next_state_{i}': next_state[i] for i in range(len(next_state))},
                
                # Key metrics for analysis
                'short_rmse_1h': metrics.get('short_rmse_1h', 0.0),
                'long_rmse_30d': metrics.get('long_rmse_30d', 0.0),
                'physics_residual': metrics.get('physics_residual', 0.0),
                'blend_short': rl_forecaster.blend_weights['short'],
                'blend_long': rl_forecaster.blend_weights['long'],
                'blend_physics': rl_forecaster.blend_weights['physics']
            }
            
            transitions.append(transition)
            pbar.update(1)
            
            # Checkpoint: save periodically
            if (sample_idx + 1) % checkpoint_freq == 0:
                df = pd.DataFrame(transitions)
                checkpoint_path = save_path.parent / f"{save_path.stem}_checkpoint_{sample_idx+1}.parquet"
                df.to_parquet(checkpoint_path)
                logger.info(f"Checkpoint saved: {checkpoint_path} ({len(df)} samples)")
        
        except Exception as e:
            logger.error(f"Error at sample {sample_idx}: {e}", exc_info=True)
            continue
    
    pbar.close()
    
    # Final save
    df = pd.DataFrame(transitions)
    df.to_parquet(save_path)
    logger.info(f"✅ Collection complete! Saved {len(df)} transitions to {save_path}")
    
    return df


def analyze_transitions(df: pd.DataFrame):
    """Print summary statistics of collected data."""
    if len(df) == 0:
        logger.warning("No transitions collected!")
        return
    
    logger.info("\n" + "="*60)
    logger.info("COLLECTION SUMMARY")
    logger.info("="*60)
    
    logger.info(f"Total samples: {len(df)}")
    logger.info(f"Time range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    
    logger.info("\nAction distribution:")
    action_counts = df['action'].value_counts().sort_index()
    for action_idx, count in action_counts.items():
        action_name = df[df['action'] == action_idx]['action_name'].iloc[0]
        logger.info(f"  Action {action_idx} ({action_name}): {count} ({100*count/len(df):.1f}%)")
    
    logger.info("\nReward statistics:")
    logger.info(f"  Mean: {df['reward'].mean():.3f}")
    logger.info(f"  Std: {df['reward'].std():.3f}")
    logger.info(f"  Min: {df['reward'].min():.3f}")
    logger.info(f"  Max: {df['reward'].max():.3f}")
    
    logger.info("\nPerformance metrics:")
    logger.info(f"  Short RMSE (1h): {df['short_rmse_1h'].mean():.4f} ± {df['short_rmse_1h'].std():.4f}")
    logger.info(f"  Long RMSE (30d): {df['long_rmse_30d'].mean():.4f} ± {df['long_rmse_30d'].std():.4f}")
    logger.info(f"  Physics residual: {df['physics_residual'].mean():.4f} ± {df['physics_residual'].std():.4f}")
    
    logger.info("\nBlend weights (mean):")
    logger.info(f"  Short: {df['blend_short'].mean():.3f}")
    logger.info(f"  Long: {df['blend_long'].mean():.3f}")
    logger.info(f"  Physics: {df['blend_physics'].mean():.3f}")
    
    logger.info("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Collect RL training data from heuristic mode")
    
    # Model paths - CANONICAL HARDCODED
    parser.add_argument('--short-ckpt', type=str, 
                       default='/home/dwijenayake/pv_forecast_30d/V1.0_FINAL_TFT/shorthead_seed42/checkpoints/best.ckpt',
                       help='Short-head TFT checkpoint')
    parser.add_argument('--long-ckpt', type=str,
                       default='/home/dwijenayake/pv_forecast_30d/V1.0_FINAL_TFT/longhead_seed43/checkpoints/best.ckpt',
                       help='Long-head TFT checkpoint')
    parser.add_argument('--plant-meta', type=str,
                       default='/home/dwijenayake/pv_forecast_30d/V1.0_FINAL_TFT/plant_metadata/plant_03.json',
                       help='Plant metadata JSON')
    
    # Data paths - CANONICAL HARDCODED
    parser.add_argument('--short-train', type=str,
                       default='/home/dwijenayake/pv_forecast_30d/data/processed/plant_level/plant_03/15min_pca32/train.parquet',
                       help='Short-head training data (for TFT initialization)')
    parser.add_argument('--long-train', type=str,
                       default='/home/dwijenayake/pv_forecast_30d/data/processed/plant_level/plant_03/hourly_longhead/train.parquet',
                       help='Long-head training data')
    parser.add_argument('--test-data', type=str,
                       default='/home/dwijenayake/pv_forecast_30d/data/processed/plant_level/plant_03/15min_pca32/test.parquet',
                       help='Test data for rolling forecasts')
    
    # Collection params
    parser.add_argument('--num-samples', type=int, default=1000,
                       help='Number of samples to collect')
    parser.add_argument('--output', type=str, 
                       default='/home/dwijenayake/pv_forecast_30d/data/rl_transitions/heuristic_run_001.parquet',
                       help='Output parquet file')
    parser.add_argument('--checkpoint-freq', type=int, default=100,
                       help='Save checkpoint every N samples')
    
    # Hardware
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device for models')
    
    args = parser.parse_args()
    
    # Convert to paths
    short_ckpt = Path(args.short_ckpt)
    long_ckpt = Path(args.long_ckpt)
    plant_meta = Path(args.plant_meta)
    short_train = Path(args.short_train)
    long_train = Path(args.long_train)
    test_data = Path(args.test_data)
    output_path = Path(args.output)
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Verify paths
    for path in [short_ckpt, long_ckpt, plant_meta, short_train, long_train, test_data]:
        if not path.exists():
            logger.error(f"Path not found: {path}")
            return
    
    logger.info("="*60)
    logger.info("MiRACLE RL DATA COLLECTION")
    logger.info("="*60)
    logger.info(f"Target samples: {args.num_samples}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Device: {args.device}")
    logger.info("="*60 + "\n")
    
    # Initialize forecaster
    logger.info("Loading PhysicsAwareForecaster...")
    forecaster = PhysicsAwareForecaster(
        short_ckpt=short_ckpt,
        long_ckpt=long_ckpt,
        plant_metadata=plant_meta,
        short_train_parquet=short_train,
        long_train_parquet=long_train,
        device=args.device
    )
    logger.info("✅ Forecaster loaded\n")
    
    # Wrap with RL (HEURISTIC MODE)
    logger.info("Initializing RL system in HEURISTIC mode...")
    rl_forecaster = RLIntegratedForecaster(
        forecaster=forecaster,
        rl_mode="heuristic",  # ← Rule-based decisions, DDQN observes
        checkpoint_dir=Path("checkpoints/rl")
    )
    logger.info("✅ RL system ready\n")
    
    # Load test data windows
    test_windows = load_test_data(test_data, num_samples=args.num_samples)
    
    # Collect transitions
    logger.info("Starting data collection...")
    logger.info("💡 TIP: Open dashboard in another terminal to watch live:")
    logger.info("    streamlit run src/rl/monitoring_dashboard.py\n")
    
    df = collect_transitions(
        rl_forecaster=rl_forecaster,
        test_windows=test_windows,
        num_samples=args.num_samples,
        save_path=output_path,
        checkpoint_freq=args.checkpoint_freq
    )
    
    # Analyze results
    analyze_transitions(df)
    
    logger.info(f"\n🎉 All done! Data saved to: {output_path}")
    logger.info(f"📊 Ready for offline DDQN training (Option D)")
    logger.info(f"\nNext steps:")
    logger.info(f"  1. Verify data: python -c 'import pandas as pd; print(pd.read_parquet(\"{output_path}\").head())'")
    logger.info(f"  2. Train DDQN: python scripts/train_rl_offline.py --data {output_path}")


if __name__ == "__main__":
    main()
