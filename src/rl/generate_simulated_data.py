"""
Generate RL training data from historical test set using ground truth + noise.

Since TFT models aren't trained yet, we simulate realistic forecast errors
by adding calibrated noise to ground truth. This gives us realistic RMSE 
distributions for RL training.

Once TFT models are ready, replace with actual forecast pipeline.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from tqdm import tqdm

from src.rl.reward import compute_reward as canonical_compute_reward

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def simulate_forecast_errors(ground_truth: np.ndarray, horizon: int) -> Tuple[np.ndarray, float]:
    """
    Simulate realistic forecast by adding noise to ground truth.
    
    Error increases with horizon:
        - 1h: RMSE ~0.03-0.05
        - 24h: RMSE ~0.06-0.08  
        - 30d: RMSE ~0.10-0.15
    """
    gt_slice = ground_truth[:horizon]
    
    # Base noise level (increases with horizon)
    if horizon <= 4:  # 1 hour
        noise_std = 0.03
    elif horizon <= 96:  # 24 hours
        noise_std = 0.05
    elif horizon <= 672:  # 7 days
        noise_std = 0.08
    else:  # 30 days
        noise_std = 0.12
    
    # Add Gaussian noise
    noise = np.random.normal(0, noise_std, len(gt_slice))
    forecast = gt_slice + noise
    
    # Clip to physical bounds [0, 1] for normalized power
    forecast = np.clip(forecast, 0.0, 1.0)
    
    # Compute RMSE
    rmse = np.sqrt(np.mean((forecast - gt_slice) ** 2))
    
    return forecast, float(rmse)


def compute_metrics_from_ground_truth(
    ground_truth: np.ndarray,
    window_data: pd.DataFrame
) -> Dict:
    """
    Simulate forecast metrics using ground truth + realistic noise.
    """
    # Simulate forecasts at different horizons
    _, rmse_1h = simulate_forecast_errors(ground_truth, horizon=4)
    _, rmse_6h = simulate_forecast_errors(ground_truth, horizon=24)
    _, rmse_24h = simulate_forecast_errors(ground_truth, horizon=96)
    _, rmse_7d = simulate_forecast_errors(ground_truth, horizon=672)
    _, rmse_30d = simulate_forecast_errors(ground_truth, horizon=2880)
    
    # Physics residual (compare ground truth vs PVLib)
    if 'pvlib_ac_kw' in window_data.columns and len(ground_truth) >= 96:
        pvlib_baseline = window_data['pvlib_ac_kw'].values[:96]
        gt_slice = ground_truth[:96]
        physics_residual = np.sqrt(np.mean((gt_slice - pvlib_baseline) ** 2))
    else:
        physics_residual = np.random.uniform(0.02, 0.05)
    
    metrics = {
        'short_rmse_1h': rmse_1h,
        'short_rmse_6h': rmse_6h,
        'short_rmse_24h': rmse_24h,
        'long_rmse_7d': rmse_7d,
        'long_rmse_30d': rmse_30d,
        'physics_residual': float(physics_residual),
        'success': True
    }
    
    return metrics


def build_rl_state(metrics: Dict, context: Dict) -> np.ndarray:
    """Build 35-dimensional RL state vector from metrics."""
    state = np.zeros(35, dtype=np.float32)
    
    # [0-4] RMSE components
    state[0] = metrics['short_rmse_1h']
    state[1] = metrics['long_rmse_30d']
    state[2] = metrics['physics_residual']
    state[3] = metrics.get('short_rmse_6h', 0.06)
    state[4] = metrics.get('short_rmse_24h', 0.08)
    
    # [5-6] Confidence (inverse of RMSE)
    state[5] = 1.0 - min(state[0] * 10, 0.9)  # short confidence
    state[6] = 1.0 - min(state[1] * 10, 0.9)  # long confidence
    
    # [7-9] Drift indicators
    state[7] = context.get('data_drift', 0.0)
    state[8] = abs(state[0] - state[1])  # short-long mismatch
    state[9] = 1.0 if metrics['success'] else 0.0
    
    # [10-14] Blend weights
    state[10] = context.get('blend_short', 0.33)
    state[11] = context.get('blend_long', 0.33)
    state[12] = context.get('blend_physics', 0.34)
    state[13] = context.get('actions_since_retrain', 0) / 100.0
    state[14] = 0.0  # retrain count
    
    # [15-19] Temporal context
    forecast_time = context.get('forecast_start')
    if forecast_time:
        state[15] = forecast_time.hour / 24.0
        state[16] = 1.0 if (forecast_time.hour < 6 or forecast_time.hour > 20) else 0.0
        state[17] = (forecast_time.month - 1) / 12.0
        state[18] = forecast_time.dayofweek / 7.0
        state[19] = 1.0 if forecast_time.dayofweek >= 5 else 0.0
    
    # [20-24] Weather
    state[20] = context.get('cloud_cover', 0.5)
    state[21] = context.get('ghi', 0.3)
    state[22] = context.get('dni', 0.3)
    state[23] = context.get('temperature', 0.5)
    state[24] = 1.0  # weather quality
    
    # [25-29] Compute budget
    state[25] = 1.0  # budget remaining
    state[26] = 0.5  # priority
    state[27] = 1.0  # API status
    state[28] = 0.95  # API agreement
    state[29] = 0.0  # cost
    
    # [30-34] Physics baseline
    state[30] = 1.0 - min(metrics['physics_residual'] * 10, 0.9)
    state[31] = 0.0  # calibration age
    state[32] = context.get('solar_zenith', 0.5)
    state[33] = context.get('clearsky_ghi', 0.5)
    state[34] = 1.0  # pvlib confidence
    
    return state


def simulate_heuristic_action(state: np.ndarray) -> Tuple[int, str]:
    """Simulate heuristic action based on state."""
    short_rmse = state[0]
    long_rmse = state[1]
    physics_res = state[2]
    
    if short_rmse > 0.10 and long_rmse > 0.12:
        return 7, "SUGGEST_RETRAIN"
    elif short_rmse > 0.08:
        return 1, "FINE_TUNE_SHORT"
    elif long_rmse > 0.10:
        return 2, "FINE_TUNE_LONG"
    elif physics_res > 0.05:
        return 3, "RECALIBRATE_PVLIB"
    elif physics_res < 0.03:
        return 6, "BLEND_HIGH_PHYSICS"
    else:
        return 5, "BLEND_MEDIUM"


def compute_reward(state: np.ndarray, action: int, next_state: np.ndarray) -> float:
    """
    Canonical reward wrapper.

    This file must NOT define its own action_costs or alternate reward logic.
    Single source of truth lives in src/rl/reward.py
    """
    return canonical_compute_reward(state=state, action=action, next_state=next_state)



def generate_rl_transitions(
    test_data: pd.DataFrame,
    num_samples: int = 100,
    stride_days: int = 1
) -> List[Dict]:
    """Generate RL transitions from historical test data."""
    transitions = []
    
    window_size = 2880  # 30 days @ 15min
    stride = 96 * stride_days
    
    logger.info(f"Generating {num_samples} transitions with {stride_days}-day stride")
    logger.info(f"Test data: {len(test_data)} samples ({test_data['timestamp_utc'].min()} to {test_data['timestamp_utc'].max()})")
    
    pbar = tqdm(total=num_samples, desc="Generating RL data")
    
    for i in range(0, len(test_data) - window_size, stride):
        if len(transitions) >= num_samples:
            break
        
        # Extract window
        window = test_data.iloc[i:i+window_size].copy()
        forecast_start = window['timestamp_utc'].iloc[0]
        ground_truth = window['power_norm'].values
        
        # Compute metrics with simulated forecast errors
        metrics = compute_metrics_from_ground_truth(ground_truth, window)
        
        # Build context
        context = {
            'forecast_start': forecast_start,
            'blend_short': 0.33,
            'blend_long': 0.33,
            'blend_physics': 0.34,
            'actions_since_retrain': i // stride,
            'cloud_cover': float(window['cloud_cover'].iloc[0]) if 'cloud_cover' in window else 0.5,
            'temperature': float(window['temperature_2m'].iloc[0]) if 'temperature_2m' in window else 20.0,
            'data_drift': np.random.uniform(0.0, 0.05)  # Simulate drift
        }
        
        # Build state
        state = build_rl_state(metrics, context)
        
        # Simulate action
        action, action_name = simulate_heuristic_action(state)
        
        # Next state (look ahead 1 day)
        if i + stride + window_size < len(test_data):
            next_window = test_data.iloc[i+stride:i+stride+window_size].copy()
            next_forecast_start = next_window['timestamp_utc'].iloc[0]
            next_ground_truth = next_window['power_norm'].values
            
            next_metrics = compute_metrics_from_ground_truth(next_ground_truth, next_window)
            next_context = context.copy()
            next_context['forecast_start'] = next_forecast_start
            next_state = build_rl_state(next_metrics, next_context)
        else:
            next_state = state.copy()
        
        # Compute reward
        reward = compute_reward(state, action, next_state)
        
        # Build transition
        transition = {
            'sample_idx': len(transitions),
            'timestamp': pd.Timestamp.now(tz='UTC').isoformat(),
            'forecast_start': forecast_start.isoformat(),
            'action': action,
            'action_name': action_name,
            'reward': reward,
            'action_success': True,
            **{f'state_{j}': state[j] for j in range(len(state))},
            **{f'next_state_{j}': next_state[j] for j in range(len(next_state))},
            'short_rmse_1h': metrics['short_rmse_1h'],
            'long_rmse_30d': metrics['long_rmse_30d'],
            'physics_residual': metrics['physics_residual'],
            'blend_short': context['blend_short'],
            'blend_long': context['blend_long'],
            'blend_physics': context['blend_physics']
        }
        
        transitions.append(transition)
        pbar.update(1)
    
    pbar.close()
    logger.info(f"Generated {len(transitions)} transitions")
    return transitions


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate RL training data (simulated forecast)")
    parser.add_argument('--test-data', type=str, 
                       default='/home/dwijenayake/pv_forecast_30d/data/processed/plant_level/plant_03/15min_pca32/test.parquet')
    parser.add_argument('--num-samples', type=int, default=200)
    parser.add_argument('--stride-days', type=int, default=1)
    parser.add_argument('--output', type=str, 
                       default='/home/dwijenayake/pv_forecast_30d/data/rl_transitions/historical_batch.parquet')
    
    args = parser.parse_args()
    
    # Load test data
    logger.info(f"Loading test data from {args.test_data}")
    test_data = pd.read_parquet(args.test_data)
    test_data = test_data.sort_values('timestamp_utc').reset_index(drop=True)
    
    # Generate transitions
    transitions = generate_rl_transitions(
        test_data=test_data,
        num_samples=args.num_samples,
        stride_days=args.stride_days
    )
    
    if len(transitions) == 0:
        logger.error("No transitions generated!")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(transitions)
    
    # Print statistics
    logger.info(f"\n{'='*60}")
    logger.info(f"RL Training Data Summary")
    logger.info(f"{'='*60}")
    logger.info(f"Total transitions: {len(df)}")
    logger.info(f"\nAction distribution:")
    print(df['action_name'].value_counts())
    logger.info(f"\nReward statistics:")
    logger.info(f"  Mean: {df['reward'].mean():.4f}")
    logger.info(f"  Std:  {df['reward'].std():.4f}")
    logger.info(f"  Min:  {df['reward'].min():.4f}")
    logger.info(f"  Max:  {df['reward'].max():.4f}")
    logger.info(f"  Positive: {(df['reward'] > 0).sum()}/{len(df)} ({100*(df['reward']>0).sum()/len(df):.1f}%)")
    logger.info(f"\nState variance (RMSE components):")
    logger.info(f"  short_rmse_1h:  {df['short_rmse_1h'].mean():.4f} ± {df['short_rmse_1h'].std():.4f}")
    logger.info(f"  long_rmse_30d:  {df['long_rmse_30d'].mean():.4f} ± {df['long_rmse_30d'].std():.4f}")
    logger.info(f"  physics_residual: {df['physics_residual'].mean():.4f} ± {df['physics_residual'].std():.4f}")
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    
    logger.info(f"\n✅ Saved {len(df)} transitions to {output_path}")
    logger.info(f"   Size: {output_path.stat().st_size / 1024:.1f} KB")
    logger.info(f"\nReady for DDQN training!")
    logger.info(f"  python scripts/train_rl_offline.py --data {output_path}")


if __name__ == "__main__":
    main()
