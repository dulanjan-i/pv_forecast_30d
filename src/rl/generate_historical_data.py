"""
Generate RL training data from historical test set.

Strategy:
1. Load historical test data (Oct-Nov 2023)
2. Run PhysicsAwareForecaster in sliding windows (daily steps)
3. Compute actual RMSE (short-term 1h, long-term 30d, physics residual)
4. Build RL transitions: (state with real RMSE, action, reward, next_state)
5. Save to parquet for DDQN training

This gives us realistic state distributions based on actual forecast performance.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from tqdm import tqdm
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.physics_aware_forecaster import PhysicsAwareForecaster
from src.rl.rl_meta_controller import RLMetaControllerSystem

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_test_data(test_path: str) -> pd.DataFrame:
    """Load and prepare test data."""
    logger.info(f"Loading test data from {test_path}")
    df = pd.read_parquet(test_path)
    
    # Ensure timestamp is sorted
    df = df.sort_values('timestamp_utc').reset_index(drop=True)
    
    logger.info(f"Loaded {len(df)} samples from {df['timestamp_utc'].min()} to {df['timestamp_utc'].max()}")
    return df


def compute_rmse(forecast: np.ndarray, ground_truth: np.ndarray, horizon: int) -> float:
    """Compute RMSE over a specific horizon."""
    if len(forecast) < horizon or len(ground_truth) < horizon:
        return 0.05  # Default fallback
    
    forecast_h = forecast[:horizon]
    gt_h = ground_truth[:horizon]
    
    rmse = np.sqrt(np.mean((forecast_h - gt_h) ** 2))
    return float(rmse)


def run_forecast_and_compute_metrics(
    forecaster: PhysicsAwareForecaster,
    window_data: pd.DataFrame,
    ground_truth: np.ndarray,
    forecast_start: pd.Timestamp
) -> Dict:
    """
    Run forecast and compute all RL state metrics.
    
    Returns:
        metrics dict with RMSE values, residuals, etc.
    """
    try:
        # Run forecast (2880 steps = 30 days @ 15min)
        forecast = forecaster.predict_30d(
            forecast_start=forecast_start,
            weather_df=window_data,
            historical_df=window_data  # Use window as historical context
        )
        
        # Convert to numpy if needed
        if torch.is_tensor(forecast):
            forecast = forecast.cpu().numpy()
        elif isinstance(forecast, pd.Series):
            forecast = forecast.values
        
        # Compute RMSE at different horizons
        # 1 hour = 4 steps @ 15min
        rmse_1h = compute_rmse(forecast, ground_truth, horizon=4)
        
        # 6 hours = 24 steps
        rmse_6h = compute_rmse(forecast, ground_truth, horizon=24)
        
        # 24 hours = 96 steps
        rmse_24h = compute_rmse(forecast, ground_truth, horizon=96)
        
        # 7 days = 672 steps
        rmse_7d = compute_rmse(forecast, ground_truth, horizon=672)
        
        # 30 days = 2880 steps
        rmse_30d = compute_rmse(forecast, ground_truth, horizon=2880)
        
        # Physics residual (compare TFT blend vs PVLib)
        # For now, use relative difference at 24h horizon
        if 'pvlib_ac_kw' in window_data.columns and len(ground_truth) >= 96:
            pvlib_baseline = window_data['pvlib_ac_kw'].values[:96]
            physics_residual = np.mean(np.abs(ground_truth[:96] - pvlib_baseline))
        else:
            physics_residual = 0.02  # Default
        
        metrics = {
            'short_rmse_1h': rmse_1h,
            'short_rmse_6h': rmse_6h,
            'short_rmse_24h': rmse_24h,
            'long_rmse_7d': rmse_7d,
            'long_rmse_30d': rmse_30d,
            'physics_residual': physics_residual,
            'forecast_length': len(forecast),
            'success': True
        }
        
        return metrics
        
    except Exception as e:
        logger.warning(f"Forecast failed: {e}")
        # Return default fallback metrics
        return {
            'short_rmse_1h': 0.05,
            'short_rmse_6h': 0.06,
            'short_rmse_24h': 0.08,
            'long_rmse_7d': 0.10,
            'long_rmse_30d': 0.12,
            'physics_residual': 0.03,
            'forecast_length': 0,
            'success': False
        }


def build_rl_state(metrics: Dict, context: Dict) -> np.ndarray:
    """
    Build 35-dimensional RL state vector from metrics.
    
    State encoding (matching RLMetaControllerSystem.collect_metrics):
        [0-4]:   RMSE (short_1h, short_6h, short_24h, long_7d, long_30d)
        [5-6]:   Confidence scores
        [7-9]:   Drift indicators
        [10-14]: Blend weights & history
        [15-19]: Temporal context (hour, day, season, etc.)
        [20-24]: Weather features
        [25-29]: Compute budget
        [30-34]: Physics baseline quality
    """
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
    
    # [7-9] Drift (placeholder - would need historical comparison)
    state[7] = 0.0  # data drift
    state[8] = 0.0  # short-long mismatch
    state[9] = context.get('forecast_success', 1.0)
    
    # [10-14] Blend weights (default equal blend initially)
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
        state[17] = (forecast_time.month - 1) / 12.0  # season
        state[18] = forecast_time.dayofweek / 7.0
        state[19] = 1.0 if forecast_time.dayofweek >= 5 else 0.0  # weekend
    
    # [20-24] Weather (from context if available)
    state[20] = context.get('cloud_cover', 0.5)
    state[21] = context.get('ghi', 0.3)
    state[22] = context.get('dni', 0.3)
    state[23] = context.get('temperature', 0.5)
    state[24] = 1.0  # weather quality
    
    # [25-29] Compute budget (normalized)
    state[25] = 1.0  # budget remaining
    state[26] = 0.5  # priority score
    state[27] = 1.0  # API status
    state[28] = 0.95  # API agreement
    state[29] = 0.0  # cost accumulated
    
    # [30-34] Physics baseline
    state[30] = 1.0 - min(metrics['physics_residual'] * 10, 0.9)
    state[31] = 0.0  # pvlib calibration age
    state[32] = context.get('solar_zenith', 0.5)
    state[33] = context.get('clearsky_ghi', 0.5)
    state[34] = 1.0  # pvlib confidence
    
    return state


def simulate_heuristic_action(state: np.ndarray) -> Tuple[int, str]:
    """
    Simulate heuristic action based on state (mimics LocalAdvisors).
    
    Heuristic rules:
    - If short RMSE high (>0.08) → FINE_TUNE_SHORT (1)
    - If long RMSE high (>0.10) → FINE_TUNE_LONG (2)
    - If physics residual high (>0.05) → RECALIBRATE_PVLIB (3)
    - If both RMSE high → SUGGEST_RETRAIN (7)
    - Otherwise → BLEND_HIGH_PHYSICS (6) or BLEND_MEDIUM (5)
    """
    short_rmse = state[0]
    long_rmse = state[1]
    physics_res = state[2]
    
    # Thresholds
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
    Compute reward based on RMSE improvement.
    
    R = w1*(RMSE_improvement_short) + w2*(RMSE_improvement_long) - w3*action_cost
    """
    # Weights
    w_short = 1.0
    w_long = 0.5
    w_cost = 0.2
    
    # RMSE improvement
    rmse_improve_short = (state[0] - next_state[0]) / 0.01
    rmse_improve_long = (state[1] - next_state[1]) / 0.01
    
    # Action costs (from RLMetaControllerSystem.ACTION_COSTS)
    action_costs = {
        0: 0.1,  # MAINTAIN
        1: 0.3,  # FINE_TUNE_SHORT
        2: 0.4,  # FINE_TUNE_LONG
        3: 0.5,  # RECALIBRATE_PVLIB
        4: 0.2,  # BLEND_SHORT
        5: 0.2,  # BLEND_MEDIUM
        6: 0.2,  # BLEND_HIGH_PHYSICS
        7: 1.0,  # SUGGEST_RETRAIN
    }
    
    reward = (
        w_short * rmse_improve_short +
        w_long * rmse_improve_long -
        w_cost * action_costs.get(action, 0.3)
    )
    
    return float(reward)


def generate_rl_transitions(
    test_data: pd.DataFrame,
    forecaster: PhysicsAwareForecaster,
    num_samples: int = 100,
    stride_days: int = 1
) -> List[Dict]:
    """
    Generate RL transitions by running forecaster on sliding windows.
    
    Args:
        test_data: Historical test data with ground truth
        forecaster: Initialized PhysicsAwareForecaster
        num_samples: Number of transitions to generate
        stride_days: Days to move forward for each sample (96 steps @ 15min)
        
    Returns:
        List of transition dicts with (state, action, reward, next_state)
    """
    transitions = []
    
    # Window size: 30 days = 2880 steps @ 15min
    window_size = 2880
    stride = 96 * stride_days  # 1 day stride
    
    logger.info(f"Generating {num_samples} transitions with {stride_days}-day stride")
    logger.info(f"Window size: {window_size} steps (30 days @ 15min)")
    
    pbar = tqdm(total=num_samples, desc="Generating RL data")
    
    for i in range(0, len(test_data) - window_size, stride):
        if len(transitions) >= num_samples:
            break
        
        # Extract window
        window = test_data.iloc[i:i+window_size].copy()
        forecast_start = window['timestamp_utc'].iloc[0]
        ground_truth = window['power_norm'].values
        
        # Run forecast and compute metrics
        metrics = run_forecast_and_compute_metrics(
            forecaster, window, ground_truth, forecast_start
        )
        
        # Build context
        context = {
            'forecast_start': forecast_start,
            'forecast_success': 1.0 if metrics['success'] else 0.0,
            'blend_short': 0.33,
            'blend_long': 0.33,
            'blend_physics': 0.34,
            'actions_since_retrain': i // stride,
            'cloud_cover': float(window['cloud_cover'].iloc[0]) if 'cloud_cover' in window else 0.5,
            'temperature': float(window['temperature_2m'].iloc[0]) if 'temperature_2m' in window else 20.0,
        }
        
        # Build state
        state = build_rl_state(metrics, context)
        
        # Simulate heuristic action
        action, action_name = simulate_heuristic_action(state)
        
        # For next_state, look ahead 1 day if available
        if i + stride + window_size < len(test_data):
            next_window = test_data.iloc[i+stride:i+stride+window_size].copy()
            next_forecast_start = next_window['timestamp_utc'].iloc[0]
            next_ground_truth = next_window['power_norm'].values
            
            next_metrics = run_forecast_and_compute_metrics(
                forecaster, next_window, next_ground_truth, next_forecast_start
            )
            
            next_context = context.copy()
            next_context['forecast_start'] = next_forecast_start
            next_context['forecast_success'] = 1.0 if next_metrics['success'] else 0.0
            
            next_state = build_rl_state(next_metrics, next_context)
        else:
            # Last sample: next_state = current state
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
            'action_success': metrics['success'],
            
            # State (35 dims)
            **{f'state_{j}': state[j] for j in range(len(state))},
            
            # Next state (35 dims)
            **{f'next_state_{j}': next_state[j] for j in range(len(next_state))},
            
            # Key metrics for analysis
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
    
    parser = argparse.ArgumentParser(description="Generate RL training data from historical test set")
    parser.add_argument('--test-data', type=str, 
                       default='/home/dwijenayake/pv_forecast_30d/data/processed/plant_level/plant_03/15min_pca32/test.parquet',
                       help='Path to historical test data')
    parser.add_argument('--num-samples', type=int, default=100,
                       help='Number of RL transitions to generate')
    parser.add_argument('--stride-days', type=int, default=1,
                       help='Days to stride forward for each sample')
    parser.add_argument('--output', type=str, 
                       default='/home/dwijenayake/pv_forecast_30d/data/rl_transitions/historical_batch.parquet',
                       help='Output parquet file')
    
    args = parser.parse_args()
    
    # Load test data
    test_data = load_test_data(args.test_data)
    
    # Initialize forecaster with CANONICAL HARDCODED PATHS
    logger.info("Initializing PhysicsAwareForecaster...")
    forecaster = PhysicsAwareForecaster(
        short_ckpt="/home/dwijenayake/pv_forecast_30d/V1.0_FINAL_TFT/shorthead_seed42/checkpoints/best.ckpt",
        long_ckpt="/home/dwijenayake/pv_forecast_30d/V1.0_FINAL_TFT/longhead_seed43/checkpoints/best.ckpt",
        plant_metadata="/home/dwijenayake/pv_forecast_30d/V1.0_FINAL_TFT/plant_metadata/plant_03.json",
        short_train_parquet="/home/dwijenayake/pv_forecast_30d/data/processed/plant_level/plant_03/15min_pca32/train.parquet",
        long_train_parquet="/home/dwijenayake/pv_forecast_30d/data/processed/plant_level/plant_03/hourly_longhead/train.parquet",
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Generate transitions
    transitions = generate_rl_transitions(
        test_data=test_data,
        forecaster=forecaster,
        num_samples=args.num_samples,
        stride_days=args.stride_days
    )
    
    if len(transitions) == 0:
        logger.error("No transitions generated!")
        return
    
    # Convert to DataFrame and save
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
    logger.info(f"\nNext steps:")
    logger.info(f"  1. Verify: python -c 'import pandas as pd; print(pd.read_parquet(\"{output_path}\").info())'")
    logger.info(f"  2. Train DDQN: python scripts/train_rl_offline.py --data {output_path}")


if __name__ == "__main__":
    main()
