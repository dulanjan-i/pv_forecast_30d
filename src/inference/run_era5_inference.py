#!/usr/bin/env python3
"""
Run TFT+PVLib inference on ERA5 weather data to generate predicted PV power.

Steps:
1. Load ERA5 weather parquet (already has proper timestamps, PVLib columns)
2. For each 30-day forecast window:
   - Extract 7-day history (for long-head encoder)
   - Run PhysicsAwareForecaster.predict_30d()
   - Save predicted power time series
3. Compute RMSEs by comparing overlapping forecasts
4. Save predictions + RMSEs for RL pipeline

Author: PV Forecast Team
Date: 2026-01-03
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from tqdm import tqdm
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.inference.physics_aware_forecaster import PhysicsAwareForecaster

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_inference_on_era5(
    era5_path: str,
    forecaster: PhysicsAwareForecaster,
    stride_days: int = 7
) -> pd.DataFrame:
    """
    Run TFT+PVLib on ERA5 data in rolling windows.
    
    Args:
        era5_path: Path to ERA5 parquet
        forecaster: Initialized forecaster
        stride_days: Days between forecast starts
    
    Returns:
        DataFrame with forecasts and metadata
    """
    # Load ERA5
    logger.info(f"Loading ERA5 from {era5_path}")
    era5 = pd.read_parquet(era5_path)
    logger.info(f"  {len(era5)} timesteps from {era5['timestamp_utc'].min()} to {era5['timestamp_utc'].max()}")
    
    # Window parameters
    history_steps = 168 * 4  # 7 days @ 15min for long-head encoder
    forecast_steps = 2880    # 30 days @ 15min
    stride_steps = 96 * stride_days  # days to hours to 15min steps
    
    total_needed = history_steps + forecast_steps
    
    logger.info(f"\nForecast parameters:")
    logger.info(f"  History: {history_steps} steps (7 days)")
    logger.info(f"  Forecast: {forecast_steps} steps (30 days)")
    logger.info(f"  Stride: {stride_steps} steps ({stride_days} days)")
    logger.info(f"  Total needed per window: {total_needed} steps")
    
    # Calculate number of possible windows
    num_windows = (len(era5) - total_needed) // stride_steps + 1
    logger.info(f"  Possible windows: {num_windows}\n")
    
    all_forecasts = []
    
    for window_idx in tqdm(range(num_windows), desc="Running forecasts"):
        start_idx = window_idx * stride_steps
        
        # Check if we have enough data
        if start_idx + total_needed > len(era5):
            logger.warning(f"Window {window_idx}: Not enough data, skipping")
            break
        
        # Extract history and forecast periods
        history = era5.iloc[start_idx:start_idx + history_steps].copy()
        forecast_window = era5.iloc[start_idx + history_steps:start_idx + total_needed].copy()
        
        forecast_start = forecast_window['timestamp_utc'].iloc[0]
        
        try:
            # Run forecast
            predictions = forecaster.predict_30d(
                forecast_start=forecast_start,
                weather_df=forecast_window,
                historical_df=history
            )
            
            # Convert to numpy if tensor
            if torch.is_tensor(predictions):
                predictions = predictions.cpu().numpy()
            
            # Create forecast record
            forecast_record = {
                'window_idx': window_idx,
                'forecast_start': forecast_start,
                'forecast_end': forecast_window['timestamp_utc'].iloc[-1],
                'num_steps': len(predictions),
                'predictions': predictions,
                'timestamps': forecast_window['timestamp_utc'].values,
                'success': True
            }
            
            all_forecasts.append(forecast_record)
            
            logger.info(f"  Window {window_idx}: {forecast_start} → {len(predictions)} steps")
            
        except Exception as e:
            logger.error(f"  Window {window_idx}: Failed - {e}")
            continue
    
    logger.info(f"\n✓ Completed {len(all_forecasts)} forecasts")
    
    return all_forecasts


def compute_cross_forecast_rmses(forecasts: List[Dict]) -> pd.DataFrame:
    """
    Compute RMSEs by comparing overlapping forecasts.
    
    For each pair of forecasts with overlapping periods:
    - Compute RMSE in the overlap region
    - This gives us forecast consistency metrics
    
    Also compute:
    - Short-term RMSE (1h, 6h, 24h between consecutive forecasts)
    - Long-term RMSE (7d, 30d extrapolation drift)
    """
    logger.info("\n" + "="*70)
    logger.info("COMPUTING CROSS-FORECAST RMSEs")
    logger.info("="*70)
    
    rmse_records = []
    
    # For each forecast, compare with next forecast in overlapping region
    for i in range(len(forecasts) - 1):
        curr = forecasts[i]
        next_f = forecasts[i + 1]
        
        curr_times = pd.to_datetime(curr['timestamps'])
        next_times = pd.to_datetime(next_f['timestamps'])
        
        # Find overlap
        overlap_start = max(curr_times.min(), next_times.min())
        overlap_end = min(curr_times.max(), next_times.max())
        
        if overlap_end <= overlap_start:
            continue
        
        # Extract overlapping predictions
        curr_mask = (curr_times >= overlap_start) & (curr_times <= overlap_end)
        next_mask = (next_times >= overlap_start) & (next_times <= overlap_end)
        
        curr_pred = curr['predictions'][curr_mask]
        next_pred = next_f['predictions'][next_mask]
        
        # Align lengths
        min_len = min(len(curr_pred), len(next_pred))
        curr_pred = curr_pred[:min_len]
        next_pred = next_pred[:min_len]
        
        if len(curr_pred) == 0:
            continue
        
        # Compute RMSEs at different horizons
        rmse_1h = np.sqrt(np.mean((curr_pred[:4] - next_pred[:4])**2)) if len(curr_pred) >= 4 else 0.0
        rmse_6h = np.sqrt(np.mean((curr_pred[:24] - next_pred[:24])**2)) if len(curr_pred) >= 24 else 0.0
        rmse_24h = np.sqrt(np.mean((curr_pred[:96] - next_pred[:96])**2)) if len(curr_pred) >= 96 else 0.0
        rmse_7d = np.sqrt(np.mean((curr_pred[:672] - next_pred[:672])**2)) if len(curr_pred) >= 672 else 0.0
        rmse_30d = np.sqrt(np.mean((curr_pred - next_pred)**2))
        
        record = {
            'forecast_pair': f"{i}-{i+1}",
            'forecast_start_1': curr['forecast_start'],
            'forecast_start_2': next_f['forecast_start'],
            'overlap_steps': min_len,
            'overlap_hours': min_len * 0.25,  # 15min = 0.25h
            'rmse_1h': rmse_1h,
            'rmse_6h': rmse_6h,
            'rmse_24h': rmse_24h,
            'rmse_7d': rmse_7d,
            'rmse_30d': rmse_30d,
        }
        
        rmse_records.append(record)
    
    rmse_df = pd.DataFrame(rmse_records)
    
    # Print summary statistics
    logger.info(f"\nRMSE Summary ({len(rmse_df)} forecast pairs):")
    logger.info(f"{'='*70}")
    logger.info(f"  1-hour RMSE:  {rmse_df['rmse_1h'].mean():.4f} ± {rmse_df['rmse_1h'].std():.4f}")
    logger.info(f"  6-hour RMSE:  {rmse_df['rmse_6h'].mean():.4f} ± {rmse_df['rmse_6h'].std():.4f}")
    logger.info(f"  24-hour RMSE: {rmse_df['rmse_24h'].mean():.4f} ± {rmse_df['rmse_24h'].std():.4f}")
    logger.info(f"  7-day RMSE:   {rmse_df['rmse_7d'].mean():.4f} ± {rmse_df['rmse_7d'].std():.4f}")
    logger.info(f"  30-day RMSE:  {rmse_df['rmse_30d'].mean():.4f} ± {rmse_df['rmse_30d'].std():.4f}")
    logger.info(f"{'='*70}\n")
    
    return rmse_df


def save_predictions(forecasts: List[Dict], output_path: str):
    """Save all predictions to parquet."""
    # Flatten forecasts into time series
    records = []
    
    for fc in forecasts:
        for i, (ts, pred) in enumerate(zip(fc['timestamps'], fc['predictions'])):
            records.append({
                'timestamp_utc': pd.Timestamp(ts),
                'window_idx': fc['window_idx'],
                'forecast_start': fc['forecast_start'],
                'step_ahead': i,
                'hours_ahead': i * 0.25,
                'predicted_power': float(pred)
            })
    
    df = pd.DataFrame(records)
    df.to_parquet(output_path, index=False)
    logger.info(f"✓ Saved {len(df)} predictions to {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--era5', type=str,
                       default='/home/dwijenayake/pv_forecast_30d/data/processed/era5_2023_2025_extended.parquet',
                       help='Path to ERA5 weather data')
    parser.add_argument('--output-predictions', type=str,
                       default='/home/dwijenayake/pv_forecast_30d/data/processed/era5_predictions.parquet',
                       help='Output path for predictions')
    parser.add_argument('--output-rmses', type=str,
                       default='/home/dwijenayake/pv_forecast_30d/data/processed/era5_rmses.parquet',
                       help='Output path for RMSEs')
    parser.add_argument('--stride-days', type=int, default=7,
                       help='Days between forecast starts')
    
    args = parser.parse_args()
    
    # Initialize forecaster
    logger.info("="*70)
    logger.info("INITIALIZING TFT+PVLIB FORECASTER")
    logger.info("="*70)
    
    forecaster = PhysicsAwareForecaster(
        short_ckpt="/home/dwijenayake/pv_forecast_30d/V1.0_FINAL_TFT/shorthead_seed42/checkpoints/best.ckpt",
        long_ckpt="/home/dwijenayake/pv_forecast_30d/V1.0_FINAL_TFT/longhead_seed43/checkpoints/best.ckpt",
        plant_metadata="/home/dwijenayake/pv_forecast_30d/V1.0_FINAL_TFT/plant_metadata/plant_03.json",
        short_train_parquet="/home/dwijenayake/pv_forecast_30d/data/processed/plant_level/plant_03/15min_pca32/train.parquet",
        long_train_parquet="/home/dwijenayake/pv_forecast_30d/data/processed/plant_level/plant_03/hourly_longhead/train.parquet",
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    logger.info("✓ Forecaster ready\n")
    
    # Run inference
    logger.info("="*70)
    logger.info("STEP 3: RUNNING TFT+PVLIB INFERENCE ON ERA5")
    logger.info("="*70)
    
    forecasts = run_inference_on_era5(
        era5_path=args.era5,
        forecaster=forecaster,
        stride_days=args.stride_days
    )
    
    if len(forecasts) == 0:
        logger.error("No forecasts generated!")
        return
    
    # Save predictions
    logger.info("="*70)
    logger.info("STEP 4: SAVING PREDICTIONS")
    logger.info("="*70)
    save_predictions(forecasts, args.output_predictions)
    
    # Compute RMSEs
    logger.info("="*70)
    logger.info("STEP 5: COMPUTING RMSES")
    logger.info("="*70)
    rmse_df = compute_cross_forecast_rmses(forecasts)
    rmse_df.to_parquet(args.output_rmses, index=False)
    logger.info(f"✓ Saved RMSEs to {args.output_rmses}")
    
    # Final summary
    logger.info("\n" + "="*70)
    logger.info("INFERENCE COMPLETE")
    logger.info("="*70)
    logger.info(f"Forecasts generated: {len(forecasts)}")
    logger.info(f"Predictions saved: {args.output_predictions}")
    logger.info(f"RMSEs saved: {args.output_rmses}")
    logger.info(f"\nNext step: Generate RL transitions using these predictions + RMSEs")
    logger.info("="*70)


if __name__ == "__main__":
    main()
