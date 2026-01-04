#!/usr/bin/env python3
"""
Fetch ERA5 Reanalysis for 2024-2025 Extended Evaluation

Downloads ERA5 historical weather data for the 764-day period after our test set
(Dec 2023 - Dec 2025) to create extended evaluation dataset for:
1. Testing TFT performance on diverse seasonal conditions
2. Generating RL training data with realistic weather variation
3. Thesis validation beyond the original test period

Author: PV Forecast Team
Date: 2026-01-03
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List
import json
import sys
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.inference.weather_client import WeatherClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ERA5Fetcher:
    """Fetch ERA5 reanalysis for extended evaluation."""
    
    def __init__(self):
        self.client = WeatherClient()
        
        # Load plant metadata
        meta_path = Path("/home/dwijenayake/pv_forecast_30d/V1.0_FINAL_TFT/plant_metadata/plant_03.json")
        with open(meta_path) as f:
            self.metadata = json.load(f)
        
        self.lat = self.metadata['latitude']
        self.lon = self.metadata['longitude']
        self.tilt = self.metadata['tilt_deg']
        self.azimuth = self.metadata['azimuth_deg']
    
    def fetch_era5_chunk(
        self,
        start_date: str,
        days: int = 30
    ) -> pd.DataFrame:
        """
        Fetch a chunk of ERA5 reanalysis data.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            days: Number of days to fetch
        
        Returns:
            DataFrame with ERA5 weather @ 15-min resolution
        """
        logger.info(f"Fetching ERA5: {start_date} (+{days}d)")
        
        # Fetch historical data (OpenMeteo uses ERA5 for historical)
        data = self.client.fetch_and_prepare(
            latitude=self.lat,
            longitude=self.lon,
            start_time=start_date,
            days=days,
            tilt=self.tilt,
            azimuth=self.azimuth,
            resolution="15min",
            auto_select=False  # Force historical data, not forecast
        )
        
        data['data_source'] = 'era5'
        
        logger.info(f"  ✓ Fetched {len(data)} steps @ 15min")
        return data
    
    def fetch_reanalysis(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
cy_days: int = 7  # Issue forecast every N days
    ) -> pd.DataFrame:
        """
        Generate complete forecast evaluation dataset.
        
        Args:
            start_date: First forecast issue date
            end_date: Last forecast issue date
            issue_frequency_days: Days between forecast issues (7 = weekly)
        
        Returns:
            DataFrame with forecasts, reanalysis, and errors for entire period
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"GENERATING FORECAST EVALUATION DATASET")
        logger.info(f"Period: {start_date} → {end_date}")
        logger.info(f"Forecast frequency: Every {issue_frequency_days} days")
        logger.info(f"{'='*70}\n")
        
        all_data = []
        
        # Generate issue dates
        current = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        issue_dates = []
        
        while current <= end:
            issue_dates.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=issue_frequency_days)
        
        logger.info(f"Will fetch {len(issue_dates)} forecasts (every {issue_frequency_days} days)")
        
        # Fetch forecasts + reanalysis for each issue date
        for i, issue_date in enumerate(issue_dates):
            logger.info(f"\n[{i+1}/{len(issue_dates)}] Processing forecast issued {issue_date}")
            
            try:
                # Fetch 30-day forecast
                forecast = self.fetch_historical_forecast(issue_date, horizon_days=30)
                
                # Fetch corresponding reanalysis
                forecast_end = (pd.to_datetime(issue_date) + timedelta(days=30)).strftime("%Y-%m-%d")
                reanalysis = self.fetch_reanalysis(issue_date, forecast_end)
                
                # Compute errors
                merged = self.compute_forecast_errors(forecast, reanalysis)
                
                # Add issue metadata
                merged['issue_idx'] = i
                merged['issue_date'] = issue_date
                
                all_data.append(merged)
                
            except Exception as e:
                logger.error(f"  ✗ Failed to fetch {issue_date}: {e}")
                continue
        
        # Combine all forecasts
        if not all_data:
            raise RuntimeError("No forecasts fetched successfully!")
        
        full_dataset = pd.concat(all_data, ignore_index=True)
        
        logger.info(f"\n{'='*70}")
        logger.info(f"DATASET GENERATION COMPLETE")
        logger.info(f"{'='*70}")
        logger.info(f"Total timesteps: {len(full_dataset)}")
        logger.info(f"Total forecasts: {len(issue_dates)}")
        logger.info(f"Date range: {full_dataset['timestamp_utc'].min()} → {full_dataset['timestamp_utc'].max()}")
        logger.info(f"\nForecast Error Statistics:")
        logger.infra5_dataset(
        self,
        start_date: str = "2023-12-01",
        end_date: str = "2025-12-31",
        chunk_days: int = 30
    ) -> pd.DataFrame:
        """
        Fetch ERA5 reanalysis in chunks and combine.
       Fetch ERA5 reanalysis for 2023-2025 extended evaluation."""
    
    fetcher = ERA5Fetcher()
    
    # Fetch ERA5 (Dec 2023 - Dec 2025 = 25 months)
    dataset = fetcher.generate_era5_dataset(
        start_date="2023-12-01",
        end_date="2025-12-31",
        chunk_days=30  # Fetch in 30-day chunks
    )
    
    # Save
    output_path = Path("/home/dwijenayake/pv_forecast_30d/data/processed/era5_2023_2025_extended.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(output_path, index=False)
    
    logger.info(f"\n✅ Saved to: {output_path}")
    logger.info(f"   Size: {output_path.stat().st_size / (1024**2):.2f} MB")
    logger.info(f"   Timesteps: {len(dataset):,}")
    logger.info(f"\nNext steps:")
    logger.info(f"  1. Generate RL transitions: python src/rl/generate_era5_eval_transitions.py")
    logger.info(f"  2. Merge datasets: Combine with existing 33 samples")
    logger.info(f"  3. Train DDQN: python src/training/train_rl_offline.py --epochs 5000
        logger.info(f"Total days to fetch: {total_days}")
        logger.info(f"Estimated chunks: {total_days // chunk_days + 1}\n")
        
        chunk_idx = 0
        while current <= end_dt:
            chunk_idx += 1
            
            # Calculate days for this chunk
            days_left = (end_dt - current).days + 1
            days_to_fetch = min(chunk_days, days_left)
            
            try:
                logger.info(f"[Chunk {chunk_idx}] {current.strftime('%Y-%m-%d')} (+{days_to_fetch}d)")
                
                # Fetch chunk
                chunk_data = self.fetch_era5_chunk(
                    start_date=current.strftime("%Y-%m-%d"),
                    days=days_to_fetch
                )
                
                chunk_data['chunk_idx'] = chunk_idx
                all_data.append(chunk_data)
                
                logger.info(f"  ✓ Chunk {chunk_idx} complete ({len(chunk_data)} steps)\n")
                
            except Exception as e:
                logger.error(f"  ✗ Failed chunk {chunk_idx}: {e}\n")
                # Try smaller chunk on error
                if days_to_fetch > 7:
                    logger.info(f"  Retrying with smaller chunk (7 days)...")
                    try:
                        chunk_data = self.fetch_era5_chunk(
                            start_date=current.strftime("%Y-%m-%d"),
                            days=7
                        )
                        chunk_data['chunk_idx'] = chunk_idx
                        all_data.append(chunk_data)
                        logger.info(f"  ✓ Retry successful\n")
                        current += timedelta(days=7)
                        continue
                    except:
                        pass
            
            # Move to next chunk
            current += timedelta(days=days_to_fetch)
        
        # Combine all chunks
        if not all_data:
            raise RuntimeError("No ERA5 data fetched!")
        
        full_dataset = pd.concat(all_data, ignore_index=True)
        
        # Remove duplicates (chunk overlaps)
        full_dataset = full_dataset.drop_duplicates(subset=['timestamp_utc']).sort_values('timestamp_utc').reset_index(drop=True)
        
        logger.info(f"\n{'='*70}")
        logger.info(f"ERA5 FETCH COMPLETE")
        logger.info(f"{'='*70}")
        logger.info(f"Total timesteps: {len(full_dataset):,}")
        logger.info(f"Date range: {full_dataset['timestamp_utc'].min()} → {full_dataset['timestamp_utc'].max()}")
        logger.info(f"Duration: {(full_dataset['timestamp_utc'].max() - full_dataset['timestamp_utc'].min()).days} days")
        logger.info(f"\nWeather Statistics:")
        logger.info(f"  GHI mean: {full_dataset['shortwave_radiation_instant'].mean():.1f} W/m²")
        logger.info(f"  GHI max: {full_dataset['shortwave_radiation_instant'].max():.1f} W/m²")
        logger.info(f"  Temp mean: {full_dataset['temperature_2m'].mean():.1f} °C")
        logger.info(f"  Temp range: {full_dataset['temperature_2m'].min():.1f} to {full_dataset['temperature_2m'].max():.1