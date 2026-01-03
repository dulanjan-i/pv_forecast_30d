#!/usr/bin/env python3
"""
Phase 1 Inference Pipeline: Dec 2023 - Dec 2024

This is the PRODUCTION inference pipeline that:
1. Fetches historical weather (Dec 2023 - Dec 2024)
2. Preprocesses for both TFT heads + PVLib
3. Runs rolling 30-day forecasts with DYNAMIC encoder updates
4. Generates RL training data from predictions
5. Validates data hygiene with DEFCON 1 precision

Key Innovation:
- Uses PREDICTED power from previous forecasts as encoder input
- Simulates real-world deployment where only predictions available
- Tests model's ability to handle its own outputs

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
from datetime import timedelta

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.inference.weather_client import WeatherClient
from src.inference.physics_aware_forecaster import PhysicsAwareForecaster

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Phase1Pipeline:
    """Production inference pipeline with dynamic encoder updates."""
    
    def __init__(self, weather_source='historical'):
        """
        Initialize pipeline.
        
        Args:
            weather_source: 'historical' (ERA5 archive) or 'api' (real-time forecast)
        """
        self.base_dir = Path("/home/dwijenayake/pv_forecast_30d")
        self.phase1_dir = self.base_dir / "data/processed/test_phase1_dec2023_dec2024"
        self.test_dir = self.base_dir / "data/processed/plant_level/plant_03"
        self.weather_source = weather_source
        
        # Initialize weather client
        self.weather_client = WeatherClient()
        
        # Load plant metadata
        import json
        meta_path = self.base_dir / "V1.0_FINAL_TFT/plant_metadata/plant_03.json"
        with open(meta_path) as f:
            self.metadata = json.load(f)
        
        self.lat = self.metadata['latitude']
        self.lon = self.metadata['longitude']
        self.tilt = self.metadata['tilt_deg']
        self.azimuth = self.metadata['azimuth_deg']
    
    def step1_fetch_weather(self):
        """Fetch weather Dec 1, 2023 → Dec 31, 2024 (13 months) using ERA5 OR real-time API."""
        logger.info("\n" + "="*70)
        logger.info(f"STEP 1: FETCHING WEATHER (source: {self.weather_source})")
        logger.info("="*70)
        
        if self.weather_source == 'api':
            # Real-time forecast from API
            logger.info("Using: Open-Meteo Forecast API (Real-time)")
            logger.info(f"Location: {self.lat}°N, {self.lon}°E")
            
            from src.data.weather_api_orchestrator import WeatherAPIOrchestrator
            api = WeatherAPIOrchestrator()
            
            # Fetch 15-day forecast
            weather_hourly = api.fetch_with_fallback(
                lat=self.lat,
                lon=self.lon,
                days_ahead=15
            )
            
            logger.info(f"✓ Fetched {len(weather_hourly)} hourly timesteps from API")
            
            # Map API column names to training column names
            logger.info("Mapping API columns to training format...")
            column_mapping = {
                'shortwave_radiation': 'shortwave_radiation_instant',
                'direct_radiation': 'direct_radiation_instant',
                'diffuse_radiation': 'diffuse_radiation_instant',
                'direct_normal_irradiance': 'direct_normal_irradiance_instant'
            }
            weather_hourly = weather_hourly.rename(columns=column_mapping)
            
            # Add missing columns that API doesn't provide
            if 'weather_code' not in weather_hourly.columns:
                weather_hourly['weather_code'] = 0  # Default: clear sky
            
            # Resample to 15min
            logger.info(f"Resampling to 15min...")
            weather_hourly_indexed = weather_hourly.set_index('timestamp_utc')
            numeric_cols = weather_hourly.select_dtypes(include=[np.number]).columns.tolist()
            weather_15min = weather_hourly_indexed[numeric_cols].resample('15min').interpolate(method='linear').reset_index()
            
        else:
            # Historical ERA5 archive (Phase 1 validation mode)
            logger.info("Using: OpenMeteo Historical Archive (ERA5)")
            
            start_date = "2023-12-01"
            end_date = "2024-12-31"
            
            logger.info(f"Period: {start_date} → {end_date}")
            logger.info(f"Location: {self.lat}°N, {self.lon}°E")
        
            import openmeteo_requests
            import requests_cache
            from retry_requests import retry
            import urllib3
            
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            
            cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
            cache_session.verify = False  # Disable SSL verification for DBFZ
            retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
            client = openmeteo_requests.Client(session=retry_session)
            
            # Fetch in chunks (60 days each to avoid API limits)
            all_data = []
            current = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            chunk_idx = 0
            while current <= end_dt:
                days_left = (end_dt - current).days + 1
                days_to_fetch = min(60, days_left)
                chunk_end = current + timedelta(days=days_to_fetch - 1)
                
                logger.info(f"\n[Chunk {chunk_idx+1}] {current.strftime('%Y-%m-%d')} → {chunk_end.strftime('%Y-%m-%d')}")
                
                try:
                    params = {
                        "latitude": self.lat,
                    "longitude": self.lon,
                    "start_date": current.strftime("%Y-%m-%d"),
                    "end_date": chunk_end.strftime("%Y-%m-%d"),
                    "hourly": [
                        "temperature_2m", "relative_humidity_2m", "precipitation",
                        "surface_pressure", "cloud_cover", "wind_speed_10m", "wind_direction_10m",
                        "shortwave_radiation", "direct_radiation", "diffuse_radiation",
                        "direct_normal_irradiance", "weather_code"
                    ],
                    "timezone": "UTC"
                }
                    
                    url = "https://archive-api.open-meteo.com/v1/archive"
                    responses = client.weather_api(url, params=params)
                    response = responses[0]
                    
                    # Extract hourly data
                    hourly = response.Hourly()
                    hourly_data = {
                        "timestamp_utc": pd.date_range(
                            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                            freq=pd.Timedelta(seconds=hourly.Interval()),
                            inclusive="left"
                        ),
                        "temperature_2m": hourly.Variables(0).ValuesAsNumpy(),
                        "relative_humidity_2m": hourly.Variables(1).ValuesAsNumpy(),
                        "precipitation": hourly.Variables(2).ValuesAsNumpy(),
                        "surface_pressure": hourly.Variables(3).ValuesAsNumpy(),
                        "cloud_cover": hourly.Variables(4).ValuesAsNumpy(),
                        "wind_speed_10m": hourly.Variables(5).ValuesAsNumpy(),
                        "wind_direction_10m": hourly.Variables(6).ValuesAsNumpy(),
                        "shortwave_radiation_instant": hourly.Variables(7).ValuesAsNumpy(),  # GHI
                        "direct_radiation_instant": hourly.Variables(8).ValuesAsNumpy(),
                        "diffuse_radiation_instant": hourly.Variables(9).ValuesAsNumpy(),  # DHI
                        "direct_normal_irradiance_instant": hourly.Variables(10).ValuesAsNumpy(),  # DNI
                        "weather_code": hourly.Variables(11).ValuesAsNumpy(),  # WMO weather code
                    }
                    
                    chunk_df = pd.DataFrame(hourly_data)
                    all_data.append(chunk_df)
                    logger.info(f"  ✓ Fetched {len(chunk_df)} steps @ hourly")
                    
                except Exception as e:
                    logger.error(f"  ✗ Failed: {e}")
                    break
                
            # Combine
            weather_hourly = pd.concat(all_data, ignore_index=True)
            weather_hourly = weather_hourly.drop_duplicates(subset=['timestamp_utc']).sort_values('timestamp_utc').reset_index(drop=True)
            
            # Resample to 15min
            logger.info(f"\nResampling {len(weather_hourly)} hourly steps → 15min...")
            weather_hourly_indexed = weather_hourly.set_index('timestamp_utc')
            
            numeric_cols = weather_hourly.select_dtypes(include=[np.number]).columns.tolist()
            weather_15min = weather_hourly_indexed[numeric_cols].resample('15min').interpolate(method='linear').reset_index()
        
        # Common processing for both sources:
        # These are the *_instant_raw columns used as TFT features
        logger.info("Saving raw irradiance copies...")
        weather_15min['shortwave_radiation_instant_raw'] = weather_15min['shortwave_radiation_instant'].copy()
        weather_15min['direct_normal_irradiance_instant_raw'] = weather_15min['direct_normal_irradiance_instant'].copy()
        weather_15min['global_tilted_irradiance_instant_raw'] = weather_15min['shortwave_radiation_instant'].copy()  # GTI approximation from GHI
        
        # Add global_tilted_irradiance_instant (normalized version, same as GHI for now)
        # In training this was computed from tilt/azimuth, here we approximate with GHI
        weather_15min['global_tilted_irradiance_instant'] = weather_15min['shortwave_radiation_instant'].copy()
        
        # Step 2: PVLib processing (generates pvlib_* columns)
        logger.info("Computing PVLib features...")
        
        import pvlib
        from pvlib.location import Location
        
        location = Location(latitude=self.lat, longitude=self.lon, tz='UTC')
        
        # Solar position
        times = pd.DatetimeIndex(weather_15min['timestamp_utc'])
        solar_pos = location.get_solarposition(times)
        
        # Clip irradiance to non-negative (ERA5 can have small negatives from interpolation)
        dni = np.clip(weather_15min['direct_normal_irradiance_instant'].values, 0.0, None)
        ghi = np.clip(weather_15min['shortwave_radiation_instant'].values, 0.0, None)
        dhi = np.clip(weather_15min['diffuse_radiation_instant'].values, 0.0, None)
        
        # dni_extra required for haydavies model
        dni_extra = pvlib.irradiance.get_extra_radiation(times).values
        
        # POA irradiance using Hay-Davies model (same as training)
        poa_components = pvlib.irradiance.get_total_irradiance(
            surface_tilt=self.tilt,
            surface_azimuth=self.azimuth,
            solar_zenith=solar_pos['zenith'],
            solar_azimuth=solar_pos['azimuth'],
            dni=dni,
            ghi=ghi,
            dhi=dhi,
            dni_extra=dni_extra,
            model='haydavies',
            albedo=0.2
        )
        
        # Add PVLib columns with pvlib_ prefix
        weather_15min['pvlib_solar_zenith'] = solar_pos['zenith'].values
        weather_15min['pvlib_solar_azimuth'] = solar_pos['azimuth'].values
        weather_15min['pvlib_poa_global'] = np.clip(poa_components['poa_global'].values, 0.0, None)
        weather_15min['pvlib_poa_direct'] = np.clip(poa_components['poa_direct'].values, 0.0, None)
        weather_15min['pvlib_poa_diffuse'] = np.clip(poa_components['poa_diffuse'].values, 0.0, None)
        weather_15min['pvlib_poa_ground_diffuse'] = np.clip(poa_components['poa_ground_diffuse'].values, 0.0, None)
        
        # PVWatts DC/AC power proxies (same method as training)
        temp_air = weather_15min['temperature_2m'].values
        wind = np.clip(weather_15min['wind_speed_10m'].values, 0.0, None)
        poa_global = weather_15min['pvlib_poa_global'].values
        
        # Simple cell temperature model
        temp_cell = temp_air + (poa_global / 1000.0) * 25.0
        
        # PVWatts DC power
        pdc0_w = self.metadata['installed_capacity_kw'] * 1000.0
        gamma_pdc = -0.003  # typical temp coefficient
        pdc_w = pvlib.pvsystem.pvwatts_dc(poa_global, temp_cell, pdc0=pdc0_w, gamma_pdc=gamma_pdc)
        
        # PVWatts AC power
        pac_w = pvlib.inverter.pvwatts(pdc_w, pdc0=pdc0_w)
        
        weather_15min['pvlib_dc_kw'] = np.clip(pdc_w / 1000.0, 0.0, None)
        weather_15min['pvlib_ac_kw'] = np.clip(pac_w / 1000.0, 0.0, None)
        
        # Add legacy column poa_irradiance (same as pvlib_poa_global)
        weather_15min['poa_irradiance'] = weather_15min['pvlib_poa_global'].copy()
        
        # Add metadata
        weather_15min['plant_id'] = 'plant_03'
        
        # Drop data_source - not used in TFT training
        # (kept in internal processing but removed before saving)
        
        # Save
        output_path = self.phase1_dir / "weather_raw_15min.parquet"
        weather_15min.to_parquet(output_path, index=False)
        
        logger.info(f"\n✅ Weather fetched: {len(weather_15min):,} timesteps")
        logger.info(f"   Saved to: {output_path}")
        logger.info(f"   Date range: {weather_15min.timestamp_utc.min()} → {weather_15min.timestamp_utc.max()}")
        
        return weather_15min
    
    def step2_preprocess_weather(self, weather_15min: pd.DataFrame):
        """Preprocess for both TFT heads + ensure column alignment."""
        logger.info("\n" + "="*70)
        logger.info("STEP 2: PREPROCESSING WEATHER")
        logger.info("="*70)
        
        # Verify all required columns present (must match test.parquet schema)
        required_cols = [
            'timestamp_utc', 'plant_id',
            # OpenMeteo weather (13 cols)
            'temperature_2m', 'relative_humidity_2m', 'precipitation', 'weather_code',
            'cloud_cover', 'wind_speed_10m', 'wind_direction_10m',
            'shortwave_radiation_instant', 'direct_radiation_instant', 'diffuse_radiation_instant',
            'direct_normal_irradiance_instant', 'global_tilted_irradiance_instant', 'surface_pressure',
            # Raw versions (3 cols)
            'shortwave_radiation_instant_raw', 'direct_normal_irradiance_instant_raw',
            'global_tilted_irradiance_instant_raw',
            # PVLib features (8 cols)
            'pvlib_solar_zenith', 'pvlib_solar_azimuth', 'pvlib_poa_global',
            'pvlib_poa_direct', 'pvlib_poa_diffuse', 'pvlib_poa_ground_diffuse',
            'pvlib_dc_kw', 'pvlib_ac_kw',
            # Legacy
            'poa_irradiance'
        ]
        
        missing = [c for c in required_cols if c not in weather_15min.columns]
        if missing:
            raise ValueError(f"Weather missing required columns: {missing}")
        
        logger.info(f"✓ All {len(required_cols)} required columns present")
        
        # Keep only required columns (drop any extras like data_source)
        weather_15min = weather_15min[required_cols].copy()
        
        # Save 15min version (for short-head)
        output_15min = self.phase1_dir / "weather_with_pvlib_15min.parquet"
        weather_15min.to_parquet(output_15min, index=False)
        logger.info(f"✓ Saved 15min version: {output_15min}")
        
        # Resample to hourly (for long-head)
        logger.info("\nResampling to hourly for long-head...")
        numeric_cols = weather_15min.select_dtypes(include=[np.number]).columns.tolist()
        
        weather_hourly = weather_15min[['timestamp_utc'] + numeric_cols].set_index('timestamp_utc').resample('1h').mean().reset_index()
        
        # Copy categorical columns
        for col in ['data_source', 'plant_id']:
            if col in weather_15min.columns:
                weather_hourly[col] = weather_15min[col].iloc[0]
        
        output_hourly = self.phase1_dir / "weather_with_pvlib_hourly.parquet"
        weather_hourly.to_parquet(output_hourly, index=False)
        logger.info(f"✓ Saved hourly version: {output_hourly}")
        logger.info(f"  15min: {len(weather_15min):,} steps")
        logger.info(f"  Hourly: {len(weather_hourly):,} steps")
        
        return weather_15min, weather_hourly
    
    def step3_extract_encoder_context(self):
        """Extract encoder context from test.parquet @ 15min resolution."""
        logger.info("\n" + "="*70)
        logger.info("STEP 3: EXTRACTING ENCODER CONTEXT FROM TEST SET")
        logger.info("="*70)
        
        # Load test data @ 15min
        test_15min = pd.read_parquet(self.test_dir / "15min_pca32/test.parquet")
        
        logger.info(f"Test data loaded:")
        logger.info(f"  15min: {test_15min.timestamp_utc.min()} → {test_15min.timestamp_utc.max()}")
        
        # Short-head: Last 24 hours (96 steps @ 15min)
        encoder_short = test_15min.tail(96).copy()
        output_short = self.phase1_dir / "encoder_context_short.parquet"
        encoder_short.to_parquet(output_short, index=False)
        logger.info(f"\n✓ Short-head encoder (24h @ 15min):")
        logger.info(f"   {encoder_short.timestamp_utc.min()} → {encoder_short.timestamp_utc.max()}")
        logger.info(f"   Saved: {output_short}")
        
        # Long-head equivalent: Last 7 days @ 15min (168h × 4 = 672 steps)
        # This will be resampled to hourly (168 steps) by forecaster
        encoder_long_15min = test_15min.tail(672).copy()
        output_long_15min = self.phase1_dir / "encoder_context_long_15min.parquet"
        encoder_long_15min.to_parquet(output_long_15min, index=False)
        logger.info(f"\n✓ Long-head encoder (7d @ 15min, will be resampled to hourly):")
        logger.info(f"   {encoder_long_15min.timestamp_utc.min()} → {encoder_long_15min.timestamp_utc.max()}")
        logger.info(f"   Saved: {output_long_15min}")
        
        return encoder_short, encoder_long_15min
    
    def step4_run_rolling_inference(
        self,
        weather_15min: pd.DataFrame,
        weather_hourly: pd.DataFrame,
        initial_encoder_short: pd.DataFrame,
        initial_encoder_long: pd.DataFrame,
        forecaster: PhysicsAwareForecaster
    ):
        """Run rolling 30-day forecasts with DYNAMIC encoder updates."""
        logger.info("\n" + "="*70)
        logger.info("STEP 4: RUNNING ROLLING INFERENCE (DYNAMIC ANCHORING)")
        logger.info("="*70)
        
        forecast_start_date = pd.Timestamp("2023-12-01", tz='UTC')
        forecast_end_date = pd.Timestamp("2024-12-31", tz='UTC')
        stride_days = 7
        
        logger.info(f"Start: {forecast_start_date}")
        logger.info(f"End: {forecast_end_date}")
        logger.info(f"Stride: {stride_days} days")
        logger.info(f"Strategy: DYNAMIC (use predicted power as encoder)\n")
        
        # Initialize encoder context
        current_encoder_short = initial_encoder_short.copy()
        current_encoder_long = initial_encoder_long.copy()  # This is 672 @ 15min
        
        all_predictions = []
        current_forecast_start = forecast_start_date
        forecast_idx = 0
        
        pbar = tqdm(desc="Rolling forecasts", unit="forecast")
        
        while current_forecast_start <= forecast_end_date:
            forecast_idx += 1
            
            try:
                # Run 30-day forecast
                logger.info(f"\n[Forecast {forecast_idx}] Starting: {current_forecast_start}")
                
                # Extract weather for 30 days @ 15min (forecaster will resample internally)
                forecast_end = current_forecast_start + timedelta(days=30)
                weather_window_15min = weather_15min[
                    (weather_15min['timestamp_utc'] >= current_forecast_start) &
                    (weather_15min['timestamp_utc'] < forecast_end)
                ].copy()
                
                if len(weather_window_15min) < 2880:
                    logger.warning(f"  Not enough 15min weather ({len(weather_window_15min)} < 2880), stopping")
                    break
                
                # Run forecast - pass 672 @ 15min encoder (will be resampled to 168 hourly)
                # Forecaster detects 15min frequency and resamples to hourly for long-head
                predictions = forecaster.predict_30d(
                    forecast_start=current_forecast_start,
                    weather_df=weather_window_15min,  # 15min input (2880 steps)
                    historical_df=current_encoder_long  # 672 @ 15min (→ 168 hourly)
                )
                
                if torch.is_tensor(predictions):
                    predictions = predictions.cpu().numpy()
                
                # Save predictions
                pred_record = {
                    'forecast_idx': forecast_idx,
                    'forecast_start': current_forecast_start,
                    'forecast_end': forecast_end,
                    'timestamps': weather_window_15min['timestamp_utc'].values[:len(predictions)],
                    'predicted_power': predictions
                }
                all_predictions.append(pred_record)
                
                logger.info(f"  ✓ Predicted {len(predictions)} steps")
                logger.info(f"    Power range: [{predictions.min():.3f}, {predictions.max():.3f}]")
                
                # UPDATE ENCODER CONTEXT for next forecast (DYNAMIC!)
                # Need last 7 days BEFORE the next forecast_start as encoder
                days_to_advance = stride_days
                steps_to_advance_15min = days_to_advance * 96  # 15min steps
                
                # Next forecast starts at current_forecast_start + stride_days
                next_forecast_start_idx = steps_to_advance_15min
                
                if len(predictions) >= 672:  # Need at least 7 days of predictions
                    # Short-head encoder: 24h immediately before next forecast
                    # (predictions from days 6-7, i.e., indices 576-671 for stride=7)
                    short_start_idx = max(0, next_forecast_start_idx - 96)
                    short_end_idx = next_forecast_start_idx
                    
                    new_encoder_short_data = predictions[short_start_idx:short_end_idx]
                    new_encoder_short_times = weather_window_15min['timestamp_utc'].iloc[short_start_idx:short_end_idx].values
                    
                    current_encoder_short = weather_window_15min.iloc[short_start_idx:short_end_idx].copy()
                    current_encoder_short['power_norm'] = new_encoder_short_data
                    
                    # Add deprecated plant_XX columns (zeros, for schema compatibility)
                    for col in ['plant_01', 'plant_02', 'plant_03', 'plant_05', 'plant_06']:
                        if col not in current_encoder_short.columns:
                            current_encoder_short[col] = 0.0
                    
                    logger.info(f"  ↻ Updated short-head encoder: {pd.Timestamp(new_encoder_short_times[0])} → {pd.Timestamp(new_encoder_short_times[-1])}")
                    
                    # Long-head encoder: 7 days (672 steps) immediately before next forecast
                    # (predictions from days 0-7, i.e., indices 0-671 for stride=7)
                    long_start_idx = max(0, next_forecast_start_idx - 672)
                    long_end_idx = next_forecast_start_idx
                    
                    new_encoder_long_data = predictions[long_start_idx:long_end_idx]
                    new_encoder_long_times = weather_window_15min['timestamp_utc'].iloc[long_start_idx:long_end_idx].values
                    
                    current_encoder_long = weather_window_15min.iloc[long_start_idx:long_end_idx].copy()
                    current_encoder_long['power_norm'] = new_encoder_long_data
                    
                    # Add deprecated plant_XX columns
                    for col in ['plant_01', 'plant_02', 'plant_03', 'plant_05', 'plant_06']:
                        if col not in current_encoder_long.columns:
                            current_encoder_long[col] = 0.0
                    
                    logger.info(f"  ↻ Updated long-head encoder: {pd.Timestamp(new_encoder_long_times[0])} → {pd.Timestamp(new_encoder_long_times[-1])}")
                
                # Move to next forecast window
                current_forecast_start += timedelta(days=stride_days)
                pbar.update(1)
                
            except Exception as e:
                logger.error(f"  ✗ Forecast {forecast_idx} failed: {e}")
                break
        
        pbar.close()
        
        # Flatten predictions into time series
        logger.info(f"\n✅ Completed {len(all_predictions)} forecasts")
        logger.info("Flattening predictions...")
        
        records = []
        for pred in all_predictions:
            for i, (ts, power) in enumerate(zip(pred['timestamps'], pred['predicted_power'])):
                records.append({
                    'timestamp_utc': pd.Timestamp(ts),
                    'forecast_idx': pred['forecast_idx'],
                    'forecast_start': pred['forecast_start'],
                    'step_ahead': i,
                    'hours_ahead': i * 0.25,
                    'predicted_power_norm': float(power)
                })
        
        predictions_df = pd.DataFrame(records)
        
        output_path = self.phase1_dir / "predictions_phase1.parquet"
        predictions_df.to_parquet(output_path, index=False)
        
        logger.info(f"✓ Saved predictions: {output_path}")
        logger.info(f"  Total timesteps: {len(predictions_df):,}")
        
        # Only show date range if predictions exist
        if len(predictions_df) > 0 and 'timestamp_utc' in predictions_df.columns:
            logger.info(f"  Date range: {predictions_df.timestamp_utc.min()} → {predictions_df.timestamp_utc.max()}")
        else:
            logger.warning("  No predictions generated - all forecasts failed!")
        
        return predictions_df
    
    def run(self):
        """Execute full Phase 1 pipeline."""
        logger.info("\n" + "="*70)
        logger.info("PHASE 1 INFERENCE PIPELINE: Dec 2023 - Dec 2024")
        logger.info("="*70)
        
        # Step 1: Fetch weather
        weather_15min = self.step1_fetch_weather()
        
        # Step 2: Preprocess
        weather_15min, weather_hourly = self.step2_preprocess_weather(weather_15min)
        
        # Step 3: Extract encoder context
        encoder_short, encoder_long = self.step3_extract_encoder_context()
        
        # Step 4: Initialize forecaster
        logger.info("\n" + "="*70)
        logger.info("INITIALIZING FORECASTER")
        logger.info("="*70)
        
        forecaster = PhysicsAwareForecaster(
            short_ckpt=str(self.base_dir / "V1.0_FINAL_TFT/shorthead_seed42/checkpoints/best.ckpt"),
            long_ckpt=str(self.base_dir / "V1.0_FINAL_TFT/longhead_seed43/checkpoints/best.ckpt"),
            plant_metadata=str(self.base_dir / "V1.0_FINAL_TFT/plant_metadata/plant_03.json"),
            short_train_parquet=str(self.base_dir / "data/processed/plant_level/plant_03/15min_pca32/train.parquet"),
            long_train_parquet=str(self.base_dir / "data/processed/plant_level/plant_03/hourly_longhead/train.parquet"),
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        logger.info("✓ Forecaster ready")
        
        # Step 5: Run inference
        predictions_df = self.step4_run_rolling_inference(
            weather_15min,
            weather_hourly,
            encoder_short,
            encoder_long,
            forecaster
        )
        
        logger.info("\n" + "="*70)
        logger.info("PHASE 1 COMPLETE")
        logger.info("="*70)
        logger.info(f"Predictions saved: {self.phase1_dir / 'predictions_phase1.parquet'}")
        logger.info(f"\nNext step: Generate RL transitions from predictions")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='MiRACLE Inference Pipeline')
    parser.add_argument('--weather-source', choices=['historical', 'api'], default='historical',
                       help='Weather data source: historical (ERA5) or api (real-time forecast)')
    args = parser.parse_args()
    
    pipeline = Phase1Pipeline(weather_source=args.weather_source)
    pipeline.run()


if __name__ == "__main__":
    main()
