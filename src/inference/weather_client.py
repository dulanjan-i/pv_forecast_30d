#!/usr/bin/env python3
"""
OpenMeteo Weather API Client for Real-Time Forecasting

Fetches 30-day weather forecasts from OpenMeteo Forecast API with ALL required variables
for TFT inference. Optimized for plant_03 (Germany) but supports any location.

Rate Limits (OpenMeteo):
- 600 calls/min
- 5,000 calls/hour  
- 10,000 calls/day
- 300,000 calls/month

Author: PV Forecast Team
Date: 2026-01-02
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple, Union
from datetime import datetime, timedelta
import json

import openmeteo_requests
import requests_cache
from retry_requests import retry


class WeatherClient:
    """
    OpenMeteo Weather API client for real-time forecasting.
    
    Fetches ALL weather variables used in TFT training:
    - Irradiance: shortwave, direct, diffuse, DNI, GTI (raw + instant)
    - Meteorology: temp, humidity, pressure, wind speed/direction
    - Conditions: cloud cover, precipitation, weather code
    
    Returns data at hourly resolution, resamples to 15-min for TFT.
    """
    
    def __init__(
        self,
        cache_dir: str = ".cache",
        cache_expire_hours: int = 1,
        retry_count: int = 5,
        retry_backoff: float = 0.2
    ):
        """
        Initialize weather client with caching and retry logic.
        
        Args:
            cache_dir: Directory for response caching
            cache_expire_hours: Cache expiration (1 hour = fresh forecasts)
            retry_count: Number of retries on API failure
            retry_backoff: Exponential backoff factor
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Load ECMWF credentials if available
        self.ecmwf_creds = self._load_ecmwf_credentials()
        
        # Setup OpenMeteo client with cache and retry
        # Note: SSL verification disabled for environments with cert issues
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        cache_session = requests_cache.CachedSession(
            str(self.cache_dir / 'weather_cache'), 
            expire_after=cache_expire_hours * 3600
        )
        cache_session.verify = False  # Disable SSL verification
        
        retry_session = retry(
            cache_session, 
            retries=retry_count, 
            backoff_factor=retry_backoff
        )
        self.client = openmeteo_requests.Client(session=retry_session)
        
        # OpenMeteo API endpoints
        self.forecast_url = "https://api.open-meteo.com/v1/forecast"  # 7-day optimal, 15-day max
        self.ensemble_url = "https://ensemble-api.open-meteo.com/v1/ensemble"  # 35-day (BROKEN: returns anomalies)
        self.ecmwf_url = "https://api.open-meteo.com/v1/ecmwf"  # 15-day standard, 46-day extended range
        self.gfs_url = "https://api.open-meteo.com/v1/gfs"  # 16-day max, NOAA GFS
        self.gfs_url = "https://api.open-meteo.com/v1/gfs"  # 16-day max, NOAA GFS
        
        # ALL weather variables from training data (16 features)
        self.weather_vars = [
            "temperature_2m",
            "relative_humidity_2m", 
            "precipitation",
            "weather_code",
            "cloud_cover",
            "wind_speed_10m",
            "wind_direction_10m",
            "shortwave_radiation_instant",      # GHI equivalent
            "direct_radiation_instant",          # Direct horizontal
            "diffuse_radiation_instant",         # DHI
            "direct_normal_irradiance_instant",  # DNI
            "global_tilted_irradiance_instant",  # GTI (needs tilt/azimuth)
            "surface_pressure"
        ]
    
    def _load_ecmwf_credentials(self) -> Optional[Dict]:
        """Load ECMWF API credentials from .ecmwf_credentials.json"""
        cred_file = Path(".ecmwf_credentials.json")
        if cred_file.exists():
            try:
                with open(cred_file) as f:
                    return json.load(f)
            except Exception as e:
                print(f"[WeatherClient] Warning: Could not load ECMWF credentials: {e}")
        return None
    
    def select_best_api(self, days: int) -> Tuple[str, str]:
        """
        Smart API selector based on forecast horizon.
        
        Strategy:
        - 0-7 days: Forecast API (high resolution, free, very accurate)
        - 8-15 days: ECMWF (high quality for Europe, best medium-range)
        - 16-35 days: GFS (long-range capability)
        
        Returns:
            (api_name, api_url) tuple
        
        Notes:
            This strategy leverages each API's strengths:
            - Forecast API: Best for short-term (hourly updates)
            - ECMWF: Best for medium-term Europe (0.25° resolution)
            - GFS: Best for long-term global (only option for 30+ days)
            
            RL meta-controller can later blend predictions from multiple
            horizons for optimal uncertainty quantification.
        """
        if days <= 7:
            return "Forecast", self.forecast_url
        elif days <= 15:
            return "ECMWF", self.ecmwf_url
        else:
            # For 30+ days, use GFS (not ensemble which returns anomalies)
            return "GFS", self.gfs_url
    
    def fetch_forecast(
        self,
        latitude: float,
        longitude: float,
        start_time: str | pd.Timestamp,
        days: int = 30,
        tilt: float = 25.0,
        azimuth: float = 180.0,
        timezone: str = "UTC",
        use_ensemble: bool = False,
        use_ecmwf: bool = False,
        auto_select: bool = True
    ) -> pd.DataFrame:
        """
        Fetch weather forecast from OpenMeteo (Forecast, Ensemble, or ECMWF API).
        
        Args:
            latitude: Plant latitude (°N)
            longitude: Plant longitude (°E)
            start_time: Forecast start (ISO format or Timestamp)
            days: Forecast horizon in days (default 30)
            tilt: Panel tilt angle in degrees (default 25°)
            azimuth: Panel azimuth in degrees (180° = south)
            timezone: Output timezone (default UTC)
            use_ensemble: If True, force Ensemble API (BROKEN - returns anomalies)
            use_ecmwf: If True, force ECMWF API
            auto_select: If True (default), automatically choose best API for horizon
        
        Returns:
            DataFrame with columns:
            - timestamp_utc: UTC timestamps @ 1-hour resolution
            - All 16 weather variables
            - Shape: (days*24, 17) for hourly data
        
        Raises:
            RuntimeError: If API call fails after retries
            ValueError: If response is invalid
        
        Notes:
            Smart API Selection (auto_select=True):
            - 0-7 days: Forecast API (high accuracy, free)
            - 8-15 days: ECMWF (best for Europe medium-range)
            - 16+ days: GFS (long-range, lower resolution)
            
            Manual Override: Set use_ecmwf=True or use_ensemble=True to force specific API
        """
        # Convert start_time to string
        if isinstance(start_time, pd.Timestamp):
            start_time = start_time.strftime("%Y-%m-%d")
        elif isinstance(start_time, datetime):
            start_time = start_time.strftime("%Y-%m-%d")
        
        # Calculate end date
        start_dt = pd.to_datetime(start_time)
        end_dt = start_dt + pd.Timedelta(days=days)
        
        # Choose API endpoint (smart selection or manual override)
        if auto_select and not use_ecmwf and not use_ensemble:
            api_name, api_url = self.select_best_api(days)
            print(f"                [Auto-selected {api_name} API for {days}-day forecast]")
        elif use_ecmwf:
            api_url = self.ecmwf_url
            api_name = "ECMWF"
        elif use_ensemble:
            api_url = self.ensemble_url
            api_name = "Ensemble"
        else:
            api_url = self.forecast_url
            api_name = "Forecast"
        
        end_time = end_dt.strftime("%Y-%m-%d")
        
        print(f"[WeatherClient] Fetching {days}-day {api_name} forecast")
        print(f"                Location: ({latitude:.4f}°N, {longitude:.4f}°E)")
        print(f"                Period: {start_time} → {end_time}")
        print(f"                Tilt/Azimuth: {tilt}° / {azimuth}°")
        print(f"                API: {api_name}")
        
        # Build API request parameters
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_time,
            "end_date": end_time,
            "hourly": self.weather_vars,
            "timezone": timezone,
            "tilt": tilt,
            "azimuth": azimuth
        }
        
        # Add model-specific parameters
        if use_ecmwf or api_name == "ECMWF":
            # ECMWF API doesn't need models parameter
            pass
        elif use_ensemble:
            # gfs_seamless: Global, up to 35 days (50km resolution)
            params["models"] = "gfs_seamless"
        elif api_name == "GFS":
            # GFS API: use best_match for automatic model selection
            params["models"] = "best_match"
        else:
            params["models"] = "best_match"
        
        try:
            # Call OpenMeteo API (Forecast or Ensemble)
            responses = self.client.weather_api(api_url, params=params)
            response = responses[0]
            
            print(f"                ✓ {api_name} API response: {response.Latitude():.4f}°N, {response.Longitude():.4f}°E")
            print(f"                ✓ Elevation: {response.Elevation():.0f}m")
            
            # Extract hourly data
            hourly = response.Hourly()
            
            # Parse timestamps
            timestamps = pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left"
            )
            
            # Build DataFrame with all variables
            data = {"timestamp_utc": timestamps}
            
            # Extract each variable (order matches self.weather_vars)
            for idx, var in enumerate(self.weather_vars):
                values = hourly.Variables(idx).ValuesAsNumpy()
                data[var] = values
            
            df = pd.DataFrame(data)
            
            # Add "raw" versions (some features have _raw suffix in training)
            df['shortwave_radiation_instant_raw'] = df['shortwave_radiation_instant']
            df['direct_normal_irradiance_instant_raw'] = df['direct_normal_irradiance_instant']
            df['global_tilted_irradiance_instant_raw'] = df['global_tilted_irradiance_instant']
            
            print(f"                ✓ Fetched {len(df)} hourly steps")
            print(f"                ✓ Columns: {len(df.columns)}")
            
            return df
            
        except Exception as e:
            raise RuntimeError(f"OpenMeteo API call failed: {e}")
    
    def resample_to_resolution(
        self, 
        hourly_df: pd.DataFrame, 
        target_resolution: str = "15min"
    ) -> pd.DataFrame:
        """
        Resample hourly weather data to target resolution via linear interpolation.
        
        Args:
            hourly_df: DataFrame with hourly timestamps
            target_resolution: Target resolution ('15min' for short-head, '1h' for long-head)
        
        Returns:
            DataFrame @ target resolution
            
        Notes:
            - Short-head TFT (96 steps): needs 15-min resolution (24h × 4 = 96)
            - Long-head TFT (720 steps): needs 1-hour resolution (30d × 24 = 720)
            - API returns hourly → we resample down (15-min) or keep as-is (1-hour)
        """
        if target_resolution == "1h":
            # Long-head: keep hourly data as-is, just ensure proper indexing
            print(f"[WeatherClient] Keeping 1-hour resolution (long-head)...")
            print(f"                ✓ {len(hourly_df)} hourly steps")
            return hourly_df.copy()
        
        print(f"[WeatherClient] Resampling to {target_resolution} resolution (short-head)...")
        
        df = hourly_df.copy()
        df = df.set_index('timestamp_utc')
        
        # Resample to target resolution
        df_resampled = df.resample(target_resolution).asfreq()
        
        # Interpolate continuous variables (most weather vars)
        continuous_vars = [
            'temperature_2m', 'relative_humidity_2m', 'precipitation',
            'cloud_cover', 'wind_speed_10m', 'wind_direction_10m',
            'shortwave_radiation_instant', 'direct_radiation_instant',
            'diffuse_radiation_instant', 'direct_normal_irradiance_instant',
            'global_tilted_irradiance_instant', 'surface_pressure',
            'shortwave_radiation_instant_raw', 'direct_normal_irradiance_instant_raw',
            'global_tilted_irradiance_instant_raw'
        ]
        
        for var in continuous_vars:
            if var in df_resampled.columns:
                df_resampled[var] = df_resampled[var].interpolate(method='linear')
        
        # Forward-fill discrete variables (weather_code)
        if 'weather_code' in df_resampled.columns:
            df_resampled['weather_code'] = df_resampled['weather_code'].ffill()
        
        df_resampled = df_resampled.reset_index()
        print(f"                ✓ Resampled to {len(df_resampled)} steps @ {target_resolution}")
        return df_resampled
    
    def resample_to_15min(
        self,
        hourly_df: pd.DataFrame,
        method: str = "interpolate"
    ) -> pd.DataFrame:
        """Legacy wrapper for 15-min resampling (short-head compatibility)"""
        return self.resample_to_resolution(hourly_df, target_resolution="15min")
    
    def fetch_and_prepare(
        self,
        latitude: float,
        longitude: float,
        start_time: str | pd.Timestamp,
        days: int = 30,
        tilt: float = 25.0,
        azimuth: float = 180.0,
        resolution: str = "15min",
        use_ensemble: bool = False,
        use_ecmwf: bool = False,
        auto_select: bool = True
    ) -> pd.DataFrame:
        """
        One-shot: Fetch forecast and resample to target resolution.
        
        Args:
            latitude, longitude: Plant location
            start_time: Forecast start
            days: Forecast horizon (default 30)
            tilt, azimuth: Panel orientation
            resolution: '15min' (short-head) or '1h' (long-head)
            use_ensemble: Force Ensemble API (BROKEN)
            use_ecmwf: Force ECMWF API
            auto_select: Auto-select best API for horizon (recommended)
        
        Returns:
            DataFrame @ target resolution with all weather variables
            Shape: (days * 96, N_cols) for 15-min OR (days * 24, N_cols) for 1-hour
        """
        # Fetch hourly
        hourly = self.fetch_forecast(
            latitude=latitude,
            longitude=longitude,
            start_time=start_time,
            days=days,
            tilt=tilt,
            azimuth=azimuth,
            use_ensemble=use_ensemble,
            use_ecmwf=use_ecmwf,
            auto_select=auto_select
        )
        
        # Resample to target resolution
        forecast_resampled = self.resample_to_resolution(hourly, target_resolution=resolution)
        
        # Trim to exact requested duration
        # 15-min: days * 96 steps, 1-hour: days * 24 steps
        steps_per_day = 96 if resolution == "15min" else 24
        expected_steps = days * steps_per_day
        if len(forecast_resampled) > expected_steps:
            forecast_resampled = forecast_resampled.iloc[:expected_steps].copy()
        
        # Map to TFT expected names (ghi, dni, dhi for PVLib)
        forecast_resampled['ghi'] = forecast_resampled['shortwave_radiation_instant']
        forecast_resampled['dni'] = forecast_resampled['direct_normal_irradiance_instant']
        forecast_resampled['dhi'] = forecast_resampled['diffuse_radiation_instant']
        
        return forecast_resampled
    
    def fetch_30d_multi_chunk(
        self,
        latitude: float,
        longitude: float,
        start_time: Union[str, pd.Timestamp],
        tilt: float,
        azimuth: float
    ) -> pd.DataFrame:
        """
        Fetch 30-day forecast by combining two 15-day chunks.
        OpenMeteo Forecast API limits queries to 15 days max.
        
        Args:
            latitude: Plant latitude (degrees)
            longitude: Plant longitude (degrees)
            start_time: Forecast start time
            tilt: Panel tilt angle (degrees from horizontal)
            azimuth: Panel azimuth angle (degrees, 180=south)
        
        Returns:
            30-day forecast at 15-min resolution (~2880 steps)
        """
        if isinstance(start_time, str):
            start_time = pd.Timestamp(start_time, tz='UTC')
        elif not isinstance(start_time, pd.Timestamp):
            start_time = pd.Timestamp(start_time).tz_localize('UTC')
        
        # Fetch first 15 days
        chunk1 = self.fetch_and_prepare(
            latitude=latitude,
            longitude=longitude,
            start_time=start_time,
            days=15,
            tilt=tilt,
            azimuth=azimuth
        )
        
        # Fetch next 15 days
        start_time_chunk2 = start_time + pd.Timedelta(days=15)
        chunk2 = self.fetch_and_prepare(
            latitude=latitude,
            longitude=longitude,
            start_time=start_time_chunk2,
            days=15,
            tilt=tilt,
            azimuth=azimuth
        )
        
        # Concatenate and remove duplicate boundary timestamp
        forecast_30d = pd.concat([chunk1, chunk2], ignore_index=True)
        forecast_30d = forecast_30d.drop_duplicates(subset=['timestamp_utc'], keep='first')
        
        return forecast_30d
    
    def validate_forecast(self, df: pd.DataFrame, expected_days: int = 30) -> Dict[str, bool]:
        """
        Validate forecast DataFrame for completeness and sanity.
        
        Args:
            df: Weather forecast DataFrame
            expected_days: Expected forecast length in days (default: 30)
        
        Returns:
            Dict of validation checks (all should be True)
        """
        checks = {}
        
        # Check shape (expected_days @ 15min = expected_days * 96 steps, with tolerance)
        expected_rows = expected_days * 96
        checks['shape_correct'] = abs(len(df) - expected_rows) <= 5  # Allow ±5 for interpolation edge
        
        # Check required columns
        required = ['timestamp_utc', 'ghi', 'dni', 'dhi', 'temperature_2m', 'wind_speed_10m']
        checks['columns_present'] = all(c in df.columns for c in required)
        
        # Check for NaNs in critical columns
        checks['no_nans_ghi'] = not df['ghi'].isna().any()
        checks['no_nans_dni'] = not df['dni'].isna().any()
        checks['no_nans_temp'] = not df['temperature_2m'].isna().any()
        
        # Check irradiance ranges (W/m²: 0-1500)
        checks['ghi_range_valid'] = (df['ghi'].min() >= 0) and (df['ghi'].max() <= 1500)
        checks['dni_range_valid'] = (df['dni'].min() >= 0) and (df['dni'].max() <= 1200)
        
        # Check temperature range (-50 to 60°C)
        checks['temp_range_valid'] = (df['temperature_2m'].min() >= -50) and (df['temperature_2m'].max() <= 60)
        
        return checks


def fetch_weather_for_plant(
    plant_metadata_path: str | Path,
    forecast_start: str | pd.Timestamp,
    days: int = 30
) -> pd.DataFrame:
    """
    Convenience function: Fetch weather for a specific plant using metadata.
    
    Args:
        plant_metadata_path: Path to plant JSON metadata
        forecast_start: Forecast start time
        days: Forecast horizon (default 30)
    
    Returns:
        Weather forecast DataFrame @ 15-min resolution
    """
    # Load plant metadata
    with open(plant_metadata_path) as f:
        meta = json.load(f)
    
    client = WeatherClient()
    
    return client.fetch_and_prepare(
        latitude=meta['latitude'],
        longitude=meta['longitude'],
        start_time=forecast_start,
        days=days,
        tilt=meta['tilt_deg'],
        azimuth=meta['azimuth_deg']
    )


if __name__ == "__main__":
    """Test weather client with plant_03."""
    
    print("="*70)
    print("WEATHER CLIENT TEST - Plant 03")
    print("="*70)
    
    # Plant 03 metadata (hardcoded for testing)
    plant_03_meta = {
        'latitude': 48.694644,
        'longitude': 12.597587,
        'tilt_deg': 25.0,
        'azimuth_deg': 180.0
    }
    
    # Use today's date for forecast (OpenMeteo only allows future dates)
    today = pd.Timestamp.now(tz='UTC').floor('D')
    
    # Initialize client
    client = WeatherClient()
    
    # Test 1: Fetch hourly forecast
    print(f"\n[TEST 1] Fetching 7-day hourly forecast from {today.date()}...")
    hourly = client.fetch_forecast(
        latitude=plant_03_meta['latitude'],
        longitude=plant_03_meta['longitude'],
        start_time=today,
        days=7,
        tilt=plant_03_meta['tilt_deg'],
        azimuth=plant_03_meta['azimuth_deg']
    )
    print(f"✓ Hourly shape: {hourly.shape}")
    print(f"✓ Columns: {list(hourly.columns)}")
    print(f"✓ Time range: {hourly.timestamp_utc.min()} → {hourly.timestamp_utc.max()}")
    
    # Test 2: Resample to 15-min
    print("\n[TEST 2] Resampling to 15-min...")
    forecast_15min = client.resample_to_15min(hourly)
    print(f"✓ 15-min shape: {forecast_15min.shape}")
    print(f"✓ Expected: {7*96} steps for 7 days")
    
    # Test 3: Full pipeline (15 days max allowed by API)
    print("\n[TEST 3] Full 15-day pipeline (API max)...")
    forecast_15d = client.fetch_and_prepare(
        latitude=plant_03_meta['latitude'],
        longitude=plant_03_meta['longitude'],
        start_time=today,
        days=15,
        tilt=plant_03_meta['tilt_deg'],
        azimuth=plant_03_meta['azimuth_deg']
    )
    print(f"✓ 15-day shape: {forecast_15d.shape}")
    print(f"✓ Expected: ({15 * 96}, N) for 15 days @ 15min")
    print(f"✓ Time range: {forecast_15d['timestamp_utc'].iloc[0]} → {forecast_15d['timestamp_utc'].iloc[-1]}")
    
    # Test 4: Validation
    print("\n[TEST 4] Validating forecast...")
    checks = client.validate_forecast(forecast_15d, expected_days=15)
    for check, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {check}: {passed}")
    
    if all(checks.values()):
        print("\n✅ ALL TESTS PASSED - Weather client ready!")
    else:
        print("\n⚠️  Some validation checks failed")
    
    print("\n" + "="*70)
    print("Sample data (first 5 rows):")
    print(forecast_15d[['timestamp_utc', 'ghi', 'dni', 'dhi', 'temperature_2m', 'wind_speed_10m']].head())
