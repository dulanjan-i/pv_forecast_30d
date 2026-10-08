"""
Multi-source weather API orchestrator with intelligent fallback strategy.

Priority:
1. Open-Meteo (free, 15-day forecast)
2. NOAA GFS (free, longer range)
3. VisualCrossing (paid fallback, high resolution)

Implements caching, rate limiting, and cost optimization.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import time
from typing import Dict, List, Optional, Tuple
import hashlib


class WeatherAPIOrchestrator:
    """
    Coordinates multiple weather APIs with intelligent fallback.
    Optimizes for cost and reliability.
    """
    
    def __init__(self, cache_dir: str = "data/cache/weather_api"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # API endpoints
        self.open_meteo_url = "https://api.open-meteo.com/v1/forecast"
        self.noaa_gfs_url = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
        
        # Rate limiting
        self.last_call = {}
        self.min_interval = {
            'open_meteo': 1.0,  # 1 second between calls
            'noaa': 2.0,        # 2 seconds
            'visual_crossing': 5.0  # 5 seconds (paid, conservative)
        }
        
        # Cost tracking
        self.api_calls = {'open_meteo': 0, 'noaa': 0, 'visual_crossing': 0}
        
    def _get_cache_key(self, lat: float, lon: float, start_date: str, end_date: str, source: str) -> str:
        """Generate cache key for API request."""
        key_str = f"{source}_{lat}_{lon}_{start_date}_{end_date}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cached(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Retrieve cached weather data if available and fresh (< 6 hours old)."""
        cache_file = self.cache_dir / f"{cache_key}.parquet"
        if cache_file.exists():
            # Check age
            age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
            if age_hours < 6:  # Cache valid for 6 hours
                print(f"   ✓ Using cached data (age: {age_hours:.1f}h)")
                return pd.read_parquet(cache_file)
        return None
    
    def _save_cache(self, cache_key: str, data: pd.DataFrame):
        """Save weather data to cache."""
        cache_file = self.cache_dir / f"{cache_key}.parquet"
        data.to_parquet(cache_file, index=False)
    
    def _rate_limit(self, source: str):
        """Enforce rate limiting for API calls."""
        if source in self.last_call:
            elapsed = time.time() - self.last_call[source]
            if elapsed < self.min_interval[source]:
                sleep_time = self.min_interval[source] - elapsed
                time.sleep(sleep_time)
        self.last_call[source] = time.time()
    
    def fetch_open_meteo(self, lat: float, lon: float, 
                         start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Fetch weather from Open-Meteo (free tier).
        
        Args:
            lat: Latitude
            lon: Longitude
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with weather variables or None if failed
        """
        cache_key = self._get_cache_key(lat, lon, start_date, end_date, 'open_meteo')
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached
        
        self._rate_limit('open_meteo')
        
        params = {
            'latitude': lat,
            'longitude': lon,
            'start_date': start_date,
            'end_date': end_date,
            'hourly': [
                'temperature_2m',
                'relative_humidity_2m',
                'dew_point_2m',
                'precipitation',
                'rain',
                'snowfall',
                'snow_depth',
                'cloud_cover',
                'cloud_cover_low',
                'cloud_cover_mid',
                'cloud_cover_high',
                'wind_speed_10m',
                'wind_direction_10m',
                'wind_gusts_10m',
                'surface_pressure',
                'shortwave_radiation',
                'direct_radiation',
                'diffuse_radiation',
                'direct_normal_irradiance'
            ],
            'timezone': 'UTC'
        }
        
        try:
            print(f"   📡 Calling Open-Meteo API...")
            response = requests.get(self.open_meteo_url, params=params, timeout=30, verify=False)
            response.raise_for_status()
            
            data = response.json()
            hourly = data['hourly']
            
            # Convert to DataFrame
            df = pd.DataFrame({
                'timestamp_utc': pd.to_datetime(hourly['time']),
                'temperature_2m': hourly['temperature_2m'],
                'relative_humidity_2m': hourly['relative_humidity_2m'],
                'dew_point_2m': hourly['dew_point_2m'],
                'precipitation': hourly['precipitation'],
                'rain': hourly['rain'],
                'snowfall': hourly['snowfall'],
                'snow_depth': hourly['snow_depth'],
                'cloud_cover': hourly['cloud_cover'],
                'cloud_cover_low': hourly['cloud_cover_low'],
                'cloud_cover_mid': hourly['cloud_cover_mid'],
                'cloud_cover_high': hourly['cloud_cover_high'],
                'wind_speed_10m': hourly['wind_speed_10m'],
                'wind_direction_10m': hourly['wind_direction_10m'],
                'wind_gusts_10m': hourly['wind_gusts_10m'],
                'surface_pressure': hourly['surface_pressure'],
                'shortwave_radiation': hourly['shortwave_radiation'],
                'direct_radiation': hourly['direct_radiation'],
                'diffuse_radiation': hourly['diffuse_radiation'],
                'direct_normal_irradiance': hourly['direct_normal_irradiance']
            })
            
            self.api_calls['open_meteo'] += 1
            self._save_cache(cache_key, df)
            
            print(f"   ✓ Open-Meteo: {len(df)} hourly records")
            return df
            
        except Exception as e:
            print(f"   ✗ Open-Meteo failed: {e}")
            return None
    
    def fetch_noaa_gfs(self, lat: float, lon: float,
                       start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Fetch NOAA GFS forecast (free, longer range).
        
        Note: This is a simplified implementation. Real NOAA access requires
        parsing GRIB2 files which needs additional libraries (pygrib).
        """
        print(f"   ⚠️  NOAA GFS: Requires GRIB2 parsing (not implemented yet)")
        return None
    
    def fetch_visual_crossing(self, lat: float, lon: float,
                              start_date: str, end_date: str,
                              api_key: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Fetch from VisualCrossing (PAID fallback, high resolution).
        
        Args:
            lat: Latitude
            lon: Longitude
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            api_key: VisualCrossing API key
            
        Returns:
            DataFrame with weather or None
        """
        if api_key is None:
            print(f"   ⚠️  VisualCrossing: No API key provided (skipping paid API)")
            return None
        
        cache_key = self._get_cache_key(lat, lon, start_date, end_date, 'visual_crossing')
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached
        
        self._rate_limit('visual_crossing')
        
        url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{lat},{lon}/{start_date}/{end_date}"
        
        params = {
            'key': api_key,
            'unitGroup': 'metric',
            'include': 'hours',
            'elements': 'datetime,temp,humidity,dew,precip,snow,snowdepth,windspeed,winddir,cloudcover,pressure,solarradiation,solarenergy'
        }
        
        try:
            print(f"   💰 Calling VisualCrossing API (PAID)...")
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Parse hourly data
            records = []
            for day in data['days']:
                for hour in day['hours']:
                    records.append({
                        'timestamp_utc': pd.to_datetime(f"{day['datetime']} {hour['datetime']}"),
                        'temperature_2m': hour.get('temp'),
                        'relative_humidity_2m': hour.get('humidity'),
                        'dew_point_2m': hour.get('dew'),
                        'precipitation': hour.get('precip', 0),
                        'snowfall': hour.get('snow', 0),
                        'snow_depth': hour.get('snowdepth', 0),
                        'cloud_cover': hour.get('cloudcover'),
                        'wind_speed_10m': hour.get('windspeed'),
                        'wind_direction_10m': hour.get('winddir'),
                        'surface_pressure': hour.get('pressure'),
                        'shortwave_radiation': hour.get('solarradiation'),
                    })
            
            df = pd.DataFrame(records)
            
            self.api_calls['visual_crossing'] += 1
            self._save_cache(cache_key, df)
            
            print(f"   ✓ VisualCrossing: {len(df)} hourly records (${0.25:.2f} cost)")
            return df
            
        except Exception as e:
            print(f"   ✗ VisualCrossing failed: {e}")
            return None
    
    def fetch_with_fallback(self, lat: float, lon: float,
                           days_ahead: int = 15,
                           visual_crossing_key: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch weather with intelligent fallback strategy.
        
        Priority:
        1. Open-Meteo (free, reliable, 15 days)
        2. NOAA GFS (free, longer range - not implemented yet)
        3. VisualCrossing (paid, high quality)
        
        Args:
            lat: Latitude
            lon: Longitude
            days_ahead: Number of days to forecast
            visual_crossing_key: Optional paid API key
            
        Returns:
            DataFrame with weather forecast
        """
        print(f"\n🌤️  Fetching {days_ahead}-day weather forecast for ({lat:.4f}, {lon:.4f})")
        print("=" * 70)
        
        start_date = datetime.now().strftime('%Y-%m-%d')
        end_date = (datetime.now() + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
        
        # Try Open-Meteo first (free, best for 15 days)
        if days_ahead <= 15:
            df = self.fetch_open_meteo(lat, lon, start_date, end_date)
            if df is not None and len(df) > 0:
                print(f"✓ SUCCESS: Open-Meteo (free API)")
                return df
        
        # Try NOAA GFS (free, longer range)
        df = self.fetch_noaa_gfs(lat, lon, start_date, end_date)
        if df is not None and len(df) > 0:
            print(f"✓ SUCCESS: NOAA GFS (free API)")
            return df
        
        # Fallback to VisualCrossing (paid)
        if visual_crossing_key:
            df = self.fetch_visual_crossing(lat, lon, start_date, end_date, visual_crossing_key)
            if df is not None and len(df) > 0:
                print(f"✓ SUCCESS: VisualCrossing (PAID API - ${0.25 * (days_ahead / 1000):.4f})")
                return df
        
        raise RuntimeError("All weather APIs failed! Cannot proceed with inference.")
    
    def print_cost_summary(self):
        """Print API usage and cost summary."""
        print("\n" + "=" * 70)
        print("💰 API USAGE SUMMARY")
        print("=" * 70)
        print(f"Open-Meteo calls:      {self.api_calls['open_meteo']:3d} (FREE)")
        print(f"NOAA GFS calls:        {self.api_calls['noaa']:3d} (FREE)")
        print(f"VisualCrossing calls:  {self.api_calls['visual_crossing']:3d} (PAID)")
        
        vc_cost = self.api_calls['visual_crossing'] * 0.25
        print(f"\nTotal cost: ${vc_cost:.2f}")
        print("=" * 70)


if __name__ == "__main__":
    # Test the orchestrator
    orchestrator = WeatherAPIOrchestrator()
    
    # Plant 03 coordinates (Germany)
    lat, lon = 51.3397, 12.3731
    
    # Fetch 15-day forecast
    weather_df = orchestrator.fetch_with_fallback(lat, lon, days_ahead=15)
    
    print(f"\n✓ Retrieved {len(weather_df)} hourly timesteps")
    print(f"  Columns: {weather_df.columns.tolist()}")
    print(f"\n  Sample data:")
    print(weather_df.head())
    
    orchestrator.print_cost_summary()
