# src/inference/pvlib_predictor.py
"""
PVLib Physics-Based Power Predictor for MiRACLE.

Purpose:
    Generate pure physics-based PV power forecasts using PVLib solar models.
    This serves as a baseline prediction for physics-aware blending with TFT ML models.
    Uses clear-sky or weather-forecast-based irradiance to compute DC power output.

How:
    1. Load plant metadata (lat, lon, tilt, azimuth, capacity)
    2. Calculate solar position (zenith, azimuth) from timestamps
    3. Compute plane-of-array (POA) irradiance from weather forecast GHI/DNI/DHI
    4. Convert irradiance to DC power using simple efficiency model
    5. Normalize to [0, 1] scale matching TFT training data

Why:
    - Provides physics-based constraint for ML predictions (prevent unrealistic forecasts)
    - Serves as fallback when ML confidence is low
    - Used in upsampling to distribute hourly predictions across 15-min intervals
    - Enables physics-aware blending: final = α × ML + (1-α) × PVLib

Input/Output Paths:
    INPUT (required):
        - Plant metadata JSON: "data/metadata/germany/plant_03.json"
          Keys: plant_id, latitude, longitude, tilt_deg, azimuth_deg, 
                installed_capacity_kw, timezone
        
        - Weather forecast DataFrame (passed as parameter, NOT a file path):
          * For validation: read from test parquet (historical weather as "forecast")
            Example: pd.read_parquet("data/processed/plant_level/plant_03/15min_pca32/test.parquet")
          * For production: fetch from weather API or load CSV
            Example: weather_df = fetch_from_api() or pd.read_csv("forecast.csv")
          Required columns: timestamp_utc, ghi, dni, dhi
    
    OUTPUT (auto-created):
        - PVLib predictions: "outputs/pvlib_forecasted/<plant_id>_<timestamp>.parquet"
          Columns: timestamp_utc, plant_id, power_norm_pvlib, power_kw_pvlib
    
    NOTE: weather_df is a DataFrame parameter, NOT a CLI argument!
          Use this class inside a script that handles weather data loading.

Usage:
    from src.inference.pvlib_predictor import PVLibPredictor
    
    # Initialize with plant metadata
    predictor = PVLibPredictor("data/metadata/germany/plant_03.json")
    
    # Option 1: From forecasted weather (validation with historical data)
    weather_df = pd.read_parquet("data/processed/.../test.parquet")
    pvlib_power = predictor.predict_from_weather(weather_df)
    
    # Option 2: Clear-sky baseline (no weather needed)
    pvlib_power = predictor.predict_clear_sky(start_time, num_steps=2880, freq="15min")
    
    # Save predictions
    predictor.save_prediction(pvlib_power, timestamps, "outputs/pvlib_forecasted/out.parquet")

CLI Usage (demo only):
    # Run built-in demo from repo root
    python -m src.inference.pvlib_predictor
    
    # For custom CLI script with weather file input, create wrapper:
    # scripts/run_pvlib_prediction.py --weather_csv forecast.csv --plant plant_03

Dependencies:
    - pvlib>=0.9.0
    - pandas
    - numpy
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import pvlib
from pvlib.location import Location
from pvlib.pvsystem import PVSystem


class PVLibPredictor:
    """
    Physics-based PV power predictor using PVLib solar models.
    
    Attributes:
        plant_id: Unique plant identifier
        location: PVLib Location object (lat, lon, altitude, timezone)
        system: PVLib PVSystem object (tilt, azimuth)
        capacity_dc: Installed DC capacity in kW
    """
    
    def __init__(self, plant_metadata_path: str | Path):
        """
        Initialize predictor with plant metadata.
        
        Args:
            plant_metadata_path: Path to plant JSON metadata file
                Expected keys: plant_id, latitude, longitude, tilt_deg, 
                              azimuth_deg, installed_capacity_kw, timezone
        """
        self.metadata_path = Path(plant_metadata_path)
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Plant metadata not found: {self.metadata_path}")
        
        # Load plant configuration
        with open(self.metadata_path, "r") as f:
            config = json.load(f)
        
        self.plant_id = config["plant_id"]
        self.capacity_dc = float(config["installed_capacity_kw"])
        self.tilt_deg = float(config["tilt_deg"])
        self.azimuth_deg = float(config["azimuth_deg"])
        
        # Create PVLib Location object (for solar position calculations)
        self.location = Location(
            latitude=float(config["latitude"]),
            longitude=float(config["longitude"]),
            tz=config.get("timezone", "UTC"),
            altitude=config.get("altitude_m", 0.0)  # Default sea level if not specified
        )
        
        # Create PVLib PVSystem object (for irradiance calculations)
        # Note: PVLib uses 180° = south, 90° = east, 270° = west
        self.system = PVSystem(
            surface_tilt=float(config["tilt_deg"]),
            surface_azimuth=float(config["azimuth_deg"]),
            # Simple model: no inverter, no temperature effects (can be extended later)
        )
    
    def predict_from_weather(
        self,
        weather_df: pd.DataFrame,
        time_col: str = "timestamp_utc",
        ghi_col: str = "ghi",
        dni_col: str = "dni",
        dhi_col: str = "dhi"
    ) -> np.ndarray:
        """
        Predict DC power from forecasted weather irradiance components.
        
        Args:
            weather_df: DataFrame with forecasted weather
                Required columns: timestamp_utc, ghi, dni, dhi
                Optional: temp_air (for temperature corrections)
            time_col: Name of timestamp column
            ghi_col: Global Horizontal Irradiance column (W/m²)
            dni_col: Direct Normal Irradiance column (W/m²)
            dhi_col: Diffuse Horizontal Irradiance column (W/m²)
        
        Returns:
            dc_power_norm: Normalized DC power [0, 1], shape (N,)
        """
        df = weather_df.copy()
        
        # Ensure timestamps are timezone-aware
        if time_col not in df.columns:
            raise ValueError(f"Time column '{time_col}' not found in weather_df")
        df[time_col] = pd.to_datetime(df[time_col], utc=True)
        
        # Check required irradiance columns
        required_cols = [ghi_col, dni_col, dhi_col]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required irradiance columns: {missing}")
        
        # Step 1: Calculate solar position for each timestamp
        # Returns: apparent_zenith, zenith, apparent_elevation, elevation, azimuth, equation_of_time
        solar_position = self.location.get_solarposition(df[time_col])
        
        # Step 2: Calculate plane-of-array (POA) irradiance
        # Transforms horizontal irradiance (GHI/DNI/DHI) to tilted plane
        poa_irradiance = pvlib.irradiance.get_total_irradiance(
            surface_tilt=self.tilt_deg,
            surface_azimuth=self.azimuth_deg,
            dni=df[dni_col],
            ghi=df[ghi_col],
            dhi=df[dhi_col],
            solar_zenith=solar_position["apparent_zenith"],
            solar_azimuth=solar_position["azimuth"]
        )
        # Returns dict with keys: poa_global, poa_direct, poa_diffuse, poa_sky_diffuse, poa_ground_diffuse
        
        # Step 3: Convert POA irradiance to DC power
        # Simple model: P_dc = (Irradiance / 1000 W/m²) × Capacity_dc
        # This assumes:
        # - Standard Test Conditions (STC): 1000 W/m² irradiance
        # - No temperature derating (can be added with temp_air)
        # - No soiling, aging, or mismatch losses
        poa_global = poa_irradiance["poa_global"].values  # W/m²
        
        # Convert to kW/m² and multiply by capacity
        dc_power_kw = (poa_global / 1000.0) * self.capacity_dc
        
        # Step 4: Normalize to [0, 1] scale (matches TFT training data)
        dc_power_norm = dc_power_kw / self.capacity_dc
        
        # Step 5: Clip to valid range [0, 1]
        # Negative values can occur at night or with bad weather data
        # Values > 1.0 should not occur with correct capacity, but clip as safety
        dc_power_norm = np.clip(dc_power_norm, 0.0, 1.0)
        
        return dc_power_norm
    
    def predict_clear_sky(
        self,
        start_time: pd.Timestamp | str,
        num_steps: int = 2880,
        freq: str = "15min"
    ) -> np.ndarray:
        """
        Predict DC power using clear-sky model (no cloud cover, ideal conditions).
        
        Useful for:
        - Testing/debugging without weather forecast data
        - Upper bound baseline (best case scenario)
        - Fallback when weather API is unavailable
        
        Args:
            start_time: Forecast start timestamp (timezone-aware)
            num_steps: Number of timesteps to predict (default: 2880 = 30 days @ 15-min)
            freq: Time frequency ('15min', '1H', etc.)
        
        Returns:
            dc_power_norm: Normalized DC power [0, 1], shape (num_steps,)
        """
        # Create timestamp range
        if isinstance(start_time, str):
            start_time = pd.Timestamp(start_time, tz="UTC")
        
        timestamps = pd.date_range(start=start_time, periods=num_steps, freq=freq)
        
        # Step 1: Calculate solar position
        solar_position = self.location.get_solarposition(timestamps)
        
        # Step 2: Calculate clear-sky irradiance
        # Uses Ineichen clear-sky model (simplified model, no atmospheric turbidity needed)
        clearsky = self.location.get_clearsky(timestamps, model="ineichen")
        # Returns DataFrame with columns: ghi, dni, dhi
        
        # Step 3: Calculate POA irradiance for tilted panel
        poa_irradiance = pvlib.irradiance.get_total_irradiance(
            surface_tilt=self.tilt_deg,
            surface_azimuth=self.azimuth_deg,
            dni=clearsky["dni"],
            ghi=clearsky["ghi"],
            dhi=clearsky["dhi"],
            solar_zenith=solar_position["apparent_zenith"],
            solar_azimuth=solar_position["azimuth"]
        )
        
        # Step 4: Convert to DC power (same as predict_from_weather)
        poa_global = poa_irradiance["poa_global"].values
        dc_power_kw = (poa_global / 1000.0) * self.capacity_dc
        dc_power_norm = dc_power_kw / self.capacity_dc
        dc_power_norm = np.clip(dc_power_norm, 0.0, 1.0)
        
        return dc_power_norm
    
    def save_prediction(
        self,
        power: np.ndarray,
        timestamps: pd.DatetimeIndex,
        output_path: str | Path,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save PVLib predictions to parquet file.
        
        Args:
            power: Normalized power predictions [0, 1], shape (N,)
            timestamps: Corresponding timestamps
            output_path: Output parquet file path
            metadata: Optional metadata to include in output
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Build output DataFrame
        df = pd.DataFrame({
            "timestamp_utc": timestamps,
            "plant_id": self.plant_id,
            "power_norm_pvlib": power,
            "power_kw_pvlib": power * self.capacity_dc
        })
        
        # Add metadata columns if provided
        if metadata:
            for key, val in metadata.items():
                df[key] = val
        
        # Save to parquet
        df.to_parquet(output_path, index=False, engine="pyarrow")
        print(f"[INFO] Saved PVLib predictions: {output_path}")
        print(f"       Rows: {len(df):,}, Columns: {len(df.columns)}")


def demo_usage():
    """
    Demonstration of PVLibPredictor usage for testing.
    
    Can be run standalone: python -m src.inference.pvlib_predictor
    """
    import sys
    
    # Example plant metadata path
    metadata_path = "data/metadata/germany/plant_03.json"
    
    if not Path(metadata_path).exists():
        print(f"[ERROR] Metadata file not found: {metadata_path}")
        print("[INFO] Please run from repository root: python -m src.inference.pvlib_predictor")
        sys.exit(1)
    
    # Initialize predictor
    print("[INFO] Initializing PVLibPredictor...")
    predictor = PVLibPredictor(metadata_path)
    print(f"       Plant: {predictor.plant_id}")
    print(f"       Location: {predictor.location.latitude:.4f}°N, {predictor.location.longitude:.4f}°E")
    print(f"       Tilt: {predictor.tilt_deg}°, Azimuth: {predictor.azimuth_deg}°")
    print(f"       Capacity: {predictor.capacity_dc:.1f} kW")
    
    # Example 1: Clear-sky prediction
    print("\n[DEMO 1] Clear-sky prediction (30 days @ 15-min)")
    start_time = pd.Timestamp("2023-11-01 00:00:00", tz="UTC")
    timestamps = pd.date_range(start=start_time, periods=2880, freq="15min")
    
    power_clearsky = predictor.predict_clear_sky(start_time, num_steps=2880, freq="15min")
    
    print(f"         Output shape: {power_clearsky.shape}")
    print(f"         Min: {power_clearsky.min():.4f}, Max: {power_clearsky.max():.4f}, Mean: {power_clearsky.mean():.4f}")
    print(f"         Non-zero steps: {(power_clearsky > 0.01).sum()} / {len(power_clearsky)}")
    
    # Save clear-sky output
    output_dir = Path("outputs/pvlib_forecasted")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    predictor.save_prediction(
        power=power_clearsky,
        timestamps=timestamps,
        output_path=output_dir / f"{predictor.plant_id}_clearsky_30d_15min.parquet",
        metadata={"model": "clear_sky_ineichen", "forecast_start": str(start_time)}
    )
    
    # Example 2: From synthetic weather forecast
    print("\n[DEMO 2] Prediction from synthetic weather forecast (7 days @ 1-hour)")
    timestamps_hourly = pd.date_range(start=start_time, periods=168, freq="1h")
    
    # Create synthetic weather forecast (normally from API)
    # Using clear-sky as synthetic "perfect forecast" for demo
    clearsky_hourly = predictor.location.get_clearsky(timestamps_hourly, model="ineichen")
    weather_df = pd.DataFrame({
        "timestamp_utc": timestamps_hourly,
        "ghi": clearsky_hourly["ghi"],
        "dni": clearsky_hourly["dni"],
        "dhi": clearsky_hourly["dhi"]
    })
    
    power_from_weather = predictor.predict_from_weather(weather_df)
    
    print(f"         Output shape: {power_from_weather.shape}")
    print(f"         Min: {power_from_weather.min():.4f}, Max: {power_from_weather.max():.4f}, Mean: {power_from_weather.mean():.4f}")
    
    predictor.save_prediction(
        power=power_from_weather,
        timestamps=timestamps_hourly,
        output_path=output_dir / f"{predictor.plant_id}_weather_7d_1h.parquet",
        metadata={"model": "pvlib_from_weather", "forecast_start": str(start_time)}
    )
    
    print("\n[SUCCESS] Demo completed! Check outputs/pvlib_forecasted/")


if __name__ == "__main__":
    demo_usage()
