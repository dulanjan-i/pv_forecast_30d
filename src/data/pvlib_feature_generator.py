"""
PVLib Feature Generator
Adds solar position and irradiance features to weather dataframes.
"""

import warnings

import pandas as pd
import numpy as np
import pvlib


def _first_col(df: pd.DataFrame, names):
    """First existing column from `names` (as a Series), else None."""
    for n in names:
        if n in df.columns:
            return pd.to_numeric(df[n], errors="coerce")
    return None


def generate_pvlib_features(weather_df: pd.DataFrame,
                            latitude: float,
                            longitude: float,
                            altitude: float = 200,
                            tilt: float = 30,
                            azimuth: float = 180,
                            timezone: str = 'UTC') -> pd.DataFrame:
    """
    Add PVLib-computed solar features to weather dataframe.
    
    Args:
        weather_df: Weather forecast with timestamp_utc column
        latitude: Plant latitude (degrees)
        longitude: Plant longitude (degrees)
        altitude: Elevation above sea level (meters)
        tilt: Panel tilt angle (degrees, 0=flat)
        azimuth: Panel azimuth (degrees, 180=south)
        timezone: Timezone string
        
    Returns:
        DataFrame with added PVLib features:
        - solar_zenith: Solar zenith angle (degrees)
        - solar_azimuth: Solar azimuth angle (degrees)
        - solar_elevation: Solar elevation angle (degrees)
        - poa_global: Plane-of-array global irradiance (W/m²)
        - poa_direct: POA direct irradiance (W/m²)
        - poa_diffuse: POA diffuse irradiance (W/m²)
    """
    df = weather_df.copy()
    
    # Ensure timestamp column exists
    if 'timestamp_utc' not in df.columns:
        if 'timestamp' in df.columns:
            df['timestamp_utc'] = df['timestamp']
        else:
            raise ValueError("DataFrame must have 'timestamp_utc' or 'timestamp' column")
    
    # Create location object
    location = pvlib.location.Location(
        latitude=latitude,
        longitude=longitude,
        altitude=altitude,
        tz=timezone
    )
    
    # Get timestamps
    times = pd.DatetimeIndex(pd.to_datetime(df['timestamp_utc'], utc=True))
    
    # Calculate solar position
    solar_position = location.get_solarposition(times)
    
    df['solar_zenith'] = solar_position['zenith'].values
    df['solar_azimuth'] = solar_position['azimuth'].values
    df['solar_elevation'] = solar_position['elevation'].values
    
    # Resolve irradiance columns. Open-Meteo's `direct_radiation` is the *horizontal* beam component,
    # not DNI, so it must never be passed as `dni`; use the *_normal_irradiance column instead.
    ghi_s = _first_col(df, ["shortwave_radiation_instant", "shortwave_radiation", "ghi"])
    dhi_s = _first_col(df, ["diffuse_radiation_instant", "diffuse_radiation", "dhi"])
    dni_s = _first_col(df, ["direct_normal_irradiance_instant", "direct_normal_irradiance", "dni"])

    if ghi_s is not None and dhi_s is not None:
        ghi = ghi_s.clip(lower=0.0).to_numpy(dtype=float)
        dhi = dhi_s.clip(lower=0.0).to_numpy(dtype=float)
        if dni_s is not None:
            dni = dni_s.clip(lower=0.0).to_numpy(dtype=float)
        else:
            # Derive DNI from closure: DNI = (GHI - DHI) / cos(zenith), only for a well-defined sun position
            cosz = np.cos(np.radians(solar_position['zenith'].to_numpy(dtype=float)))
            with np.errstate(divide="ignore", invalid="ignore"):
                dni = np.where(cosz > 0.05, (ghi - dhi) / cosz, 0.0)
            dni = np.clip(dni, 0.0, 1400.0)
    else:
        warnings.warn(
            "No GHI/DHI columns found (looked for shortwave_radiation[_instant]/ghi and "
            "diffuse_radiation[_instant]/dhi); falling back to a clear-sky model.",
            stacklevel=2,
        )
        clearsky = location.get_clearsky(times)
        ghi = clearsky['ghi'].to_numpy(dtype=float)
        dni = clearsky['dni'].to_numpy(dtype=float)
        dhi = clearsky['dhi'].to_numpy(dtype=float)

    poa_irradiance = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt,
        surface_azimuth=azimuth,
        solar_zenith=solar_position['zenith'],
        solar_azimuth=solar_position['azimuth'],
        dni=dni,
        ghi=ghi,
        dhi=dhi
    )
    df['poa_global'] = poa_irradiance['poa_global'].values
    df['poa_direct'] = poa_irradiance['poa_direct'].values
    df['poa_diffuse'] = poa_irradiance['poa_diffuse'].values

    # Add time features
    df['hour_sin'] = np.sin(2 * np.pi * times.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * times.hour / 24)
    df['day_of_year'] = times.dayofyear
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    print(f"   ✓ Added PVLib features: solar position, POA irradiance, time encoding")
    
    return df


if __name__ == "__main__":
    # Test
    import pandas as pd
    
    # Create test weather data
    times = pd.date_range('2024-01-01', periods=96, freq='15min', tz='UTC')
    weather = pd.DataFrame({
        'timestamp_utc': times,
        'shortwave_radiation': np.random.rand(96) * 500,
        'direct_radiation': np.random.rand(96) * 400,
        'diffuse_radiation': np.random.rand(96) * 200
    })
    
    # Generate features
    result = generate_pvlib_features(
        weather_df=weather,
        latitude=51.3397,
        longitude=12.3731
    )
    
    print("\nGenerated features:")
    print(result.columns.tolist())
    print("\nSample data:")
    print(result.head())
