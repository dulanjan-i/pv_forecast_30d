#!/usr/bin/env python3
"""
Test ECMWF Open Data API (Free Real-Time Forecasts)

ECMWF Open Data provides:
- IFS HRES: 0-10 days, 0.25° resolution (deterministic)
- IFS ENS: 0-15 days, 0.5° resolution, 51 ensemble members
- No authentication required!
- Updated 4x daily (00, 06, 12, 18 UTC)

Docs: https://www.ecmwf.int/en/forecasts/datasets/open-data
API: https://github.com/ecmwf/ecmwf-opendata

Author: PV Forecast Team
Date: 2026-01-03
"""

import json
from pathlib import Path
from datetime import datetime
from ecmwf.opendata import Client
import numpy as np

def test_ecmwf_opendata_ifs_hres():
    """Test ECMWF IFS HRES (High Resolution deterministic)."""
    print("\n" + "="*70)
    print("Testing ECMWF Open Data - IFS HRES (10-day deterministic)")
    print("="*70)
    
    # Load plant metadata
    meta_path = Path("/home/dwijenayake/pv_forecast_30d/V1.0_FINAL_TFT/plant_metadata/plant_03.json")
    with open(meta_path) as f:
        metadata = json.load(f)
    
    lat = metadata['latitude']
    lon = metadata['longitude']
    
    print(f"Location: {lat:.4f}°N, {lon:.4f}°E (plant_03)")
    
    # Initialize client
    client = Client(source="ecmwf")  # Can also use "aws", "azure", "gcp"
    
    # Available parameters for surface level
    parameters = [
        "2t",      # 2m temperature
        "10u",     # 10m u-wind
        "10v",     # 10m v-wind
        "sp",      # Surface pressure
        "tp",      # Total precipitation
        "tcc",     # Total cloud cover
        # Solar radiation (may not be directly available - need derived)
        "ssrd",    # Surface solar radiation downwards (accumulated)
        "strd",    # Surface thermal radiation downwards
    ]
    
    try:
        print(f"\nRequesting IFS HRES forecast...")
        print(f"  Model: IFS HRES (High Resolution)")
        print(f"  Horizon: 0-10 days (240 hours)")
        print(f"  Resolution: 0.25° (~28km)")
        print(f"  Parameters: {', '.join(parameters)}")
        
        # Request latest forecast
        # Note: ECMWF Open Data uses GRIB2 format
        result = client.retrieve(
            step=list(range(0, 241, 3)),  # 0-240 hours, 3-hourly
            type="fc",  # Forecast
            param=parameters,
            target="ecmwf_ifs_hres.grib2"
        )
        
        print(f"\n✅ IFS HRES retrieved successfully!")
        print(f"  Saved to: ecmwf_ifs_hres.grib2")
        print(f"  Steps: 0-240h @ 3-hourly (81 timesteps)")
        
        # TODO: Parse GRIB2 with cfgrib/xarray
        print(f"\n⚠️  GRIB2 parsing requires cfgrib package")
        print(f"  Install: pip install cfgrib")
        
        return True
        
    except Exception as e:
        print(f"\n❌ IFS HRES failed: {e}")
        return False


def test_ecmwf_opendata_ifs_ens():
    """Test ECMWF IFS ENS (Ensemble forecast)."""
    print("\n" + "="*70)
    print("Testing ECMWF Open Data - IFS ENS (15-day ensemble)")
    print("="*70)
    
    # Load plant metadata
    meta_path = Path("/home/dwijenayake/pv_forecast_30d/V1.0_FINAL_TFT/plant_metadata/plant_03.json")
    with open(meta_path) as f:
        metadata = json.load(f)
    
    lat = metadata['latitude']
    lon = metadata['longitude']
    
    print(f"Location: {lat:.4f}°N, {lon:.4f}°E (plant_03)")
    
    client = Client(source="ecmwf")
    
    parameters = ["2t", "10u", "10v", "tp", "tcc"]
    
    try:
        print(f"\nRequesting IFS ENS forecast...")
        print(f"  Model: IFS ENS (Ensemble)")
        print(f"  Horizon: 0-15 days (360 hours)")
        print(f"  Resolution: 0.5° (~55km)")
        print(f"  Members: 51 (control + 50 perturbed)")
        
        # Request ensemble forecast
        result = client.retrieve(
            step=list(range(0, 361, 6)),  # 0-360 hours, 6-hourly
            type="ef",  # Ensemble forecast
            param=parameters,
            number=[0] + list(range(1, 51)),  # Control + perturbed members
            target="ecmwf_ifs_ens.grib2"
        )
        
        print(f"\n✅ IFS ENS retrieved successfully!")
        print(f"  Saved to: ecmwf_ifs_ens.grib2")
        print(f"  Steps: 0-360h @ 6-hourly (61 timesteps × 51 members)")
        
        return True
        
    except Exception as e:
        print(f"\n❌ IFS ENS failed: {e}")
        return False


def print_comparison():
    """Print comparison with Open-Meteo."""
    print("\n" + "="*70)
    print("ECMWF OPEN DATA vs OPEN-METEO COMPARISON")
    print("="*70)
    
    print("\n📊 ECMWF Open Data (Direct):")
    print("  ✅ FREE (no credentials)")
    print("  ✅ IFS HRES: 10 days @ 0.25° (best quality)")
    print("  ✅ IFS ENS: 15 days @ 0.5° (51 members)")
    print("  ✅ Updated 4x daily (00, 06, 12, 18 UTC)")
    print("  ⚠️  GRIB2 format (requires cfgrib parsing)")
    print("  ⚠️  Limited solar variables (ssrd accumulated, needs processing)")
    print("  ⚠️  500 simultaneous connection limit")
    
    print("\n📊 Open-Meteo (Current):")
    print("  ✅ FREE (generous rate limits)")
    print("  ✅ JSON API (easy parsing)")
    print("  ✅ Forecast: 15 days @ high quality")
    print("  ✅ ECMWF proxy: 15 days @ 0.25°")
    print("  ✅ GFS: 16 days @ 50km")
    print("  ✅ Solar variables pre-computed (GHI, DNI, DHI, GTI)")
    print("  ✅ Panel tilt/azimuth support built-in")
    
    print("\n🎯 RECOMMENDATION:")
    print("  For PV forecasting in Germany, Open-Meteo is OPTIMAL because:")
    print("    1. Already provides ECMWF IFS data (via proxy)")
    print("    2. Pre-computed solar irradiance (GHI/DNI/DHI/GTI)")
    print("    3. Panel-aware calculations (tilt/azimuth)")
    print("    4. JSON API (no GRIB parsing needed)")
    print("    5. Fast response (<0.1s vs multi-second GRIB downloads)")
    
    print("\n  ECMWF Open Data is better for:")
    print("    - Research requiring raw meteorological fields")
    print("    - Ensemble uncertainty quantification (51 members)")
    print("    - Custom processing pipelines")
    
    print("\n  🚀 ACTION: Keep current Open-Meteo setup (no change needed)")
    print("="*70 + "\n")


def main():
    """Run ECMWF Open Data tests."""
    print("\n" + "#"*70)
    print("# ECMWF OPEN DATA API TEST")
    print("# Free real-time forecasts (no credentials required)")
    print("#"*70)
    
    # Test IFS HRES
    hres_ok = test_ecmwf_opendata_ifs_hres()
    
    # Test IFS ENS
    # ens_ok = test_ecmwf_opendata_ifs_ens()  # Skip for now (large download)
    
    # Print comparison
    print_comparison()
    
    if hres_ok:
        print("✅ ECMWF Open Data is accessible!")
        print("   However, Open-Meteo is still recommended for PV forecasting.")
    else:
        print("⚠️  ECMWF Open Data had issues (connection limit or format)")
        print("   Stick with Open-Meteo (proven and working).")


if __name__ == "__main__":
    main()
