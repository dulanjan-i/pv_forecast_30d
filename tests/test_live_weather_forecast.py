#!/usr/bin/env python3
"""
End-to-End Test: Live Weather → TFT → 15-Day Forecast

Tests the complete pipeline with real OpenMeteo API weather data.
NOTE: OpenMeteo Forecast API limited to 15 days (2025-10-01 to 2026-01-17).

Author: PV Forecast Team
Date: 2026-01-02
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.inference.physics_aware_forecaster import PhysicsAwareForecaster


def test_live_weather_15day():
    """
    Test 15-day forecast with live OpenMeteo weather data.
    
    Uses:
        - Plant 03 (Germany, hardcoded test plant)
        - Live weather from OpenMeteo Forecast API
        - Real TFT models (short + long head)
        - PVLib physics baseline
    
    Expected:
        - Forecast shape: (1440,) for 15 days @ 15-min
        - Output range: [0.0, 1.0] (normalized)
        - No NaNs, no negative values
        - Realistic diurnal pattern (zero at night)
    """
    print("="*70)
    print("TEST: 15-Day Live Weather Forecast (Plant 03)")
    print("="*70)
    
    # Paths (V1.0 FINAL - Verified Seeds: Short=42, Long=43)
    short_ckpt = Path("V1.0_FINAL_TFT/shorthead_seed42/checkpoints/best.pt")
    long_ckpt = Path("V1.0_FINAL_TFT/longhead_seed43/checkpoints/best.pt")
    plant_metadata = Path("V1.0_FINAL_TFT/plant_metadata/plant_03.json")
    short_train = Path("data/processed/plant_level/plant_03/15min_pca32/train.parquet")
    long_train = Path("data/processed/plant_level/plant_03/hourly_longhead/train.parquet")
    
    # Check files exist
    missing = []
    for p in [short_ckpt, long_ckpt, plant_metadata, short_train, long_train]:
        if not p.exists():
            missing.append(str(p))
    
    if missing:
        print("[ERROR] Missing files:")
        for f in missing:
            print(f"  - {f}")
        print("\n[SKIP] Run TFT training first to generate checkpoints")
        return False
    
    # Initialize forecaster
    print("\n[1/4] Initializing forecaster...")
    forecaster = PhysicsAwareForecaster(
        short_ckpt=short_ckpt,
        long_ckpt=long_ckpt,
        plant_metadata=plant_metadata,
        short_train_parquet=short_train,
        long_train_parquet=long_train,
        device='cpu'  # Force CPU for testing
    )
    
    # Forecast start (today)
    forecast_start = pd.Timestamp.now(tz='UTC').floor('D')
    print(f"\n[2/4] Forecast start: {forecast_start}")
    
    # Run forecast with live weather
    print("\n[3/4] Running forecast with LIVE WEATHER from OpenMeteo API...")
    print("      This will fetch real-time 15-day weather forecast")
    
    try:
        forecast_15d = forecaster.predict_30d(
            forecast_start=forecast_start,
            use_live_weather=True,
            return_components=False
        )
    except Exception as e:
        print(f"[ERROR] Forecast failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Validate output
    print("\n[4/4] Validating forecast output...")
    
    expected_steps = 1440  # 15 days @ 15-min (adjusted from 30d)
    checks = {
        'shape': forecast_15d.shape == (expected_steps,),
        'no_nans': not np.isnan(forecast_15d).any(),
        'no_negatives': (forecast_15d >= 0).all(),
        'range_valid': (forecast_15d.max() <= 1.5),  # Allow some overshoot
        'has_variation': forecast_15d.std() > 0.01,
        'has_zeros': (forecast_15d == 0).sum() > 100  # Nighttime zeros
    }
    
    print("\n" + "-"*70)
    print("Validation Results:")
    for check, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {check}: {passed}")
    
    print("\nForecast Statistics:")
    print(f"  Shape: {forecast_15d.shape}")
    print(f"  Range: [{forecast_15d.min():.4f}, {forecast_15d.max():.4f}]")
    print(f"  Mean: {forecast_15d.mean():.4f}")
    print(f"  Std: {forecast_15d.std():.4f}")
    print(f"  Zeros: {(forecast_15d == 0).sum()} / {len(forecast_15d)}")
    
    # Sample output
    print("\nSample forecast (first 10 steps):")
    for i in range(10):
        ts = forecast_start + pd.Timedelta(minutes=15*i)
        print(f"  {ts}: {forecast_15d[i]:.4f}")
    
    all_passed = all(checks.values())
    print("\n" + "="*70)
    if all_passed:
        print("✅ TEST PASSED - Live weather forecast working!")
    else:
        print("⚠️  TEST PARTIALLY PASSED - Some checks failed")
    print("="*70)
    
    return all_passed


if __name__ == "__main__":
    success = test_live_weather_15day()
    sys.exit(0 if success else 1)
