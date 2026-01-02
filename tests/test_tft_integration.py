#!/usr/bin/env python3
"""
Test TFT integration with PhysicsAwareForecaster.
Validates short-head and long-head inference with real checkpoints.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.inference.physics_aware_forecaster import PhysicsAwareForecaster


def test_short_head_single_day():
    """Test short-head TFT inference on Day 0."""
    print("\n" + "="*70)
    print("TEST: Short-Head TFT Inference (Single Day)")
    print("="*70)
    
    # Checkpoint paths
    short_ckpt = Path("experiments/tft/runs/germany/plant_03/15min/pvlib_warmstart_from_global_noleak/20251229_151100/checkpoints/best_state_dict.pt")
    long_ckpt = Path("experiments/tft/runs/germany/plant_03/longhead/hourly720/warm/lr8e-4_do0.15_bs64_acc8_seed43/20251231_104405/checkpoints/best_state_dict.pt")
    metadata = Path("data/metadata/germany/plant_03.json")
    short_train = Path("data/processed/plant_level/plant_03/15min_pca32/train.parquet")
    long_train = Path("data/processed/plant_level/plant_03/hourly_longhead/train.parquet")
    
    # Check files exist
    missing = []
    for p in [short_ckpt, long_ckpt, metadata, short_train, long_train]:
        if not p.exists():
            missing.append(str(p))
    
    if missing:
        print("❌ Missing files:")
        for m in missing:
            print(f"   - {m}")
        return False
    
    print("\n✓ All required files found")
    
    try:
        # Initialize forecaster
        print("\n[1/4] Initializing PhysicsAwareForecaster...")
        forecaster = PhysicsAwareForecaster(
            short_ckpt=short_ckpt,
            long_ckpt=long_ckpt,
            plant_metadata=metadata,
            short_train_parquet=short_train,
            long_train_parquet=long_train,
            device="cpu"  # Use CPU for testing
        )
        print("✓ Forecaster initialized")
        
        # Load test data
        print("\n[2/4] Loading test data...")
        test_df = pd.read_parquet("data/processed/plant_level/plant_03/15min_pca32/test.parquet")
        print(f"✓ Test data loaded: {test_df.shape}")
        print(f"   Time range: {test_df.timestamp_utc.min()} → {test_df.timestamp_utc.max()}")
        
        # Test Day 2 prediction (skip first day to have enough history)
        print("\n[3/4] Running short-head inference for Day 2...")
        # Start at Day 2 to ensure we have 96 steps of history (Day 1)
        day2_start = pd.Timestamp("2023-10-14 00:00:00", tz="UTC")
        
        # Extract historical (encoder) and forecast (decoder) windows
        historical_df = test_df.copy()  # Full test set for history
        weather_df = test_df.copy()  # Full test set for weather
        
        print(f"   Day 2 start: {day2_start}")
        print(f"   Test data range: {test_df.timestamp_utc.min()} → {test_df.timestamp_utc.max()}")
        
        pred_day2 = forecaster._predict_short_head_for_day(
            day_start=day2_start,
            day_idx=2,
            historical_df=historical_df,
            weather_df=weather_df
        )
        
        print(f"✓ Short-head prediction complete!")
        print(f"   Shape: {pred_day2.shape}")
        print(f"   Range: [{pred_day2.min():.4f}, {pred_day2.max():.4f}]")
        print(f"   Mean: {pred_day2.mean():.4f}")
        
        # Compare vs offline_predict_tft.py baseline
        print("\n[4/4] Comparing vs offline baseline...")
        offline_preds = pd.read_parquet("outputs/plant03_shorthead_test_preds.parquet")
        
        # Find Day 2 predictions (rows 192-288, since 96 steps per day @ 15min)
        offline_day2 = offline_preds.iloc[192:288]['y_hat_q50'].values
        
        # Calculate MAE
        mae = np.abs(pred_day2 - offline_day2).mean()
        rmse = np.sqrt(((pred_day2 - offline_day2) ** 2).mean())
        max_diff = np.abs(pred_day2 - offline_day2).max()
        
        print(f"   Offline baseline shape: {offline_day2.shape}")
        print(f"   MAE vs offline: {mae:.6f}")
        print(f"   RMSE vs offline: {rmse:.6f}")
        print(f"   Max difference: {max_diff:.6f}")
        
        # Success criteria
        if mae < 0.01:  # Very close match
            print("\n✅ SHORT-HEAD TEST PASSED!")
            print("   Predictions match offline baseline (MAE < 0.01)")
            return True
        elif mae < 0.05:  # Reasonable match
            print("\n⚠️  SHORT-HEAD TEST ACCEPTABLE")
            print(f"   Minor differences (MAE={mae:.4f} < 0.05)")
            return True
        else:
            print("\n❌ SHORT-HEAD TEST FAILED")
            print(f"   Large differences (MAE={mae:.4f} >= 0.05)")
            return False
            
    except Exception as e:
        print(f"\n❌ SHORT-HEAD TEST FAILED WITH ERROR:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_long_head():
    """Test long-head TFT inference."""
    print("\n" + "="*70)
    print("TEST: Long-Head TFT Inference (30 Days)")
    print("="*70)
    
    # Checkpoint paths
    short_ckpt = Path("experiments/tft/runs/germany/plant_03/15min/pvlib_warmstart_from_global_noleak/20251229_151100/checkpoints/best_state_dict.pt")
    long_ckpt = Path("experiments/tft/runs/germany/plant_03/longhead/hourly720/warm/lr8e-4_do0.15_bs64_acc8_seed43/20251231_104405/checkpoints/best_state_dict.pt")
    metadata = Path("data/metadata/germany/plant_03.json")
    short_train = Path("data/processed/plant_level/plant_03/15min_pca32/train.parquet")
    long_train = Path("data/processed/plant_level/plant_03/hourly_longhead/train.parquet")
    
    try:
        # Initialize forecaster
        print("\n[1/3] Initializing PhysicsAwareForecaster...")
        forecaster = PhysicsAwareForecaster(
            short_ckpt=short_ckpt,
            long_ckpt=long_ckpt,
            plant_metadata=metadata,
            short_train_parquet=short_train,
            long_train_parquet=long_train,
            device="cpu"
        )
        print("✓ Forecaster initialized")
        
        # Load test data
        print("\n[2/3] Loading hourly test data...")
        test_df = pd.read_parquet("data/processed/plant_level/plant_03/hourly_longhead/test.parquet")
        print(f"✓ Test data loaded: {test_df.shape}")
        
        # Test long-head prediction (start a few days in to have history)
        print("\n[3/3] Running long-head inference (720 hours)...")
        forecast_start = pd.Timestamp("2023-10-19 00:00:00", tz="UTC")  # Start 7 days in for history
        
        historical_df = test_df.copy()  # Full test set for history
        weather_df = test_df.copy()  # Full test set for weather
        
        print(f"   Forecast start: {forecast_start}")
        print(f"   Test data range: {test_df.timestamp_utc.min()} → {test_df.timestamp_utc.max()}")
        
        pred_long = forecaster._predict_long_head(
            forecast_start=forecast_start,
            historical_df=historical_df,
            weather_df=weather_df
        )
        
        print(f"✓ Long-head prediction complete!")
        print(f"   Shape: {pred_long.shape}")
        print(f"   Range: [{pred_long.min():.4f}, {pred_long.max():.4f}]")
        print(f"   Mean: {pred_long.mean():.4f}")
        
        # Validation checks
        checks = []
        if pred_long.shape == (720,):
            checks.append("✓ Shape correct (720,)")
        else:
            checks.append(f"✗ Shape wrong: {pred_long.shape}")
        
        if pred_long.min() >= 0 and pred_long.max() <= 1.0:
            checks.append(f"✓ Range valid [{pred_long.min():.4f}, {pred_long.max():.4f}]")
        else:
            checks.append(f"✗ Range invalid")
        
        print("\nValidation:")
        for check in checks:
            print(f"   {check}")
        
        if all("✓" in c for c in checks):
            print("\n✅ LONG-HEAD TEST PASSED!")
            return True
        else:
            print("\n❌ LONG-HEAD TEST FAILED")
            return False
            
    except Exception as e:
        print(f"\n❌ LONG-HEAD TEST FAILED WITH ERROR:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all TFT integration tests."""
    print("\n" + "="*70)
    print("TFT INTEGRATION TEST SUITE")
    print("="*70)
    print("Testing real TFT inference in hierarchical forecasting pipeline")
    print("="*70)
    
    results = {
        'short_head': test_short_head_single_day(),
        'long_head': test_long_head()
    }
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:20s}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n" + "="*70)
        print("🎉 ALL TFT INTEGRATION TESTS PASSED!")
        print("="*70)
        print("\n✅ Short-head TFT inference working")
        print("✅ Long-head TFT inference working")
        print("✅ Ready for full 30-day hierarchical pipeline test")
        print("\nNext: Run test_hierarchical_pipeline.py with real TFT")
        return 0
    else:
        print("\n" + "="*70)
        print("❌ SOME TESTS FAILED")
        print("="*70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
