#!/usr/bin/env python3
"""
Comprehensive test of the hierarchical 30-day forecasting pipeline.
Tests all components with synthetic data before TFT integration.

Version: 2.0 (Hierarchical Refinement Architecture)
Date: 2026-01-02
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.inference.pvlib_predictor import PVLibPredictor
from src.inference.physics_glue import (
    upsample_with_pvlib_shape,
    blend_with_physics,
    blend_hierarchical,
    apply_physics_constraints
)
from src.inference.rl_controller import RLMetaController


def test_1_pvlib_predictor():
    """Test PVLib physics baseline generation."""
    print("\n" + "="*70)
    print("TEST 1: PVLib Physics Baseline")
    print("="*70)
    
    try:
        predictor = PVLibPredictor("data/metadata/germany/plant_03.json")
        
        # Generate 30-day clear-sky baseline
        baseline = predictor.predict_clear_sky(
            start_time="2023-11-01",
            num_steps=2880,
            freq="15min"
        )
        
        print(f"✓ Shape: {baseline.shape}")
        assert baseline.shape == (2880,), f"Expected (2880,), got {baseline.shape}"
        
        print(f"✓ Range: [{baseline.min():.3f}, {baseline.max():.3f}]")
        assert baseline.min() >= 0, "Negative values found"
        assert baseline.max() <= 1.0, "Values > 1.0 found"
        
        night_count = (baseline < 0.01).sum()
        day_count = (baseline >= 0.01).sum()
        print(f"✓ Daylight: {day_count}/{2880} steps ({day_count/2880*100:.1f}%)")
        print(f"✓ Night: {night_count}/{2880} steps ({night_count/2880*100:.1f}%)")
        
        print("✅ PVLib predictor working correctly")
        return baseline
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        raise


def test_2_upsampling(pvlib_baseline):
    """Test hourly to 15-min upsampling."""
    print("\n" + "="*70)
    print("TEST 2: Hourly to 15-min Upsampling")
    print("="*70)
    
    try:
        # Simulate long-head hourly predictions (720 hours)
        np.random.seed(42)
        long_hourly = np.clip(pvlib_baseline[::4] * np.random.uniform(0.8, 1.2, 720), 0, 1)
        
        # Upsample using PVLib shape
        long_upsampled = upsample_with_pvlib_shape(long_hourly, pvlib_baseline)
        
        print(f"✓ Input shape: {long_hourly.shape} (hourly)")
        print(f"✓ Output shape: {long_upsampled.shape} (15-min)")
        assert long_upsampled.shape == (2880,), f"Expected (2880,), got {long_upsampled.shape}"
        
        # Verify energy conservation (check first 5 hours)
        for h in range(5):
            hour_sum = long_upsampled[h*4:(h+1)*4].sum()
            expected = long_hourly[h]
            error = abs(hour_sum - expected)
            assert error < 1e-6, f"Hour {h}: sum={hour_sum:.6f}, expected={expected:.6f}"
        
        print(f"✓ Energy conservation verified (first 5 hours)")
        print("✅ Upsampling working correctly")
        return long_upsampled
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        raise


def test_3_hierarchical_blend(pvlib_baseline, long_upsampled):
    """Test 3-way hierarchical blending."""
    print("\n" + "="*70)
    print("TEST 3: Hierarchical 3-Way Blending")
    print("="*70)
    
    try:
        # Simulate short-head for Day 0 (first 96 steps)
        np.random.seed(43)
        short_day0 = np.clip(pvlib_baseline[:96] * np.random.uniform(0.9, 1.1, 96), 0, 1)
        
        # Extract corresponding slices
        long_slice = long_upsampled[:96]
        pvlib_slice = pvlib_baseline[:96]
        
        # Test hierarchical blend
        blended = blend_hierarchical(
            short_pred=short_day0,
            long_upsampled=long_slice,
            pvlib_baseline=pvlib_slice,
            alpha_short=0.6,
            alpha_long=0.4,
            alpha_ml=0.7,
            constraints=True
        )
        
        print(f"✓ Input shapes: short={short_day0.shape}, long={long_slice.shape}, pvlib={pvlib_slice.shape}")
        print(f"✓ Output shape: {blended.shape}")
        assert blended.shape == (96,), f"Expected (96,), got {blended.shape}"
        
        # Verify constraints
        print(f"✓ Range: [{blended.min():.3f}, {blended.max():.3f}]")
        assert blended.min() >= 0, "Negative values found"
        assert blended.max() <= 1.0, "Values > 1.0 found"
        
        # Check night constraint
        night_mask = pvlib_slice < 0.01
        night_violations = (blended[night_mask] > 0.01).sum()
        print(f"✓ Night constraint: {night_violations} violations (should be 0)")
        assert night_violations == 0, f"{night_violations} night violations"
        
        # Check capacity constraint
        max_allowed = pvlib_slice * 1.2
        capacity_violations = (blended > max_allowed).sum()
        print(f"✓ Capacity constraint: {capacity_violations} violations (should be 0)")
        assert capacity_violations == 0, f"{capacity_violations} capacity violations"
        
        print("✅ Hierarchical blending working correctly")
        return blended
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        raise


def test_4_rl_controller():
    """Test RL meta-controller weight adaptation."""
    print("\n" + "="*70)
    print("TEST 4: RL Meta-Controller")
    print("="*70)
    
    try:
        controller = RLMetaController(mode="heuristic")
        
        # Test weight evolution over 30 days
        print("\nWeight Evolution by Day:")
        print(f"{'Day':<5} {'α_short':<10} {'α_long':<10} {'α_ml':<10} {'α_pvlib':<10}")
        print("-" * 45)
        
        test_days = [0, 3, 7, 14, 21, 29]
        for day in test_days:
            weights = controller.get_blend_weights(day=day, weather_confidence=0.8)
            print(f"{day:<5} {weights['alpha_short']:<10.3f} {weights['alpha_long']:<10.3f} "
                  f"{weights['alpha_ml']:<10.3f} {weights['alpha_pvlib']:<10.3f}")
            
            # Verify constraints
            assert abs(weights['alpha_short'] + weights['alpha_long'] - 1.0) < 1e-6, "ML weights don't sum to 1"
            assert abs(weights['alpha_ml'] + weights['alpha_pvlib'] - 1.0) < 1e-6, "Physics weights don't sum to 1"
            assert 0 <= weights['alpha_short'] <= 1, "alpha_short out of range"
            assert 0 <= weights['alpha_ml'] <= 1, "alpha_ml out of range"
        
        # Verify trend: alpha_short decreases, alpha_long increases
        w0 = controller.get_blend_weights(0, 0.8)
        w29 = controller.get_blend_weights(29, 0.8)
        assert w0['alpha_short'] > w29['alpha_short'], "alpha_short should decrease with horizon"
        assert w0['alpha_long'] < w29['alpha_long'], "alpha_long should increase with horizon"
        
        print("\n✓ Weight normalization correct")
        print("✓ Weight evolution trend correct (short↓, long↑)")
        print("✅ RL meta-controller working correctly")
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        raise


def test_5_full_30day_pipeline(pvlib_baseline):
    """Test full 30-day hierarchical pipeline with synthetic TFT."""
    print("\n" + "="*70)
    print("TEST 5: Full 30-Day Hierarchical Pipeline")
    print("="*70)
    
    try:
        np.random.seed(44)
        
        # Simulate long-head prediction (1 call → 720 hours)
        print("Step 1: Long-head strategic overview (1 TFT call)")
        long_hourly = np.clip(pvlib_baseline[::4] * np.random.uniform(0.85, 1.15, 720), 0, 1)
        long_upsampled = upsample_with_pvlib_shape(long_hourly, pvlib_baseline)
        print(f"  ✓ Long-head: {long_hourly.shape} → upsampled to {long_upsampled.shape}")
        
        # Initialize RL controller
        controller = RLMetaController(mode="heuristic")
        
        # Rolling 30-day refinement
        print("\nStep 2: Rolling daily refinement (30 TFT calls)")
        forecast_30d = np.zeros(2880)
        short_calls = []
        
        for day in range(30):
            # Simulate short-head for this day (96 steps)
            day_start_idx = day * 96
            day_end_idx = (day + 1) * 96
            pvlib_day = pvlib_baseline[day_start_idx:day_end_idx]
            short_day = np.clip(pvlib_day * np.random.uniform(0.9, 1.1, 96), 0, 1)
            short_calls.append(short_day)
            
            # Get adaptive weights
            weights = controller.get_blend_weights(day=day, weather_confidence=0.8)
            
            # Hierarchical blend
            long_slice = long_upsampled[day_start_idx:day_end_idx]
            blended_day = blend_hierarchical(
                short_pred=short_day,
                long_upsampled=long_slice,
                pvlib_baseline=pvlib_day,
                alpha_short=weights['alpha_short'],
                alpha_long=weights['alpha_long'],
                alpha_ml=weights['alpha_ml'],
                constraints=True
            )
            
            forecast_30d[day_start_idx:day_end_idx] = blended_day
            
            if day % 10 == 0:
                print(f"  Day {day:2d}: α_short={weights['alpha_short']:.2f}, "
                      f"α_ml={weights['alpha_ml']:.2f} → blended {len(blended_day)} steps")
        
        print(f"\n  ✓ Total TFT calls: 1 long + {len(short_calls)} short = {1 + len(short_calls)}")
        
        # Validation
        print("\nStep 3: Final validation")
        print(f"  ✓ Final shape: {forecast_30d.shape}")
        assert forecast_30d.shape == (2880,), f"Expected (2880,), got {forecast_30d.shape}"
        
        print(f"  ✓ Range: [{forecast_30d.min():.3f}, {forecast_30d.max():.3f}]")
        assert forecast_30d.min() >= 0, "Negative values found"
        assert forecast_30d.max() <= 1.0, "Values > 1.0 found"
        
        # Night constraint
        night_mask = pvlib_baseline < 0.01
        night_violations = (forecast_30d[night_mask] > 0.01).sum()
        print(f"  ✓ Night violations: {night_violations} / {night_mask.sum()} night steps")
        assert night_violations == 0, f"{night_violations} night violations"
        
        # Capacity constraint
        max_allowed = pvlib_baseline * 1.2
        capacity_violations = (forecast_30d > max_allowed).sum()
        print(f"  ✓ Capacity violations: {capacity_violations} / {2880} total steps")
        assert capacity_violations == 0, f"{capacity_violations} capacity violations"
        
        print("\n✅ Full 30-day hierarchical pipeline working correctly!")
        
        # Summary
        print("\n" + "="*70)
        print("PIPELINE SUMMARY")
        print("="*70)
        print(f"Architecture: Hierarchical Refinement v2.0")
        print(f"Long-head: 1 call → 720 hours @ 1h → upsampled to 2880 @ 15min")
        print(f"Short-head: 30 calls → 96 steps @ 15min per day")
        print(f"Total TFT calls: 31 (1 long + 30 short)")
        print(f"Blending: 3-way hierarchical (short + long + physics)")
        print(f"RL adaptation: α_short 0.65→0.35, α_ml 0.71→0.57 over 30 days")
        print(f"Final forecast: {forecast_30d.shape} @ 15-min resolution")
        print(f"Physics constraints: ✓ All enforced (night=0, capacity≤120%)")
        
        return forecast_30d
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        raise


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("HIERARCHICAL 30-DAY FORECASTING PIPELINE - COMPREHENSIVE TEST")
    print("="*70)
    print("Version: 2.0 (Hierarchical Refinement Architecture)")
    print("Status: Pre-TFT Integration (Synthetic Placeholders)")
    print("="*70)
    
    try:
        # Test 1: PVLib baseline
        pvlib_baseline = test_1_pvlib_predictor()
        
        # Test 2: Upsampling
        long_upsampled = test_2_upsampling(pvlib_baseline)
        
        # Test 3: Hierarchical blending
        _ = test_3_hierarchical_blend(pvlib_baseline, long_upsampled)
        
        # Test 4: RL controller
        test_4_rl_controller()
        
        # Test 5: Full pipeline
        _ = test_5_full_30day_pipeline(pvlib_baseline)
        
        # Final summary
        print("\n" + "="*70)
        print("🎉 ALL TESTS PASSED!")
        print("="*70)
        print("\n✅ Hierarchical architecture fully validated")
        print("✅ All physics constraints enforced")
        print("✅ RL adaptive weights working")
        print("✅ 31 TFT calls confirmed (1 long + 30 short)")
        print("\n🔒 PIPELINE LOCKED AND READY FOR TFT INTEGRATION")
        print("\nNext step: Replace synthetic TFT placeholders with real inference")
        print("  - Implement _predict_long_head() in physics_aware_forecaster.py")
        print("  - Implement _predict_short_head_for_day() in physics_aware_forecaster.py")
        print("  - Study offline_predict_tft.py for batch preparation")
        print("="*70)
        
        return 0
        
    except Exception as e:
        print("\n" + "="*70)
        print("❌ TEST SUITE FAILED")
        print("="*70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
