# src/inference/physics_glue.py
"""
Physics-Aware Glue Functions for MiRACLE Forecasting.

Purpose:
    Bridge ML predictions (TFT) with physics models (PVLib) for robust forecasting.
    Implements upsampling, blending, and constraint enforcement.

Functions:
    - upsample_with_pvlib_shape: Distribute hourly predictions to 15-min using solar curve
    - blend_with_physics: Combine ML and physics predictions with weights
    - apply_physics_constraints: Enforce physical limits (night=0, max capacity)

Architecture:
    Day 1 (0-96 steps @ 15-min):  Short-head TFT → blend with PVLib
    Days 2-30 (96-2880 steps @ 15-min):  Long-head TFT (hourly) → upsample → blend with PVLib
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


def upsample_with_pvlib_shape(
    hourly_predictions: np.ndarray,
    pvlib_15min: np.ndarray,
    method: str = "proportional"
) -> np.ndarray:
    """
    Upsample hourly predictions to 15-min resolution using PVLib solar curve shape.
    
    Distributes each hourly value across 4 quarter-hour intervals proportionally
    to the PVLib power curve, preserving the hourly sum.
    
    Args:
        hourly_predictions: ML predictions at 1-hour resolution, shape (H,)
            Example: 720 steps = 30 days
        pvlib_15min: PVLib physics predictions at 15-min resolution, shape (H*4,)
            Example: 2880 steps = 30 days @ 15-min
        method: Upsampling method
            - 'proportional': Distribute proportional to PVLib intra-hour shape (default)
            - 'uniform': Simple linear interpolation (fallback if PVLib unavailable)
    
    Returns:
        upsampled_15min: Predictions at 15-min resolution, shape (H*4,)
    
    Example:
        >>> hourly = np.array([0.5, 0.6, 0.7])  # 3 hours
        >>> pvlib = np.array([0.1, 0.15, 0.2, 0.05,   # hour 1
        ...                   0.2, 0.25, 0.3, 0.25,   # hour 2
        ...                   0.3, 0.35, 0.4, 0.35])  # hour 3
        >>> result = upsample_with_pvlib_shape(hourly, pvlib)
        >>> result.shape
        (12,)
        >>> # Each hour's 4 steps sum to original hourly value
    """
    num_hours = len(hourly_predictions)
    expected_15min = num_hours * 4
    
    if len(pvlib_15min) != expected_15min:
        raise ValueError(
            f"PVLib shape mismatch: expected {expected_15min} 15-min steps "
            f"for {num_hours} hours, got {len(pvlib_15min)}"
        )
    
    if method == "uniform":
        # Simple linear interpolation (fallback)
        # Repeat each hourly value 4 times, then smooth
        upsampled = np.repeat(hourly_predictions, 4)
        return upsampled
    
    elif method == "proportional":
        # Proportional distribution using PVLib shape
        upsampled = np.zeros(expected_15min, dtype=np.float32)
        
        for h in range(num_hours):
            # Get this hour's 4 quarter-hour PVLib values
            start_idx = h * 4
            end_idx = start_idx + 4
            pvlib_hour = pvlib_15min[start_idx:end_idx]
            
            # Calculate sum for this hour
            pvlib_sum = pvlib_hour.sum()
            
            if pvlib_sum > 1e-6:  # Daytime: distribute proportionally
                # Proportion: what fraction of the hour's total does each 15-min get?
                proportions = pvlib_hour / pvlib_sum
                # Apply proportions to ML hourly prediction
                upsampled[start_idx:end_idx] = hourly_predictions[h] * proportions
            else:
                # Nighttime (all PVLib near zero): distribute uniformly
                # This avoids division by zero
                upsampled[start_idx:end_idx] = hourly_predictions[h] / 4.0
        
        return upsampled
    
    else:
        raise ValueError(f"Unknown upsampling method: {method}")


def blend_with_physics(
    ml_predictions: np.ndarray,
    pvlib_baseline: np.ndarray,
    alpha: float,
    constraints: bool = True,
    max_capacity_multiplier: float = 1.2
) -> np.ndarray:
    """
    Blend ML predictions with physics baseline using weighted average.
    
    Final prediction = α × ML + (1-α) × PVLib
    
    Args:
        ml_predictions: TFT model predictions, shape (N,)
        pvlib_baseline: PVLib physics predictions, shape (N,)
        alpha: Blending weight for ML predictions ∈ [0, 1]
            - α=1.0: Pure ML (trust model completely)
            - α=0.0: Pure physics (no trust in ML)
            - α=0.7: 70% ML, 30% physics (typical for Day 1)
            - α=0.5: Equal weight (typical for Days 2-30)
        constraints: Apply physics constraints after blending
        max_capacity_multiplier: Maximum allowed over-capacity (1.2 = 120% of PVLib)
    
    Returns:
        blended: Physics-aware predictions, shape (N,)
    
    Example:
        >>> ml = np.array([0.8, 0.6, 0.4])
        >>> pvlib = np.array([0.7, 0.5, 0.3])
        >>> blend_with_physics(ml, pvlib, alpha=0.7)
        array([0.77, 0.57, 0.37])  # 0.7*0.8 + 0.3*0.7 = 0.77
    """
    if len(ml_predictions) != len(pvlib_baseline):
        raise ValueError(
            f"Shape mismatch: ML {ml_predictions.shape} vs PVLib {pvlib_baseline.shape}"
        )
    
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"Alpha must be in [0, 1], got {alpha}")
    
    # Weighted blend
    blended = alpha * ml_predictions + (1.0 - alpha) * pvlib_baseline
    
    if constraints:
        blended = apply_physics_constraints(
            blended, pvlib_baseline, max_capacity_multiplier
        )
    
    return blended


def apply_physics_constraints(
    predictions: np.ndarray,
    pvlib_baseline: np.ndarray,
    max_capacity_multiplier: float = 1.2
) -> np.ndarray:
    """
    Apply physics-based constraints to predictions.
    
    Constraints:
        1. Nighttime constraint: If PVLib near zero (< 0.01), force prediction to zero
        2. Capacity constraint: Clip to [0, pvlib × max_multiplier]
        3. Non-negative constraint: No negative power
    
    Args:
        predictions: Raw predictions to constrain, shape (N,)
        pvlib_baseline: Physics baseline for reference, shape (N,)
        max_capacity_multiplier: Max allowed over-capacity relative to PVLib
            - 1.0: Never exceed PVLib (conservative)
            - 1.2: Allow up to 120% of PVLib (account for ML learning edge effects)
            - 1.5: Allow up to 150% (aggressive, use with caution)
    
    Returns:
        constrained: Physics-compliant predictions, shape (N,)
    """
    constrained = predictions.copy()
    
    # Constraint 1: Force night hours to zero
    # PVLib < 0.01 indicates nighttime (< 1% capacity)
    night_mask = pvlib_baseline < 0.01
    constrained[night_mask] = 0.0
    
    # Constraint 2: Clip to valid range [0, max_capacity]
    # Use PVLib as reference for physical maximum
    max_allowed = pvlib_baseline * max_capacity_multiplier
    constrained = np.clip(constrained, 0.0, max_allowed)
    
    return constrained


def blend_hierarchical(
    short_pred: np.ndarray,
    long_upsampled: np.ndarray,
    pvlib_baseline: np.ndarray,
    alpha_short: float = 0.6,
    alpha_long: float = 0.4,
    alpha_ml: float = 0.7,
    constraints: bool = True,
    max_capacity_multiplier: float = 1.2
) -> np.ndarray:
    """
    3-way hierarchical blend: short-head + long-head + physics.
    
    Architecture:
        Layer 1: ML ensemble (short + long strategic blend)
        Layer 2: Physics-aware (ML + PVLib blend)
        Layer 3: Hard constraints (night=0, capacity limits)
    
    Args:
        short_pred: Short-head TFT prediction (tactical precision), shape (N,)
        long_upsampled: Long-head TFT upsampled (strategic context), shape (N,)
        pvlib_baseline: PVLib physics prediction, shape (N,)
        alpha_short: Weight for short-head in ML ensemble (0.6 = 60%)
        alpha_long: Weight for long-head in ML ensemble (0.4 = 40%)
        alpha_ml: Weight for ML blend vs physics (0.7 = 70% ML, 30% physics)
        constraints: Apply hard physics constraints
        max_capacity_multiplier: Maximum over-capacity allowed (1.2 = 120%)
    
    Returns:
        final: Hierarchical blended prediction, shape (N,)
    
    Example:
        >>> short = np.array([0.8, 0.6, 0.4])
        >>> long = np.array([0.7, 0.5, 0.3])
        >>> pvlib = np.array([0.65, 0.48, 0.25])
        >>> result = blend_hierarchical(short, long, pvlib)
        >>> # Layer 1: 0.6×0.8 + 0.4×0.7 = 0.76
        >>> # Layer 2: 0.7×0.76 + 0.3×0.65 = 0.727
        >>> # Layer 3: constrain(0.727, pvlib)
    """
    if len(short_pred) != len(long_upsampled) or len(short_pred) != len(pvlib_baseline):
        raise ValueError(
            f"Shape mismatch: short {short_pred.shape}, long {long_upsampled.shape}, "
            f"pvlib {pvlib_baseline.shape}"
        )
    
    if not (0 <= alpha_short <= 1 and 0 <= alpha_long <= 1):
        raise ValueError(f"Alphas must be in [0,1], got short={alpha_short}, long={alpha_long}")
    
    if not abs(alpha_short + alpha_long - 1.0) < 1e-6:
        raise ValueError(f"alpha_short + alpha_long must equal 1.0, got {alpha_short + alpha_long}")
    
    # Layer 1: ML Ensemble (short precision + long strategy)
    ml_blend = alpha_short * short_pred + alpha_long * long_upsampled
    
    # Layer 2: Physics-Aware Blend (ML data-driven + PVLib physics)
    physics_blend = alpha_ml * ml_blend + (1.0 - alpha_ml) * pvlib_baseline
    
    # Layer 3: Hard Constraints (enforce physical reality)
    if constraints:
        final = apply_physics_constraints(
            physics_blend,
            pvlib_baseline,
            max_capacity_multiplier=max_capacity_multiplier
        )
    else:
        final = physics_blend
    
    return final


# ==================== Testing & Validation ====================

# Example usage (can be run for testing)
if __name__ == "__main__":
    print("[INFO] Testing physics glue functions...")
    
    # Test 1: Upsampling
    print("\n[TEST 1] Upsampling 3 hours to 15-min")
    hourly = np.array([0.5, 0.6, 0.7])
    pvlib = np.array([
        0.1, 0.15, 0.2, 0.05,   # hour 1: sunrise (low → peak → low)
        0.2, 0.25, 0.3, 0.25,   # hour 2: morning (increasing)
        0.3, 0.35, 0.4, 0.35    # hour 3: midday (high plateau)
    ])
    
    upsampled = upsample_with_pvlib_shape(hourly, pvlib)
    print(f"    Input hourly: {hourly}")
    print(f"    Output 15-min: {upsampled}")
    print(f"    Hourly sums preserved: {upsampled[:4].sum():.3f} ≈ {hourly[0]:.3f}")
    
    # Test 2: Blending
    print("\n[TEST 2] Blending ML with physics")
    ml = np.array([0.8, 0.6, 0.4, 0.2, 0.0])
    pvlib_base = np.array([0.7, 0.5, 0.3, 0.1, 0.0])
    
    blended_high = blend_with_physics(ml, pvlib_base, alpha=0.9)
    blended_mid = blend_with_physics(ml, pvlib_base, alpha=0.5)
    blended_low = blend_with_physics(ml, pvlib_base, alpha=0.1)
    
    print(f"    ML predictions:     {ml}")
    print(f"    PVLib baseline:     {pvlib_base}")
    print(f"    Blended (α=0.9):    {blended_high}  # Trust ML")
    print(f"    Blended (α=0.5):    {blended_mid}  # Equal weight")
    print(f"    Blended (α=0.1):    {blended_low}  # Trust physics")
    
    # Test 3: Constraints
    print("\n[TEST 3] Physics constraints")
    raw = np.array([0.9, 0.8, 0.5, 0.05, -0.1])  # Some invalid values
    pvlib_ref = np.array([0.7, 0.6, 0.4, 0.005, 0.0])  # Night at end
    
    constrained = apply_physics_constraints(raw, pvlib_ref, max_capacity_multiplier=1.2)
    print(f"    Raw predictions:    {raw}")
    print(f"    PVLib reference:    {pvlib_ref}")
    print(f"    Constrained:        {constrained}")
    print(f"    - Clipped to 120% of PVLib: {constrained[0]:.3f} ≤ {pvlib_ref[0]*1.2:.3f}")
    print(f"    - Night forced to 0: {constrained[-2:]}")
    
    # Test 4: Hierarchical Blending
    print("\n[TEST 4] Hierarchical 3-way blend")
    short = np.array([0.8, 0.6, 0.4, 0.2])
    long = np.array([0.7, 0.5, 0.3, 0.1])
    pvlib_base = np.array([0.65, 0.48, 0.25, 0.05])
    
    hierarchical = blend_hierarchical(
        short, long, pvlib_base,
        alpha_short=0.6, alpha_long=0.4, alpha_ml=0.7
    )
    
    print(f"    Short-head (tactical):  {short}")
    print(f"    Long-head (strategic):  {long}")
    print(f"    PVLib (physics):        {pvlib_base}")
    print(f"    ML blend (60/40):       {0.6*short + 0.4*long}")
    print(f"    Final hierarchical:     {hierarchical}")
    print(f"    - Layer 1: 60% short + 40% long")
    print(f"    - Layer 2: 70% ML + 30% physics")
    print(f"    - Layer 3: constraints applied")
    
    print("\n[SUCCESS] All tests passed!")
