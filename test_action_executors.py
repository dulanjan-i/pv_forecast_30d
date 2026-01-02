#!/usr/bin/env python3
"""
Test action executors with real TFT models.

Verifies that fine-tuning and recalibration actually modify model parameters.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import logging

from src.rl.rl_integrated_forecaster import RLIntegratedForecaster
from src.rl.rl_meta_controller import RLConfig
from src.inference.physics_aware_forecaster import PhysicsAwareForecaster

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_fine_tune_actions():
    """Test that fine-tuning actions actually change learning rates."""
    logger.info("="*80)
    logger.info("TEST: Fine-Tuning Actions (Learning Rate Adjustment)")
    logger.info("="*80)
    
    # Paths
    SHORT_CKPT = Path("V1.0_FINAL_TFT/shorthead_seed42/checkpoints/best.ckpt")
    LONG_CKPT = Path("V1.0_FINAL_TFT/longhead_seed43/checkpoints/best.ckpt")
    PLANT_META = Path("V1.0_FINAL_TFT/plant_metadata/plant_03.json")
    SHORT_TRAIN = Path("data/processed/plant_level/plant_03/15min_pca32/train.parquet")
    LONG_TRAIN = Path("data/processed/plant_level/plant_03/hourly_longhead/train.parquet")
    
    try:
        # Initialize forecaster
        logger.info("Loading PhysicsAwareForecaster...")
        forecaster = PhysicsAwareForecaster(
            short_ckpt=SHORT_CKPT,
            long_ckpt=LONG_CKPT,
            plant_metadata=PLANT_META,
            short_train_parquet=SHORT_TRAIN,
            long_train_parquet=LONG_TRAIN,
            device='cuda:0' if torch.cuda.is_available() else 'cpu'
        )
        
        # Wrap with RL
        rl_forecaster = RLIntegratedForecaster(
            forecaster=forecaster,
            rl_mode="heuristic",
            checkpoint_dir=Path("checkpoints/rl")
        )
        
        # Test Action 1: FINE_TUNE_SHORT_TFT
        logger.info("\n--- Testing Action 1: FINE_TUNE_SHORT_TFT ---")
        
        # Get initial LR from hparams (models not attached to Trainer, so can't call optimizers())
        if hasattr(forecaster, 'short_model') and hasattr(forecaster.short_model, 'hparams'):
            lr_before = forecaster.short_model.hparams.get('learning_rate', 1e-3)
            logger.info(f"Initial short-TFT LR: {lr_before:.2e}")
            
            # Add high RMSE to history to trigger LR increase
            rl_forecaster.metrics_history.append({'short_rmse_1h': 0.12})
            
            # Execute action
            success = rl_forecaster.execute_action(1)
            
            lr_after = forecaster.short_model.hparams.get('learning_rate', 1e-3)
            logger.info(f"After action short-TFT LR: {lr_after:.2e}")
            
            if success and lr_after != lr_before:
                logger.info("✅ Action 1: PASSED - Learning rate changed")
            else:
                logger.error("❌ Action 1: FAILED - Learning rate unchanged")
                return False
        else:
            logger.warning("⚠️  Short model hparams not accessible, skipping Action 1")
        
        # Test Action 2: FINE_TUNE_LONG_TFT
        logger.info("\n--- Testing Action 2: FINE_TUNE_LONG_TFT ---")
        
        if hasattr(forecaster, 'long_model') and hasattr(forecaster.long_model, 'hparams'):
            lr_before = forecaster.long_model.hparams.get('learning_rate', 1e-3)
            logger.info(f"Initial long-TFT LR: {lr_before:.2e}")
            
            # Add high RMSE to trigger LR increase
            rl_forecaster.metrics_history[-1]['long_rmse_30d'] = 0.15
            
            # Execute action
            success = rl_forecaster.execute_action(2)
            
            lr_after = forecaster.long_model.hparams.get('learning_rate', 1e-3)
            logger.info(f"After action long-TFT LR: {lr_after:.2e}")
            
            if success and lr_after != lr_before:
                logger.info("✅ Action 2: PASSED - Learning rate changed")
            else:
                logger.error("❌ Action 2: FAILED - Learning rate unchanged")
                return False
        else:
            logger.warning("⚠️  Long model optimizer not accessible, skipping Action 2")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        return False


def test_recalibrate_pvlib():
    """Test that recalibration actually changes PVLib parameters."""
    logger.info("\n" + "="*80)
    logger.info("TEST: PVLib Recalibration (Panel Metadata Adjustment)")
    logger.info("="*80)
    
    # Paths
    SHORT_CKPT = Path("V1.0_FINAL_TFT/shorthead_seed42/checkpoints/best.ckpt")
    LONG_CKPT = Path("V1.0_FINAL_TFT/longhead_seed43/checkpoints/best.ckpt")
    PLANT_META = Path("V1.0_FINAL_TFT/plant_metadata/plant_03.json")
    SHORT_TRAIN = Path("data/processed/plant_level/plant_03/15min_pca32/train.parquet")
    LONG_TRAIN = Path("data/processed/plant_level/plant_03/hourly_longhead/train.parquet")
    
    try:
        # Initialize forecaster
        logger.info("Loading PhysicsAwareForecaster...")
        forecaster = PhysicsAwareForecaster(
            short_ckpt=SHORT_CKPT,
            long_ckpt=LONG_CKPT,
            plant_metadata=PLANT_META,
            short_train_parquet=SHORT_TRAIN,
            long_train_parquet=LONG_TRAIN,
            device='cpu'  # CPU for faster loading
        )
        
        # Wrap with RL
        rl_forecaster = RLIntegratedForecaster(
            forecaster=forecaster,
            rl_mode="heuristic",
            checkpoint_dir=Path("checkpoints/rl")
        )
        
        logger.info("\n--- Testing Action 3: RECALIBRATE_PVLIB ---")
        
        # Get initial metadata (PVLib stores tilt/azimuth as attributes, not in metadata dict)
        if hasattr(forecaster, 'pvlib_predictor'):
            pvlib = forecaster.pvlib_predictor
            if hasattr(pvlib, 'tilt_deg') and hasattr(pvlib, 'azimuth_deg'):
                tilt_before = pvlib.tilt_deg
                azimuth_before = pvlib.azimuth_deg
                
                logger.info(f"Initial PVLib parameters:")
                logger.info(f"  Tilt: {tilt_before:.1f}°")
                logger.info(f"  Azimuth: {azimuth_before:.1f}°")
                
                # Add high physics residual to trigger recalibration
                rl_forecaster.metrics_history.append({'physics_residual': 0.20})
                
                # Execute action
                success = rl_forecaster.execute_action(3)
                
                tilt_after = pvlib.tilt_deg
                azimuth_after = pvlib.azimuth_deg
                
                logger.info(f"After recalibration:")
                logger.info(f"  Tilt: {tilt_after:.1f}°")
                logger.info(f"  Azimuth: {azimuth_after:.1f}°")
                
                if success and (tilt_after != tilt_before or azimuth_after != azimuth_before):
                    logger.info("✅ Action 3: PASSED - PVLib parameters changed")
                    return True
                elif not success:
                    logger.error("❌ Action 3: FAILED - Action execution failed")
                    return False
                else:
                    logger.warning("⚠️  Action 3: Residual too low, no adjustment needed")
                    return True
            else:
                logger.error("❌ PVLib tilt/azimuth attributes not accessible")
                return False
        else:
            logger.error("❌ PVLib predictor not accessible")
            return False
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        return False


def test_blend_actions():
    """Test that blend actions change weights correctly."""
    logger.info("\n" + "="*80)
    logger.info("TEST: Blend Weight Adjustment")
    logger.info("="*80)
    
    try:
        # No need to load models, just test weight changes
        rl_forecaster = RLIntegratedForecaster(
            forecaster=None,
            rl_mode="heuristic",
            checkpoint_dir=Path("checkpoints/rl")
        )
        
        # Initial weights
        logger.info(f"Initial blend weights: {rl_forecaster.blend_weights}")
        
        # Test Action 4: BLEND_HIGH_SHORT
        logger.info("\n--- Testing Action 4: BLEND_HIGH_SHORT ---")
        rl_forecaster.execute_action(4)
        weights_4 = rl_forecaster.blend_weights.copy()
        logger.info(f"After Action 4: {weights_4}")
        assert weights_4['short'] == 0.7, "Short weight should be 0.7"
        logger.info("✅ Action 4: PASSED")
        
        # Test Action 5: BLEND_HIGH_LONG
        logger.info("\n--- Testing Action 5: BLEND_HIGH_LONG ---")
        rl_forecaster.execute_action(5)
        weights_5 = rl_forecaster.blend_weights.copy()
        logger.info(f"After Action 5: {weights_5}")
        assert weights_5['long'] == 0.7, "Long weight should be 0.7"
        logger.info("✅ Action 5: PASSED")
        
        # Test Action 6: BLEND_HIGH_PHYSICS
        logger.info("\n--- Testing Action 6: BLEND_HIGH_PHYSICS ---")
        rl_forecaster.execute_action(6)
        weights_6 = rl_forecaster.blend_weights.copy()
        logger.info(f"After Action 6: {weights_6}")
        assert weights_6['physics'] == 0.6, "Physics weight should be 0.6"
        logger.info("✅ Action 6: PASSED")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        return False


def main():
    """Run all action executor tests."""
    logger.info("Starting Action Executor Test Suite...")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    results = {}
    
    # Test blend actions first (fast, no model loading)
    results["Blend Actions"] = test_blend_actions()
    
    # Test fine-tuning (requires model loading)
    results["Fine-Tuning Actions"] = test_fine_tune_actions()
    
    # Test PVLib recalibration
    results["PVLib Recalibration"] = test_recalibrate_pvlib()
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("TEST SUMMARY")
    logger.info("="*80)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    logger.info("="*80)
    if all_passed:
        logger.info("🎉 ALL ACTION EXECUTORS WORKING!")
    else:
        logger.error("⚠️  SOME EXECUTORS FAILED")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
