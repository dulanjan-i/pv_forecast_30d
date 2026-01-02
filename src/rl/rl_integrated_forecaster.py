#!/usr/bin/env python3
"""
Practical Integration of RL Meta-Controller with PhysicsAwareForecaster

This module bridges the RL meta-controller with the existing forecasting pipeline,
handling real-time metric collection, action execution, and human confirmation flow.

Author: PV Forecast Team
Date: 2026-01-02
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging
from collections import deque
import json

from src.rl.rl_meta_controller import RLMetaController, RLConfig
from src.inference.physics_aware_forecaster import PhysicsAwareForecaster

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RLIntegratedForecaster:
    """
    PhysicsAwareForecaster with RL Meta-Controller integration.
    
    Features:
    - Real-time metrics collection
    - RL-driven dynamic blending
    - Human-in-the-loop retrain confirmations
    - Performance tracking and logging
    """
    
    def __init__(
        self,
        forecaster: PhysicsAwareForecaster,
        rl_mode: str = "heuristic",
        rl_config: Optional[RLConfig] = None,
        checkpoint_dir: Optional[Path] = None
    ):
        """
        Initialize RL-integrated forecaster.
        
        Args:
            forecaster: Existing PhysicsAwareForecaster instance
            rl_mode: "heuristic" (rule-based), "rl" (learned), or "hybrid"
            rl_config: RL hyperparameters
            checkpoint_dir: Directory for RL checkpoints
        """
        self.forecaster = forecaster
        
        # Initialize RL meta-controller
        if rl_config is None:
            rl_config = RLConfig(mode=rl_mode)
        else:
            rl_config.mode = rl_mode
        
        self.rl_controller = RLMetaController(
            config=rl_config,
            checkpoint_dir=checkpoint_dir
        )
        
        # Metrics tracking
        self.metrics_history = deque(maxlen=1000)
        self.forecast_history = deque(maxlen=100)
        self.action_history = deque(maxlen=100)
        
        # Performance baselines (for normalization)
        self.baseline_rmse = 0.05  # 50W baseline
        self.baseline_drift = 0.1
        
        # Ground truth buffer (for computing actual RMSE)
        self.ground_truth_buffer = deque(maxlen=2880)  # 30 days @ 15-min
        self.prediction_buffer = deque(maxlen=2880)
        
        logger.info(f"[RLIntegratedForecaster] Initialized in '{rl_mode}' mode")
    
    def collect_metrics(
        self,
        forecast_short: Optional[np.ndarray] = None,
        forecast_long: Optional[np.ndarray] = None,
        forecast_physics: Optional[np.ndarray] = None,
        ground_truth: Optional[np.ndarray] = None,
        weather_data: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Collect comprehensive system metrics for RL state.
        
        Args:
            forecast_short: Short-head TFT predictions
            forecast_long: Long-head TFT predictions  
            forecast_physics: PVLib predictions
            ground_truth: Actual measurements (if available)
            weather_data: Current weather features
        
        Returns:
            metrics: Dict of all system metrics
        """
        metrics = {}
        
        # Timestamp
        metrics['timestamp'] = pd.Timestamp.now()
        
        # Performance metrics (if ground truth available)
        if ground_truth is not None:
            if forecast_short is not None:
                rmse_short = np.sqrt(np.mean((forecast_short[:len(ground_truth)] - ground_truth) ** 2))
                metrics['short_rmse_1h'] = rmse_short
                metrics['short_rmse_24h'] = rmse_short  # Simplified
            
            if forecast_long is not None:
                rmse_long = np.sqrt(np.mean((forecast_long[:len(ground_truth)] - ground_truth) ** 2))
                metrics['long_rmse_24h'] = rmse_long
                metrics['long_rmse_7d'] = rmse_long
                metrics['long_rmse_30d'] = rmse_long
            
            if forecast_physics is not None:
                rmse_physics = np.sqrt(np.mean((forecast_physics[:len(ground_truth)] - ground_truth) ** 2))
                metrics['physics_residual'] = rmse_physics
        else:
            # No ground truth: use previous metrics or defaults
            metrics['short_rmse_1h'] = self.metrics_history[-1].get('short_rmse_1h', 0.05) if self.metrics_history else 0.05
            metrics['short_rmse_24h'] = self.metrics_history[-1].get('short_rmse_24h', 0.05) if self.metrics_history else 0.05
            metrics['long_rmse_24h'] = self.metrics_history[-1].get('long_rmse_24h', 0.05) if self.metrics_history else 0.05
            metrics['long_rmse_7d'] = self.metrics_history[-1].get('long_rmse_7d', 0.05) if self.metrics_history else 0.05
            metrics['long_rmse_30d'] = self.metrics_history[-1].get('long_rmse_30d', 0.05) if self.metrics_history else 0.05
            metrics['physics_residual'] = self.metrics_history[-1].get('physics_residual', 0.05) if self.metrics_history else 0.05
        
        # Consistency: Short-long mismatch in first 24h
        if forecast_short is not None and forecast_long is not None:
            # Align resolutions (short is 15-min, long is 1-hour)
            short_24h = forecast_short[:96]  # 24h @ 15-min
            long_24h = forecast_long[:24]    # 24h @ 1-hour
            
            # Resample short to hourly for comparison
            short_hourly = short_24h.reshape(24, 4).mean(axis=1)
            mismatch = np.abs(short_hourly - long_24h).mean()
            metrics['short_long_mismatch'] = mismatch
        else:
            metrics['short_long_mismatch'] = 0.0
        
        # Drift (simplified: use input distribution shift)
        if weather_data is not None and len(self.metrics_history) > 0:
            prev_weather = self.metrics_history[-1].get('weather_mean', 0.0)
            curr_weather = weather_data['ghi'].mean() if 'ghi' in weather_data else 0.0
            drift = abs(curr_weather - prev_weather) / (prev_weather + 1e-6)
            metrics['data_drift_score'] = min(drift, 1.0)
            metrics['weather_mean'] = curr_weather
        else:
            metrics['data_drift_score'] = 0.0
            metrics['weather_mean'] = 0.0
        
        # Confidence (simplified: use prediction variance as proxy)
        if forecast_short is not None:
            metrics['short_confidence'] = 1.0 - min(forecast_short.std() / 0.1, 1.0)
        else:
            metrics['short_confidence'] = 0.5
        
        if forecast_long is not None:
            metrics['long_confidence'] = 1.0 - min(forecast_long.std() / 0.1, 1.0)
        else:
            metrics['long_confidence'] = 0.5
        
        # Context
        now = pd.Timestamp.now()
        metrics['hour_of_day'] = now.hour
        metrics['is_night'] = 1.0 if (now.hour < 6 or now.hour > 20) else 0.0
        metrics['season'] = (now.month - 1) // 3  # 0-3
        
        # Weather API info
        if weather_data is not None:
            metrics['weather_api_used'] = 1  # Assume ECMWF (from current system)
            metrics['api_agreement'] = 0.95  # Simplified
            metrics['cloud_cover'] = weather_data.get('cloud_cover', [0.0])[0] if 'cloud_cover' in weather_data else 0.0
            metrics['ghi'] = weather_data.get('ghi', [0.0])[0] if 'ghi' in weather_data else 0.0
            metrics['dni'] = weather_data.get('dni', [0.0])[0] if 'dni' in weather_data else 0.0
            metrics['temperature'] = weather_data.get('temperature_2m', [20.0])[0] if 'temperature_2m' in weather_data else 20.0
            metrics['weather_quality'] = 1.0
        else:
            metrics['weather_api_used'] = 0
            metrics['api_agreement'] = 1.0
            metrics['cloud_cover'] = 0.0
            metrics['ghi'] = 0.0
            metrics['dni'] = 0.0
            metrics['temperature'] = 20.0
            metrics['weather_quality'] = 1.0
        
        # Cost metrics
        metrics['compute_budget'] = 1.0  # Simplified
        metrics['retrain_count_24h'] = 0  # Track from action history
        metrics['retrain_count_short_24h'] = 0
        metrics['retrain_count_long_24h'] = 0
        
        # Current blend weights (from forecaster)
        metrics['current_weight_short'] = 0.33
        metrics['current_weight_long'] = 0.33
        metrics['current_weight_physics'] = 0.33
        
        # Previous actions
        if self.action_history:
            last_action = self.action_history[-1]
            metrics['last_action'] = last_action.get('short_tft', 0)
            metrics['last_meta_action'] = last_action.get('meta_action', 13)
        else:
            metrics['last_action'] = 0
            metrics['last_meta_action'] = 13  # Default (balanced)
        
        # Additional
        metrics['forecast_age_hours'] = 0  # Time since last retrain
        metrics['forecast_horizon'] = 24.0  # Typical horizon
        
        # Ensemble RMSE (weighted combination)
        if ground_truth is not None and all(f is not None for f in [forecast_short, forecast_long, forecast_physics]):
            w_s, w_l, w_p = 0.33, 0.33, 0.34
            ensemble = (
                w_s * forecast_short[:len(ground_truth)] +
                w_l * forecast_long[:len(ground_truth)] +
                w_p * forecast_physics[:len(ground_truth)]
            )
            metrics['ensemble_rmse'] = np.sqrt(np.mean((ensemble - ground_truth) ** 2))
        else:
            metrics['ensemble_rmse'] = 0.05
        
        # PVLib specific
        metrics['tilt_angle'] = 25.0
        metrics['azimuth'] = 180.0
        metrics['last_calibration_hours'] = 168.0  # 1 week default
        
        # Drift for sub-models
        metrics['short_drift'] = metrics['data_drift_score']
        metrics['long_drift'] = metrics['data_drift_score']
        
        return metrics
    
    def forecast_with_rl(
        self,
        weather_data: pd.DataFrame,
        ground_truth: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Generate forecast with RL-driven adaptive blending.
        
        Args:
            weather_data: Weather features for prediction
            ground_truth: Actual measurements (for online learning)
        
        Returns:
            forecast: Final blended prediction
            info: Dict with diagnostics
        """
        # Get individual model predictions (mock for now - integrate with forecaster)
        # In production: call self.forecaster's internal methods
        forecast_short = np.random.rand(96) * 0.5  # Mock 24h @ 15-min
        forecast_long = np.random.rand(720) * 0.5   # Mock 30d @ 1-hour
        forecast_physics = np.random.rand(96) * 0.5  # Mock physics
        
        # Collect metrics
        metrics = self.collect_metrics(
            forecast_short=forecast_short,
            forecast_long=forecast_long,
            forecast_physics=forecast_physics,
            ground_truth=ground_truth,
            weather_data=weather_data
        )
        
        # RL meta-controller step
        actions = self.rl_controller.step(metrics)
        
        # Apply dynamic blend weights
        blend_weights = actions['blend_weights']
        
        # Blend forecasts (align resolutions first)
        short_24h = forecast_short[:96]
        long_24h_resampled = np.repeat(forecast_long[:24], 4)  # Upsample 1h → 15min
        physics_24h = forecast_physics[:96]
        
        forecast_blended = (
            blend_weights['short'] * short_24h +
            blend_weights['long'] * long_24h_resampled +
            blend_weights['physics'] * physics_24h
        )
        
        # Store for next iteration
        self.metrics_history.append(metrics)
        self.action_history.append(actions)
        self.forecast_history.append(forecast_blended)
        
        # If ground truth available: update RL (online learning)
        if ground_truth is not None and len(self.metrics_history) > 1:
            metrics_prev = self.metrics_history[-2]
            actions_prev = self.action_history[-2]
            
            self.rl_controller.update(
                metrics_prev=metrics_prev,
                actions=actions_prev,
                metrics_next=metrics,
                done=False
            )
        
        # Build info dict
        info = {
            'blend_weights': blend_weights,
            'actions': actions,
            'metrics': metrics,
            'rl_diagnostics': self.rl_controller.get_diagnostics(),
            'retrain_queue': self.rl_controller.retrain_queue
        }
        
        return forecast_blended, info
    
    def confirm_retrain(self, model: str, approve: bool = False):
        """
        Human confirmation for retrain suggestion.
        
        Args:
            model: 'short_tft', 'long_tft', or 'pvlib'
            approve: True to execute retrain, False to reject
        """
        queue = self.rl_controller.retrain_queue.get(model, [])
        
        if not queue:
            logger.warning(f"[Retrain] No pending requests for {model}")
            return
        
        request = queue.pop(0)
        
        if approve:
            logger.info(f"[Retrain] APPROVED for {model}")
            logger.info(f"  Reason: {request['reason']}")
            logger.info(f"  Timestamp: {request['timestamp']}")
            # TODO: Execute actual retraining here
        else:
            logger.info(f"[Retrain] REJECTED for {model}")
    
    def save_checkpoint(self, path: Optional[Path] = None):
        """Save RL checkpoint."""
        self.rl_controller.save_checkpoint(path)
    
    def load_checkpoint(self, path: Path):
        """Load RL checkpoint."""
        self.rl_controller.load_checkpoint(path)
    
    def get_status(self) -> Dict:
        """Get comprehensive system status."""
        return {
            'rl_mode': self.rl_controller.config.mode,
            'metrics_count': len(self.metrics_history),
            'forecast_count': len(self.forecast_history),
            'pending_retrains': {
                k: len(v) for k, v in self.rl_controller.retrain_queue.items()
            },
            'rl_diagnostics': self.rl_controller.get_diagnostics(),
            'latest_metrics': self.metrics_history[-1] if self.metrics_history else {},
            'latest_actions': self.action_history[-1] if self.action_history else {}
        }


# ============================================================================
# Helper: Action Interpretations
# ============================================================================

ACTION_NAMES = {
    0: 'maintain',
    1: 'fine_tune_hyperparams',
    2: 'suggest_retrain',
    3: 'rollback_checkpoint',
    4: 'defer_to_others'
}


def interpret_action(action: int) -> str:
    """Convert action index to human-readable string."""
    return ACTION_NAMES.get(action, f"unknown_{action}")
