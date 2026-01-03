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

from src.rl.rl_meta_controller import RLMetaControllerSystem, RLConfig
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
        
        self.rl_system = RLMetaControllerSystem(
            config=rl_config,
            checkpoint_dir=checkpoint_dir
        )
        
        # Current blend weights (managed by RL)
        self.blend_weights = {'short': 0.33, 'long': 0.33, 'physics': 0.34}
        
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
        
        # Logging for dashboard
        self.log_dir = checkpoint_dir / "logs" if checkpoint_dir else Path("/home/dwijenayake/pv_forecast_30d/checkpoints/rl/logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_log_file = self.log_dir / "metrics.jsonl"
        self.rl_state_file = self.log_dir / "rl_state.json"
        
        logger.info(f"[RLIntegratedForecaster] Initialized in '{rl_mode}' mode")
        logger.info(f"[RLIntegratedForecaster] Logging to {self.log_dir}")
    
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
        metrics['timestamp'] = pd.Timestamp.now(tz='UTC')
        
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
        now = pd.Timestamp.now(tz='UTC')
        metrics['hour_of_day'] = now.hour
        metrics['is_night'] = 1.0 if (now.hour < 6 or now.hour > 20) else 0.0
        metrics['season'] = (now.month - 1) // 3  # 0-3
        
        # Weather API info
        if weather_data is not None:
            metrics['weather_api_used'] = 1  # Assume ECMWF (from current system)
            metrics['api_agreement'] = 0.95  # Simplified
            
            # Handle both dict and DataFrame inputs
            if isinstance(weather_data, pd.DataFrame):
                metrics['cloud_cover'] = float(weather_data['cloud_cover'].iloc[0]) if 'cloud_cover' in weather_data else 0.0
                # Use available irradiance columns: poa_irradiance, global_tilted_irradiance_instant, direct_normal_irradiance_instant
                metrics['ghi'] = float(weather_data['global_tilted_irradiance_instant'].iloc[0]) if 'global_tilted_irradiance_instant' in weather_data else 0.0
                metrics['dni'] = float(weather_data['direct_normal_irradiance_instant'].iloc[0]) if 'direct_normal_irradiance_instant' in weather_data else 0.0
                metrics['temperature'] = float(weather_data['temperature_2m'].iloc[0]) if 'temperature_2m' in weather_data else 20.0
            else:
                # Dict-like access (backward compatibility)
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
    
    def execute_action(self, action_index: int, model_name: str = None):
        """
        Execute RL action with safety bounds.
        
        Args:
            action_index: Action from meta-controller (0-7)
            model_name: Which model to act on (for actions 1-3)
        
        Returns:
            success: Whether action executed successfully
        """
        try:
            if action_index == 0:  # MAINTAIN
                logger.info("[Action] MAINTAIN - no changes")
                return True
            
            elif action_index == 1:  # FINE_TUNE_SHORT_TFT
                logger.info("[Action] FINE_TUNE_SHORT_TFT - adjusting hyperparameters")
                if self.forecaster is None:
                    logger.warning("  No forecaster loaded, skipping")
                    return False
                
                # Get current learning rate (stored in model's hparams)
                if hasattr(self.forecaster, 'short_model') and hasattr(self.forecaster.short_model, 'hparams'):
                    current_lr = self.forecaster.short_model.hparams.get('learning_rate', 1e-3)
                    
                    # Decide adjustment based on recent performance
                    if len(self.metrics_history) > 0:
                        recent_rmse = self.metrics_history[-1].get('short_rmse_1h', 0.05)
                        if recent_rmse > 0.08:
                            # High error: increase LR (learn faster)
                            new_lr = min(current_lr * 1.2, 1e-2)  # Cap at 0.01
                            logger.info(f"  High RMSE ({recent_rmse:.4f}), increasing LR: {current_lr:.2e} → {new_lr:.2e}")
                        else:
                            # Low error: decrease LR (fine-tune)
                            new_lr = max(current_lr * 0.8, 1e-5)  # Floor at 1e-5
                            logger.info(f"  Low RMSE ({recent_rmse:.4f}), decreasing LR: {current_lr:.2e} → {new_lr:.2e}")
                        
                        # Update hparams (will be picked up by optimizer on next fit)
                        self.forecaster.short_model.hparams['learning_rate'] = new_lr
                        return True
                    else:
                        logger.warning("  No metrics history to guide adjustment")
                        return False
                else:
                    logger.warning("  Short model hparams not accessible")
                    return False
            
            elif action_index == 2:  # FINE_TUNE_LONG_TFT
                logger.info("[Action] FINE_TUNE_LONG_TFT - adjusting hyperparameters")
                if self.forecaster is None:
                    logger.warning("  No forecaster loaded, skipping")
                    return False
                
                # Get current learning rate (stored in model's hparams)
                if hasattr(self.forecaster, 'long_model') and hasattr(self.forecaster.long_model, 'hparams'):
                    current_lr = self.forecaster.long_model.hparams.get('learning_rate', 1e-3)
                    
                    # Decide adjustment based on recent performance
                    if len(self.metrics_history) > 0:
                        recent_rmse = self.metrics_history[-1].get('long_rmse_30d', 0.05)
                        if recent_rmse > 0.10:
                            # High error: increase LR
                            new_lr = min(current_lr * 1.2, 1e-2)
                            logger.info(f"  High RMSE ({recent_rmse:.4f}), increasing LR: {current_lr:.2e} → {new_lr:.2e}")
                        else:
                            # Low error: decrease LR
                            new_lr = max(current_lr * 0.8, 1e-5)
                            logger.info(f"  Low RMSE ({recent_rmse:.4f}), decreasing LR: {current_lr:.2e} → {new_lr:.2e}")
                        
                        # Update hparams
                        self.forecaster.long_model.hparams['learning_rate'] = new_lr
                        return True
                    else:
                        logger.warning("  No metrics history to guide adjustment")
                        return False
                else:
                    logger.warning("  Long model hparams not accessible")
                    return False
            
            elif action_index == 3:  # RECALIBRATE_PVLIB
                logger.info("[Action] RECALIBRATE_PVLIB - adjusting panel metadata")
                if self.forecaster is None or not hasattr(self.forecaster, 'pvlib_predictor'):
                    logger.warning("  No PVLib predictor loaded, skipping")
                    return False
                
                # Access PVLib predictor
                pvlib = self.forecaster.pvlib_predictor
                
                # Check if we have tilt/azimuth attributes
                if hasattr(pvlib, 'tilt_deg') and hasattr(pvlib, 'azimuth_deg'):
                    # Decide adjustments based on recent physics residual
                    if len(self.metrics_history) > 0:
                        residual = self.metrics_history[-1].get('physics_residual', 0.05)
                        
                        if residual > 0.15:
                            # High residual: adjust parameters
                            old_tilt = pvlib.tilt_deg
                            old_azimuth = pvlib.azimuth_deg
                            
                            # Small random adjustments within bounds
                            import random
                            tilt_delta = random.uniform(-3, 3)  # ±3° (within ±5° bound)
                            azimuth_delta = random.uniform(-5, 5)  # ±5° (within ±10° bound)
                            
                            new_tilt = np.clip(old_tilt + tilt_delta, 10, 60)
                            new_azimuth = np.clip(old_azimuth + azimuth_delta, 90, 270)
                            
                            # Update both stored attributes and PVSystem
                            pvlib.tilt_deg = new_tilt
                            pvlib.azimuth_deg = new_azimuth
                            pvlib.system.surface_tilt = new_tilt
                            pvlib.system.surface_azimuth = new_azimuth
                            
                            logger.info(f"  High residual ({residual:.4f}), adjusting:")
                            logger.info(f"    Tilt: {old_tilt:.1f}° → {new_tilt:.1f}°")
                            logger.info(f"    Azimuth: {old_azimuth:.1f}° → {new_azimuth:.1f}°")
                            
                            return True
                        else:
                            logger.info(f"  Residual acceptable ({residual:.4f}), no adjustment needed")
                            return True
                    else:
                        logger.warning("  No metrics history to guide adjustment")
                        return False
                else:
                    logger.warning("  PVLib tilt/azimuth attributes not accessible")
                    return False
            
            elif action_index in [4, 5, 6]:  # BLEND adjustments
                preset = self.rl_system.meta_controller.BLEND_PRESETS.get(action_index, {})
                self.blend_weights = preset
                logger.info(f"[Action] BLEND adjusted: {preset}")
                return True
            
            elif action_index == 7:  # SUGGEST_RETRAIN
                logger.warning("[Action] SUGGEST_RETRAIN - requires human confirmation")
                # Add to retrain queue
                if model_name:
                    self.rl_system.retrain_queue[model_name].append({
                        'timestamp': pd.Timestamp.now(tz='UTC'),
                        'reason': 'RL meta-controller suggestion',
                        'metrics': self.metrics_history[-1] if self.metrics_history else {}
                    })
                return True
            
            else:
                logger.error(f"[Action] Unknown action index: {action_index}")
                return False
                
        except Exception as e:
            logger.error(f"[Action] Execution failed: {e}")
            return False
    
    def forecast_with_rl(
        self,
        weather_data: pd.DataFrame,
        forecast_start: pd.Timestamp,
        historical_data: Optional[pd.DataFrame] = None,
        ground_truth: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Generate forecast with RL-driven adaptive blending.
        
        Args:
            weather_data: Weather features for prediction (30 days)
            forecast_start: Starting timestamp
            historical_data: Recent history for TFT encoder
            ground_truth: Actual measurements (for online learning)
        
        Returns:
            forecast: Final blended prediction (2880 steps @ 15-min)
            info: Dict with diagnostics
        """
        # Get individual model predictions from PhysicsAwareForecaster
        try:
            # Use existing forecaster's predict_30d method
            full_forecast = self.forecaster.predict_30d(
                forecast_start=forecast_start,
                weather_df=weather_data,
                historical_df=historical_data
            )
            
            # Extract component forecasts (if available from forecaster internals)
            # For now: use full forecast as baseline
            forecast_short = full_forecast[:96]   # Day 1
            forecast_long = full_forecast[96:]    # Days 2-30
            forecast_physics = full_forecast[:96] # Approximation
            
        except Exception as e:
            logger.error(f"[Forecast] Forecaster failed: {e}, using fallback")
            # Fallback: zeros
            forecast_short = np.zeros(96)
            forecast_long = np.zeros(2784)  # 30d - 1d = 29d @ 96/day
            forecast_physics = np.zeros(96)
            full_forecast = np.zeros(2880)
        
        # Collect comprehensive metrics
        metrics = self.collect_metrics(
            forecast_short=forecast_short,
            forecast_long=forecast_long,
            forecast_physics=forecast_physics,
            ground_truth=ground_truth,
            weather_data=weather_data
        )
        
        # RL meta-controller: decide action
        action_info = self.rl_system.step(metrics)
        action_index = action_info.get('action_index', 0)
        meta_state = self.rl_system.current_state
        
        # Execute action (with safety bounds)
        action_success = self.execute_action(action_index)
        
        # Get current blend weights (updated by execute_action if blend action)
        blend_weights = self.blend_weights
        
        # Apply dynamic blending to create final forecast
        # NOTE: For full 30-day, we blend Day 1 differently than Days 2-30
        # Day 1: blend short + physics
        # Days 2-30: use long-head as-is (already physics-aware from training)
        
        short_24h = forecast_short[:96]
        physics_24h = forecast_physics[:96]
        
        # Day 1 blended (normalize weights for 2-component blend)
        w_short_norm = blend_weights['short'] / (blend_weights['short'] + blend_weights['physics'])
        w_physics_norm = blend_weights['physics'] / (blend_weights['short'] + blend_weights['physics'])
        
        day1_blended = (
            w_short_norm * short_24h +
            w_physics_norm * physics_24h
        )
        
        # Days 2-30: use long-head forecast (already good from training)
        days_2_30 = forecast_long
        
        # Concatenate
        forecast_final = np.concatenate([day1_blended, days_2_30])
        
        # Store for next iteration
        self.metrics_history.append(metrics)
        self.action_history.append({
            'action_index': action_index,
            'action_name': self.rl_system.meta_controller.get_action_name(action_index),
            'blend_weights': blend_weights,
            'success': action_success
        })
        self.forecast_history.append(forecast_final)
        
        # Compute reward and update RL (online learning)
        reward = 0.0
        if ground_truth is not None and len(self.metrics_history) > 1:
            metrics_prev = self.metrics_history[-1]  # Previous step
            reward = self.rl_system.compute_reward(metrics_prev, metrics)
            
            # Update RL system
            self.rl_system.update(metrics, done=False)
            
            logger.info(f"[RL Update] Reward={reward:.3f}, Action={action_index}, ε={self.rl_system.meta_controller.epsilon:.3f}")
        
        # Log to dashboard files
        self.log_metrics_to_file(metrics, action_index, reward)
        self.log_rl_state_to_file()
        
        # Build info dict
        info = {
            'blend_weights': blend_weights,
            'action_index': action_index,
            'action_name': self.rl_system.meta_controller.get_action_name(action_index),
            'action_success': action_success,
            'metrics': metrics,
            'meta_state': meta_state,
            'rl_diagnostics': self.rl_system.get_status(),
            'retrain_queue': {k: len(v) for k, v in self.rl_system.retrain_queue.items()}
        }
        
        return forecast_final, info
    
    def confirm_retrain(self, model: str, approve: bool = False):
        """
        Human confirmation for retrain suggestion.
        
        Args:
            model: 'short_tft', 'long_tft', or 'pvlib'
            approve: True to execute retrain, False to reject
        """
        queue = self.rl_system.retrain_queue.get(model, [])
        
        if not queue:
            logger.warning(f"[Retrain] No pending requests for {model}")
            return
        
        request = queue.pop(0)
        
        if approve:
            logger.info(f"[Retrain] APPROVED for {model}")
            logger.info(f"  Reason: {request['reason']}")
            logger.info(f"  Timestamp: {request['timestamp']}")
            # TODO: Execute actual retraining here
            # self.forecaster.retrain_model(model)
        else:
            logger.info(f"[Retrain] REJECTED for {model}")
    
    def save_checkpoint(self, path: Optional[Path] = None):
        """Save RL checkpoint."""
        self.rl_system.save_checkpoint(path)
    
    def load_checkpoint(self, path: Path):
        """Load RL checkpoint."""
        return self.rl_system.load_checkpoint(path)
    
    def get_status(self) -> Dict:
        """Get comprehensive system status."""
        return {
            'rl_mode': self.rl_system.config.mode,
            'metrics_count': len(self.metrics_history),
            'forecast_count': len(self.forecast_history),
            'pending_retrains': {
                k: len(v) for k, v in self.rl_system.retrain_queue.items()
            },
            'rl_status': self.rl_system.get_status(),
            'latest_metrics': self.metrics_history[-1] if self.metrics_history else {},
            'latest_actions': self.action_history[-1] if self.action_history else {},
            'current_blend_weights': self.blend_weights
        }
    
    def collect_episode_data(self, num_steps: int = 100) -> pd.DataFrame:
        """
        Collect episode data for offline training.
        
        Args:
            num_steps: Number of forecasting steps to collect
        
        Returns:
            episode_data: DataFrame with (state, action, reward, next_state)
        """
        logger.info(f"[Data Collection] Starting {num_steps}-step episode in heuristic mode")
        
        # Force heuristic mode
        original_mode = self.rl_system.config.mode
        self.rl_system.config.mode = "heuristic"
        
        episode_records = []
        
        for step in range(num_steps):
            # Mock weather data (in production: query real API)
            weather_mock = pd.DataFrame({
                'ghi': np.random.rand(2880) * 800,
                'dni': np.random.rand(2880) * 900,
                'temperature_2m': np.random.rand(2880) * 20 + 15
            })
            
            forecast_start = pd.Timestamp.now(tz='UTC') + pd.Timedelta(days=step)
            
            # Generate forecast
            _, info = self.forecast_with_rl(
                weather_data=weather_mock,
                forecast_start=forecast_start
            )
            
            # Record transition
            if len(self.metrics_history) > 1:
                episode_records.append({
                    'step': step,
                    'state': info['meta_state'],
                    'action': info['action_index'],
                    'metrics': info['metrics'],
                    'timestamp': pd.Timestamp.now(tz='UTC')
                })
            
            if (step + 1) % 10 == 0:
                logger.info(f"[Data Collection] Step {step+1}/{num_steps} complete")
        
        # Restore mode
        self.rl_system.config.mode = original_mode
        
        logger.info(f"[Data Collection] Collected {len(episode_records)} transitions")
        return pd.DataFrame(episode_records)
    
    def log_metrics_to_file(self, metrics: Dict, action: int, reward: float):
        """Save metrics to JSONL for dashboard."""
        # Convert any non-JSON-serializable types to strings
        clean_metrics = {}
        for k, v in metrics.items():
            if hasattr(v, 'isoformat'):  # Timestamp
                clean_metrics[k] = v.isoformat()
            elif isinstance(v, (np.integer, np.floating)):
                clean_metrics[k] = float(v)
            elif isinstance(v, np.ndarray):
                clean_metrics[k] = v.tolist()
            else:
                clean_metrics[k] = v
        
        log_entry = {
            'timestamp': pd.Timestamp.now(tz='UTC').isoformat(),
            'action': int(action),
            'reward': float(reward),
            **clean_metrics,
            'blend_short': float(self.blend_weights['short']),
            'blend_long': float(self.blend_weights['long']),
            'blend_physics': float(self.blend_weights['physics'])
        }
        
        with open(self.metrics_log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def log_rl_state_to_file(self):
        """Save current RL state for dashboard."""
        meta = self.rl_system.meta_controller
        
        # Safe Q-value extraction
        try:
            if hasattr(meta, 'last_state') and meta.last_state is not None:
                import torch
                state_tensor = torch.FloatTensor(meta.last_state).unsqueeze(0).to(meta.device)
                q_values = meta.q_network(state_tensor).detach().cpu().numpy()[0]
                q_max = float(np.max(q_values))
            else:
                q_max = 0.0
        except Exception:
            q_max = 0.0
        
        rl_state = {
            'timestamp': pd.Timestamp.now(tz='UTC').isoformat(),
            'epsilon': float(meta.epsilon),
            'epsilon_delta': float(meta.epsilon - getattr(meta, 'epsilon_min', 0.1)),
            'last_action': int(self.action_history[-1]['action_index']) if len(self.action_history) > 0 else 0,
            'q_max': q_max,
            'buffer_size': len(meta.replay_buffer),
            'buffer_capacity': meta.replay_buffer.capacity,
            'total_steps': int(meta.steps)
        }
        
        with open(self.rl_state_file, 'w') as f:
            json.dump(rl_state, f, indent=2)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    """
    Example: Initialize and run RL-integrated forecaster.
    """
    from pathlib import Path
    
    # Paths (update to your actual paths)
    SHORT_CKPT = Path("experiments/tft/shorthead/best.ckpt")
    LONG_CKPT = Path("experiments/tft/longhead/best.ckpt")
    PLANT_META = Path("data/metadata/germany/plant_03.json")
    SHORT_TRAIN = Path("data/processed/short_train.parquet")
    LONG_TRAIN = Path("data/processed/long_train.parquet")
    
    # Initialize base forecaster
    forecaster = PhysicsAwareForecaster(
        short_ckpt=SHORT_CKPT,
        long_ckpt=LONG_CKPT,
        plant_metadata=PLANT_META,
        short_train_parquet=SHORT_TRAIN,
        long_train_parquet=LONG_TRAIN
    )
    
    # Wrap with RL integration
    rl_forecaster = RLIntegratedForecaster(
        forecaster=forecaster,
        rl_mode="heuristic",  # Start with heuristic baseline
        checkpoint_dir=Path("/home/dwijenayake/pv_forecast_30d/checkpoints/rl")
    )
    
    # Example forecast
    weather_df = pd.DataFrame({
        'timestamp': pd.date_range('2026-01-02', periods=2880, freq='15min'),
        'ghi': np.random.rand(2880) * 800,
        'dni': np.random.rand(2880) * 900,
        'temperature_2m': np.random.rand(2880) * 20 + 15
    })
    
    forecast, info = rl_forecaster.forecast_with_rl(
        weather_data=weather_df,
        forecast_start=pd.Timestamp('2026-01-02 00:00:00')
    )
    
    print(f"Forecast shape: {forecast.shape}")
    print(f"Action taken: {info['action_name']}")
    print(f"Blend weights: {info['blend_weights']}")
    print(f"RL status: {rl_forecaster.get_status()}")
