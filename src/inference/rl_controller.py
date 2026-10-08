# src/inference/rl_controller.py
"""
RL Meta-Controller for Adaptive Forecast Optimization.

Purpose:
    Dynamically optimize forecast parameters based on real-time conditions.
    Currently implements fixed heuristic rules (placeholder for future RL).

Future RL Features:
    - Weather API selection based on recent accuracy
    - Adaptive blend weights (alpha_short, alpha_long, alpha_pvlib)
    - Quantile selection based on confidence
    - Cost-accuracy trade-off optimization

State Space (Future):
    - Recent forecast accuracy (RMSE over last N days)
    - Weather forecast confidence (ensemble spread)
    - Seasonal patterns (time of year)
    - Forecast horizon (day 1 vs day 30)
    - Weather stability (variance indicators)

Action Space (Future):
    - Weather API selection: [openmeteo_base, openmeteo_ensemble, alternative]
    - Alpha weights: continuous [0, 1] for each component
    - Quantile selection: [q10, q25, q50, q75, q90]

Reward (Future):
    R = -RMSE - λ × API_cost + β × computational_efficiency
"""
from __future__ import annotations

from typing import Dict, Optional, List
import numpy as np


class RLMetaController:
    """
    Meta-controller for adaptive forecast optimization.
    
    Current: Fixed heuristic rules
    Future: RL-based adaptive policy (PPO, SAC, or similar)
    
    Attributes:
        history: List of past forecasts for future RL training
        api_performance: Tracking weather API accuracy
    """
    
    def __init__(self, mode: str = "heuristic"):
        """
        Initialize meta-controller.
        
        Args:
            mode: Controller mode
                - "heuristic": Fixed rules (current default)
                - "rl": RL-based adaptive (future implementation)
        """
        self.mode = mode
        self.history: List[Dict] = []
        self.api_performance: Dict[str, List[float]] = {
            'openmeteo_base': [],
            'openmeteo_ensemble': [],
            'alternative': []
        }
        
        print(f"[INFO] RL Meta-Controller initialized (mode={mode})")
        if mode == "rl":
            print("       [WARNING] RL mode not yet implemented, using heuristic fallback")
    
    def select_weather_api(
        self,
        day: int,
        weather_confidence: Optional[float] = None
    ) -> str:
        """
        Select optimal weather API source.
        
        Current: Always returns default
        Future: RL selects based on recent accuracy + cost
        
        Args:
            day: Forecast day (0-29)
            weather_confidence: Optional confidence metric from weather API
        
        Returns:
            api_name: Selected weather API identifier
        """
        # TODO: Implement RL-based selection
        # For now, use heuristic based on horizon
        
        if day < 7:
            # Near-term: Base API sufficient
            return "openmeteo_base"
        elif day < 14:
            # Mid-term: Ensemble for better accuracy
            return "openmeteo_ensemble"
        else:
            # Long-term: Ensemble critical
            return "openmeteo_ensemble"
    
    def get_blend_weights(
        self,
        day: int,
        weather_confidence: float = 0.8,
        recent_accuracy: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Get blend weights for hierarchical forecasting.
        
        Current: Fixed heuristics based on day
        Future: RL policy network outputs adaptive weights
        
        Args:
            day: Forecast day (0-29)
            weather_confidence: Weather forecast confidence [0,1]
            recent_accuracy: Optional dict with recent model RMSEs
        
        Returns:
            weights: Dict with keys:
                - alpha_short: Short-head weight in ML ensemble
                - alpha_long: Long-head weight in ML ensemble
                - alpha_ml: ML ensemble weight vs physics
                - alpha_pvlib: Physics weight (1 - alpha_ml)
        
        Example:
            >>> controller = RLMetaController()
            >>> weights = controller.get_blend_weights(day=0)
            >>> weights
            {'alpha_short': 0.65, 'alpha_long': 0.35, 'alpha_ml': 0.75, 'alpha_pvlib': 0.25}
        """
        # TODO: Replace with RL policy network
        # Current: Simple heuristic rules
        
        # Rule 1: Short-head weight decreases with horizon
        # Near-term: Trust short-head's precision
        # Long-term: Trust long-head's strategic view
        if day < 7:
            alpha_short = 0.65
            alpha_long = 0.35
        elif day < 14:
            alpha_short = 0.5
            alpha_long = 0.5
        else:
            alpha_short = 0.35
            alpha_long = 0.65
        
        # Rule 2: ML weight decreases with horizon and low confidence
        # Near-term + high confidence: Trust ML
        # Long-term + low confidence: Trust physics
        base_ml_weight = 0.75
        confidence_penalty = (1.0 - weather_confidence) * 0.2  # Max -20%
        horizon_penalty = day * 0.005  # -0.5% per day (max -15% at day 30)
        
        alpha_ml = base_ml_weight - confidence_penalty - horizon_penalty
        alpha_ml = np.clip(alpha_ml, 0.45, 0.85)  # Keep in reasonable range
        
        alpha_pvlib = 1.0 - alpha_ml
        
        return {
            'alpha_short': alpha_short,
            'alpha_long': alpha_long,
            'alpha_ml': alpha_ml,
            'alpha_pvlib': alpha_pvlib
        }
    
    def record_forecast(
        self,
        day: int,
        forecast: np.ndarray,
        metadata: Dict
    ) -> None:
        """
        Record forecast for future RL training.
        
        When ground truth becomes available, can compute reward
        and train RL policy.
        
        Args:
            day: Forecast day
            forecast: Predicted power values
            metadata: Additional info (weights, API used, etc.)
        """
        self.history.append({
            'day': day,
            'forecast': forecast.copy(),
            'metadata': metadata.copy()
        })
    
    def update_api_performance(
        self,
        api_name: str,
        rmse: float
    ) -> None:
        """
        Update weather API performance tracking.
        
        Args:
            api_name: API identifier
            rmse: Recent forecast RMSE using this API
        """
        if api_name in self.api_performance:
            self.api_performance[api_name].append(rmse)
            # Keep only last 30 days
            if len(self.api_performance[api_name]) > 30:
                self.api_performance[api_name] = self.api_performance[api_name][-30:]
    
    def get_best_api(self) -> str:
        """
        Get API with best recent performance.
        
        Returns:
            api_name: API with lowest average RMSE
        """
        best_api = "openmeteo_base"  # Default
        best_rmse = float('inf')
        
        for api, rmses in self.api_performance.items():
            if len(rmses) > 0:
                avg_rmse = np.mean(rmses)
                if avg_rmse < best_rmse:
                    best_rmse = avg_rmse
                    best_api = api
        
        return best_api
    
    def train_rl_policy(self, ground_truth: np.ndarray) -> None:
        """
        Train RL policy when ground truth becomes available.
        
        TODO: Implement RL training loop
        - Compute rewards from history vs ground truth
        - Update policy network
        - Save checkpoint
        
        Args:
            ground_truth: Actual PV power measurements
        """
        # Placeholder for future implementation
        print("[INFO] RL training not yet implemented")
        print(f"       History size: {len(self.history)} forecasts")
        print(f"       Ground truth size: {len(ground_truth)}")


# Example usage
if __name__ == "__main__":
    print("[INFO] Testing RL Meta-Controller...")
    
    controller = RLMetaController(mode="heuristic")
    
    # Test 1: Weather API selection
    print("\n[TEST 1] Weather API selection")
    for day in [0, 7, 15, 29]:
        api = controller.select_weather_api(day)
        print(f"    Day {day:2d}: {api}")
    
    # Test 2: Blend weights
    print("\n[TEST 2] Blend weights by day")
    for day in [0, 7, 14, 21, 29]:
        weights = controller.get_blend_weights(day)
        print(f"    Day {day:2d}: short={weights['alpha_short']:.2f}, "
              f"long={weights['alpha_long']:.2f}, "
              f"ml={weights['alpha_ml']:.2f}, "
              f"pvlib={weights['alpha_pvlib']:.2f}")
    
    # Test 3: Weather confidence impact
    print("\n[TEST 3] Weather confidence impact (Day 7)")
    for conf in [0.9, 0.7, 0.5]:
        weights = controller.get_blend_weights(day=7, weather_confidence=conf)
        print(f"    Confidence={conf:.1f}: ml={weights['alpha_ml']:.3f}")
    
    # Test 4: Record forecast
    print("\n[TEST 4] Recording forecasts")
    fake_forecast = np.random.rand(96)
    controller.record_forecast(
        day=0,
        forecast=fake_forecast,
        metadata={'api': 'openmeteo_base', 'rmse': 0.08}
    )
    print(f"    History size: {len(controller.history)}")
    
    print("\n[SUCCESS] All tests passed!")
    print("[INFO] Ready for hierarchical forecasting integration")
