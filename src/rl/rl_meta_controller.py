#!/usr/bin/env python3
"""
RL Meta-Controller for MiRACLE PV Forecasting System

Hierarchical DQN-based adaptive control for ensemble forecasting with:
- 3 Local Agents: Short-TFT, Long-TFT, PVLib (Weather API is rule-based)
- 1 Meta-Agent: Dynamic ensemble blending weights
- Human-in-the-loop for retrain confirmations

Architecture adapted from MiRACLE paper with modifications for dual-TFT design.

Author: PV Forecast Team
Date: 2026-01-02
Version: 1.0
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
from typing import Dict, Tuple, Optional, List
import json
from pathlib import Path
from dataclasses import dataclass, field
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class RLConfig:
    """RL Meta-Controller hyperparameters (from paper)."""
    
    # DQN hyperparameters
    learning_rate: float = 1e-4
    gamma: float = 0.95  # Discount factor
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay: int = 10000
    batch_size: int = 64
    target_update_freq: int = 1  # Soft target update frequency (every step for smooth weather)
    tau: float = 0.005  # Soft update coefficient (0.5% per step)
    
    # Regularization (anti-overfitting)
    dropout_rate: float = 0.4  # Dropout probability (aggressive to prevent memorization)
    weight_decay: float = 1e-3  # L2 regularization (strong penalty)
    
    # Replay buffer
    buffer_capacity: int = 10000
    prioritized_replay: bool = True
    alpha: float = 0.6  # Prioritization exponent
    beta_start: float = 0.4
    beta_frames: int = 100000
    
    # Reward function weights (from paper)
    w_accuracy: float = 1.0     # Primary: RMSE
    w_consistency: float = 0.3  # Short-long alignment
    w_stability: float = 0.2    # Drift penalty
    w_efficiency: float = 0.1   # Compute cost
    
    # Convergence criteria (from paper)
    convergence_threshold: float = 1e-3
    convergence_episodes: int = 50
    
    # Operating mode
    mode: str = "heuristic"  # "heuristic" or "rl" or "hybrid"
    human_confirm_retrain: bool = True  # Human-in-the-loop for retraining


# ============================================================================
# Experience Replay
# ============================================================================

Transition = namedtuple('Transition', 
                        ('state', 'action', 'reward', 'next_state', 'done'))


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer.
    
    Samples transitions with probability proportional to TD-error,
    giving higher priority to "surprising" transitions.
    """
    
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        
    def push(self, *args):
        """Store transition with maximum priority."""
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(Transition(*args))
        else:
            self.buffer[self.position] = Transition(*args)
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int, beta: float = 0.4):
        """Sample batch with prioritized sampling."""
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]
        
        # Compute sampling probabilities
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        samples = [self.buffer[idx] for idx in indices]
        
        # Importance sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        return samples, indices, weights
    
    def update_priorities(self, indices, priorities):
        """Update priorities based on TD-errors."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
    
    def __len__(self):
        return len(self.buffer)


# ============================================================================
# DQN Network
# ============================================================================

class DQN(nn.Module):
    """
    Deep Q-Network for action-value approximation.
    
    Architecture: 3-layer MLP with ReLU activations + Dropout.
    Reduced capacity (64) and dropout (0.4) to prevent overfitting.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64, dropout: float = 0.4):
        super(DQN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass: state → Q-values for all actions."""
        return self.network(state)


# ============================================================================
# Local Advisor (Rule-Based Monitoring)
# ============================================================================

class LocalAdvisor:
    """
    Rule-based advisor for a single sub-model (Short-TFT, Long-TFT, or PVLib).
    
    NO LEARNING - Pure heuristic monitoring and state reporting.
    Advisors provide state signals to the meta-controller, which makes all decisions.
    
    Alerts (advisory only, not actions):
    - "ok": Normal operation
    - "high_rmse": Performance degradation detected
    - "high_drift": Input distribution shift detected
    - "calibration_drift": Physics model needs recalibration (PVLib only)
    - "horizon_degradation": Long-term forecast quality declining (Long-TFT only)
    """
    
    def __init__(self, name: str, state_dim: int):
        self.name = name
        self.state_dim = state_dim
        self.steps = 0
        
        # History tracking for trend analysis
        self.rmse_history = deque(maxlen=100)
        self.drift_history = deque(maxlen=100)
        
        logger.info(f"[{self.name}] Rule-based advisor initialized (no learning)")
    
    def get_advisory_state(self, metrics: Dict) -> np.ndarray:
        """
        Build state vector for meta-controller consumption.
        
        This is the ONLY function that matters - advisors don't act,
        they just report what they see to the meta-controller.
        
        Args:
            metrics: Performance metrics from forecaster
        
        Returns:
            state: State vector for this advisor
        """
        if self.name == "short_tft":
            return self._build_short_tft_state(metrics)
        elif self.name == "long_tft":
            return self._build_long_tft_state(metrics)
        elif self.name == "pvlib":
            return self._build_pvlib_state(metrics)
        else:
            return np.zeros(self.state_dim)
    
    def _build_short_tft_state(self, metrics: Dict) -> np.ndarray:
        """Build 10-dim state for short-term TFT advisor."""
        rmse_1h = metrics.get('short_rmse_1h', 0.0)
        rmse_24h = metrics.get('short_rmse_24h', 0.0)
        
        self.rmse_history.append(rmse_1h)
        
        # Compute trend (slope of last 20 samples)
        rmse_trend = 0.0
        if len(self.rmse_history) >= 20:
            recent = list(self.rmse_history)[-20:]
            x = np.arange(len(recent))
            rmse_trend = np.polyfit(x, recent, 1)[0]  # Slope
        
        return np.array([
            rmse_1h,
            rmse_24h,
            metrics.get('short_confidence', 0.5),
            metrics.get('short_drift', 0.0),
            metrics.get('forecast_age_hours', 0) / 24.0,
            metrics.get('retrain_count_24h', 0) / 10.0,
            1.0 if metrics.get('last_fine_tune_success', False) else 0.0,
            rmse_trend,
            metrics.get('night_performance_gap', 0.0),
            metrics.get('weather_quality', 1.0)
        ])
    
    def _build_long_tft_state(self, metrics: Dict) -> np.ndarray:
        """Build 10-dim state for long-term TFT advisor."""
        rmse_24h = metrics.get('long_rmse_24h', 0.0)
        rmse_7d = metrics.get('long_rmse_7d', 0.0)
        rmse_30d = metrics.get('long_rmse_30d', 0.0)
        
        # Horizon degradation: how much worse is 30d vs 24h?
        horizon_rmse_trend = (rmse_30d - rmse_24h) if rmse_24h > 0 else 0.0
        
        return np.array([
            rmse_24h,
            rmse_7d,
            rmse_30d,
            metrics.get('long_confidence', 0.5),
            metrics.get('long_drift', 0.0),
            metrics.get('forecast_horizon', 30) / 30.0,
            metrics.get('api_agreement', 1.0),
            metrics.get('retrain_count_24h', 0) / 10.0,
            horizon_rmse_trend,
            metrics.get('api_switch_count_24h', 0) / 10.0
        ])
    
    def _build_pvlib_state(self, metrics: Dict) -> np.ndarray:
        """Build 8-dim state for PVLib physics advisor."""
        return np.array([
            metrics.get('physics_residual', 0.0),
            metrics.get('ghi', 0.0) / 1500.0,
            metrics.get('dni', 0.0) / 1200.0,
            metrics.get('temperature', 20.0) / 60.0,
            metrics.get('last_calibration_hours', 0) / 168.0,  # Normalize by 1 week
            metrics.get('calibration_drift', 0.0),
            1.0 if metrics.get('is_night', False) else 0.0,
            metrics.get('cloud_cover', 0.0) / 100.0
        ])
    
    def check_alert(self, state: np.ndarray) -> str:
        """
        Rule-based alert check (advisory only, not an action).
        
        Returns alert level to help meta-controller prioritize.
        """
        if self.name == "short_tft":
            rmse_1h = state[0]
            drift = state[3]
            night_gap = state[8]
            
            if rmse_1h > 0.15 and drift > 0.5:
                return "high_rmse_and_drift"
            elif night_gap > 0.10:
                return "night_degradation"
            return "ok"
        
        elif self.name == "long_tft":
            horizon_degradation = state[8]
            api_agreement = state[6]
            
            if horizon_degradation > 0.05:
                return "horizon_degradation"
            elif api_agreement < 0.6:
                return "weather_api_disagreement"
            return "ok"
        
        elif self.name == "pvlib":
            physics_residual = state[0]
            calibration_drift = state[5]
            last_cal = state[4] * 168  # Convert back to hours
            
            if physics_residual > 0.30:
                return "severe_physics_mismatch"
            elif physics_residual > 0.20 and last_cal > 168:
                return "calibration_drift"
            return "ok"
        
        return "ok"
    
    def get_diagnostics(self) -> Dict:
        """Return advisor diagnostics (for logging only)."""
        return {
            'name': self.name,
            'steps': self.steps,
            'rmse_history_size': len(self.rmse_history)
        }


# ============================================================================
# Meta-Controller (Global DDQN Agent)
# ============================================================================

class MetaController:
    """
    Meta-controller for global system coordination using DDQN.
    
    THIS IS THE ONLY LEARNING AGENT - advisors are rule-based.
    
    Action Space (8 discrete actions):
    - A0: MAINTAIN (do nothing)
    - A1: FINE_TUNE_SHORT_TFT (adjust short-head hyperparams)
    - A2: FINE_TUNE_LONG_TFT (adjust long-head hyperparams)
    - A3: RECALIBRATE_PVLIB (update panel metadata)
    - A4: ADJUST_BLEND_WEIGHTS_HIGH_SHORT (favor short-term)
    - A5: ADJUST_BLEND_WEIGHTS_HIGH_LONG (favor long-term)
    - A6: ADJUST_BLEND_WEIGHTS_HIGH_PHYSICS (favor physics)
    - A7: SUGGEST_RETRAIN (request full retrain - human approval)
    """
    
    # Action constants
    ACTION_MAINTAIN = 0
    ACTION_FINE_TUNE_SHORT = 1
    ACTION_FINE_TUNE_LONG = 2
    ACTION_RECALIBRATE_PVLIB = 3
    ACTION_BLEND_HIGH_SHORT = 4
    ACTION_BLEND_HIGH_LONG = 5
    ACTION_BLEND_HIGH_PHYSICS = 6
    ACTION_SUGGEST_RETRAIN = 7
    
    # Action costs (for reward computation)
    ACTION_COSTS = {
        0: 0.0,    # maintain (free)
        1: 0.1,    # fine_tune_short
        2: 0.15,   # fine_tune_long
        3: 0.05,   # recalibrate_pvlib
        4: 0.0,    # blend adjustments (free)
        5: 0.0,
        6: 0.0,
        7: 1.0     # suggest_retrain (expensive)
    }
    
    # Blend weight presets
    BLEND_PRESETS = {
        4: {'short': 0.7, 'long': 0.2, 'physics': 0.1},  # High short
        5: {'short': 0.2, 'long': 0.7, 'physics': 0.1},  # High long
        6: {'short': 0.2, 'long': 0.2, 'physics': 0.6},  # High physics
    }
    
    def __init__(
        self,
        state_dim: int,
        config: Optional[RLConfig] = None
    ):
        self.config = config or RLConfig()
        
        # DDQN with 8 discrete actions
        self.action_dim = 8
        self.config = config or RLConfig()
        
        # DDQN with 8 discrete actions
        self.action_dim = 8
        
        # Q-networks (policy + target for DDQN) - reduced capacity (64) with dropout (0.4)
        self.policy_net = DQN(state_dim, self.action_dim, 
                             hidden_dim=64, dropout=self.config.dropout_rate)
        self.target_net = DQN(state_dim, self.action_dim, 
                             hidden_dim=64, dropout=self.config.dropout_rate)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer with weight decay (L2 regularization)
        self.optimizer = optim.Adam(self.policy_net.parameters(),
                                     lr=self.config.learning_rate,
                                     weight_decay=self.config.weight_decay)
        
        # Replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=20000,  # Larger for meta-controller
            alpha=self.config.alpha
        )
        
        # Exploration
        self.epsilon = self.config.epsilon_start
        self.steps = 0
        
        # Tracking
        self.q_values_history = deque(maxlen=1000)
        self.loss_history = deque(maxlen=1000)
        
        # Current blend weights (managed by actions A4-A6)
        self.current_weights = {'short': 0.33, 'long': 0.33, 'physics': 0.34}
        
        logger.info("[MetaController] Initialized with 8 system actions (DDQN)")
    
    def get_action_name(self, action: int) -> str:
        """Convert action index to human-readable name."""
        names = [
            "MAINTAIN",
            "FINE_TUNE_SHORT_TFT",
            "FINE_TUNE_LONG_TFT",
            "RECALIBRATE_PVLIB",
            "BLEND_HIGH_SHORT",
            "BLEND_HIGH_LONG",
            "BLEND_HIGH_PHYSICS",
            "SUGGEST_RETRAIN"
        ]
        return names[action] if 0 <= action < 8 else "UNKNOWN"
    
    def select_action(self, state: np.ndarray, mode: str = "rl") -> int:
        """
        Select action using ε-greedy policy (DDQN).
        
        Args:
            state: 35-dim state vector (aggregated from advisors)
            mode: "rl" (learned) or "heuristic" (rule-based)
        
        Returns:
            action: Integer action index (0-7)
        """
        if mode == "heuristic":
            return self._heuristic_action(state)
        
        # ε-greedy exploration
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_dim)
        
        # Exploitation: argmax Q(s,a)
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.policy_net(state_t)
            self.q_values_history.append(q_values.max().item())
            return q_values.argmax().item()
    
    def _heuristic_action(self, state: np.ndarray) -> int:
        """
        Heuristic baseline policy (rule-based).
        
        Simple rules based on state observations:
        - High RMSE → fine-tune or retrain
        - High drift → fine-tune
        - High physics residual → recalibrate PVLib
        - Night → adjust blend to physics
        - Otherwise → maintain
        """
        # State layout: [short_advisory(10), long_advisory(10), pvlib_advisory(8), context(7)]
        short_rmse_1h = state[0] if len(state) > 0 else 0.0
        short_drift = state[3] if len(state) > 3 else 0.0
        long_rmse_30d = state[12] if len(state) > 12 else 0.0
        horizon_degradation = state[18] if len(state) > 18 else 0.0
        physics_residual = state[20] if len(state) > 20 else 0.0
        is_night = state[26] if len(state) > 26 else 0.0
        data_drift_global = state[30] if len(state) > 30 else 0.0
        total_retrain_count = state[34] if len(state) > 34 else 0.0
        
        # Priority-based decision tree
        
        # 1. Check if night → favor physics
        if is_night > 0.5 and physics_residual < 0.15:
            return self.ACTION_BLEND_HIGH_PHYSICS
        
        # 2. Check physics calibration drift
        if physics_residual > 0.25:
            return self.ACTION_RECALIBRATE_PVLIB
        
        # 3. Check for severe performance collapse
        if short_rmse_1h > 0.15 and total_retrain_count < 2:
            return self.ACTION_SUGGEST_RETRAIN
        
        # 4. Check short-term degradation
        if short_rmse_1h > 0.10 and short_drift > 0.5:
            return self.ACTION_FINE_TUNE_SHORT
        
        # 5. Check long-term horizon degradation
        if horizon_degradation > 0.05 or long_rmse_30d > 0.12:
            return self.ACTION_FINE_TUNE_LONG
        
        # 6. Check global drift
        if data_drift_global > 0.6:
            # Favor short-term if drift detected (more adaptive)
            return self.ACTION_BLEND_HIGH_SHORT
        
        # 7. Default: maintain
        return self.ACTION_MAINTAIN
    
    def execute_action(self, action: int) -> Dict:
        """
        Execute meta-controller action (returns action metadata).
        
        Actual execution happens in RLIntegratedForecaster,
        this just provides action interpretation.
        """
        result = {
            'action': action,
            'action_name': self.get_action_name(action),
            'cost': self.ACTION_COSTS[action],
            'requires_human_approval': (action == self.ACTION_SUGGEST_RETRAIN)
        }
        
        # Update blend weights if action is A4-A6
        if action in self.BLEND_PRESETS:
            self.current_weights = self.BLEND_PRESETS[action].copy()
            result['blend_weights'] = self.current_weights
        else:
            result['blend_weights'] = self.current_weights
        
        return result
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update(self) -> Optional[float]:
        """
        Perform DDQN gradient update.
        
        Returns:
            loss: Training loss (None if insufficient data)
        """
        if len(self.replay_buffer) < self.config.batch_size:
            return None
        
        # Sample batch with prioritization
        beta = min(1.0, self.config.beta_start + self.steps * 
                   (1.0 - self.config.beta_start) / self.config.beta_frames)
        
        transitions, indices, weights = self.replay_buffer.sample(
            self.config.batch_size, beta
        )
        
        batch = Transition(*zip(*transitions))
        
        # Convert to tensors
        state_batch = torch.FloatTensor(np.array(batch.state))
        action_batch = torch.LongTensor(batch.action).unsqueeze(1)
        reward_batch = torch.FloatTensor(batch.reward)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state))
        done_batch = torch.FloatTensor(batch.done)
        weights_batch = torch.FloatTensor(weights)
        
        # Compute Q(s,a)
        q_values = self.policy_net(state_batch).gather(1, action_batch).squeeze(1)
        
        # Compute target: r + γ * max_a' Q_target(s', a')
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0]
            target_q_values = reward_batch + (1 - done_batch) * self.config.gamma * next_q_values
        
        # Compute TD-errors for priority update
        td_errors = torch.abs(q_values - target_q_values).detach().numpy()
        self.replay_buffer.update_priorities(indices, td_errors + 1e-6)
        
        # Weighted loss (importance sampling)
        loss = (weights_batch * (q_values - target_q_values) ** 2).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Soft target update (every step for smooth weather tracking)
        if self.steps % self.config.target_update_freq == 0:
            self._soft_update_target()
        
        # Decay epsilon
        self.epsilon = max(
            self.config.epsilon_end,
            self.config.epsilon_start - self.steps / self.config.epsilon_decay
        )
        
        self.steps += 1
        self.loss_history.append(loss.item())
        
        return loss.item()
    
    def _soft_update_target(self):
        """Soft update: θ_target = τ*θ_policy + (1-τ)*θ_target"""
        for target_param, policy_param in zip(self.target_net.parameters(), 
                                                self.policy_net.parameters()):
            target_param.data.copy_(
                self.config.tau * policy_param.data + 
                (1 - self.config.tau) * target_param.data
            )
    
    def get_diagnostics(self) -> Dict:
        """Return diagnostics for monitoring."""
        return {
            'epsilon': self.epsilon,
            'steps': self.steps,
            'buffer_size': len(self.replay_buffer),
            'avg_q_value': np.mean(self.q_values_history) if self.q_values_history else 0.0,
            'avg_loss': np.mean(self.loss_history) if self.loss_history else 0.0,
            'current_blend_weights': self.current_weights
        }
        for target_param, policy_param in zip(self.target_net.parameters(),
                                                self.policy_net.parameters()):
            target_param.data.copy_(
                self.config.tau * policy_param.data +
                (1 - self.config.tau) * target_param.data
            )


# ============================================================================
# Main RL System (1 DDQN Meta-Controller + 3 Rule-Based Advisors)
# ============================================================================

class RLMetaControllerSystem:
    """
    Hierarchical RL system for MiRACLE PV forecasting.
    
    Architecture (CORRECTED to match original paper):
    - 3 Rule-Based Advisors: Short-TFT, Long-TFT, PVLib (NO learning)
    - 1 DDQN Meta-Controller: Global system coordination (LEARNS)
    - Weather API Router: Rule-based (not RL)
    
    Operating modes:
    - "heuristic": Rule-based meta-controller (baseline)
    - "rl": Learned DDQN meta-controller
    
    State Space: 35 dimensions total
    - Short-TFT advisory: 10 dims
    - Long-TFT advisory: 10 dims
    - PVLib advisory: 8 dims
    - Meta-context: 7 dims
    
    Action Space: 8 discrete system actions
    - A0: MAINTAIN
    - A1: FINE_TUNE_SHORT_TFT
    - A2: FINE_TUNE_LONG_TFT
    - A3: RECALIBRATE_PVLIB
    - A4-A6: ADJUST_BLEND_WEIGHTS
    - A7: SUGGEST_RETRAIN
    """
    
    def __init__(
        self,
        config: Optional[RLConfig] = None,
        checkpoint_dir: Optional[Path] = None
    ):
        self.config = config or RLConfig()
        self.checkpoint_dir = checkpoint_dir or Path("checkpoints/rl")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # State dimensions
        self.short_tft_state_dim = 10
        self.long_tft_state_dim = 10
        self.pvlib_state_dim = 8
        self.meta_context_dim = 7
        self.total_state_dim = 35  # Sum of all advisors + context
        
        # Initialize rule-based advisors (NO learning)
        self.advisor_short_tft = LocalAdvisor(
            name="short_tft",
            state_dim=self.short_tft_state_dim
        )
        
        self.advisor_long_tft = LocalAdvisor(
            name="long_tft",
            state_dim=self.long_tft_state_dim
        )
        
        self.advisor_pvlib = LocalAdvisor(
            name="pvlib",
            state_dim=self.pvlib_state_dim
        )
        
        # Initialize DDQN meta-controller (ONLY learning agent)
        self.meta_controller = MetaController(
            state_dim=self.total_state_dim,
            config=self.config
        )
        
        # Human confirmation queue for retrain suggestions
        self.retrain_queue = {
            'short_tft': [],
            'long_tft': [],
            'pvlib': []
        }
        
        # Performance tracking
        self.episode_rewards = deque(maxlen=100)
        self.episode_count = 0
        self.prev_rmse = None
        
        logger.info("[RLMetaControllerSystem] Initialized (%s mode)", self.config.mode)
        logger.info("  - 3 rule-based advisors (no learning)")
        logger.info("  - 1 DDQN meta-controller (learns)")
        logger.info("  - Total state: 35 dims, Actions: 8")
    
    def build_meta_state(self, metrics: Dict) -> np.ndarray:
        """
        Build full 35-dim state for meta-controller by aggregating advisor states.
        
        Layout: [short_advisory(10), long_advisory(10), pvlib_advisory(8), context(7)]
        """
        # Get advisor states
        short_state = self.advisor_short_tft.get_advisory_state(metrics)
        long_state = self.advisor_long_tft.get_advisory_state(metrics)
        pvlib_state = self.advisor_pvlib.get_advisory_state(metrics)
        
        # Build meta-context (7 dims)
        meta_context = np.array([
            metrics.get('ensemble_rmse', 0.0),
            metrics.get('short_long_mismatch', 0.0),
            metrics.get('data_drift_score', 0.0),
            metrics.get('compute_budget', 1.0),
            metrics.get('hour_of_day', 12) / 24.0,
            metrics.get('season', 0) / 3.0,
            metrics.get('total_retrain_count_7d', 0) / 10.0
        ])
        
        # Concatenate all: 10 + 10 + 8 + 7 = 35 dims
        state = np.concatenate([short_state, long_state, pvlib_state, meta_context])
        
        return state
    
    def step(self, metrics: Dict) -> Dict:
        """
        Execute one control step.
        
        Flow:
        1. Advisors build state vectors from metrics
        2. Meta-controller selects system action
        3. Execute action and return instructions
        
        Args:
            metrics: Performance metrics from forecaster (39+ features)
        
        Returns:
            actions: Dict with meta action and execution info
        """
        # Build state
        state = self.build_meta_state(metrics)
        
        # Check advisor alerts
        short_alert = self.advisor_short_tft.check_alert(state[:10])
        long_alert = self.advisor_long_tft.check_alert(state[10:20])
        pvlib_alert = self.advisor_pvlib.check_alert(state[20:28])
        
        # Meta-controller selects action
        action = self.meta_controller.select_action(state, mode=self.config.mode)
        
        # Execute action
        action_info = self.meta_controller.execute_action(action)
        
        # Add action index to response
        action_info['action_index'] = action
        
        # Add advisor alerts to response
        action_info['advisor_alerts'] = {
            'short_tft': short_alert,
            'long_tft': long_alert,
            'pvlib': pvlib_alert
        }
        
        # Store state for learning
        self.current_state = state
        self.current_action = action
        
        return action_info
    
    def compute_reward(self, metrics: Dict, metrics_next: Dict) -> float:
        """
        Compute reward aligned with original MiRACLE formulation.
        
        R_t = w₁(−RMSE_t) + w₂(−Drift_t) + w₃(−Cost_t) + w₄(−RetrainFreq_t)
        
        Args:
            metrics: Current metrics
            metrics_next: Next-step metrics
        
        Returns:
            reward: Scalar reward signal
        """
        # Component weights
        w1 = 1.0   # Accuracy
        w2 = 0.5   # Drift control
        w3 = 0.2   # Cost
        w4 = 0.3   # Retrain frequency
        
        # 1. Accuracy: reward RMSE improvement
        rmse_prev = metrics.get('ensemble_rmse', 0.0)
        rmse_next = metrics_next.get('ensemble_rmse', 0.0)
        r_accuracy = w1 * (rmse_prev - rmse_next) / 0.01  # Normalize by 10W
        
        # 2. Drift: penalize distribution shift
        drift_score = metrics_next.get('data_drift_score', 0.0)
        short_long_mismatch = metrics_next.get('short_long_mismatch', 0.0)
        r_drift = -w2 * (drift_score + short_long_mismatch) / 2.0
        
        # 3. Cost: penalize expensive actions
        action_cost = self.meta_controller.ACTION_COSTS.get(
            getattr(self, 'current_action', 0), 0.0
        )
        r_cost = -w3 * action_cost
        
        # 4. Retrain frequency: penalize excessive retraining
        retrain_count = metrics_next.get('total_retrain_count_7d', 0)
        r_retrain = -w4 * retrain_count / 10.0
        
        # Bonus: reward high API agreement
        api_agreement = metrics_next.get('api_agreement', 1.0)
        bonus = 0.1 if api_agreement > 0.9 else 0.0
        
        reward = r_accuracy + r_drift + r_cost + r_retrain + bonus
        
        return reward
    
    def update(self, metrics_next: Dict, done: bool = False):
        """
        Update meta-controller after step completion.
        
        Args:
            metrics_next: Metrics after action execution
            done: Episode termination flag
        """
        # Build next state
        next_state = self.build_meta_state(metrics_next)
        
        # Compute reward (requires previous metrics, stored in self)
        if not hasattr(self, 'current_state'):
            logger.warning("No previous state for reward computation, skipping update")
            return
        
        # For reward, we need metrics from both steps
        # In practice, RLIntegratedForecaster will call update with both metrics
        reward = 0.0  # Placeholder - actual reward computed in RLIntegratedForecaster
        
        # Store transition in meta-controller
        self.meta_controller.store_transition(
            self.current_state,
            self.current_action,
            reward,
            next_state,
            done
        )
        
        # Update meta-controller (DDQN learning)
        if self.config.mode == "rl":
            loss = self.meta_controller.update()
            if loss is not None:
                logger.debug(f"Meta-controller update: loss={loss:.4f}")
    
    def save_checkpoint(self, path: Optional[Path] = None):
        """Save meta-controller checkpoint."""
        save_path = path or self.checkpoint_dir / "meta_controller.pt"
        
        checkpoint = {
            'meta_controller_policy': self.meta_controller.policy_net.state_dict(),
            'meta_controller_target': self.meta_controller.target_net.state_dict(),
            'meta_controller_optimizer': self.meta_controller.optimizer.state_dict(),
            'meta_controller_epsilon': self.meta_controller.epsilon,
            'meta_controller_steps': self.meta_controller.steps,
            'episode_count': self.episode_count,
            'config': self.config
        }
        
        torch.save(checkpoint, save_path)
        logger.info(f"Checkpoint saved: {save_path}")
    
    def load_checkpoint(self, path: Path):
        """Load meta-controller checkpoint."""
        if not path.exists():
            logger.warning(f"Checkpoint not found: {path}")
            return False
        
        checkpoint = torch.load(path)
        
        self.meta_controller.policy_net.load_state_dict(checkpoint['meta_controller_policy'])
        self.meta_controller.target_net.load_state_dict(checkpoint['meta_controller_target'])
        self.meta_controller.optimizer.load_state_dict(checkpoint['meta_controller_optimizer'])
        self.meta_controller.epsilon = checkpoint['meta_controller_epsilon']
        self.meta_controller.steps = checkpoint['meta_controller_steps']
        self.episode_count = checkpoint.get('episode_count', 0)
        
        logger.info(f"Checkpoint loaded: {path} (steps={self.meta_controller.steps})")
        return True
    
    def get_status(self) -> Dict:
        """Return system status for monitoring."""
        meta_diag = self.meta_controller.get_diagnostics()
        
        return {
            'mode': self.config.mode,
            'episode_count': self.episode_count,
            'meta_controller': meta_diag,
            'advisors': {
                'short_tft': self.advisor_short_tft.get_diagnostics(),
                'long_tft': self.advisor_long_tft.get_diagnostics(),
                'pvlib': self.advisor_pvlib.get_diagnostics()
            },
            'retrain_queue_size': sum(len(v) for v in self.retrain_queue.values())
        }
