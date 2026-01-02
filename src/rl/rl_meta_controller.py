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
    target_update_freq: int = 1000  # Soft target update frequency
    tau: float = 0.005  # Soft update coefficient
    
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
    
    Architecture: 3-layer MLP with ReLU activations.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(DQN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass: state → Q-values for all actions."""
        return self.network(state)


# ============================================================================
# Local Agent (per sub-model)
# ============================================================================

class LocalAgent:
    """
    Local RL agent managing a single sub-model (Short-TFT, Long-TFT, or PVLib).
    
    Actions:
    - 0: maintain (do nothing)
    - 1: fine_tune_hyperparams (automated)
    - 2: suggest_retrain (requires human confirmation)
    - 3: rollback_checkpoint
    - 4: defer_to_others
    """
    
    def __init__(
        self,
        name: str,
        state_dim: int,
        action_dim: int = 5,
        config: Optional[RLConfig] = None
    ):
        self.name = name
        self.config = config or RLConfig()
        
        # Q-networks (policy + target)
        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), 
                                     lr=self.config.learning_rate)
        
        # Replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=self.config.buffer_capacity,
            alpha=self.config.alpha
        )
        
        # Exploration
        self.epsilon = self.config.epsilon_start
        self.steps = 0
        
        # Performance tracking
        self.q_values_history = deque(maxlen=1000)
        self.loss_history = deque(maxlen=1000)
        
        logger.info(f"[{self.name}] Local agent initialized")
    
    def select_action(self, state: np.ndarray, mode: str = "rl") -> int:
        """
        Select action using ε-greedy policy.
        
        Args:
            state: Current state vector
            mode: "rl" (learned) or "heuristic" (rule-based)
        
        Returns:
            action: Integer action index
        """
        if mode == "heuristic":
            return self._heuristic_action(state)
        
        # ε-greedy exploration
        if np.random.random() < self.epsilon:
            return np.random.randint(0, 5)
        
        # Exploitation: argmax Q(s,a)
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.policy_net(state_t)
            self.q_values_history.append(q_values.max().item())
            return q_values.argmax().item()
    
    def _heuristic_action(self, state: np.ndarray) -> int:
        """
        Heuristic baseline policy (rule-based).
        
        Rules:
        - If RMSE > 2x baseline: suggest retrain
        - If drift > threshold: fine-tune hyperparams
        - If recent retrain: maintain
        - Else: maintain
        """
        rmse = state[0]  # Assume first element is RMSE
        drift = state[1] if len(state) > 1 else 0.0
        recent_retrain = state[-1] if len(state) > 2 else 0.0
        
        # Thresholds (tunable)
        rmse_high = 0.10  # 100W RMSE threshold
        drift_high = 0.5
        
        if recent_retrain < 6:  # Retrained in last 6 hours
            return 0  # maintain
        elif rmse > rmse_high:
            return 2  # suggest_retrain (human confirms)
        elif drift > drift_high:
            return 1  # fine_tune_hyperparams (automated)
        else:
            return 0  # maintain
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update(self) -> Optional[float]:
        """
        Perform one gradient update step.
        
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
        
        # Soft target update
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
        """Return agent diagnostics for monitoring."""
        return {
            'epsilon': self.epsilon,
            'steps': self.steps,
            'buffer_size': len(self.replay_buffer),
            'avg_q_value': np.mean(self.q_values_history) if self.q_values_history else 0.0,
            'avg_loss': np.mean(self.loss_history) if self.loss_history else 0.0
        }


# ============================================================================
# Meta-Agent (Ensemble Blending)
# ============================================================================

class MetaAgent:
    """
    Meta-agent for dynamic ensemble weight blending.
    
    Action Space:
    - Discretized blend weights: [w_short, w_long, w_physics]
    - 10 levels per weight → 10^3 = 1000 actions (large but tractable)
    
    Simplified: Use continuous action space with policy gradient (future)
    For now: Discrete with 27 actions (3 levels per weight: low/med/high)
    """
    
    def __init__(
        self,
        state_dim: int,
        config: Optional[RLConfig] = None
    ):
        self.config = config or RLConfig()
        
        # Discretized action space: 3 weights × 3 levels = 27 actions
        # Actions encode (short_level, long_level, physics_level)
        self.action_dim = 27
        self.weight_levels = [0.1, 0.5, 0.9]  # Low, Medium, High
        
        # Q-networks
        self.policy_net = DQN(state_dim, self.action_dim, hidden_dim=256)
        self.target_net = DQN(state_dim, self.action_dim, hidden_dim=256)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(),
                                     lr=self.config.learning_rate)
        
        # Replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=20000,  # Larger for meta-agent
            alpha=self.config.alpha
        )
        
        # Exploration
        self.epsilon = self.config.epsilon_start
        self.steps = 0
        
        logger.info("[MetaAgent] Initialized with 27 discrete actions")
    
    def action_to_weights(self, action: int) -> Dict[str, float]:
        """
        Convert discrete action to blend weights.
        
        Action encoding: action = short_idx * 9 + long_idx * 3 + physics_idx
        """
        short_idx = action // 9
        long_idx = (action % 9) // 3
        physics_idx = action % 3
        
        weights_raw = {
            'short': self.weight_levels[short_idx],
            'long': self.weight_levels[long_idx],
            'physics': self.weight_levels[physics_idx]
        }
        
        # Normalize to sum=1
        total = sum(weights_raw.values())
        return {k: v / total for k, v in weights_raw.items()}
    
    def weights_to_action(self, weights: Dict[str, float]) -> int:
        """Convert blend weights to nearest discrete action."""
        # Find closest level for each weight
        short_idx = np.argmin([abs(weights['short'] - w) for w in self.weight_levels])
        long_idx = np.argmin([abs(weights['long'] - w) for w in self.weight_levels])
        physics_idx = np.argmin([abs(weights['physics'] - w) for w in self.weight_levels])
        
        return short_idx * 9 + long_idx * 3 + physics_idx
    
    def select_action(self, state: np.ndarray, mode: str = "rl") -> int:
        """Select blending action using ε-greedy policy."""
        if mode == "heuristic":
            return self._heuristic_action(state)
        
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_dim)
        
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.policy_net(state_t)
            return q_values.argmax().item()
    
    def _heuristic_action(self, state: np.ndarray) -> int:
        """
        Heuristic blending strategy.
        
        Rules:
        - Short-term (0-24h): High weight on short-TFT
        - Long-term (24h+): High weight on long-TFT
        - Night: High weight on physics (PVLib)
        - High uncertainty: Balance all three
        """
        # Assume state contains: [..., hour_of_day, is_night, forecast_horizon, ...]
        # Simplified: use fixed heuristic based on time
        hour = int(state[10]) if len(state) > 10 else 12
        is_night = state[11] > 0.5 if len(state) > 11 else False
        
        if is_night:
            # Night: Trust physics most
            weights = {'short': 0.2, 'long': 0.2, 'physics': 0.6}
        elif hour < 12:
            # Morning: Short-term important
            weights = {'short': 0.6, 'long': 0.3, 'physics': 0.1}
        else:
            # Afternoon/evening: Balance
            weights = {'short': 0.4, 'long': 0.5, 'physics': 0.1}
        
        return self.weights_to_action(weights)
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update(self) -> Optional[float]:
        """Perform gradient update (same as LocalAgent)."""
        if len(self.replay_buffer) < self.config.batch_size:
            return None
        
        beta = min(1.0, self.config.beta_start + self.steps * 
                   (1.0 - self.config.beta_start) / self.config.beta_frames)
        
        transitions, indices, weights = self.replay_buffer.sample(
            self.config.batch_size, beta
        )
        
        batch = Transition(*zip(*transitions))
        
        state_batch = torch.FloatTensor(np.array(batch.state))
        action_batch = torch.LongTensor(batch.action).unsqueeze(1)
        reward_batch = torch.FloatTensor(batch.reward)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state))
        done_batch = torch.FloatTensor(batch.done)
        weights_batch = torch.FloatTensor(weights)
        
        q_values = self.policy_net(state_batch).gather(1, action_batch).squeeze(1)
        
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0]
            target_q_values = reward_batch + (1 - done_batch) * self.config.gamma * next_q_values
        
        td_errors = torch.abs(q_values - target_q_values).detach().numpy()
        self.replay_buffer.update_priorities(indices, td_errors + 1e-6)
        
        loss = (weights_batch * (q_values - target_q_values) ** 2).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        if self.steps % self.config.target_update_freq == 0:
            self._soft_update_target()
        
        self.epsilon = max(
            self.config.epsilon_end,
            self.config.epsilon_start - self.steps / self.config.epsilon_decay
        )
        
        self.steps += 1
        return loss.item()
    
    def _soft_update_target(self):
        """Soft target update."""
        for target_param, policy_param in zip(self.target_net.parameters(),
                                                self.policy_net.parameters()):
            target_param.data.copy_(
                self.config.tau * policy_param.data +
                (1 - self.config.tau) * target_param.data
            )


# ============================================================================
# Main Meta-Controller
# ============================================================================

class RLMetaController:
    """
    Hierarchical RL Meta-Controller for MiRACLE forecasting system.
    
    Architecture:
    - 3 Local Agents: Short-TFT, Long-TFT, PVLib
    - 1 Meta-Agent: Ensemble weight blending
    - Weather API Router: Rule-based (not RL)
    
    Operating modes:
    - "heuristic": Rule-based baseline
    - "rl": Fully learned DQN policies
    - "hybrid": Local agents use RL, meta uses heuristic
    """
    
    def __init__(
        self,
        config: Optional[RLConfig] = None,
        checkpoint_dir: Optional[Path] = None
    ):
        self.config = config or RLConfig()
        self.checkpoint_dir = checkpoint_dir or Path("checkpoints/rl")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # State dimensions (carefully designed)
        self.short_tft_state_dim = 15  # Short-TFT metrics
        self.long_tft_state_dim = 15   # Long-TFT metrics
        self.pvlib_state_dim = 10      # PVLib metrics
        self.meta_state_dim = 25       # Aggregated system state
        
        # Initialize local agents
        self.agent_short_tft = LocalAgent(
            name="ShortTFT",
            state_dim=self.short_tft_state_dim,
            config=self.config
        )
        
        self.agent_long_tft = LocalAgent(
            name="LongTFT",
            state_dim=self.long_tft_state_dim,
            config=self.config
        )
        
        self.agent_pvlib = LocalAgent(
            name="PVLib",
            state_dim=self.pvlib_state_dim,
            config=self.config
        )
        
        # Initialize meta-agent
        self.meta_agent = MetaAgent(
            state_dim=self.meta_state_dim,
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
        
        logger.info("[RLMetaController] Initialized in '%s' mode", self.config.mode)
    
    def get_state_short_tft(self, metrics: Dict) -> np.ndarray:
        """Build state vector for short-TFT agent."""
        return np.array([
            metrics.get('short_rmse_1h', 0.0),
            metrics.get('short_rmse_24h', 0.0),
            metrics.get('short_confidence', 0.0),
            metrics.get('short_drift', 0.0),
            metrics.get('hour_of_day', 12) / 24.0,  # Normalize
            metrics.get('is_night', 0.0),
            metrics.get('cloud_cover', 0.0) / 100.0,
            metrics.get('retrain_count_24h', 0) / 10.0,
            metrics.get('compute_budget', 1.0),
            metrics.get('forecast_age_hours', 0.0) / 24.0,
            metrics.get('weather_quality', 1.0),
            metrics.get('data_drift_score', 0.0),
            metrics.get('last_action', 0) / 5.0,
            metrics.get('ensemble_rmse', 0.0),
            metrics.get('short_long_mismatch', 0.0)
        ])
    
    def get_state_long_tft(self, metrics: Dict) -> np.ndarray:
        """Build state vector for long-TFT agent."""
        return np.array([
            metrics.get('long_rmse_24h', 0.0),
            metrics.get('long_rmse_7d', 0.0),
            metrics.get('long_rmse_30d', 0.0),
            metrics.get('long_confidence', 0.0),
            metrics.get('long_drift', 0.0),
            metrics.get('forecast_horizon', 30.0) / 30.0,
            metrics.get('weather_api_used', 1) / 2.0,  # 0=Forecast, 1=ECMWF, 2=GFS
            metrics.get('api_agreement', 1.0),
            metrics.get('retrain_count_24h', 0) / 10.0,
            metrics.get('compute_budget', 1.0),
            metrics.get('last_action', 0) / 5.0,
            metrics.get('ensemble_rmse', 0.0),
            metrics.get('short_long_mismatch', 0.0),
            metrics.get('hour_of_day', 12) / 24.0,
            metrics.get('season', 0) / 3.0  # 0-3
        ])
    
    def get_state_pvlib(self, metrics: Dict) -> np.ndarray:
        """Build state vector for PVLib agent."""
        return np.array([
            metrics.get('physics_residual', 0.0),
            metrics.get('ghi', 0.0) / 1500.0,
            metrics.get('dni', 0.0) / 1200.0,
            metrics.get('temperature', 20.0) / 60.0,
            metrics.get('is_night', 0.0),
            metrics.get('cloud_cover', 0.0) / 100.0,
            metrics.get('tilt_angle', 25.0) / 90.0,
            metrics.get('azimuth', 180.0) / 360.0,
            metrics.get('last_calibration_hours', 0.0) / 168.0,  # Week
            metrics.get('ensemble_rmse', 0.0)
        ])
    
    def get_state_meta(self, metrics: Dict) -> np.ndarray:
        """Build state vector for meta-agent (ensemble blending)."""
        return np.array([
            # Aggregate performance
            metrics.get('ensemble_rmse', 0.0),
            metrics.get('short_rmse_24h', 0.0),
            metrics.get('long_rmse_24h', 0.0),
            metrics.get('physics_residual', 0.0),
            
            # Consistency
            metrics.get('short_long_mismatch', 0.0),
            
            # Drift
            metrics.get('data_drift_score', 0.0),
            
            # Context
            metrics.get('hour_of_day', 12) / 24.0,
            metrics.get('is_night', 0.0),
            metrics.get('forecast_horizon', 24.0) / 30.0,
            metrics.get('season', 0) / 3.0,
            
            # Weather
            metrics.get('weather_api_used', 1) / 2.0,
            metrics.get('api_agreement', 1.0),
            metrics.get('cloud_cover', 0.0) / 100.0,
            metrics.get('weather_quality', 1.0),
            
            # Current weights
            metrics.get('current_weight_short', 0.33),
            metrics.get('current_weight_long', 0.33),
            metrics.get('current_weight_physics', 0.33),
            
            # Recent actions
            metrics.get('last_meta_action', 13) / 27.0,  # Normalized
            
            # Cost
            metrics.get('compute_budget', 1.0),
            metrics.get('retrain_count_short_24h', 0) / 10.0,
            metrics.get('retrain_count_long_24h', 0) / 10.0,
            
            # Confidence
            metrics.get('short_confidence', 0.0),
            metrics.get('long_confidence', 0.0),
            
            # Additional
            metrics.get('forecast_age_hours', 0.0) / 24.0,
            metrics.get('ghi', 0.0) / 1500.0
        ])
    
    def compute_reward(
        self,
        metrics_prev: Dict,
        actions: Dict,
        metrics_next: Dict
    ) -> float:
        """
        Multi-objective reward function (from paper, adapted).
        
        R = w₁(−RMSE) + w₂(−Mismatch) + w₃(−Drift) + w₄(−Cost) + bonus
        """
        # Primary: Ensemble RMSE (lower is better)
        ensemble_rmse_prev = metrics_prev.get('ensemble_rmse', 0.05)
        ensemble_rmse_next = metrics_next.get('ensemble_rmse', 0.05)
        rmse_improvement = ensemble_rmse_prev - ensemble_rmse_next
        r_accuracy = self.config.w_accuracy * rmse_improvement / 0.01  # Normalize by 10W
        
        # Consistency: Short-long alignment in first 24h
        mismatch = metrics_next.get('short_long_mismatch', 0.0)
        r_consistency = -self.config.w_consistency * (mismatch / 0.02)  # Normalize by 20W
        
        # Stability: Data drift penalty
        drift = metrics_next.get('data_drift_score', 0.0)
        r_stability = -self.config.w_stability * drift
        
        # Efficiency: Action cost
        action_cost = 0.0
        if actions.get('short_tft') == 1:  # Fine-tune
            action_cost += 0.1
        elif actions.get('short_tft') == 2:  # Suggest retrain
            action_cost += 0.5
        
        if actions.get('long_tft') == 1:
            action_cost += 0.15
        elif actions.get('long_tft') == 2:
            action_cost += 0.8  # Long-head more expensive
        
        r_efficiency = -self.config.w_efficiency * action_cost
        
        # Bonus: Weather API agreement
        api_bonus = 0.05 if metrics_next.get('api_agreement', 0.0) > 0.9 else 0.0
        
        total_reward = r_accuracy + r_consistency + r_stability + r_efficiency + api_bonus
        
        return total_reward
    
    def step(self, metrics: Dict) -> Dict:
        """
        Execute one control step.
        
        Args:
            metrics: Current system metrics
        
        Returns:
            actions: Dict of actions for each agent
        """
        mode = self.config.mode
        
        # Get states
        state_short = self.get_state_short_tft(metrics)
        state_long = self.get_state_long_tft(metrics)
        state_pvlib = self.get_state_pvlib(metrics)
        state_meta = self.get_state_meta(metrics)
        
        # Select actions
        action_short = self.agent_short_tft.select_action(state_short, mode=mode)
        action_long = self.agent_long_tft.select_action(state_long, mode=mode)
        action_pvlib = self.agent_pvlib.select_action(state_pvlib, mode=mode)
        action_meta = self.meta_agent.select_action(state_meta, mode=mode)
        
        # Convert meta action to weights
        blend_weights = self.meta_agent.action_to_weights(action_meta)
        
        # Handle retrain suggestions (human-in-the-loop)
        if action_short == 2:  # Suggest retrain
            self.retrain_queue['short_tft'].append({
                'timestamp': metrics.get('timestamp'),
                'reason': f"RMSE={metrics.get('short_rmse_24h'):.3f}",
                'state': state_short
            })
            logger.warning("[ShortTFT] Retrain suggested - awaiting human confirmation")
        
        if action_long == 2:
            self.retrain_queue['long_tft'].append({
                'timestamp': metrics.get('timestamp'),
                'reason': f"RMSE={metrics.get('long_rmse_7d'):.3f}",
                'state': state_long
            })
            logger.warning("[LongTFT] Retrain suggested - awaiting human confirmation")
        
        return {
            'short_tft': action_short,
            'long_tft': action_long,
            'pvlib': action_pvlib,
            'blend_weights': blend_weights,
            'meta_action': action_meta
        }
    
    def update(
        self,
        metrics_prev: Dict,
        actions: Dict,
        metrics_next: Dict,
        done: bool = False
    ):
        """
        Update all agents with experience.
        
        Args:
            metrics_prev: Previous system state
            actions: Actions taken
            metrics_next: Resulting system state
            done: Episode termination flag
        """
        # Compute reward
        reward = self.compute_reward(metrics_prev, actions, metrics_next)
        
        # Get states
        state_short_prev = self.get_state_short_tft(metrics_prev)
        state_short_next = self.get_state_short_tft(metrics_next)
        
        state_long_prev = self.get_state_long_tft(metrics_prev)
        state_long_next = self.get_state_long_tft(metrics_next)
        
        state_pvlib_prev = self.get_state_pvlib(metrics_prev)
        state_pvlib_next = self.get_state_pvlib(metrics_next)
        
        state_meta_prev = self.get_state_meta(metrics_prev)
        state_meta_next = self.get_state_meta(metrics_next)
        
        # Store transitions
        self.agent_short_tft.store_transition(
            state_short_prev, actions['short_tft'], reward, state_short_next, done
        )
        
        self.agent_long_tft.store_transition(
            state_long_prev, actions['long_tft'], reward, state_long_next, done
        )
        
        self.agent_pvlib.store_transition(
            state_pvlib_prev, actions['pvlib'], reward, state_pvlib_next, done
        )
        
        self.meta_agent.store_transition(
            state_meta_prev, actions['meta_action'], reward, state_meta_next, done
        )
        
        # Perform gradient updates (if in RL mode)
        if self.config.mode in ["rl", "hybrid"]:
            loss_short = self.agent_short_tft.update()
            loss_long = self.agent_long_tft.update()
            loss_pvlib = self.agent_pvlib.update()
            loss_meta = self.meta_agent.update()
            
            if loss_short:
                logger.debug(f"[Update] Short={loss_short:.4f}, Long={loss_long:.4f}, "
                            f"PVLib={loss_pvlib:.4f}, Meta={loss_meta:.4f}")
        
        self.episode_rewards.append(reward)
    
    def save_checkpoint(self, path: Optional[Path] = None):
        """Save all agent checkpoints."""
        path = path or self.checkpoint_dir / f"rl_meta_ep{self.episode_count}.pt"
        
        checkpoint = {
            'episode': self.episode_count,
            'config': self.config.__dict__,
            'agent_short_tft': self.agent_short_tft.policy_net.state_dict(),
            'agent_long_tft': self.agent_long_tft.policy_net.state_dict(),
            'agent_pvlib': self.agent_pvlib.policy_net.state_dict(),
            'meta_agent': self.meta_agent.policy_net.state_dict(),
            'episode_rewards': list(self.episode_rewards)
        }
        
        torch.save(checkpoint, path)
        logger.info(f"[Checkpoint] Saved to {path}")
    
    def load_checkpoint(self, path: Path):
        """Load agent checkpoints."""
        checkpoint = torch.load(path, map_location='cpu')
        
        self.agent_short_tft.policy_net.load_state_dict(checkpoint['agent_short_tft'])
        self.agent_long_tft.policy_net.load_state_dict(checkpoint['agent_long_tft'])
        self.agent_pvlib.policy_net.load_state_dict(checkpoint['agent_pvlib'])
        self.meta_agent.policy_net.load_state_dict(checkpoint['meta_agent'])
        
        self.episode_count = checkpoint['episode']
        self.episode_rewards = deque(checkpoint['episode_rewards'], maxlen=100)
        
        logger.info(f"[Checkpoint] Loaded from {path} (episode {self.episode_count})")
    
    def get_diagnostics(self) -> Dict:
        """Get comprehensive system diagnostics."""
        return {
            'episode': self.episode_count,
            'mode': self.config.mode,
            'avg_reward_100ep': np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            'retrain_queue_short': len(self.retrain_queue['short_tft']),
            'retrain_queue_long': len(self.retrain_queue['long_tft']),
            'retrain_queue_pvlib': len(self.retrain_queue['pvlib']),
            'agents': {
                'short_tft': self.agent_short_tft.get_diagnostics(),
                'long_tft': self.agent_long_tft.get_diagnostics(),
                'pvlib': self.agent_pvlib.get_diagnostics()
            }
        }
