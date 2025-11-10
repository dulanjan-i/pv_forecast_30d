# src/models/rl_meta_controller.py
"""
RL Meta-Controller for MiRACLE (Role 4, DQN-based).

This module implements a generic Deep Q-Network (DQN) agent that will act as
the MiRACLE meta-controller. It does NOT hard-code PV-specific logic; instead,
it expects:
    - numeric state vectors s_t (already engineered upstream),
    - discrete action indices a_t (mapping defined elsewhere),
    - scalar rewards r_t and done flags.

High-level usage
----------------
from src.models.rl_meta_controller import (
    RLMConfig, RLMetaController,
)

cfg = RLMConfig(
    state_dim=STATE_DIM,      # e.g. len(S_t) from Role 4 design
    action_dim=NUM_ACTIONS,   # e.g. 0..6 for A0..A6
)

agent = RLMetaController(cfg)

# In your control loop:
q_action = agent.select_action(state)    # numpy / list → torch.Tensor internally
# ... execute action in MiRACLE system, observe next_state, reward, done ...
agent.store_transition(state, action, reward, next_state, done)
agent.update()  # one training step (if enough samples)

References
----------
- Mnih et al. (2015). "Human-level control through deep reinforcement learning".
  Nature. (Original DQN paper)
- Sutton & Barto (2018). "Reinforcement Learning: An Introduction".
  (For Q-learning / Bellman equation background)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import Adam


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class RLMConfig:
    """
    Configuration for the RL meta-controller (DQN agent).

    Core dimensions
    ---------------
    state_dim : int
        Dimensionality of the state vector S_t (flattened numeric features).
    action_dim : int
        Number of discrete actions A_t (e.g., len({A0,...,A6})).

    DQN architecture
    ----------------
    hidden_dim : int
        Width of hidden fully connected layers.
    num_hidden_layers : int
        How many hidden layers to use (>= 1).
    activation : str
        Non-linearity; currently supports "relu" only.

    Training hyperparameters
    ------------------------
    gamma : float
        Discount factor in [0, 1].
    lr : float
        Learning rate for Adam optimizer.
    batch_size : int
        Mini-batch size for replay updates.
    replay_capacity : int
        Maximum number of transitions stored in replay buffer.
    min_replay_size : int
        Minimum transitions before any training starts.
    target_update_interval : int
        Number of updates between target network syncs.
    max_grad_norm : float
        Gradient clipping threshold (L2 norm).

    Exploration (epsilon-greedy)
    ----------------------------
    eps_start : float
        Initial epsilon value.
    eps_end : float
        Final epsilon floor.
    eps_decay_steps : int
        Number of steps over which epsilon decays linearly from start to end.

    Device
    ------
    device : str
        "cpu" or "cuda". Can be set to "cuda" when running on GPU.
    """
    # dimensions
    state_dim: int
    action_dim: int

    # network
    hidden_dim: int = 128
    num_hidden_layers: int = 2
    activation: str = "relu"

    # training hyperparams
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    replay_capacity: int = 50_000
    min_replay_size: int = 1_000
    target_update_interval: int = 1_000
    max_grad_norm: float = 5.0

    # epsilon-greedy
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 50_000

    # device
    device: str = "cpu"


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Simple FIFO replay buffer for DQN."""

    def __init__(self, capacity: int, state_dim: int):
        self.capacity = capacity
        self.state_dim = state_dim

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

        self.idx = 0
        self.size = 0

    def store(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        i = self.idx
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.dones[i] = float(done)

        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            self.states[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.next_states[idxs],
            self.dones[idxs],
        )

    def __len__(self) -> int:
        return self.size


# ---------------------------------------------------------------------------
# DQN Network
# ---------------------------------------------------------------------------

class DQN(nn.Module):
    """Simple MLP for approximating Q(s, a)."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, num_hidden_layers: int):
        super().__init__()

        layers = []
        in_dim = state_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        # final layer to action_dim
        layers.append(nn.Linear(in_dim, action_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor of shape (batch_size, state_dim)

        Returns
        -------
        q_values : Tensor of shape (batch_size, action_dim)
        """
        return self.net(x)


# ---------------------------------------------------------------------------
# RL Meta-Controller (DQN Agent)
# ---------------------------------------------------------------------------

class RLMetaController:
    """
    DQN-based meta-controller for MiRACLE (Role 4).

    Responsibilities
    ----------------
    - Maintains Q-network and target network.
    - Stores transitions in replay buffer.
    - Selects actions via epsilon-greedy policy.
    - Performs gradient-based updates to minimize TD error.

    Integration
    -----------
    This class is intentionally generic: it does not know about PV, APIs, or
    MiRACLE internals. You wrap it in your orchestration layer which:

    - builds the state vector s_t (matching cfg.state_dim),
    - defines a fixed mapping from action indices to concrete actions
      (e.g., 0 = "use forecast API", 1 = "switch to ensemble API", ...),
    - calculates rewards based on error metrics, drift, and cost,
    - calls `select_action`, `store_transition`, and `update` at each step.
    """

    def __init__(self, cfg: RLMConfig):
        self.cfg = cfg

        self.device = torch.device(cfg.device)

        # online and target networks
        self.q_net = DQN(
            state_dim=cfg.state_dim,
            action_dim=cfg.action_dim,
            hidden_dim=cfg.hidden_dim,
            num_hidden_layers=cfg.num_hidden_layers,
        ).to(self.device)

        self.target_q_net = DQN(
            state_dim=cfg.state_dim,
            action_dim=cfg.action_dim,
            hidden_dim=cfg.hidden_dim,
            num_hidden_layers=cfg.num_hidden_layers,
        ).to(self.device)

        # initialize target with same weights
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.target_q_net.eval()

        self.optimizer = Adam(self.q_net.parameters(), lr=cfg.lr)

        self.replay = ReplayBuffer(cfg.replay_capacity, cfg.state_dim)

        # epsilon-greedy schedule
        self.eps = cfg.eps_start
        self.total_steps = 0

        # loss tracking (optional)
        self.last_loss: Optional[float] = None

    # ---------------------- Public API -------------------------------------

    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        """
        Select an action index given current state.

        Parameters
        ----------
        state : np.ndarray, shape (state_dim,)
            Current state vector S_t (normalized numeric features).
        greedy : bool
            If True, ignore epsilon and always pick argmax Q.

        Returns
        -------
        action : int
            Index in [0, action_dim-1].
        """
        state = self._to_tensor(state).unsqueeze(0)  # (1, state_dim)

        # epsilon-greedy
        if (not greedy) and (np.random.rand() < self.eps):
            action = np.random.randint(self.cfg.action_dim)
            return int(action)

        with torch.no_grad():
            q_values = self.q_net(state)  # (1, action_dim)
            action = torch.argmax(q_values, dim=1).item()
        return int(action)

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Store a transition (s, a, r, s', done) in replay buffer.
        """
        self.replay.store(state, action, reward, next_state, done)

    def update(self) -> Optional[float]:
        """
        Perform one DQN update step if enough replay samples are available.

        Returns
        -------
        loss_value : float or None
            Last loss value if an update was performed, otherwise None.
        """
        if len(self.replay) < self.cfg.min_replay_size:
            return None

        self.total_steps += 1
        self._update_epsilon()

        # Sample batch
        s, a, r, s_next, d = self.replay.sample(self.cfg.batch_size)

        s = self._to_tensor(s)
        a = self._to_tensor(a, dtype=torch.long)
        r = self._to_tensor(r)
        s_next = self._to_tensor(s_next)
        d = self._to_tensor(d)

        # Compute current Q(s,a)
        q_values = self.q_net(s)  # (B, action_dim)
        q_sa = q_values.gather(1, a.unsqueeze(1)).squeeze(1)  # (B,)

        # Compute target Q
        with torch.no_grad():
            next_q_values = self.target_q_net(s_next)  # (B, action_dim)
            max_next_q = next_q_values.max(dim=1)[0]   # (B,)
            target = r + self.cfg.gamma * (1.0 - d) * max_next_q

        # TD loss
        loss = nn.functional.mse_loss(q_sa, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), self.cfg.max_grad_norm)
        self.optimizer.step()

        self.last_loss = float(loss.item())

        # Periodically update target network
        if self.total_steps % self.cfg.target_update_interval == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        return self.last_loss

    def save(self, path: str) -> None:
        """
        Save the DQN parameters and optimizer state.

        Parameters
        ----------
        path : str
            File path to save the checkpoint (.pt or .pth).
        """
        torch.save(
            {
                "cfg": self.cfg.__dict__,
                "q_net_state_dict": self.q_net.state_dict(),
                "target_q_net_state_dict": self.target_q_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "total_steps": self.total_steps,
                "eps": self.eps,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "RLMetaController":
        """
        Load an RLMetaController from checkpoint.

        Parameters
        ----------
        path : str
            Path to checkpoint created by `save`.
        device : str
            Device to map the networks to ("cpu" or "cuda").

        Returns
        -------
        agent : RLMetaController
        """
        checkpoint = torch.load(path, map_location=device)
        cfg_dict = checkpoint["cfg"]
        cfg_dict["device"] = device
        cfg = RLMConfig(**cfg_dict)

        agent = cls(cfg)
        agent.q_net.load_state_dict(checkpoint["q_net_state_dict"])
        agent.target_q_net.load_state_dict(checkpoint["target_q_net_state_dict"])
        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        agent.total_steps = checkpoint["total_steps"]
        agent.eps = checkpoint["eps"]
        return agent

    # ---------------------- Internal helpers -------------------------------

    def _to_tensor(self, arr, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Convert numpy array or list to torch.Tensor on the agent's device."""
        if isinstance(arr, torch.Tensor):
            t = arr
        else:
            t = torch.as_tensor(arr)
        if dtype is not None:
            t = t.to(dtype)
        return t.to(self.device)

    def _update_epsilon(self) -> None:
        """Linearly decay epsilon from eps_start to eps_end over eps_decay_steps."""
        if self.cfg.eps_decay_steps <= 0:
            return

        frac = min(1.0, self.total_steps / float(self.cfg.eps_decay_steps))
        self.eps = self.cfg.eps_start + frac * (self.cfg.eps_end - self.cfg.eps_start)