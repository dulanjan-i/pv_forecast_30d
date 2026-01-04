"""
Offline RL training utilities for DDQN meta-controller.

This module contains the core training logic that should be reused across
different training workflows.
"""
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


def prepare_training_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract (state, action, reward, next_state, done) tuples from dataframe.

    Expected columns:
      - state_0..state_{D-1}
      - next_state_0..next_state_{D-1}
      - action
      - reward  (or day1_rmse, used as reward=-day1_rmse)
      - optional: done (0/1). If missing, defaults to 1 if next_state == state for all rows, else 0.

    Returns:
        (states, actions, rewards, next_states, dones)
    """
    # Keep deterministic ordering: state_0, state_1, ...
    state_cols = sorted([c for c in df.columns if c.startswith("state_") and not c.startswith("next_")],
                        key=lambda x: int(x.split("_")[1]))
    next_state_cols = sorted([c for c in df.columns if c.startswith("next_state_")],
                             key=lambda x: int(x.split("_")[2]))

    if len(state_cols) == 0 or len(next_state_cols) == 0:
        raise ValueError("Missing state columns. Need state_0.. and next_state_0..")

    if "action" not in df.columns:
        raise ValueError("Missing required column: action")

    # Reward: prefer explicit reward, else derive from day1_rmse if present
    if "reward" in df.columns:
        rewards = df["reward"].astype(np.float32).to_numpy()
    elif "day1_rmse" in df.columns:
        rewards = (-df["day1_rmse"].astype(np.float32)).to_numpy()
        logger.warning("Column 'reward' missing, using reward = -day1_rmse.")
    else:
        raise ValueError("Missing required column: reward (or day1_rmse)")

    states = df[state_cols].astype(np.float32).to_numpy()
    next_states = df[next_state_cols].astype(np.float32).to_numpy()
    actions = df["action"].astype(np.int64).to_numpy()

    # Done handling:
    # - use 'done' if present
    # - else infer a sensible default:
    #     if most rows look like one-step terminal transitions, set done=1
    #     otherwise default done=0
    if "done" in df.columns:
        dones = df["done"].astype(np.float32).to_numpy()
    else:
        same = np.all(np.isclose(states, next_states, atol=1e-6), axis=1)
        terminal_ratio = float(np.mean(same))
        if terminal_ratio > 0.95:
            dones = np.ones(len(df), dtype=np.float32)
            logger.warning("Column 'done' missing, inferred terminal transitions (done=1) for all rows.")
        else:
            dones = np.zeros(len(df), dtype=np.float32)
            logger.warning("Column 'done' missing, defaulting to non-terminal transitions (done=0).")

    # Drop any non-finite rows, do not let NaNs poison training
    mask = (
        np.isfinite(states).all(axis=1)
        & np.isfinite(next_states).all(axis=1)
        & np.isfinite(rewards)
        & np.isfinite(actions.astype(np.float32))
        & np.isfinite(dones)
    )
    dropped = int(np.sum(~mask))
    if dropped > 0:
        logger.warning("Dropping %d/%d transitions due to non-finite values.", dropped, len(df))
        states = states[mask]
        next_states = next_states[mask]
        rewards = rewards[mask]
        actions = actions[mask]
        dones = dones[mask]

    logger.info(f"State dimension: {states.shape[1]}")
    logger.info("Training data shapes:")
    logger.info(f"  States: {states.shape}")
    logger.info(f"  Actions: {actions.shape}")
    logger.info(f"  Rewards: {rewards.shape}")
    logger.info(f"  Next states: {next_states.shape}")
    logger.info(f"  Dones: {dones.shape}")

    return states, actions, rewards, next_states, dones


class DQNNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: List[int]):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = state_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransitionDataset(Dataset):
    def __init__(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ):
        self.states = torch.from_numpy(states)
        self.actions = torch.from_numpy(actions)
        self.rewards = torch.from_numpy(rewards)
        self.next_states = torch.from_numpy(next_states)
        self.dones = torch.from_numpy(dones)

    def __len__(self) -> int:
        return self.states.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "states": self.states[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_states": self.next_states[idx],
            "dones": self.dones[idx],
        }


@dataclass
class DQNTrainingConfig:
    state_dim: int
    action_dim: int
    hidden_sizes: List[int]
    lr: float = 1e-3
    gamma: float = 0.99
    batch_size: int = 128
    num_epochs: int = 50
    target_update_freq: int = 10
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay: float = 0.995
    device: str = "cuda"


class DQNTrainer:
    def __init__(self, config: DQNTrainingConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        self.q_net = DQNNetwork(config.state_dim, config.action_dim, config.hidden_sizes).to(self.device)
        self.target_net = DQNNetwork(config.state_dim, config.action_dim, config.hidden_sizes).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=config.lr)

        self.epsilon = config.epsilon_start

    def train(self, df: pd.DataFrame) -> Dict[str, List[float]]:
        states, actions, rewards, next_states, dones = prepare_training_data(df)

        dataset = TransitionDataset(states, actions, rewards, next_states, dones)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

        history = {"loss": [], "avg_reward": []}

        for epoch in range(self.config.num_epochs):
            epoch_losses: List[float] = []
            epoch_rewards: List[float] = []

            for batch in dataloader:
                loss_val = self._train_step(batch)
                epoch_losses.append(loss_val)
                epoch_rewards.append(float(batch["rewards"].mean().item()))

            avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            avg_reward = float(np.mean(epoch_rewards)) if epoch_rewards else 0.0
            history["loss"].append(avg_loss)
            history["avg_reward"].append(avg_reward)

            if (epoch + 1) % self.config.target_update_freq == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())

            self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)

            logger.info(
                "Epoch %d/%d | loss=%.6f | avg_reward=%.6f | epsilon=%.4f",
                epoch + 1,
                self.config.num_epochs,
                avg_loss,
                avg_reward,
                self.epsilon,
            )

        return history

    def _train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        states = batch["states"].to(self.device).float()
        actions = batch["actions"].to(self.device).long()
        rewards = batch["rewards"].to(self.device).float()
        next_states = batch["next_states"].to(self.device).float()
        dones = batch["dones"].to(self.device).float()

        q_values = self.q_net(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # DDQN target: online selects, target evaluates
        with torch.no_grad():
            next_actions = self.q_net(next_states).argmax(1)
            next_q_value = self.target_net(next_states).gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)
            target = rewards + (1.0 - dones) * self.config.gamma * next_q_value

        loss = F.mse_loss(q_value, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "q_net": self.q_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "config": self.config,
                "epsilon": self.epsilon,
            },
            str(p),
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.epsilon = ckpt.get("epsilon", self.config.epsilon_start)
