"""
RL training utilities:
- load_transitions(): read SARNS parquet into numpy arrays
- DQNNetwork: generic MLP used by evaluators and trainers

Expected parquet columns:
- action, reward, done (done optional)
- state_0..state_{D-1}
- next_state_0..next_state_{D-1}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# -----------------------------
# Data loading
# -----------------------------
def _ordered_state_cols(df: pd.DataFrame, prefix: str) -> list[str]:
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        raise ValueError(f"Missing columns with prefix '{prefix}'. Need {prefix}0.. etc.")

    def key(c: str) -> int:
        return int(c.split("_")[-1])

    cols = sorted(cols, key=key)

    # sanity: contiguous 0..D-1
    idxs = [key(c) for c in cols]
    if idxs != list(range(0, len(idxs))):
        raise ValueError(f"Non-contiguous {prefix} cols (expected {prefix}0..{prefix}{len(idxs)-1}).")

    return cols


def load_transitions(
    path: str,
    state_dim: Optional[int] = None,
    drop_nonfinite: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_parquet(path)

    for req in ["action", "reward"]:
        if req not in df.columns:
            raise ValueError(f"Missing required column '{req}'. Available: {list(df.columns)[:40]}")

    state_cols = _ordered_state_cols(df, "state_")
    next_state_cols = _ordered_state_cols(df, "next_state_")

    if len(state_cols) != len(next_state_cols):
        raise ValueError(f"state_dim mismatch: {len(state_cols)} vs next {len(next_state_cols)}")

    if state_dim is not None and int(state_dim) != int(len(state_cols)):
        raise ValueError(f"State dim mismatch: file has {len(state_cols)} but requested {state_dim}")

    states = df[state_cols].to_numpy(dtype=np.float32)
    next_states = df[next_state_cols].to_numpy(dtype=np.float32)
    actions = df["action"].to_numpy(dtype=np.int64)
    rewards = df["reward"].to_numpy(dtype=np.float32)

    if "done" in df.columns:
        dones = df["done"].astype(bool).to_numpy()
    else:
        dones = np.zeros(len(df), dtype=bool)

    if drop_nonfinite:
        m = (
            np.isfinite(states).all(axis=1)
            & np.isfinite(next_states).all(axis=1)
            & np.isfinite(rewards)
        )
        if m.sum() < len(df):
            states = states[m]
            next_states = next_states[m]
            actions = actions[m]
            rewards = rewards[m]
            dones = dones[m]

    return states, actions, rewards, next_states, dones


# -----------------------------
# Network (used across scripts)
# -----------------------------
class DQNNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: List[int]):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = int(state_dim)
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, int(h)))
            layers.append(nn.ReLU())
            in_dim = int(h)
        layers.append(nn.Linear(in_dim, int(action_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class OfflineTrainConfig:
    state_dim: int
    action_dim: int
    hidden_sizes: List[int]
    gamma: float = 0.95
    lr: float = 1e-3
    batch_size: int = 256
    epochs: int = 60
    target_update: int = 200
    seed: int = 42

