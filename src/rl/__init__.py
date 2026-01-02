"""
Reinforcement Learning Meta-Controller for MiRACLE

Hierarchical architecture (CORRECTED):
- 1 DDQN Meta-Controller (learns global policy)
- 3 Rule-Based Advisors (provide state signals, no learning)
"""

from .rl_meta_controller import (
    RLMetaControllerSystem,
    RLConfig,
    LocalAdvisor,
    MetaController,
    PrioritizedReplayBuffer,
    DQN
)

__all__ = [
    'RLMetaControllerSystem',
    'RLConfig',
    'LocalAdvisor',
    'MetaController',
    'PrioritizedReplayBuffer',
    'DQN'
]
