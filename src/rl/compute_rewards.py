"""
Post-process RL transitions to compute proper rewards.

Reward formula (simplified from MiRACLE):
    R_t = -RMSE_t + improvement_bonus
    
Where RMSE_t is a weighted combination of short/long horizon errors.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_rewards_from_states(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rewards from state RMSE metrics.
    
    State encoding:
        state_0: short_rmse_1h
        state_1: long_rmse_30d
        state_2: physics_residual (already normalized)
        
    Reward:
        R_t = w1*(improvement_short) + w2*(improvement_long) - w3*|physics_residual|
    
    Args:
        df: Transitions with state_0, state_1, ..., next_state_0, next_state_1, ...
        
    Returns:
        df: Updated with computed rewards
    """
    df = df.copy()
    
    # Extract RMSE components
    short_rmse = df['state_0'].values  # Current short RMSE
    long_rmse = df['state_1'].values   # Current long RMSE
    physics_res = df['state_2'].values  # Physics residual
    
    next_short_rmse = df['next_state_0'].values
    next_long_rmse = df['next_state_1'].values
    next_physics_res = df['next_state_2'].values
    
    # Weights (aligned with MiRACLE)
    w_short = 1.0   # Prioritize short-term accuracy
    w_long = 0.5    # Long-term matters but less urgent
    w_physics = 0.2  # Small penalty for physics deviation
    
    # Compute improvement (negative RMSE change = positive improvement)
    improvement_short = short_rmse - next_short_rmse
    improvement_long = long_rmse - next_long_rmse
    
    # Reward = weighted improvements - physics penalty
    rewards = (
        w_short * improvement_short +
        w_long * improvement_long -
        w_physics * np.abs(next_physics_res)
    )
    
    # Normalize to reasonable scale (RMSE typically 0.01-0.10)
    rewards = rewards / 0.01  # Scale so 0.01 improvement = +1.0 reward
    
    # Add small baseline to avoid all-negative rewards (optional)
    # rewards += 0.1
    
    df['reward'] = rewards
    
    logger.info(f"Reward statistics:")
    logger.info(f"  Mean: {rewards.mean():.4f}")
    logger.info(f"  Std:  {rewards.std():.4f}")
    logger.info(f"  Min:  {rewards.min():.4f}")
    logger.info(f"  Max:  {rewards.max():.4f}")
    logger.info(f"  Median: {np.median(rewards):.4f}")
    logger.info(f"  Positive: {(rewards > 0).sum()}/{len(rewards)} ({100*(rewards>0).sum()/len(rewards):.1f}%)")
    
    return df


def main():
    # Load collected transitions
    data_path = Path("data/rl_transitions/batch_001.parquet")
    logger.info(f"Loading transitions from {data_path}")
    df = pd.read_parquet(data_path)
    
    logger.info(f"Original shape: {df.shape}")
    logger.info(f"Original reward stats: mean={df['reward'].mean():.4f}, std={df['reward'].std():.4f}")
    
    # Compute proper rewards
    df = compute_rewards_from_states(df)
    
    # Save back
    output_path = data_path.parent / "batch_001_with_rewards.parquet"
    df.to_parquet(output_path, index=False)
    logger.info(f"✅ Saved updated transitions to {output_path}")
    
    # Also overwrite original
    df.to_parquet(data_path, index=False)
    logger.info(f"✅ Updated original file: {data_path}")


if __name__ == "__main__":
    main()
