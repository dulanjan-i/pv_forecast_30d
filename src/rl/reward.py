import numpy as np

# Must match MetaController.ACTION_COSTS
ACTION_COSTS = {
    0: 0.0,   # MAINTAIN
    1: 0.1,   # FINE_TUNE_SHORT
    2: 0.15,  # FINE_TUNE_LONG
    3: 0.05,  # RECALIBRATE_PVLIB
    4: 0.0,   # BLEND_HIGH_SHORT
    5: 0.0,   # BLEND_HIGH_LONG
    6: 0.0,   # BLEND_HIGH_PHYSICS
    7: 1.0,   # SUGGEST_RETRAIN
}

def compute_reward(state: np.ndarray, action: int, next_state: np.ndarray,
                   w_short: float = 1.0, w_long: float = 0.5, w_phys: float = 0.2, w_cost: float = 0.2,
                   scale: float = 0.01) -> float:
    """
    Canonical offline reward for MiRACLE RL.

    state[0] = short_rmse_1h (or day1 proxy)
    state[1] = long_rmse_30d
    state[2] = physics_residual
    """
    short_improve = (state[0] - next_state[0]) / scale
    long_improve  = (state[1] - next_state[1]) / scale
    phys_penalty  = np.abs(next_state[2]) / scale

    cost_penalty = ACTION_COSTS.get(int(action), 0.0)

    r = (
        w_short * short_improve +
        w_long  * long_improve -
        w_phys  * phys_penalty -
        w_cost  * cost_penalty
    )
    return float(r)
