# src/rl/reward.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


# -----------------------------
# Action costs (8-action design)
# -----------------------------
ACTION_COSTS: Dict[int, float] = {
    0: 0.0,   # MAINTAIN
    1: 0.1,   # FINE_TUNE_SHORT
    2: 0.15,  # FINE_TUNE_LONG
    3: 0.05,  # RECALIBRATE_PVLIB
    4: 0.0,   # BLEND_HIGH_SHORT
    5: 0.0,   # BLEND_HIGH_LONG
    6: 0.0,   # BLEND_HIGH_PHYSICS
    7: 1.0,   # SUGGEST_RETRAIN
}

SCALE_DENOM_DEFAULT = 0.01


@dataclass(frozen=True)
class RewardWeightsV1:
    w_short: float = 1.0
    w_long: float = 0.5
    w_phys: float = 0.2
    w_cost: float = 0.2
    scale_denom: float = SCALE_DENOM_DEFAULT


@dataclass(frozen=True)
class RewardWeightsV2:
    # keep the same core spirit as v1, but add ramp/peak/anti-smooth shaping
    w_acc: float = 1.0
    w_phys: float = 0.2
    w_smooth: float = 0.4
    w_ramp: float = 0.4
    w_peak: float = 0.2

    # stronger cost penalty than v1
    w_cost2: float = 0.6

    scale_denom: float = SCALE_DENOM_DEFAULT


def compute_reward_v1(
    state: np.ndarray,
    next_state: np.ndarray,
    action: int,
    weights: RewardWeightsV1 = RewardWeightsV1(),
) -> float:
    """
    Canonical MiRACLE reward (v1).

    Expected state layout (minimum):
      state[0]      = short_rmse_t
      state[1]      = long_rmse_t
      next_state[0] = short_rmse_{t+1}
      next_state[1] = long_rmse_{t+1}
      next_state[2] = physics_residual_{t+1}   (signed or unsigned; we use abs)
    """
    s0 = float(state[0])
    s1 = float(state[1])
    ns0 = float(next_state[0])
    ns1 = float(next_state[1])
    phys_res_next = float(next_state[2]) if len(next_state) > 2 else 0.0

    delta_short = s0 - ns0
    delta_long = s1 - ns1
    cost = ACTION_COSTS.get(int(action), 0.0)

    r = (
        weights.w_short * delta_short
        + weights.w_long * delta_long
        - weights.w_phys * abs(phys_res_next)
        - weights.w_cost * cost
    )
    return r / float(weights.scale_denom)


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0:
        return 0.0
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _tv(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    if a.size < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(a))))


def compute_reward_v2(
    state: np.ndarray,
    next_state: np.ndarray,
    action: int,
    y: np.ndarray,
    y_hat: np.ndarray,
    is_daylight: Optional[np.ndarray] = None,
    weights: RewardWeightsV2 = RewardWeightsV2(),
) -> float:
    """
    Shaped reward (v2) to avoid the PVLib-hugging / over-smoothing loophole.

    Uses:
      - accuracy improvement proxy (ΔRMSE_day1) from state -> next_state (same as v1 spirit)
      - physics residual penalty (abs next_state[2])
      - anti-smoothing penalty: max(0, TV(y) - TV(y_hat)) on daylight
      - ramp fidelity penalty: RMSE(diff(y_hat), diff(y)) on daylight
      - peak fidelity penalty: abs(max(y_hat_daylight) - max(y_daylight))
      - stronger action cost penalty
    """
    y = np.asarray(y, dtype=np.float64)
    y_hat = np.asarray(y_hat, dtype=np.float64)

    if is_daylight is None:
        mask = np.ones_like(y, dtype=bool)
    else:
        mask = np.asarray(is_daylight).astype(bool)
        if mask.shape != y.shape:
            raise ValueError(f"is_daylight shape {mask.shape} does not match y shape {y.shape}")

    y_dl = y[mask]
    yhat_dl = y_hat[mask]

    # accuracy improvement proxy (day1 rmse reduction)
    # If your state[0]/next_state[0] are already short_rmse and represent day1 quality, this is consistent.
    delta_rmse = float(state[0]) - float(next_state[0])

    # physics residual term from next_state (same as v1)
    phys_res_next = float(next_state[2]) if len(next_state) > 2 else 0.0

    # anti-smoothing: punish predictions that are smoother than reality
    tv_y = _tv(y_dl)
    tv_yhat = _tv(yhat_dl)
    smooth_deficit = max(0.0, tv_y - tv_yhat)

    # ramps: first differences (daylight-only)
    ramp_rmse = _rmse(np.diff(yhat_dl), np.diff(y_dl)) if y_dl.size >= 2 else 0.0

    # peaks: daylight max mismatch
    peak_err = abs(float(np.max(yhat_dl)) - float(np.max(y_dl))) if y_dl.size > 0 else 0.0

    cost = ACTION_COSTS.get(int(action), 0.0)

    r = (
        weights.w_acc * delta_rmse
        - weights.w_phys * abs(phys_res_next)
        - weights.w_smooth * smooth_deficit
        - weights.w_ramp * ramp_rmse
        - weights.w_peak * peak_err
        - weights.w_cost2 * cost
    )
    return r / float(weights.scale_denom)
