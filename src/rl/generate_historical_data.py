"""
Generate RL training transitions from a minimal, canonical offline environment.

Environment signals per timestamp:
- power_norm: ground truth
- predicted_power_norm: baseline forecast output (MiRACLE v1.0 Core)
- pvlib_power_norm: PVLib baseline converted to cap-normalized power

Action semantics in this minimal environment:
- action 6 (BLEND_HIGH_PHYSICS): increase physics weight by lowering beta (ML weight)
- all other actions: no-op on prediction in this minimal environment, only cost affects reward

This is scientifically defensible IF you state it clearly:
the offline RL study is restricted to "blend-weight meta-control" given frozen components.

Output schema:
forecast_start, action, reward, done, rmse_day1,
blend_short, blend_long, blend_physics,
state_0..state_34, next_state_0..next_state_34
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.rl.reward import compute_reward as canonical_compute_reward


# -----------------------------
# Metrics helpers
# -----------------------------
def rmse(y_hat: np.ndarray, y: np.ndarray) -> float:
    y_hat = np.asarray(y_hat, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    m = np.isfinite(y_hat) & np.isfinite(y)
    if m.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_hat[m] - y[m]) ** 2)))


def _ensure_datetime_utc(s: pd.Series) -> pd.Series:
    ts = pd.to_datetime(s, utc=True, errors="coerce")
    if ts.isna().any():
        raise ValueError("Failed to parse timestamps to UTC. Check timestamp_utc column.")
    return ts


def _floor_day_utc(ts: pd.Series) -> pd.Series:
    dt = pd.to_datetime(ts, utc=True)
    return dt.dt.floor("D")


# -----------------------------
# Minimal blend mapping
# -----------------------------
@dataclass(frozen=True)
class BlendParams:
    beta: float  # ML vs physics. y = beta*baseline + (1-beta)*pvlib


def action_to_params(action: int, base: BlendParams) -> BlendParams:
    a = int(action)
    if a == 6:
        return BlendParams(beta=0.5)  # more physics
    return base


def params_to_weights(p: BlendParams) -> Tuple[float, float, float]:
    # No short/long decomposition in minimal env.
    # Keep fields for schema compatibility.
    w_phys = 1.0 - p.beta
    w_ml = p.beta
    w_short = 0.0
    w_long = float(w_ml)  # treat baseline forecast as "long/ML" for bookkeeping
    return float(w_short), float(w_long), float(w_phys)


# -----------------------------
# State builder (35-dim)
# -----------------------------
def build_state_35(
    short_rmse_1h: float,
    long_rmse_30d: float,
    physics_residual: float,
    rmse_day1: float,
    w_short: float,
    w_long: float,
    w_phys: float,
    day_of_year: int,
    month: int,
) -> np.ndarray:
    """
    Minimal but consistent 35-dim state.

    Key indices (kept consistent with prior scripts):
      state_0 = short_rmse_1h
      state_1 = long_rmse_30d
      state_2 = physics_residual
      state_3 = rmse_day1
      state_10..12 = blend_short/long/physics
      state_16 = day_of_year
      state_17 = month
    """
    s = np.zeros(35, dtype=np.float32)

    s[0] = float(short_rmse_1h)
    s[1] = float(long_rmse_30d)
    s[2] = float(physics_residual)
    s[3] = float(rmse_day1)

    # blend weights
    s[10] = float(w_short)
    s[11] = float(w_long)
    s[12] = float(w_phys)

    # time context
    s[16] = float(day_of_year)
    s[17] = float(month)

    return s


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Generate RL training data from historical test set (minimal env)")
    ap.add_argument("--test-data", required=True, help="Parquet with timestamp_utc, power_norm, predicted_power_norm, pvlib_power_norm")
    ap.add_argument("--num-samples", type=int, default=None, help="Optional cap on number of forecast_start days")
    ap.add_argument("--stride-days", type=int, default=1, help="Days to stride forward for each sample")
    ap.add_argument("--output", required=True, help="Output parquet file")

    ap.add_argument("--base-beta", type=float, default=0.7, help="Base beta: ML vs physics")
    ap.add_argument("--transitions-per-day", type=int, default=8, help="Transitions per day (default 8 = all actions)")

    args = ap.parse_args()

    in_path = Path(args.test_data)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(in_path).copy()

    required = ["timestamp_utc", "power_norm", "predicted_power_norm", "pvlib_power_norm", "forecast_start"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}. Found: {list(df.columns)[:30]}")

    df["timestamp_utc"] = _ensure_datetime_utc(df["timestamp_utc"])
    df = df.sort_values(["forecast_start", "timestamp_utc"]).reset_index(drop=True)

    starts = pd.to_datetime(df["forecast_start"], utc=True, errors="coerce").dropna().drop_duplicates().tolist()
    starts = starts[:: int(args.stride_days)]
    if args.num_samples is not None:
        starts = starts[: int(args.num_samples)]

    base = BlendParams(beta=float(args.base_beta))
    all_actions = [0, 1, 2, 3, 4, 5, 6, 7]

    transitions: List[Dict] = []

    for fs in starts:
        w = df[df["forecast_start"] == fs]
        if len(w) < 96:
            continue

        y = w["power_norm"].to_numpy(dtype=np.float32)
        y_base = w["predicted_power_norm"].to_numpy(dtype=np.float32)
        y_pv = w["pvlib_power_norm"].to_numpy(dtype=np.float32)

        H = min(len(y), len(y_base), len(y_pv))
        h1 = min(96, H)
        h1h = min(4, h1)

        # State based on BASE (no action)
        p0 = base
        wS0, wL0, wP0 = params_to_weights(p0)

        y0 = (p0.beta * y_base[:H]) + ((1.0 - p0.beta) * y_pv[:H])

        rmse_day1_0 = rmse(y0[:h1], y[:h1])
        long_rmse_0 = rmse(y0[:H], y[:H])
        short_rmse_1h_0 = rmse(y0[:h1h], y[:h1h]) if h1h > 0 else float("nan")
        phys_res_0 = float(np.mean(np.abs(y0[:h1] - y_pv[:h1]))) if h1 > 0 else float("nan")

        ts_fs = pd.Timestamp(fs)
        doy = int(ts_fs.dayofyear)
        mon = int(ts_fs.month)

        state = build_state_35(
            short_rmse_1h=short_rmse_1h_0,
            long_rmse_30d=long_rmse_0,
            physics_residual=phys_res_0,
            rmse_day1=rmse_day1_0,
            w_short=wS0,
            w_long=wL0,
            w_phys=wP0,
            day_of_year=doy,
            month=mon,
        )

        actions_today = all_actions if int(args.transitions_per_day) >= 8 else list(
            np.random.choice(all_actions, size=int(args.transitions_per_day), replace=True)
        )

        for a in actions_today:
            p1 = action_to_params(a, base=base)
            wS1, wL1, wP1 = params_to_weights(p1)

            y1 = (p1.beta * y_base[:H]) + ((1.0 - p1.beta) * y_pv[:H])

            rmse_day1_1 = rmse(y1[:h1], y[:h1])
            long_rmse_1 = rmse(y1[:H], y[:H])
            short_rmse_1h_1 = rmse(y1[:h1h], y[:h1h]) if h1h > 0 else float("nan")
            phys_res_1 = float(np.mean(np.abs(y1[:h1] - y_pv[:h1]))) if h1 > 0 else float("nan")

            next_state = build_state_35(
                short_rmse_1h=short_rmse_1h_1,
                long_rmse_30d=long_rmse_1,
                physics_residual=phys_res_1,
                rmse_day1=rmse_day1_1,
                w_short=wS1,
                w_long=wL1,
                w_phys=wP1,
                day_of_year=doy,
                month=mon,
            )

            reward = float(canonical_compute_reward(state, int(a), next_state))

            transitions.append(
                {
                    "forecast_start": ts_fs.isoformat(),
                    "action": int(a),
                    "reward": reward,
                    "done": False,
                    "rmse_day1": float(rmse_day1_0),
                    "blend_short": float(wS1),
                    "blend_long": float(wL1),
                    "blend_physics": float(wP1),
                    **{f"state_{k}": float(state[k]) for k in range(35)},
                    **{f"next_state_{k}": float(next_state[k]) for k in range(35)},
                }
            )

    out_df = pd.DataFrame(transitions)
    if len(out_df) == 0:
        raise RuntimeError("No transitions generated. Check that each forecast_start has at least 96 rows.")

    out_df.to_parquet(out_path, index=False)
    print(f"OK wrote transitions: {out_path} rows={len(out_df)}")
    print("action_counts:", out_df["action"].value_counts().sort_index().to_dict())
    print("reward_desc:\n", out_df["reward"].describe())


if __name__ == "__main__":
    main()
