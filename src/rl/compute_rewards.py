"""
Recompute rewards in an existing SARNS parquet using the canonical reward function.

Single source of truth:
  src/rl/reward.py

Usage (recommended, writes a new file):
  PYTHONPATH=. python -m src.rl.compute_rewards \
    --in data/rl_transitions/batch_001.parquet \
    --out data/rl_transitions/batch_001_rew.parquet

If you really want to overwrite in-place:
  PYTHONPATH=. python -m src.rl.compute_rewards --in ... --inplace
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from src.rl.reward import compute_reward as canonical_compute_reward


def _infer_state_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("state_")]
    # keep deterministic order state_0, state_1, ...
    cols = sorted(cols, key=lambda x: int(x.split("_")[1]))
    return cols


def _infer_next_state_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("next_state_")]
    cols = sorted(cols, key=lambda x: int(x.split("_")[2]))
    return cols


def recompute_rewards(df: pd.DataFrame) -> pd.DataFrame:
    if "action" not in df.columns:
        raise ValueError("Missing required column: action")

    state_cols = _infer_state_cols(df)
    next_state_cols = _infer_next_state_cols(df)

    if not state_cols or not next_state_cols:
        raise ValueError("Missing state columns. Need state_0.. and next_state_0..")

    if len(state_cols) != len(next_state_cols):
        raise ValueError(f"State dim mismatch: {len(state_cols)} state vs {len(next_state_cols)} next_state")

    # Build matrices
    S = df[state_cols].to_numpy(dtype=np.float32)
    NS = df[next_state_cols].to_numpy(dtype=np.float32)
    A = df["action"].to_numpy(dtype=np.int64)

    # Canonical reward expects at least indices 0,1,2 exist
    if S.shape[1] < 3:
        raise ValueError(
            f"State dim={S.shape[1]} but canonical reward requires at least 3 dims "
            "(short_rmse, long_rmse, physics_residual)."
        )

    # Vectorized loop (fast enough for typical SARNS sizes)
    rewards = np.empty((S.shape[0],), dtype=np.float32)
    for i in range(S.shape[0]):
        rewards[i] = canonical_compute_reward(S[i], int(A[i]), NS[i])

    df = df.copy()
    df["reward"] = rewards
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Input parquet")
    ap.add_argument("--out", dest="out_path", default=None, help="Output parquet (default: <in>_rew.parquet)")
    ap.add_argument("--inplace", action="store_true", help="Overwrite input parquet in-place")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Missing input: {in_path}")

    df = pd.read_parquet(in_path)
    df2 = recompute_rewards(df)

    if args.inplace:
        df2.to_parquet(in_path, index=False)
        print(f"OK wrote rewards in-place: {in_path}")
        return

    out_path = Path(args.out_path) if args.out_path else in_path.with_name(in_path.stem + "_rew.parquet")
    df2.to_parquet(out_path, index=False)
    print(f"OK wrote: {out_path}")


if __name__ == "__main__":
    main()
