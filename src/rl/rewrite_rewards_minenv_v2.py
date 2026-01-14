# src/rl/rewrite_rewards_minenv_v2.py
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.rl.reward import compute_reward_v2


def _infer_state_cols(df: pd.DataFrame, prefix: str, dim: int) -> list[str]:
    cols = [f"{prefix}{i}" for i in range(dim)]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing {len(missing)} columns like {missing[:5]}")
    return cols


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sarns-in", required=True, help="Input SARNS parquet (MINENV_v1)")
    ap.add_argument("--preds", required=True, help="Predictions parquet containing y_hat and forecast_start, step_ahead")
    ap.add_argument("--gt", required=True, help="Ground truth parquet with power_norm and is_daylight")
    ap.add_argument("--out", required=True, help="Output SARNS parquet (MINENV_v2)")
    ap.add_argument("--state-dim", type=int, default=35)
    ap.add_argument("--day1-steps", type=int, default=96)
    args = ap.parse_args()

    sarns_path = Path(args.sarns_in)
    preds_path = Path(args.preds)
    gt_path = Path(args.gt)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(sarns_path)
    preds = pd.read_parquet(preds_path)
    gt = pd.read_parquet(gt_path)

    # required columns
    if "timestamp_utc" not in preds.columns or "forecast_start" not in preds.columns:
        raise ValueError("preds parquet must include timestamp_utc and forecast_start")
    if "timestamp_utc" not in gt.columns:
        raise ValueError("gt parquet must include timestamp_utc")
    if "power_norm" not in gt.columns:
        raise ValueError("gt parquet must include power_norm")
    if "is_daylight" not in gt.columns:
        raise ValueError("gt parquet must include is_daylight (0/1)")

    # figure out prediction column
    yhat_col = None
    for c in ["predicted_power_norm", "y_hat", "pred_power_norm"]:
        if c in preds.columns:
            yhat_col = c
            break
    if yhat_col is None:
        raise ValueError("Could not find prediction column in preds parquet")

    # join for day1 series per forecast_start
    # We only need day1 points: step_ahead 0..95
    if "step_ahead" not in preds.columns:
        raise ValueError("preds parquet must include step_ahead to select day1")
    preds_day1 = preds[preds["step_ahead"].between(0, args.day1_steps - 1)].copy()

    gt2 = gt[["timestamp_utc", "power_norm", "is_daylight"]].copy()
    merged = preds_day1.merge(gt2, on="timestamp_utc", how="inner")

    # cache day1 arrays per forecast_start
    grouped = merged.groupby("forecast_start", sort=True)

    state_cols = _infer_state_cols(df, "state_", args.state_dim)
    next_state_cols = _infer_state_cols(df, "next_state_", args.state_dim)

    new_rewards = []
    for row in df.itertuples(index=False):
        fs = getattr(row, "forecast_start")
        action = int(getattr(row, "action"))

        try:
            g = grouped.get_group(fs)
        except KeyError:
            raise ValueError(f"forecast_start {fs} not found in preds/gt join (day1)")

        y = g["power_norm"].to_numpy(dtype=np.float64)
        y_hat = g[yhat_col].to_numpy(dtype=np.float64)
        is_daylight = g["is_daylight"].to_numpy(dtype=np.int8)

        state = np.asarray([getattr(row, c) for c in state_cols], dtype=np.float32)
        next_state = np.asarray([getattr(row, c) for c in next_state_cols], dtype=np.float32)

        r2 = compute_reward_v2(
            state=state,
            next_state=next_state,
            action=action,
            y=y,
            y_hat=y_hat,
            is_daylight=is_daylight,
        )
        new_rewards.append(float(r2))

    df_out = df.copy()
    df_out["reward_v1_old"] = df_out["reward"]
    df_out["reward"] = np.asarray(new_rewards, dtype=np.float32)

    df_out.to_parquet(out_path, index=False)
    print(f"[OK] wrote: {out_path} rows={len(df_out)}")
    print(df_out["reward"].describe())


if __name__ == "__main__":
    main()
