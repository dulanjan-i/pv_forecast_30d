#!/usr/bin/env python3
"""
Replay Phase-1 forecast_start timestamps and collect RL transitions.

Usage:
  python src/rl/replay_phase1_sampling.py \
    --phase1-transitions data/rl_transitions/phase1_4797.parquet \
    --raw-test data/processed/plant_level/plant_03/15min_pca32/test.parquet \
    --output data/rl_transitions/phase2_run_B.parquet

This script builds historical and forecast windows for each `forecast_start` in
the Phase-1 transitions file and runs the RL forecaster once per start,
recording transitions similar to `collect_rl_data.py`.

Core fix enforced here:
- Hard-assert len(forecast_day1) == len(gt_day1) == 96
  If not, SKIP the sample. Do not assign fake rewards.

Additional fixes:
- Require a real target column in raw test parquet (no fallback to zeros).
- Save the full state dimension (not just first 5 entries).
- Timestamp alignment is robust (UTC parse then tz-naive).
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.rl.rl_integrated_forecaster import RLIntegratedForecaster
from src.inference.physics_aware_forecaster import PhysicsAwareForecaster

LOGGER = logging.getLogger(__name__)


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def compute_rmse(pred: np.ndarray, true: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float32)
    true = np.asarray(true, dtype=np.float32)
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def _to_utc_naive(ts: pd.Series) -> pd.Series:
    s = pd.to_datetime(ts, utc=True, errors="coerce")
    return s.dt.tz_convert(None)


def build_windows_from_start(
    df: pd.DataFrame,
    forecast_start: str,
    *,
    lookback: int = 672,
    window_size: int = 96,
    time_col: str = "timestamp_utc",
    target_col: str = "power_norm",
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]]:
    """
    Extract (historical_df, forecast_df, gt_day1) for a given forecast_start timestamp.
    Returns None if the start is invalid or out-of-range.
    """
    start_ts = pd.to_datetime(forecast_start, utc=True, errors="coerce")
    if pd.isna(start_ts):
        return None
    start_ts = start_ts.tz_convert(None)

    if time_col not in df.columns or target_col not in df.columns:
        return None

    idx_arr = df.index[df[time_col] == start_ts]
    if len(idx_arr) == 0:
        return None

    idx = int(idx_arr[0])

    if idx - lookback < 0:
        return None
    if idx + window_size > len(df):
        return None

    historical = df.iloc[idx - lookback : idx].copy()
    forecast = df.iloc[idx : idx + window_size].copy()
    gt = forecast[target_col].to_numpy(dtype=np.float32)

    if len(forecast) != window_size or len(gt) != window_size:
        return None
    if not np.all(np.isfinite(gt)):
        return None

    return historical, forecast, gt


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay phase1 sampling with strict transition rebuild")
    parser.add_argument("--phase1-transitions", type=str, required=True, help="Phase1 transitions parquet")
    parser.add_argument("--raw-test-parquet", type=str, required=True, help="Raw test parquet (with target)")
    parser.add_argument("--model-dir", type=str, required=True, help="Model directory")
    parser.add_argument("--plant-config", type=str, required=True, help="Plant configuration JSON")
    parser.add_argument("--output-dir", type=str, default="data/rl_transitions", help="Output directory")
    parser.add_argument("--batch-name", type=str, default=None, help="Custom output batch name")
    parser.add_argument("--lookback", type=int, default=672, help="Lookback length in rows")
    parser.add_argument("--window-size", type=int, default=96, help="Forecast window size in rows (96)")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on number of forecast_starts")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--time-col", type=str, default="timestamp_utc", help="Time column name in raw test parquet")
    parser.add_argument("--target-col", type=str, default="power_norm", help="Target column name in raw test parquet")
    args = parser.parse_args()

    setup_logging(args.log_level)

    if args.window_size != 96:
        raise ValueError("This replay script is designed for day-ahead windows. Set --window-size 96.")

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_name = args.batch_name or f"replay_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_path = output_dir / f"{batch_name}.parquet"

    LOGGER.info("Loading phase1 transitions: %s", args.phase1_transitions)
    phase1_df = pd.read_parquet(args.phase1_transitions)

    if "forecast_start" not in phase1_df.columns:
        raise ValueError("Phase1 transitions must contain a 'forecast_start' column")

    # Unique starts as strings (tz-naive)
    starts = (
        pd.to_datetime(phase1_df["forecast_start"], utc=True, errors="coerce")
        .dt.tz_convert(None)
        .dropna()
        .drop_duplicates()
        .astype(str)
        .tolist()
    )
    if args.max_samples is not None:
        starts = starts[: args.max_samples]

    LOGGER.info("Unique forecast_starts to replay: %d", len(starts))

    LOGGER.info("Loading raw test parquet: %s", args.raw_test_parquet)
    raw_df = pd.read_parquet(args.raw_test_parquet)

    if args.time_col not in raw_df.columns:
        raise ValueError(f"Raw test parquet missing time column: {args.time_col}")
    if args.target_col not in raw_df.columns:
        raise ValueError(
            f"Raw test parquet missing target column: {args.target_col}. "
            "This script will not fabricate ground truth."
        )

    raw_df = raw_df.copy()
    raw_df[args.time_col] = _to_utc_naive(raw_df[args.time_col])
    raw_df = raw_df.dropna(subset=[args.time_col]).sort_values(args.time_col).reset_index(drop=True)

    LOGGER.info("Loading plant config: %s", args.plant_config)
    with open(args.plant_config, "r", encoding="utf-8") as f:
        plant_config: Dict = json.load(f)

    LOGGER.info("Initializing forecasters...")
    physics_forecaster = PhysicsAwareForecaster(model_dir, plant_config)

    rl_forecaster = RLIntegratedForecaster(
        physics_forecaster=physics_forecaster,
        model_dir=model_dir,
        plant_config=plant_config,
    )

    # past metrics for state building
    past_metrics = {
        "short_rmse": 0.0,
        "long_rmse": 0.0,
        "physics_residual": 0.0,
        "weather_quality": 0.8,
        "seasonal_factor": 1.0,
    }

    # Reuse actions if present in phase1
    action_map = {}
    if "action" in phase1_df.columns:
        tmp = phase1_df.dropna(subset=["action", "forecast_start"]).copy()
        tmp["forecast_start_ts"] = pd.to_datetime(tmp["forecast_start"], utc=True, errors="coerce").dt.tz_convert(None)
        tmp = tmp.dropna(subset=["forecast_start_ts"])
        action_map = tmp.groupby("forecast_start_ts")["action"].first().to_dict()

    stats = {
        "attempted": 0,
        "saved": 0,
        "skip_no_window": 0,
        "skip_forecast_failed": 0,
        "skip_len_mismatch": 0,
        "skip_nonfinite": 0,
        "skip_bad_state": 0,
    }

    transitions: List[Dict] = []

    for fs in starts:
        stats["attempted"] += 1

        win = build_windows_from_start(
            raw_df,
            fs,
            lookback=args.lookback,
            window_size=args.window_size,
            time_col=args.time_col,
            target_col=args.target_col,
        )
        if win is None:
            stats["skip_no_window"] += 1
            continue

        historical_df, forecast_df, gt = win

        try:
            forecast = rl_forecaster.forecast_day1(historical_df, forecast_df)
        except Exception as e:
            LOGGER.warning("Forecast failed at %s: %s", fs, str(e))
            stats["skip_forecast_failed"] += 1
            continue

        if forecast is None:
            stats["skip_forecast_failed"] += 1
            continue

        forecast = np.asarray(forecast, dtype=np.float32)[:96]
        gt = np.asarray(gt, dtype=np.float32)

        # Core fix: strict length check, else skip (no fake reward)
        if len(forecast) != 96 or len(gt) != 96:
            stats["skip_len_mismatch"] += 1
            continue
        if not (np.all(np.isfinite(forecast)) and np.all(np.isfinite(gt))):
            stats["skip_nonfinite"] += 1
            continue

        rmse_pv = compute_rmse(forecast, gt)

        try:
            physics_residual = rl_forecaster.physics_forecaster.compute_physics_residual(forecast, gt)
        except Exception:
            physics_residual = 0.0

        current_metrics = {
            "short_rmse": rmse_pv,
            "long_rmse": rmse_pv,
            "physics_residual": float(physics_residual),
            "weather_quality": 0.8,
            "seasonal_factor": 1.0,
        }

        try:
            state = rl_forecaster.meta_controller.compute_state(past_metrics, current_metrics)
        except Exception as e:
            LOGGER.warning("State computation failed at %s: %s", fs, str(e))
            stats["skip_bad_state"] += 1
            continue

        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.size == 0 or not np.all(np.isfinite(state)):
            stats["skip_bad_state"] += 1
            continue

        fs_ts = pd.to_datetime(fs, utc=True, errors="coerce")
        action = None
        if not pd.isna(fs_ts):
            action = action_map.get(fs_ts.tz_convert(None))
        if action is None:
            action = rl_forecaster.meta_controller.select_action(state)

        reward = -rmse_pv
        next_state = state.copy()
        done = 1

        trans: Dict = {
            "forecast_start": fs,
            "action": int(action),
            "reward": float(reward),
            "done": int(done),
            "rmse_pv": float(rmse_pv),
            "physics_residual": float(physics_residual),
        }

        for i in range(int(state.size)):
            trans[f"state_{i}"] = float(state[i])
            trans[f"next_state_{i}"] = float(next_state[i])

        transitions.append(trans)

        past_metrics = current_metrics
        stats["saved"] += 1

        if stats["saved"] % 250 == 0:
            LOGGER.info("Replayed %d transitions...", stats["saved"])

    out_df = pd.DataFrame(transitions)
    out_df.to_parquet(output_path, index=False)

    LOGGER.info("Saved replay transitions to %s", str(output_path))
    LOGGER.info("Stats: %s", stats)
    LOGGER.info("Total saved transitions: %d", len(out_df))


if __name__ == "__main__":
    main()
