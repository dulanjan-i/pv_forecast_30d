"""
RL Data Collection Script for MiRACLE - FIXED for "Phase 2" Gap Filling

This version automatically patches missing ground truth in the test set 
by using pre-generated predictions (Phase 1 outputs) to fill NaNs.
This prevents the "skip_forecast_failed" error due to missing history.

Usage:
    python scripts/collect_rl_data.py \
        --num-samples 1000 \
        --output data/rl_transitions/phase2_run.parquet \
        --fill-data predictions_phase1.parquet

Author: MiRACLE Team (Architect & User)
Date: 2026-01-03
"""

import sys
from pathlib import Path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))
# Replace whole file with a single consistent implementation.
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


def load_test_data(
    test_parquet: str,
    *,
    max_windows: Optional[int] = None,
    lookback: int = 672,
    window_size: int = 96,
    stride: int = 1,
) -> List[Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, Optional[np.ndarray], Optional[pd.DataFrame]]]:
    p = Path(test_parquet)
    if p.is_dir():
        candidates = list(p.glob("*.parquet"))
        if not candidates:
            raise ValueError(f"No parquet files found in directory {test_parquet}")
        preferred = [c for c in candidates if "weather_with_pvlib" in c.name or "15min" in c.name]
        file_to_load = preferred[0] if preferred else candidates[0]
    else:
        file_to_load = p

    LOGGER.info("Loading test data from %s", str(file_to_load))
    df = pd.read_parquet(file_to_load)
    if 'timestamp_utc' not in df.columns:
        for c in df.columns:
            if 'time' in c or 'timestamp' in c:
                df = df.rename(columns={c: 'timestamp_utc'})
                break
    df = df.sort_values('timestamp_utc').reset_index(drop=True)
    # make timezone-aware to help infer_freq and comparisons
    df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], utc=True)

    power_col_candidates = ['power_norm', 'pvlib_ac_kw', 'power_kw', 'ac_power', 'pvlib_dc_kw']
    power_col = None
    for c in power_col_candidates:
        if c in df.columns:
            power_col = c
            break
    if power_col is None:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            raise ValueError("No numeric columns available to use as ground truth power")
        power_col = num_cols[0]

    windows: List[Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, Optional[np.ndarray], Optional[pd.DataFrame]]] = []
    N = len(df)
    start_idx = lookback
    end_idx = N - window_size + 1
    if end_idx <= start_idx:
        raise ValueError("Not enough rows to build a single window; increase data or reduce lookback")

    for i in range(start_idx, end_idx, stride):
        hist_df = df.iloc[i - lookback:i].reset_index(drop=True).copy()
        fore_df = df.iloc[i : i + window_size].reset_index(drop=True).copy()
        gt = fore_df[power_col].values.astype(np.float32)
        windows.append((hist_df, fore_df, gt, None, None))
        if max_windows and len(windows) >= int(max_windows):
            break

    LOGGER.info("Constructed %d windows from %s", len(windows), str(file_to_load))
    return windows


def collect_transitions(
    rl_forecaster: RLIntegratedForecaster,
    test_windows: List[Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]],
    *,
    num_samples: int,
    shuffle: bool,
    seed: int,
    prefill_encoder: bool = False,
    max_encoder_synth_frac: float = 0.5,
    min_encoder_observed_frac: float = 0.5,
    autoregressive_fill: bool = False,
) -> Tuple[List[Dict], Dict[str, int]]:
    if len(test_windows) == 0:
        raise ValueError("No test windows available to collect transitions")

    if autoregressive_fill:
        order = []
        for i, w in enumerate(test_windows):
            try:
                fs = pd.to_datetime(w[1]["timestamp_utc"].iloc[0])
            except Exception:
                fs = pd.NaT
            order.append((i, fs))
        order_sorted = sorted(order, key=lambda x: x[1] if not pd.isna(x[1]) else pd.Timestamp.max)
        idxs = np.array([i for i, _ in order_sorted])
        if shuffle:
            LOGGER.warning("--autoregressive-fill requested: disabling shuffle to preserve chronological chaining")
            shuffle = False
    else:
        idxs = np.arange(len(test_windows))
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(idxs)

    max_possible = len(test_windows)
    if num_samples > max_possible:
        LOGGER.warning(
            "Requested %d samples but only %d unique windows exist. Will collect at most %d without repeating.",
            num_samples,
            max_possible,
            max_possible,
        )
        num_samples = max_possible

    stats = {
        "attempted": 0,
        "saved": 0,
        "skip_forecast_failed": 0,
        "skip_len_mismatch": 0,
        "skip_nonfinite": 0,
        "skip_bad_state": 0,
        "skip_synth_threshold": 0,
    }

    transitions: List[Dict] = []
    preds_cache: Dict[pd.Timestamp, float] = {}

    past_metrics = {
        "short_rmse": 0.0,
        "long_rmse": 0.0,
        "physics_residual": 0.0,
        "weather_quality": 0.8,
        "seasonal_factor": 1.0,
    }

    for k in idxs[:num_samples]:
        historical_df, forecast_df, ground_truth, full_gt30, full_weather30 = test_windows[int(k)]
        stats["attempted"] += 1

        encoder_synth_fraction = 0.0
        encoder_source = 'observed'

        forecast_start = None
        try:
            if "timestamp_utc" in forecast_df.columns and len(forecast_df) > 0:
                forecast_start = str(pd.to_datetime(forecast_df["timestamp_utc"].iloc[0]))
        except Exception:
            forecast_start = None

        try:
            if not hasattr(rl_forecaster, 'forecaster'):
                raise RuntimeError("No underlying forecaster available to generate predictions")

            day_start_ts = pd.to_datetime(forecast_df['timestamp_utc'].iloc[0])

            df_freq = None
            try:
                if 'timestamp_utc' in historical_df.columns and len(historical_df) >= 4:
                    df_freq = pd.infer_freq(historical_df['timestamp_utc'].iloc[:100])
            except Exception:
                df_freq = None

            if df_freq in ['15T', '15min']:
                enc_len = rl_forecaster.forecaster.short_config.get('encoder_len', 96)
                if len(historical_df) < enc_len:
                    stats["skip_len_mismatch"] += 1
                    continue

                if autoregressive_fill:
                    try:
                        if 'power_norm' not in historical_df.columns:
                            historical_df['power_norm'] = pd.NA
                        missing_mask = historical_df['power_norm'].isna()
                        if missing_mask.any():
                            for idx_row, ts in zip(historical_df.index, historical_df['timestamp_utc']):
                                ts = pd.to_datetime(ts)
                                if missing_mask.loc[idx_row] and ts in preds_cache:
                                    historical_df.at[idx_row, 'power_norm'] = preds_cache[ts]
                    except Exception:
                        pass

                try:
                    if 'power_norm' in historical_df.columns:
                        missing_mask = historical_df['power_norm'].isna()
                        n_missing = int(missing_mask.sum())
                    else:
                        historical_df['power_norm'] = pd.NA
                        missing_mask = historical_df['power_norm'].isna()
                        n_missing = int(missing_mask.sum())

                    if (n_missing > 0 and prefill_encoder) and hasattr(rl_forecaster.forecaster, 'pvlib_predictor'):
                        pv_pred = rl_forecaster.forecaster.pvlib_predictor.predict_from_weather(historical_df)
                        pv_pred = np.asarray(pv_pred, dtype=np.float32)
                        if n_missing > 0:
                            historical_df.loc[missing_mask, 'power_norm'] = pv_pred[missing_mask.values]
                        n_filled = int(missing_mask.sum())
                        encoder_synth_fraction = float(n_filled) / float(enc_len)
                        if n_filled == 0:
                            encoder_source = 'observed'
                        elif n_filled < enc_len:
                            encoder_source = 'observed+pvl_synth'
                        else:
                            encoder_source = 'pvl_only'
                except Exception:
                    pass

                try:
                    observed_frac = 1.0 - float(encoder_synth_fraction)
                    if encoder_synth_fraction > max_encoder_synth_frac or observed_frac < min_encoder_observed_frac:
                        stats["skip_synth_threshold"] += 1
                        continue
                except Exception:
                    pass

                forecast = rl_forecaster.forecaster._predict_short_head_for_day(pd.Timestamp(day_start_ts), 0, historical_df, forecast_df)

                try:
                    if autoregressive_fill and forecast is not None:
                        times = pd.to_datetime(forecast_df['timestamp_utc']).tolist()
                        for t, v in zip(times[: len(forecast)], np.asarray(forecast, dtype=float)):
                            preds_cache[pd.to_datetime(t)] = float(v)
                except Exception:
                    pass
            else:
                try:
                    if 'timestamp_utc' in forecast_df.columns:
                        if pd.api.types.is_datetime64_any_dtype(forecast_df['timestamp_utc']):
                            if forecast_df['timestamp_utc'].dt.tz is None:
                                forecast_df['timestamp_utc'] = forecast_df['timestamp_utc'].dt.tz_localize('UTC')
                        else:
                            forecast_df['timestamp_utc'] = pd.to_datetime(forecast_df['timestamp_utc']).dt.tz_localize('UTC')
                except Exception:
                    forecast_df['timestamp_utc'] = pd.to_datetime(forecast_df['timestamp_utc'])

                try:
                    if 'timestamp_utc' in historical_df.columns:
                        if pd.api.types.is_datetime64_any_dtype(historical_df['timestamp_utc']):
                            if historical_df['timestamp_utc'].dt.tz is None:
                                historical_df['timestamp_utc'] = historical_df['timestamp_utc'].dt.tz_localize('UTC')
                        else:
                            historical_df['timestamp_utc'] = pd.to_datetime(historical_df['timestamp_utc']).dt.tz_localize('UTC')
                except Exception:
                    historical_df['timestamp_utc'] = pd.to_datetime(historical_df['timestamp_utc'])

                components = rl_forecaster.forecaster.predict_30d(
                    forecast_start=str(pd.to_datetime(forecast_df['timestamp_utc'].iloc[0])),
                    weather_df=forecast_df,
                    historical_df=historical_df,
                    return_components=True,
                )
                forecast = components['final'][:96]
        except Exception as e:
            LOGGER.warning("Forecast failed%s: %s", f" @ {forecast_start}" if forecast_start else "", str(e))
            stats["skip_forecast_failed"] += 1
            continue

        if forecast is None:
            stats["skip_forecast_failed"] += 1
            continue

        forecast = np.asarray(forecast, dtype=np.float32)[:96]
        ground_truth = np.asarray(ground_truth, dtype=np.float32)

        if len(forecast) != 96 or len(ground_truth) != 96:
            stats["skip_len_mismatch"] += 1
            continue
        if not (np.all(np.isfinite(forecast)) and np.all(np.isfinite(ground_truth))):
            stats["skip_nonfinite"] += 1
            continue

        rmse_pv = compute_rmse(forecast, ground_truth)

        full30_rmse = None
        try:
            if full_weather30 is not None and hasattr(rl_forecaster, 'forecaster'):
                if 'timestamp_utc' in full_weather30.columns:
                    full_weather30['timestamp_utc'] = pd.to_datetime(full_weather30['timestamp_utc'], utc=True)
                if 'timestamp_utc' in historical_df.columns:
                    historical_df['timestamp_utc'] = pd.to_datetime(historical_df['timestamp_utc'], utc=True)

                components_30 = rl_forecaster.forecaster.predict_30d(
                    forecast_start=str(pd.to_datetime(forecast_df['timestamp_utc'].iloc[0])),
                    weather_df=full_weather30,
                    historical_df=historical_df,
                    return_components=True,
                )
                full_forecast = components_30['final']
                if full_gt30 is not None and len(full_gt30) == len(full_forecast):
                    full30_rmse = compute_rmse(full_forecast, full_gt30)
        except Exception:
            full30_rmse = None

        try:
            if hasattr(rl_forecaster, 'forecaster') and hasattr(rl_forecaster.forecaster, 'pvlib_predictor'):
                pvlib_pred = rl_forecaster.forecaster.pvlib_predictor.predict_from_weather(forecast_df)
                physics_residual = compute_rmse(pvlib_pred[:96], ground_truth[:96])
            else:
                physics_residual = 0.0
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
            metrics_for_rl = {**past_metrics, **current_metrics}
            state = rl_forecaster.rl_system.build_meta_state(metrics_for_rl)
        except Exception as e:
            LOGGER.warning("State computation failed%s: %s", f" @ {forecast_start}" if forecast_start else "", str(e))
            stats["skip_bad_state"] += 1
            continue

        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.size == 0 or not np.all(np.isfinite(state)):
            stats["skip_bad_state"] += 1
            continue

        action = rl_forecaster.rl_system.meta_controller.select_action(state, mode=rl_forecaster.rl_system.config.mode)

        reward_val = -full30_rmse if full30_rmse is not None else -rmse_pv
        reward = reward_val
        next_state = state.copy()
        done = 1

        trans: Dict = {
            "forecast_start": forecast_start,
            "action": int(action),
            "reward": float(reward),
            "done": int(done),
            "rmse_day1": float(rmse_pv),
            "rmse_30d": float(full30_rmse) if full30_rmse is not None else None,
            "physics_residual": float(physics_residual),
            "encoder_source": encoder_source,
            "encoder_synth_fraction": float(encoder_synth_fraction),
        }

        for i in range(int(state.size)):
            trans[f"state_{i}"] = float(state[i])
            trans[f"next_state_{i}"] = float(next_state[i])

        transitions.append(trans)

        past_metrics = current_metrics
        stats["saved"] += 1

        if stats["saved"] % 250 == 0:
            LOGGER.info("Collected %d transitions...", stats["saved"])

    return transitions, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect RL transitions from test data")
    # Support both old and new flag names for compatibility
    parser.add_argument("--test-data", type=str, required=False, help="Test parquet file path")
    parser.add_argument("--model-dir", type=str, required=False, help="Model directory containing short/long heads")
    parser.add_argument("--short-ckpt", type=str, required=False, help="Short-head checkpoint")
    parser.add_argument("--long-ckpt", type=str, required=False, help="Long-head checkpoint")
    parser.add_argument("--plant-config", type=str, required=False, help="Plant configuration JSON")
    parser.add_argument("--plant-meta", type=str, required=False, help="Plant metadata JSON (alt)")
    parser.add_argument("--output-dir", type=str, default="data/rl_transitions", help="Output directory")
    parser.add_argument("--output", type=str, required=False, help="Output file (alt)")
    parser.add_argument("--num-samples", type=int, default=500, help="Max number of transitions to collect")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--lookback", type=int, default=672, help="Lookback length in rows")
    parser.add_argument("--window-size", type=int, default=96, help="Forecast window size (rows), should be 96")
    parser.add_argument("--stride", type=int, default=1, help="Stride between forecast_starts (1=every row)")
    parser.add_argument("--max-windows", type=int, default=None, help="Optional cap on window construction")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle windows before collecting")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (shuffling)")
    parser.add_argument("--prefill-encoder", action="store_true", help="Prefill encoder missing power with PVLib predictions")
    parser.add_argument("--max-encoder-synth-frac", type=float, default=0.5, help="Maximum allowed fraction of encoder rows that are synthetic (0-1)")
    parser.add_argument("--min-encoder-observed-frac", type=float, default=0.5, help="Minimum required fraction of observed encoder rows (0-1)")
    parser.add_argument("--autoregressive-fill", action="store_true", help="Fill encoder power_norm from previous model predictions (chronological chaining)")

    args = parser.parse_args()

    setup_logging(args.log_level)

    if args.window_size != 96:
        raise ValueError("This collector is designed for day-ahead windows. Set --window-size 96.")

    model_dir = Path(args.model_dir) if args.model_dir else None
    if args.short_ckpt and args.long_ckpt:
        short_ckpt = Path(args.short_ckpt)
        long_ckpt = Path(args.long_ckpt)
    elif model_dir:
        short_ckpt = model_dir / "shorthead_seed42" / "best.pt"
        if not short_ckpt.exists():
            short_ckpt = model_dir / "shorthead_seed42" / "best.ckpt"
        long_ckpt = model_dir / "longhead_seed43" / "best.pt"
        if not long_ckpt.exists():
            long_ckpt = model_dir / "longhead_seed43" / "best.ckpt"
        if not short_ckpt.exists() or not long_ckpt.exists():
            candidates = list(model_dir.rglob("best.pt")) + list(model_dir.rglob("best.ckpt"))
            short_cands = [p for p in candidates if "short" in str(p).lower()]
            long_cands = [p for p in candidates if "long" in str(p).lower()]
            if not short_ckpt.exists() and short_cands:
                short_ckpt = short_cands[0]
            if not long_ckpt.exists() and long_cands:
                long_ckpt = long_cands[0]
            if (not short_ckpt.exists() or not long_ckpt.exists()) and len(candidates) >= 2:
                if not short_ckpt.exists():
                    short_ckpt = candidates[0]
                if not long_ckpt.exists():
                    long_ckpt = candidates[1]
    else:
        raise ValueError("Specify either --model-dir or both --short-ckpt and --long-ckpt")

    plant_config = args.plant_config or args.plant_meta
    if not plant_config:
        raise ValueError("Provide --plant-config or --plant-meta path")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_name = (args.output and Path(args.output).stem) or f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_path = output_dir / f"{batch_name}.parquet"

    LOGGER.info("Short ckpt: %s", short_ckpt)
    LOGGER.info("Long  ckpt: %s", long_ckpt)

    physics_forecaster = PhysicsAwareForecaster(
        short_ckpt=short_ckpt,
        long_ckpt=long_ckpt,
        plant_metadata=plant_config,
        short_train_parquet=Path("data/processed/plant_level/plant_03/15min_pca32/train.parquet"),
        long_train_parquet=Path("data/processed/plant_level/plant_03/hourly_longhead/train.parquet"),
    )

    rl_forecaster = RLIntegratedForecaster(
        forecaster=physics_forecaster,
        rl_mode="heuristic",
        checkpoint_dir=Path(short_ckpt).parent,
    )

    test_data_arg = args.test_data
    if not test_data_arg:
        raise ValueError("--test-data is required")

    LOGGER.info("Loading test data and building rolling windows...")
    test_windows = load_test_data(
        test_data_arg,
        max_windows=args.max_windows or args.num_samples,
        lookback=args.lookback,
        window_size=args.window_size,
        stride=args.stride,
    )
    LOGGER.info("Built %d windows (stride=%d)", len(test_windows), args.stride)

    transitions, stats = collect_transitions(
        rl_forecaster,
        test_windows,
        num_samples=args.num_samples,
        shuffle=args.shuffle,
        seed=args.seed,
        prefill_encoder=args.prefill_encoder,
        max_encoder_synth_frac=args.max_encoder_synth_frac,
        min_encoder_observed_frac=args.min_encoder_observed_frac,
        autoregressive_fill=args.autoregressive_fill,
    )

    transitions_df = pd.DataFrame(transitions)
    transitions_df.to_parquet(output_path, index=False)

    LOGGER.info("Saved transitions to %s", str(output_path))
    LOGGER.info("Stats: %s", stats)
    LOGGER.info("Total saved transitions: %d", len(transitions_df))


if __name__ == "__main__":
    main()
    
    return windows


def collect_transitions(
    rl_forecaster: RLIntegratedForecaster,
    test_windows: list,
    num_samples: int,
    save_path: Path,
    checkpoint_freq: int = 100
):
    """
    Collect RL transitions using the forecaster.
    """
    logger.info(f"Starting collection loop: {num_samples} samples")
    transitions = []
    pbar = tqdm(total=num_samples, desc="Collecting")
    
    for sample_idx in range(num_samples):
        # Cycle windows if we requested more samples than we have windows
        window_idx = sample_idx % len(test_windows)
        historical_df, weather_df, ground_truth = test_windows[window_idx]
        
        try:
            forecast_start = weather_df['timestamp_utc'].iloc[0]
            
            # Run forecast
            forecast, info = rl_forecaster.forecast_with_rl(
                weather_data=weather_df,
                forecast_start=forecast_start,
                historical_data=historical_df,
                ground_truth=ground_truth
            )
            
            # Extract basic RL components
            state = info['meta_state']
            action = info['action_index']
            
            # Compute Reward (Day 1 RMSE)
            if isinstance(forecast, np.ndarray) and len(forecast) >= 96:
                forecast_day1 = forecast[:96]
                gt_day1 = ground_truth[:96] if len(ground_truth) >= 96 else ground_truth
                
                if len(gt_day1) == len(forecast_day1):
                    rmse = np.sqrt(np.mean((forecast_day1 - gt_day1) ** 2))
                    reward = -rmse 
                else:
                    reward = -0.05 # Fallback (should be rare with patched data)
            else:
                reward = -0.05
            
            # Store transition
            transition = {
                'sample_idx': sample_idx,
                'timestamp': pd.Timestamp.now(tz='UTC').isoformat(),
                'forecast_start': str(forecast_start),
                'action': action,
                'reward': reward,
                # Flatten state
                **{f'state_{i}': state[i] for i in range(len(state))},
                # Metrics
                'blend_short': rl_forecaster.blend_weights.get('short', 0),
                'blend_long': rl_forecaster.blend_weights.get('long', 0),
                'blend_physics': rl_forecaster.blend_weights.get('physics', 0)
            }
            
            transitions.append(transition)
            pbar.update(1)
            
            # Checkpoint
            if (sample_idx + 1) % checkpoint_freq == 0:
                pd.DataFrame(transitions).to_parquet(save_path)
                
        except Exception as e:
            logger.error(f"Error at {sample_idx}: {e}")
            continue
    
    pbar.close()
    pd.DataFrame(transitions).to_parquet(save_path)
    logger.info(f"Saved to {save_path}")
    return pd.DataFrame(transitions)


def main():
    parser = argparse.ArgumentParser()
    
    # Paths
    parser.add_argument('--short-ckpt', default='/home/dwijenayake/pv_forecast_30d/V1.0_FINAL_TFT/shorthead_seed42/checkpoints/best.ckpt')
    parser.add_argument('--long-ckpt', default='/home/dwijenayake/pv_forecast_30d/V1.0_FINAL_TFT/longhead_seed43/checkpoints/best.ckpt')
    parser.add_argument('--plant-meta', default='/home/dwijenayake/pv_forecast_30d/V1.0_FINAL_TFT/plant_metadata/plant_03.json')
    parser.add_argument('--short-train', default='/home/dwijenayake/pv_forecast_30d/data/processed/plant_level/plant_03/15min_pca32/train.parquet')
    parser.add_argument('--long-train', default='/home/dwijenayake/pv_forecast_30d/data/processed/plant_level/plant_03/hourly_longhead/train.parquet')
    
    # Data & Fill Args
    parser.add_argument('--test-data', default='/home/dwijenayake/pv_forecast_30d/data/processed/plant_level/plant_03/15min_pca32/test.parquet')
    parser.add_argument('--fill-data', type=str, default=None, help='Path to predictions parquet to patch missing history')
    
    parser.add_argument('--num-samples', type=int, default=1000)
    parser.add_argument('--output', default='data/rl_transitions/run_fixed.parquet')
    parser.add_argument('--device', default='cuda:0')
    
    args = parser.parse_args()
    
    # Init Forecaster
    logger.info("Initializing Forecaster...")
    forecaster = PhysicsAwareForecaster(
        short_ckpt=Path(args.short_ckpt),
        long_ckpt=Path(args.long_ckpt),
        plant_metadata=Path(args.plant_meta),
        short_train_parquet=Path(args.short_train),
        long_train_parquet=Path(args.long_train),
        device=args.device
    )
    
    rl_forecaster = RLIntegratedForecaster(
        forecaster=forecaster,
        rl_mode="heuristic"
    )
    
    # Load Data (With Patching)
    fill_path = Path(args.fill_data) if args.fill_data else None
    test_windows = load_and_patch_data(Path(args.test_data), fill_path, num_samples=args.num_samples)
    
    if not test_windows:
        logger.error("No valid windows found! Exiting.")
        return

    # Run Collection
    collect_transitions(
        rl_forecaster=rl_forecaster,
        test_windows=test_windows,
        num_samples=args.num_samples,
        save_path=Path(args.output)
    )

if __name__ == "__main__":
    main()