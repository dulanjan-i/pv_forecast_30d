#!/usr/bin/env python3
"""
Phase 2 wrapper: reuse Phase1 pipeline pieces to run 2025 forecasts

This wrapper calls the key Phase1 steps (weather fetch + preprocess + rolling
inference) but bootstraps encoder context from an existing Phase 1
predictions file (15-min). It writes outputs to a separate Phase 2 folder.
"""
import argparse
import sys
from pathlib import Path
import shutil
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description='Phase 2 wrapper (bootstrap from Phase 1 tail)')
    parser.add_argument('--start-date', type=str, default='2025-01-01')
    parser.add_argument('--end-date', type=str, default='2025-12-31')
    parser.add_argument('--weather-source', choices=['historical', 'api'], default='historical')
    parser.add_argument('--phase1-preds', type=str,
                        default='data/processed/test_phase1_dec2023_dec2024/predictions_phase1.parquet',
                        help='Path to Phase 1 predictions parquet (15-min) to use as encoder bootstrap')
    parser.add_argument('--out-dir', type=str, default='data/processed/test_phase2_2025', help='Output dir for Phase 2')
    args = parser.parse_args()

    base = Path(__file__).resolve().parents[2]
    # Ensure project root is on sys.path so `src` package imports work
    sys.path.insert(0, str(base))
    phase1_preds = Path(args.phase1_preds)
    out_dir = base / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # If Phase1 preds exist, compute Phase2 start = last Phase1 timestep + 15min
    p1_tmp = None
    start_date_str = args.start_date
    if phase1_preds.exists():
        try:
            p1_tmp = pd.read_parquet(phase1_preds)
            if 'timestamp_utc' in p1_tmp.columns:
                p1_tmp['timestamp_utc'] = pd.to_datetime(p1_tmp['timestamp_utc'], utc=True)
            elif 'timestamp' in p1_tmp.columns:
                p1_tmp = p1_tmp.rename(columns={'timestamp': 'timestamp_utc'})
                p1_tmp['timestamp_utc'] = pd.to_datetime(p1_tmp['timestamp_utc'], utc=True)
            last_ts = p1_tmp['timestamp_utc'].max()
            start_date_computed = last_ts + pd.Timedelta(minutes=15)
            start_date_str = start_date_computed.isoformat()
            print(f"[INFO] Computed Phase2 start from Phase1 last timestep: {last_ts} -> {start_date_str}")
        except Exception as e:
            print(f"[WARN] Could not compute Phase2 start from Phase1 preds: {e}; using {args.start_date}")

    # Import Phase1 pipeline class (we reuse its steps)
    from src.inference.phase1_inference_pipeline import Phase1Pipeline
    from src.inference.physics_aware_forecaster import PhysicsAwareForecaster
    import torch

    # Instantiate pipeline but redirect outputs to our Phase2 out dir
    # instantiate pipeline with computed start_date (will ensure weather fetched to args.end_date)
    pipeline = Phase1Pipeline(weather_source=args.weather_source, start_date=start_date_str, end_date=args.end_date)
    pipeline.phase1_dir = out_dir

    # Step 1: fetch weather (will save into out_dir)
    weather_15min = pipeline.step1_fetch_weather()

    # Step 2: preprocess
    weather_15min, weather_hourly = pipeline.step2_preprocess_weather(weather_15min)

    # Quick safety-fills: ensure no NaN/Inf remain in key weather fields that break TFT encoding
    safe_fill_cols = ['cloud_cover', 'shortwave_radiation_instant', 'pvlib_poa_global', 'pvlib_ac_kw']
    for col in safe_fill_cols:
        if col in weather_15min.columns:
            weather_15min[col] = weather_15min[col].fillna(0.0).replace([float('inf'), -float('inf')], 0.0)
        if col in weather_hourly.columns:
            weather_hourly[col] = weather_hourly[col].fillna(0.0).replace([float('inf'), -float('inf')], 0.0)

    # Prepare encoder context from Phase1 predictions (preferred)
    if phase1_preds.exists():
        # reuse p1_tmp if we already loaded it above
        if p1_tmp is not None:
            p1 = p1_tmp
        else:
            p1 = pd.read_parquet(phase1_preds)
        # Normalize timestamps: ensure tz-aware UTC for comparisons with weather (which is UTC)
        if 'timestamp_utc' in p1.columns:
            p1['timestamp_utc'] = pd.to_datetime(p1['timestamp_utc'], utc=True)
        if 'timestamp' in p1.columns and 'timestamp_utc' not in p1.columns:
            p1 = p1.rename(columns={'timestamp': 'timestamp_utc'})
            p1['timestamp_utc'] = pd.to_datetime(p1['timestamp_utc'], utc=True)
        if 'predicted_power_norm' in p1.columns and 'power_norm' not in p1.columns:
            p1['power_norm'] = p1['predicted_power_norm']
        p1 = p1.sort_values('timestamp_utc').reset_index(drop=True)
        # Require Phase1 predictions to explicitly cover the 7-day encoder window
        encoder_end = pd.Timestamp(start_date_str, tz='UTC')
        encoder_start = encoder_end - pd.Timedelta(days=7)
        # select rows from p1 that fall into the encoder window
        p1_window = p1[(p1.timestamp_utc >= encoder_start) & (p1.timestamp_utc < encoder_end)].copy()
        if len(p1_window) == 672:
            # Merge Phase1 predicted power with the weather dataset so encoder has full feature set
            if 'plant_id' not in p1_window.columns:
                p1_window['plant_id'] = 'plant_03'
            encoder_long_15min = p1_window.reset_index(drop=True).merge(
                weather_15min,
                on=['timestamp_utc', 'plant_id'],
                how='left',
                suffixes=('', '_w')
            )
            # Ensure power_norm from Phase1 preds takes precedence
            if 'predicted_power_norm' in p1_window.columns and 'power_norm' not in encoder_long_15min.columns:
                encoder_long_15min['power_norm'] = p1_window['predicted_power_norm'].values
            # Safety fill any NaN/Inf from the merge for weather columns
            for col in ['cloud_cover', 'shortwave_radiation_instant', 'pvlib_poa_global', 'pvlib_ac_kw']:
                if col in encoder_long_15min.columns:
                    encoder_long_15min[col] = encoder_long_15min[col].fillna(0.0).replace([float('inf'), -float('inf')], 0.0)
            encoder_short = encoder_long_15min.tail(96).copy()
            # Save for traceability
            encoder_short.to_parquet(out_dir / 'encoder_context_short.parquet', index=False)
            encoder_long_15min.to_parquet(out_dir / 'encoder_context_long_15min.parquet', index=False)
        else:
            # Phase1 preds insufficient — fall back to weather-derived encoder context
            pred_end = p1.timestamp_utc.max()
            print(f"[WARN] Phase1 predictions do not fully cover encoder window ({encoder_start} → {encoder_end}), have end={pred_end}, rows={len(p1)}; falling back to weather")
            weather_path = out_dir / "weather_with_pvlib_15min.parquet"
            if not weather_path.exists():
                raise RuntimeError(f'Cannot build encoder context: weather file missing: {weather_path}')
            weather_15min = pd.read_parquet(weather_path)
            encoder_long_15min = weather_15min[
                (weather_15min.timestamp_utc >= encoder_start) &
                (weather_15min.timestamp_utc < encoder_end)
            ].copy()
            if len(encoder_long_15min) == 0:
                raise RuntimeError(f'No weather found for encoder window: {encoder_start} → {encoder_end}')
            # Create proxy power_norm from pvlib baseline if missing
            if 'power_norm' not in encoder_long_15min.columns:
                if 'pvlib_ac_kw' in encoder_long_15min.columns:
                    cap = pipeline.metadata.get('installed_capacity_kw', 1.0)
                    encoder_long_15min['power_norm'] = (encoder_long_15min['pvlib_ac_kw'] / cap).clip(0.0, 1.0)
                else:
                    encoder_long_15min['power_norm'] = 0.0

            encoder_long_15min['power_norm'] = encoder_long_15min['power_norm'].fillna(0.0).replace([float('inf'), -float('inf')], 0.0)
            encoder_short = encoder_long_15min.tail(96).copy()
            encoder_short.to_parquet(out_dir / 'encoder_context_short.parquet', index=False)
            encoder_long_15min.to_parquet(out_dir / 'encoder_context_long_15min.parquet', index=False)
    else:
        raise FileNotFoundError(f'Phase 1 predictions not found at: {phase1_preds}')

    # Initialize forecaster (same as Phase1)
    forecaster = PhysicsAwareForecaster(
        short_ckpt=str(pipeline.base_dir / "V1.0_FINAL_TFT/shorthead_seed42/checkpoints/best.ckpt"),
        long_ckpt=str(pipeline.base_dir / "V1.0_FINAL_TFT/longhead_seed43/checkpoints/best.ckpt"),
        plant_metadata=str(pipeline.base_dir / "V1.0_FINAL_TFT/plant_metadata/plant_03.json"),
        short_train_parquet=str(pipeline.base_dir / "data/processed/plant_level/plant_03/15min_pca32/train.parquet"),
        long_train_parquet=str(pipeline.base_dir / "data/processed/plant_level/plant_03/hourly_longhead/train.parquet"),
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Run rolling inference using the prepared encoder contexts
    predictions_df = pipeline.step4_run_rolling_inference(
        weather_15min=weather_15min,
        weather_hourly=weather_hourly,
        initial_encoder_short=encoder_short,
        initial_encoder_long=encoder_long_15min,
        forecaster=forecaster
    )

    # Move/rename output file to a Phase2-appropriate name
    saved = out_dir / 'predictions_phase1.parquet'
    if saved.exists():
        target = out_dir / 'predictions_phase2.parquet'
        shutil.move(str(saved), str(target))
        print(f'✓ Phase 2 predictions saved: {target}')
    else:
        # If the function returned a DataFrame but didn't save, save here
        target = out_dir / 'predictions_phase2.parquet'
        predictions_df.to_parquet(target, index=False)
        print(f'✓ Phase 2 predictions written: {target}')


if __name__ == '__main__':
    main()
