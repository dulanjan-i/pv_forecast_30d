# src/inference/phase1_inference_pipeline.py
"""
Phase 1 inference pipeline (rolling 30d forecasts).

Key guarantees (so it stops exploding):
- All timestamp_utc columns are forced to tz-aware UTC before comparisons.
- Adds --stride-days argument.
- If power_norm is missing in encoder context, we synthesize it from pvlib_ac_kw / installed_capacity_kw.
- Skips incomplete windows instead of crashing.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import torch

from src.inference.physics_aware_forecaster import PhysicsAwareForecaster

LOGGER = logging.getLogger("phase1_inference_pipeline")


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def to_utc(series: pd.Series) -> pd.Series:
    """Force a datetime-like series to tz-aware UTC."""
    s = pd.to_datetime(series, errors="coerce")
    if getattr(s.dt, "tz", None) is None:
        # naive -> localize as UTC
        return s.dt.tz_localize("UTC")
    return s.dt.tz_convert("UTC")


def ensure_timestamp_utc(df: pd.DataFrame, col: str = "timestamp_utc") -> pd.DataFrame:
    if col not in df.columns:
        raise KeyError(f"Missing required time column: {col}")
    df = df.copy()
    df[col] = to_utc(df[col])
    return df


def ensure_plant_onehot(df: pd.DataFrame, plant_id: str) -> pd.DataFrame:
    df = df.copy()
    cols = ["plant_01", "plant_02", "plant_03", "plant_05", "plant_06"]
    for c in cols:
        if c not in df.columns:
            df[c] = 0
    if plant_id in cols:
        df[plant_id] = 1
    return df


def synth_power_norm_from_pvlib(df: pd.DataFrame, installed_capacity_kw: float) -> pd.DataFrame:
    """
    If power_norm is absent or mostly NaN, synthesize from pvlib_ac_kw/capacity.
    This is only for encoder bootstrapping when you do not have real PV measurements.
    """
    df = df.copy()
    if "pvlib_ac_kw" not in df.columns:
        # cannot synthesize
        return df

    pn = None
    if "power_norm" in df.columns:
        pn = pd.to_numeric(df["power_norm"], errors="coerce")

    need = True
    if pn is not None:
        nonnull_frac = float(pn.notna().mean())
        if nonnull_frac >= 0.8:
            need = False

    if need:
        cap = float(installed_capacity_kw) if installed_capacity_kw and installed_capacity_kw > 0 else 1.0
        pv = pd.to_numeric(df["pvlib_ac_kw"], errors="coerce").fillna(0.0)
        df["power_norm"] = (pv / cap).clip(0.0, 1.5)

    return df


@dataclass
class Paths:
    phase_dir: Path
    out_path: Path
    plant_meta: Path
    short_ckpt: Path
    long_ckpt: Path
    short_train: Path
    long_train: Path
    hist_encoder: Optional[Path]


def load_plant_meta(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_encoder_context(
    forecast_start_utc: pd.Timestamp,
    *,
    hist_encoder_df: Optional[pd.DataFrame],
    weather_df: pd.DataFrame,
    installed_capacity_kw: float,
    plant_id: str,
    lookback_steps: int = 672,
) -> pd.DataFrame:
    """
    Prefer hist_encoder_df (should contain power_norm).
    Fallback: build encoder from weather_df and synthesize power_norm from pvlib.
    """
    encoder_end = forecast_start_utc
    encoder_start = encoder_end - pd.Timedelta(minutes=15 * lookback_steps)

    if hist_encoder_df is not None:
        h = hist_encoder_df
        w = h[(h["timestamp_utc"] >= encoder_start) & (h["timestamp_utc"] < encoder_end)].copy()
        if len(w) >= lookback_steps:
            w = w.sort_values("timestamp_utc").tail(lookback_steps).copy()
            w = ensure_plant_onehot(w, plant_id)
            w = synth_power_norm_from_pvlib(w, installed_capacity_kw)
            return w

    # fallback: weather-based encoder
    w = weather_df[(weather_df["timestamp_utc"] >= encoder_start) & (weather_df["timestamp_utc"] < encoder_end)].copy()
    w = w.sort_values("timestamp_utc").tail(lookback_steps).copy()
    w = ensure_plant_onehot(w, plant_id)
    w = synth_power_norm_from_pvlib(w, installed_capacity_kw)
    return w


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 rolling inference pipeline")
    parser.add_argument("--weather-source", choices=["historical", "api"], default="historical")
    parser.add_argument("--start-date", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--stride-days", type=int, default=1, help="Rolling stride in days (default 1)")
    parser.add_argument("--phase1-dir", type=str, default="data/processed/test_phase1_dec2023_dec2024")
    parser.add_argument("--out", type=str, default=None, help="Output predictions parquet path")

    parser.add_argument("--plant-meta", type=str, default="V1.0_FINAL_TFT/plant_metadata/plant_03.json")
    parser.add_argument("--short-ckpt", type=str, default="V1.0_FINAL_TFT/shorthead_seed42/checkpoints/best.ckpt")
    parser.add_argument("--long-ckpt", type=str, default="V1.0_FINAL_TFT/longhead_seed43/checkpoints/best.ckpt")

    parser.add_argument("--short-train", type=str, default="data/processed/plant_level/plant_03/15min_pca32/train.parquet")
    parser.add_argument("--long-train", type=str, default="data/processed/plant_level/plant_03/hourly_longhead/train.parquet")

    parser.add_argument("--hist-encoder", type=str, default="data/processed/plant_level/plant_03/hist_weather_gt_15min_utc.parquet")
    parser.add_argument("--log-level", type=str, default="INFO")

    args = parser.parse_args()
    setup_logging(args.log_level)

    start_utc = pd.Timestamp(args.start_date, tz="UTC")
    end_utc = pd.Timestamp(args.end_date, tz="UTC")
    stride_days = int(args.stride_days)

    phase_dir = Path(args.phase1_dir)
    out_path = Path(args.out) if args.out else (phase_dir / "predictions_phase1.parquet")

    paths = Paths(
        phase_dir=phase_dir,
        out_path=out_path,
        plant_meta=Path(args.plant_meta),
        short_ckpt=Path(args.short_ckpt),
        long_ckpt=Path(args.long_ckpt),
        short_train=Path(args.short_train),
        long_train=Path(args.long_train),
        hist_encoder=Path(args.hist_encoder) if args.hist_encoder else None,
    )

    plant = load_plant_meta(paths.plant_meta)
    plant_id = str(plant.get("plant_id", "plant_03"))
    installed_capacity_kw = float(plant.get("installed_capacity_kw", 1.0))

    # Load weather (historical mode reads your prepared parquet)
    if args.weather_source != "historical":
        raise NotImplementedError("api mode is not wired in this drop-in file. Run historical first, then we wire api safely.")

    weather_path = paths.phase_dir / "weather_with_pvlib_15min.parquet"
    if not weather_path.exists():
        raise FileNotFoundError(f"Missing weather parquet: {weather_path}")

    weather_15min = pd.read_parquet(weather_path)
    weather_15min = ensure_timestamp_utc(weather_15min, "timestamp_utc").sort_values("timestamp_utc")
    weather_15min = ensure_plant_onehot(weather_15min, plant_id)

    hist_encoder_df: Optional[pd.DataFrame] = None
    if paths.hist_encoder and paths.hist_encoder.exists():
        hist_encoder_df = pd.read_parquet(paths.hist_encoder)
        hist_encoder_df = ensure_timestamp_utc(hist_encoder_df, "timestamp_utc").sort_values("timestamp_utc")
        hist_encoder_df = ensure_plant_onehot(hist_encoder_df, plant_id)

    forecaster = PhysicsAwareForecaster(
        short_ckpt=paths.short_ckpt,
        long_ckpt=paths.long_ckpt,
        plant_metadata=str(paths.plant_meta),
        short_train_parquet=paths.short_train,
        long_train_parquet=paths.long_train,
    )

    # Determine last valid start such that we have 2880 steps available
    max_ts = weather_15min["timestamp_utc"].max()
    latest_start = (max_ts - pd.Timedelta(minutes=15 * 2879)).floor("D")
    latest_start = pd.Timestamp(latest_start).tz_convert("UTC")

    run_end = min(end_utc, latest_start)

    LOGGER.info("Start: %s", start_utc)
    LOGGER.info("End:   %s", end_utc)
    LOGGER.info("Stride: %d days", stride_days)
    LOGGER.info("Weather coverage max_ts: %s", max_ts)
    LOGGER.info("Latest valid forecast_start: %s", latest_start)
    LOGGER.info("Will run until: %s", run_end)

    forecast_starts = pd.date_range(start_utc, run_end, freq=f"{stride_days}D", tz="UTC")

    rows: List[Dict[str, Any]] = []
    skipped_incomplete = 0
    total = 0

    for fs in forecast_starts:
        total += 1
        window_end = fs + pd.Timedelta(minutes=15 * 2880)
        w = weather_15min[(weather_15min["timestamp_utc"] >= fs) & (weather_15min["timestamp_utc"] < window_end)].copy()
        w = w.sort_values("timestamp_utc")

        if len(w) != 2880:
            skipped_incomplete += 1
            continue

        enc = build_encoder_context(
            fs,
            hist_encoder_df=hist_encoder_df,
            weather_df=weather_15min,
            installed_capacity_kw=installed_capacity_kw,
            plant_id=plant_id,
            lookback_steps=672,
        )

        preds = forecaster.predict_30d(
            forecast_start=str(fs),
            weather_df=w,
            historical_df=enc,
            return_components=False,
        )

        if torch.is_tensor(preds):
            preds = preds.detach().cpu().numpy()
        preds = np.asarray(preds, dtype=np.float32).reshape(-1)
        if len(preds) != 2880:
            skipped_incomplete += 1
            continue

        for i in range(2880):
            rows.append(
                {
                    "timestamp_utc": w["timestamp_utc"].iloc[i],
                    "forecast_start": fs,
                    "step_ahead": i,
                    "hours_ahead": float(i) * 0.25,
                    "predicted_power_norm": float(preds[i]),
                }
            )

        if total % 25 == 0:
            LOGGER.info("Progress: %d/%d starts processed", total, len(forecast_starts))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(rows)
    out_df.to_parquet(out_path, index=False)

    LOGGER.info("WROTE: %s", str(out_path))
    LOGGER.info("forecast_starts attempted: %d", len(forecast_starts))
    LOGGER.info("skipped incomplete windows: %d", skipped_incomplete)
    LOGGER.info("rows written: %d", len(out_df))


if __name__ == "__main__":
    main()
