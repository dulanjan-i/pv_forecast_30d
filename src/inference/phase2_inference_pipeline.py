# src/inference/phase2_inference_pipeline.py
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

# Allow running as a script: `python src/inference/phase2_inference_pipeline.py ...`
# without "ModuleNotFoundError: No module named 'src'"
if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]  # .../pv_forecast_30d
    sys.path.insert(0, str(repo_root))

from src.inference.physics_aware_forecaster import PhysicsAwareForecaster  # noqa: E402


LOGGER = logging.getLogger("phase2_inference_pipeline")


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
    weather_15min: Path


def load_plant_meta(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _as_utc_midnight(s: str) -> pd.Timestamp:
    # Accept YYYY-MM-DD
    t = pd.to_datetime(s, utc=True)
    # Normalize to midnight UTC
    return pd.Timestamp(year=t.year, month=t.month, day=t.day, tz="UTC")


def _ensure_ts_utc(df: pd.DataFrame, col: str = "timestamp_utc") -> pd.DataFrame:
    out = df.copy()
    out[col] = pd.to_datetime(out[col], utc=True, errors="coerce")
    out = out.dropna(subset=[col])
    return out


def _window_15min(weather_15m: pd.DataFrame, fs: pd.Timestamp) -> pd.DataFrame:
    # 30 days at 15-min resolution = 2880 steps (30*24*4)
    end = fs + pd.Timedelta(days=30) - pd.Timedelta(minutes=15)
    w = weather_15m[(weather_15m["timestamp_utc"] >= fs) & (weather_15m["timestamp_utc"] <= end)]
    return w


def _hist_window(hist: pd.DataFrame, fs: pd.Timestamp, days: int = 7) -> pd.DataFrame:
    start = fs - pd.Timedelta(days=days)
    h = hist[(hist["timestamp_utc"] >= start) & (hist["timestamp_utc"] < fs)].copy()
    if len(h) == 0:
        # fallback: just take last 672 rows before fs (7 days * 24h * 4 = 672)
        h2 = hist[hist["timestamp_utc"] < fs].tail(days * 24 * 4).copy()
        return h2
    return h


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="Phase 2 daily rolling inference (typically 2025).")

    parser.add_argument("--weather-source", choices=["historical", "api"], default="historical")
    parser.add_argument("--start-date", type=str, default="2024-12-29")
    parser.add_argument("--end-date", type=str, default="2025-12-31")
    parser.add_argument("--stride-days", type=int, default=1)

    parser.add_argument(
        "--phase-dir",
        type=str,
        default="data/processed/test_phase2_2025",
        help="Folder containing phase2 weather parquet(s) and where outputs can be written.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Output parquet path. If empty, writes <phase-dir>/predictions_phase2_daily.parquet",
    )

    # Model and metadata paths
    parser.add_argument("--plant-meta", type=str, default="V1.0_FINAL_TFT/plant_metadata/plant_03.json")
    parser.add_argument("--short-ckpt", type=str, required=True)
    parser.add_argument("--long-ckpt", type=str, required=True)
    parser.add_argument("--short-train", type=str, default="data/processed/plant_level/plant_03/15min_pca32/train.parquet")
    parser.add_argument("--long-train", type=str, default="data/processed/plant_level/plant_03/hourly_longhead/train.parquet")

    # Encoder context (optional but recommended)
    parser.add_argument(
        "--hist-encoder",
        type=str,
        default="data/processed/plant_level/plant_03/hist_weather_gt_15min_utc.parquet",
        help="15-min historical context parquet that includes the same features as training (usually includes power_norm).",
    )

    # Weather parquet (15-min with pvlib columns already present)
    parser.add_argument(
        "--weather-15min",
        type=str,
        default="",
        help="15-min weather parquet with pvlib columns. If empty, uses <phase-dir>/weather_with_pvlib_15min.parquet",
    )

    args = parser.parse_args()

    phase_dir = Path(args.phase_dir)
    phase_dir.mkdir(parents=True, exist_ok=True)

    out_path = Path(args.out) if args.out else (phase_dir / "predictions_phase2_daily.parquet")
    weather_15min_path = Path(args.weather_15min) if args.weather_15min else (phase_dir / "weather_with_pvlib_15min.parquet")

    paths = Paths(
        phase_dir=phase_dir,
        out_path=out_path,
        plant_meta=Path(args.plant_meta),
        short_ckpt=Path(args.short_ckpt),
        long_ckpt=Path(args.long_ckpt),
        short_train=Path(args.short_train),
        long_train=Path(args.long_train),
        hist_encoder=Path(args.hist_encoder) if args.hist_encoder else None,
        weather_15min=weather_15min_path,
    )

    plant = load_plant_meta(paths.plant_meta)
    plant_id = str(plant.get("plant_id", "plant_03"))

    LOGGER.info("plant_id: %s", plant_id)
    LOGGER.info("phase_dir: %s", paths.phase_dir)
    LOGGER.info("out_path: %s", paths.out_path)
    LOGGER.info("weather_15min: %s", paths.weather_15min)

    if not paths.weather_15min.exists():
        raise FileNotFoundError(f"Missing weather parquet: {paths.weather_15min}")

    # Load weather
    weather_15m = pd.read_parquet(paths.weather_15min)
    weather_15m = _ensure_ts_utc(weather_15m, "timestamp_utc")
    weather_15m = weather_15m.sort_values("timestamp_utc").drop_duplicates("timestamp_utc")

    max_ts = weather_15m["timestamp_utc"].max()
    min_ts = weather_15m["timestamp_utc"].min()
    LOGGER.info("Weather coverage: %s -> %s (%d rows)", min_ts, max_ts, len(weather_15m))

    # Load historical encoder context if provided
    hist_df = None
    if paths.hist_encoder and paths.hist_encoder.exists():
        hist_df = pd.read_parquet(paths.hist_encoder)
        hist_df = _ensure_ts_utc(hist_df, "timestamp_utc")
        hist_df = hist_df.sort_values("timestamp_utc").drop_duplicates("timestamp_utc")
        LOGGER.info("Hist encoder coverage: %s -> %s (%d rows)", hist_df["timestamp_utc"].min(), hist_df["timestamp_utc"].max(), len(hist_df))
    else:
        LOGGER.warning("hist-encoder missing or not provided, will fallback to training parquet inside forecaster.")

    # Init forecaster once
    forecaster = PhysicsAwareForecaster(
        short_ckpt=paths.short_ckpt,
        long_ckpt=paths.long_ckpt,
        plant_metadata=paths.plant_meta,
        short_train_parquet=paths.short_train,
        long_train_parquet=paths.long_train,
    )

    start = _as_utc_midnight(args.start_date)
    end = _as_utc_midnight(args.end_date)
    stride = int(args.stride_days)

    LOGGER.info("Start: %s", start)
    LOGGER.info("End:   %s", end)
    LOGGER.info("Stride: %d days", stride)

    attempted = 0
    skipped = 0
    rows = []

    fs = start
    while fs <= end:
        attempted += 1

        w = _window_15min(weather_15m, fs)
        if len(w) != 2880:
            skipped += 1
            fs = fs + pd.Timedelta(days=stride)
            continue

        h = None
        if hist_df is not None:
            h = _hist_window(hist_df, fs, days=7)

        try:
            out = forecaster.predict_30d(weather_df=w, historical_df=h, forecast_start=fs)
            y = out["final"]
            if y is None or len(y) != 2880:
                skipped += 1
                fs = fs + pd.Timedelta(days=stride)
                continue

            # Flatten to rows
            base = fs
            for step in range(2880):
                ts = base + pd.Timedelta(minutes=15 * step)
                rows.append(
                    {
                        "timestamp_utc": ts,
                        "forecast_start": fs,
                        "step_ahead": step,
                        "hours_ahead": step * 0.25,
                        "predicted_power_norm": float(y[step]),
                    }
                )
        except Exception:
            skipped += 1

        fs = fs + pd.Timedelta(days=stride)

    paths.out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(rows)
    out_df.to_parquet(paths.out_path, index=False)

    LOGGER.info("WROTE: %s", paths.out_path)
    LOGGER.info("forecast_starts attempted: %d", attempted)
    LOGGER.info("skipped incomplete/failed: %d", skipped)
    LOGGER.info("rows written: %d", len(out_df))


if __name__ == "__main__":
    main()
