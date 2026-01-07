from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from src.inference.physics_aware_forecaster import PhysicsAwareForecaster

LOGGER = logging.getLogger("phase1_inference_pipeline_v3")


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def to_utc(series: pd.Series) -> pd.Series:
    s = pd.to_datetime(series, errors="coerce")
    if getattr(s.dt, "tz", None) is None:
        return s.dt.tz_localize("UTC")
    return s.dt.tz_convert("UTC")


def ensure_timestamp_utc(df: pd.DataFrame, col: str = "timestamp_utc") -> pd.DataFrame:
    if col not in df.columns:
        raise KeyError(f"Missing required time column: {col}")
    out = df.copy()
    out[col] = to_utc(out[col])
    return out


def ensure_plant_onehot(df: pd.DataFrame, plant_id: str) -> pd.DataFrame:
    out = df.copy()
    cols = ["plant_01", "plant_02", "plant_03", "plant_05", "plant_06"]
    for c in cols:
        if c not in out.columns:
            out[c] = 0
    if plant_id in cols:
        out[plant_id] = 1
    return out


def synth_power_norm_from_pvlib(df: pd.DataFrame, installed_capacity_kw: float) -> pd.DataFrame:
    out = df.copy()
    if "pvlib_ac_kw" not in out.columns:
        return out

    pn = None
    if "power_norm" in out.columns:
        pn = pd.to_numeric(out["power_norm"], errors="coerce")

    need = True
    if pn is not None and float(pn.notna().mean()) >= 0.8:
        need = False

    if need:
        cap = float(installed_capacity_kw) if installed_capacity_kw and installed_capacity_kw > 0 else 1.0
        pv = pd.to_numeric(out["pvlib_ac_kw"], errors="coerce").fillna(0.0)
        out["power_norm"] = (pv / cap).clip(0.0, 1.5)

    return out


def add_lstm_pca_lags(df: pd.DataFrame, lag_steps: int = 96) -> pd.DataFrame:
    """
    If df contains columns like lstm_enc_pca_000 .. lstm_enc_pca_031,
    ensure lagged versions lstm_enc_pca_000_lag96 exist by shifting lag_steps.
    This prevents KeyError for tft_lstm runs expecting lagged PCA encoding features.
    """
    out = df.copy()
    if "timestamp_utc" in out.columns:
        out = out.sort_values("timestamp_utc")

    base_cols = [c for c in out.columns if c.startswith("lstm_enc_pca_") and "_lag" not in c]
    if not base_cols:
        return out

    lag_suffix = f"_lag{lag_steps}"
    created = 0
    for c in base_cols:
        lag_c = f"{c}{lag_suffix}"
        if lag_c not in out.columns:
            out[lag_c] = out[c].shift(lag_steps)
            created += 1

    if created > 0:
        # Fill missing lags at the beginning with zeros (safe fallback)
        lag_cols = [f"{c}{lag_suffix}" for c in base_cols]
        out[lag_cols] = out[lag_cols].fillna(0.0)

    return out


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

    w = weather_df[(weather_df["timestamp_utc"] >= encoder_start) & (weather_df["timestamp_utc"] < encoder_end)].copy()
    w = w.sort_values("timestamp_utc").tail(lookback_steps).copy()
    w = ensure_plant_onehot(w, plant_id)
    w = synth_power_norm_from_pvlib(w, installed_capacity_kw)
    return w


def _tft_only_from_components(components: Dict[str, Any]) -> np.ndarray:
    """
    Combine short + long only (no physics) using alpha_short/alpha_long from blend_weights if present.
    """
    short_daily = np.asarray(components["short_head_daily"], dtype=np.float32)  # (30,96)
    long_up = np.asarray(components["long_upsampled"], dtype=np.float32).reshape(-1)  # (2880,)
    weights_daily = components.get("blend_weights", None)

    out = np.zeros(2880, dtype=np.float32)

    for day in range(30):
        s = short_daily[day].reshape(-1)
        lo = long_up[day * 96 : (day + 1) * 96]

        if weights_daily is not None and day < len(weights_daily):
            w = weights_daily[day]
            a_s = float(w.get("alpha_short", 0.5))
            a_l = float(w.get("alpha_long", 0.5))
        else:
            a_s, a_l = 0.5, 0.5

        denom = a_s + a_l
        if denom <= 0:
            a_s, a_l = 0.5, 0.5
        else:
            a_s /= denom
            a_l /= denom

        out[day * 96 : (day + 1) * 96] = a_s * s + a_l * lo

    return np.clip(out, 0.0, 1.5)


def shard_list(items: Sequence[pd.Timestamp], shard_idx: int, num_shards: int) -> List[pd.Timestamp]:
    if num_shards <= 1:
        return list(items)
    if shard_idx < 0 or shard_idx >= num_shards:
        raise ValueError(f"Invalid shard_idx={shard_idx} for num_shards={num_shards}")
    return list(items)[shard_idx::num_shards]


@dataclass
class Paths:
    phase1_dir: Path
    out_path: Path
    plant_meta: Path
    short_ckpt: Path
    long_ckpt: Path
    short_train: Path
    long_train: Path
    hist_encoder: Optional[Path]


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--weather-source", choices=["historical"], default="historical")
    ap.add_argument("--start-date", required=True, type=str)
    ap.add_argument("--end-date", required=True, type=str)
    ap.add_argument("--stride-days", type=int, default=1)
    ap.add_argument("--phase1-dir", required=True, type=str)
    ap.add_argument("--out", required=True, type=str)

    ap.add_argument("--plant-meta", required=True, type=str)
    ap.add_argument("--short-ckpt", required=True, type=str)
    ap.add_argument("--long-ckpt", required=True, type=str)
    ap.add_argument("--short-train", required=True, type=str)
    ap.add_argument("--long-train", required=True, type=str)
    ap.add_argument("--hist-encoder", type=str, default="")

    ap.add_argument(
        "--pred-mode",
        choices=["hybrid", "pvlib_only", "short_only", "long_only", "tft_only"],
        default="hybrid",
    )
    ap.add_argument("--save-components", type=int, default=0)

    ap.add_argument("--shard-idx", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)

    ap.add_argument("--log-level", type=str, default="INFO")
    args = ap.parse_args()
    setup_logging(args.log_level)

    paths = Paths(
        phase1_dir=Path(args.phase1_dir),
        out_path=Path(args.out),
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

    weather_path = paths.phase1_dir / "weather_with_pvlib_15min.parquet"
    if not weather_path.exists():
        raise FileNotFoundError(f"Missing weather parquet: {weather_path}")

    weather_15min = pd.read_parquet(weather_path)
    weather_15min = ensure_timestamp_utc(weather_15min, "timestamp_utc")
    weather_15min = ensure_plant_onehot(weather_15min, plant_id)
    weather_15min = add_lstm_pca_lags(weather_15min, lag_steps=96)
    weather_15min = weather_15min.sort_values("timestamp_utc")

    hist_encoder_df: Optional[pd.DataFrame] = None
    if paths.hist_encoder and paths.hist_encoder.exists():
        hist_encoder_df = pd.read_parquet(paths.hist_encoder)
        hist_encoder_df = ensure_timestamp_utc(hist_encoder_df, "timestamp_utc")
        hist_encoder_df = ensure_plant_onehot(hist_encoder_df, plant_id)
        hist_encoder_df = add_lstm_pca_lags(hist_encoder_df, lag_steps=96)
        hist_encoder_df = hist_encoder_df.sort_values("timestamp_utc")

    forecaster = PhysicsAwareForecaster(
        short_ckpt=paths.short_ckpt,
        long_ckpt=paths.long_ckpt,
        plant_metadata=str(paths.plant_meta),
        short_train_parquet=str(paths.short_train),
        long_train_parquet=str(paths.long_train),
    )

    start_utc = pd.Timestamp(args.start_date, tz="UTC")
    end_utc = pd.Timestamp(args.end_date, tz="UTC")
    stride_days = int(args.stride_days)

    max_ts = weather_15min["timestamp_utc"].max()
    latest_start = (max_ts - pd.Timedelta(minutes=15 * 2879)).floor("D")
    latest_start = pd.Timestamp(latest_start).tz_convert("UTC")
    run_end = min(end_utc, latest_start)

    all_starts = pd.date_range(start_utc, run_end, freq=f"{stride_days}D", tz="UTC")
    starts = shard_list(all_starts, int(args.shard_idx), int(args.num_shards))

    LOGGER.info("pred_mode=%s save_components=%s", args.pred_mode, bool(args.save_components))
    LOGGER.info("shard_idx=%d num_shards=%d starts=%d", args.shard_idx, args.num_shards, len(starts))
    LOGGER.info("out=%s", str(paths.out_path))

    rows: List[Dict[str, Any]] = []
    skipped_incomplete = 0

    for k, fs in enumerate(starts, start=1):
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

        need_components = (args.pred_mode != "hybrid") or bool(args.save_components)

        if need_components:
            comp = forecaster.predict_30d(
                forecast_start=str(fs),
                weather_df=w,
                historical_df=enc,
                return_components=True,
                blend_weights=None,
            )

            pred_hybrid = np.asarray(comp["final"], dtype=np.float32).reshape(-1)
            pred_pvlib = np.asarray(comp["pvlib_15min"], dtype=np.float32).reshape(-1)
            pred_short = np.asarray(comp["short_head_daily"], dtype=np.float32).reshape(-1)  # 30*96 -> 2880
            pred_long = np.asarray(comp["long_upsampled"], dtype=np.float32).reshape(-1)

            if args.pred_mode == "hybrid":
                preds = pred_hybrid
            elif args.pred_mode == "pvlib_only":
                preds = np.clip(pred_pvlib, 0.0, 1.5)
            elif args.pred_mode == "short_only":
                preds = np.clip(pred_short, 0.0, 1.5)
            elif args.pred_mode == "long_only":
                preds = np.clip(pred_long, 0.0, 1.5)
            elif args.pred_mode == "tft_only":
                preds = _tft_only_from_components(comp)
            else:
                raise ValueError(f"Unknown pred-mode: {args.pred_mode}")

        else:
            preds = forecaster.predict_30d(
                forecast_start=str(fs),
                weather_df=w,
                historical_df=enc,
                return_components=False,
            )
            if torch.is_tensor(preds):
                preds = preds.detach().cpu().numpy()
            preds = np.asarray(preds, dtype=np.float32).reshape(-1)

            pred_pvlib = None
            pred_short = None
            pred_long = None
            pred_hybrid = None

        if len(preds) != 2880:
            skipped_incomplete += 1
            continue

        save_components = bool(args.save_components) and (pred_pvlib is not None)

        for i in range(2880):
            r: Dict[str, Any] = {
                "timestamp_utc": w["timestamp_utc"].iloc[i],
                "forecast_start": fs,
                "step_ahead": int(i),
                "hours_ahead": float(i) * 0.25,
                "predicted_power_norm": float(preds[i]),
            }
            if save_components:
                r["pred_pvlib_norm"] = float(pred_pvlib[i])
                r["pred_short_norm"] = float(pred_short[i])
                r["pred_long_norm"] = float(pred_long[i])
                r["pred_hybrid_norm"] = float(pred_hybrid[i])
            rows.append(r)

        if k % 25 == 0:
            LOGGER.info("progress %d/%d in shard", k, len(starts))

    paths.out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(rows)
    out_df.to_parquet(paths.out_path, index=False)

    LOGGER.info("WROTE %s", str(paths.out_path))
    LOGGER.info("starts=%d skipped_incomplete=%d rows=%d", len(starts), skipped_incomplete, len(out_df))


if __name__ == "__main__":
    main()
