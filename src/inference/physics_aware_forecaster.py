# src/inference/physics_aware_forecaster.py
"""
Physics-Aware Forecaster: Complete 30-day PV Power Forecasting Pipeline.

Integrates:
    - Short-head TFT (96 steps @ 15-min, 24 hours)
    - Long-head TFT (720 steps @ 1-hour, 30 days)
    - PVLib physics baseline
    - Physics-aware upsampling and blending

Backtest-safe behavior:
    - If weather_df is provided, no live API calls are made, even if use_live_weather=True.
    - Live weather fetch happens only if use_live_weather=True AND weather_df is None.
    - Offline/backtest requires a full 30-day (2880 steps @ 15-min) weather window.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TimeSeriesDataSet

from src.inference.pvlib_predictor import PVLibPredictor
from src.inference.physics_glue import (
    blend_hierarchical,
    upsample_with_pvlib_shape,
)
from src.inference.rl_controller import RLMetaController
from src.inference.tft_utils import (
    load_tft_config,
    create_training_dataset,
    load_tft_model,
    extract_q50_prediction,
    create_inference_dataframe,
    validate_inference_window,
)


class PhysicsAwareForecaster:
    def __init__(
        self,
        short_ckpt: str | Path,
        long_ckpt: str | Path,
        plant_metadata: str | Path,
        short_train_parquet: str | Path,
        long_train_parquet: str | Path,
        device: Optional[str] = None,
    ):
        self.short_ckpt = Path(short_ckpt)
        self.long_ckpt = Path(long_ckpt)
        self.plant_metadata_path = Path(plant_metadata)
        self.short_train_parquet = Path(short_train_parquet)
        self.long_train_parquet = Path(long_train_parquet)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"[INFO] Initializing PhysicsAwareForecaster on {self.device}")

        def _resolve_ckpt_and_run_dir(p: Path):
            p = Path(p)
            if p.exists() and p.is_file():
                if p.parent.name == "checkpoints" and p.parent.parent.exists():
                    return p, p.parent.parent
                return p, p.parent

            cand = p.parent / "checkpoints" / p.name
            if cand.exists():
                return cand, cand.parent.parent if cand.parent.name == "checkpoints" and cand.parent.parent.exists() else cand.parent

            for name in ("best.ckpt", "best.pt", "best_state_dict.pt"):
                cand2 = p.parent / "checkpoints" / name
                if cand2.exists():
                    return cand2, cand2.parent.parent

            if p.exists() and p.is_dir():
                for name in (
                    "checkpoints/best.ckpt",
                    "checkpoints/best.pt",
                    "checkpoints/best_state_dict.pt",
                    "best.ckpt",
                    "best.pt",
                ):
                    cand3 = p / name
                    if cand3.exists():
                        return cand3, p

            matches = list(p.parent.rglob("best.ckpt")) if p.parent.exists() else []
            if matches:
                ck = matches[0]
                run_dir = ck.parent.parent if ck.parent.name == "checkpoints" and ck.parent.parent.exists() else ck.parent
                return ck, run_dir

            return p, p.parent

        print("[INFO] Loading TFT configurations...")
        short_ckpt_file, short_run_dir = _resolve_ckpt_and_run_dir(self.short_ckpt)
        long_ckpt_file, long_run_dir = _resolve_ckpt_and_run_dir(self.long_ckpt)

        self.short_ckpt = short_ckpt_file
        self.long_ckpt = long_ckpt_file

        self.short_config = load_tft_config(short_run_dir)
        self.long_config = load_tft_config(long_run_dir)

        print("[INFO] Loading training datasets for normalization...")
        short_train_df = pd.read_parquet(self.short_train_parquet)
        self.short_train_ds = create_training_dataset(
            short_train_df,
            self.short_config["roles"],
            self.short_config["encoder_len"],
            self.short_config["pred_len"],
        )
        print(f"       Short-head dataset: encoder={self.short_config['encoder_len']}, pred={self.short_config['pred_len']}")

        long_train_df = pd.read_parquet(self.long_train_parquet)
        self.long_train_ds = create_training_dataset(
            long_train_df,
            self.long_config["roles"],
            self.long_config["encoder_len"],
            self.long_config["pred_len"],
        )
        print(f"       Long-head dataset: encoder={self.long_config['encoder_len']}, pred={self.long_config['pred_len']}")

        print("[INFO] Loading TFT models...")
        self.short_model = load_tft_model(
            self.short_config,
            self.short_train_ds,
            self.short_ckpt.parent,
            strict=True,
            device=str(self.device),
        )
        print(f"       Short-head model loaded from {self.short_ckpt.parent.name}")

        self.long_model = load_tft_model(
            self.long_config,
            self.long_train_ds,
            self.long_ckpt.parent,
            strict=True,
            device=str(self.device),
        )
        print(f"       Long-head model loaded from {self.long_ckpt.parent.name}")

        self.pvlib_predictor = PVLibPredictor(self.plant_metadata_path)
        self.plant_id = self.pvlib_predictor.plant_id
        print(f"[INFO] PVLib predictor ready for {self.plant_id}")

        if self.device.type == "cuda":
            print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
            print(f"[INFO] Models confirmed on GPU: {next(self.short_model.parameters()).device}")

        self.rl_controller = RLMetaController(mode="heuristic")

        with open(self.plant_metadata_path, "r") as f:
            self.plant_metadata = json.load(f)

        print("[INFO] PhysicsAwareForecaster initialized successfully")

    def _ensure_encoder_power_norm(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        cap_kw = float(self.plant_metadata.get("installed_capacity_kw", 0.0) or 0.0)

        if "power_norm" not in df.columns:
            df["power_norm"] = np.nan

        df["power_norm"] = pd.to_numeric(df["power_norm"], errors="coerce")
        df.loc[~np.isfinite(df["power_norm"]), "power_norm"] = np.nan

        if df["power_norm"].isna().any():
            fill = None
            if cap_kw > 0 and "pvlib_ac_kw" in df.columns:
                pv = pd.to_numeric(df["pvlib_ac_kw"], errors="coerce")
                fill = (pv / cap_kw).clip(0.0, 1.5)
            elif cap_kw > 0 and "pvlib_dc_kw" in df.columns:
                pv = pd.to_numeric(df["pvlib_dc_kw"], errors="coerce")
                fill = (pv / cap_kw).clip(0.0, 1.5)

            if fill is not None:
                df["power_norm"] = df["power_norm"].fillna(fill)

        df["power_norm"] = df["power_norm"].fillna(0.0)
        return df

    def predict_30d(
        self,
        forecast_start: pd.Timestamp | str,
        weather_df: Optional[pd.DataFrame] = None,
        historical_df: Optional[pd.DataFrame] = None,
        use_live_weather: bool = False,
        return_components: bool = False,
        blend_weights: Optional[Dict[str, float]] = None,
    ) -> np.ndarray | Dict[str, np.ndarray]:
        """
        Generate 30-day physics-aware forecast with hierarchical refinement.

        blend_weights contract:
            - If blend_weights is provided, it MUST be a dict with keys: 'short','long','physics'
            - It is applied for blending (see loop). This code guarantees no None weights anywhere.
        """
        if isinstance(forecast_start, str):
            forecast_start = pd.Timestamp(forecast_start, tz="UTC")
        else:
            if forecast_start.tz is None:
                forecast_start = forecast_start.tz_localize("UTC")
            else:
                forecast_start = forecast_start.tz_convert("UTC")

        print(f"\n[INFO] Hierarchical 30-day forecast starting {forecast_start}")

        # 0) Weather acquisition (live fetch only if weather_df is missing)
        if use_live_weather and weather_df is None:
            print("[INFO] Fetching live weather from OpenMeteo API...")
            from .weather_client import WeatherClient

            with open(self.plant_metadata_path, "r") as f:
                plant_meta = json.load(f)

            client = WeatherClient()
            weather_df = client.fetch_and_prepare(
                latitude=plant_meta["latitude"],
                longitude=plant_meta["longitude"],
                start_time=forecast_start,
                days=15,
                tilt=plant_meta["tilt_deg"],
                azimuth=plant_meta["azimuth_deg"],
            )
            print(f"[WARN] OpenMeteo limited to 15 days (fetched {len(weather_df)} steps)")
            print("[WARN] For 30-day forecasts, use ECMWF/Ensemble in production")

            if len(weather_df) < 2880:
                steps_needed = 2880 - len(weather_df)
                last_day = weather_df.iloc[-96:].copy()
                repeats = (steps_needed + 95) // 96
                for i in range(repeats):
                    repeated = last_day.copy()
                    repeated["timestamp_utc"] = repeated["timestamp_utc"] + pd.Timedelta(days=i + 1)
                    weather_df = pd.concat([weather_df, repeated], ignore_index=True)
                weather_df = weather_df.iloc[:2880].copy()
                print(f"[WARN] Extended live forecast by repeating last day to {len(weather_df)} steps")

        elif use_live_weather and weather_df is not None:
            print("[WARN] use_live_weather=True but offline weather_df provided, ignoring live fetch (backtest-safe).")

        if weather_df is None:
            raise RuntimeError("weather_df is required for offline/backtest runs.")

        if "timestamp_utc" not in weather_df.columns:
            raise RuntimeError("weather_df missing required column: timestamp_utc")

        # 1) Historical/encoder data
        if historical_df is None:
            print("[INFO] No historical data provided, using training data for encoder")
            historical_df = pd.read_parquet(self.short_train_parquet)
            historical_df = historical_df.iloc[-200:].copy()

        historical_df = self._ensure_encoder_power_norm(historical_df)

        print("       Architecture: Long-head (strategic) + 30× Short-head (tactical) + Physics")

        # 2) Prepare hourly views for long-head
        print("\n[PREP] Preparing data for multi-resolution inference...")
        if "timestamp_utc" in historical_df.columns:
            try:
                df_freq = pd.infer_freq(pd.to_datetime(historical_df["timestamp_utc"].iloc[:100]))
            except Exception:
                df_freq = None
            print(f"       Input frequency detected: {df_freq}")

            if df_freq in ["15T", "15min"]:
                print("       Resampling to hourly for long-head...")

                hist_numeric_cols = historical_df.select_dtypes(include=[np.number]).columns.tolist()
                hist_cat_cols = [c for c in historical_df.columns if c not in hist_numeric_cols + ["timestamp_utc"]]

                weather_numeric_cols = weather_df.select_dtypes(include=[np.number]).columns.tolist()
                weather_cat_cols = [c for c in weather_df.columns if c not in weather_numeric_cols + ["timestamp_utc"]]

                hourly_hist = (
                    historical_df[["timestamp_utc"] + hist_numeric_cols]
                    .set_index("timestamp_utc")
                    .resample("1h")
                    .mean()
                    .reset_index()
                )
                hourly_weather = (
                    weather_df[["timestamp_utc"] + weather_numeric_cols]
                    .set_index("timestamp_utc")
                    .resample("1h")
                    .mean()
                    .reset_index()
                )

                for col in hist_cat_cols:
                    hourly_hist[col] = historical_df[col].iloc[0]
                for col in weather_cat_cols:
                    hourly_weather[col] = weather_df[col].iloc[0]

                print(f"       Hourly shape: historical={hourly_hist.shape}, weather={hourly_weather.shape}")
            else:
                hourly_hist = historical_df
                hourly_weather = weather_df
        else:
            hourly_hist = historical_df
            hourly_weather = weather_df

        # 3) Strict 30-day weather window (fail fast offline)
        forecast_end = forecast_start + pd.Timedelta(days=30)
        weather_window = weather_df[
            (weather_df["timestamp_utc"] >= forecast_start)
            & (weather_df["timestamp_utc"] < forecast_end)
        ].copy()

        if len(weather_window) != 2880:
            avail_min = weather_df["timestamp_utc"].min()
            avail_max = weather_df["timestamp_utc"].max()
            raise RuntimeError(
                f"Offline weather window incomplete for forecast_start={forecast_start}. "
                f"Need 2880 rows, got {len(weather_window)}. "
                f"weather_df range: {avail_min} -> {avail_max}"
            )

        # Step 1: PVLib baseline
        print("\n[STEP 1/4] Computing PVLib physics baseline (2880 steps @ 15-min)...")
        print(
            f"          Weather window: {len(weather_window)} steps "
            f"({weather_window.timestamp_utc.min()} -> {weather_window.timestamp_utc.max()})"
        )

        if "pvlib_ac_kw" in weather_window.columns:
            print("          Using pre-computed PVLib baseline (pvlib_ac_kw column)")
            capacity_kw = float(self.plant_metadata["installed_capacity_kw"])
            pvlib_15min = (weather_window["pvlib_ac_kw"].values / capacity_kw).astype(float)
            pvlib_15min = np.clip(pvlib_15min, 0.0, 1.0)
        else:
            print("          Computing fresh PVLib predictions...")
            pvlib_15min = self.pvlib_predictor.predict_from_weather(weather_window)

        print(
            f"          PVLib shape: {pvlib_15min.shape}, "
            f"range: [{np.nanmin(pvlib_15min):.3f}, {np.nanmax(pvlib_15min):.3f}]"
        )

        # Step 2: Long-head inference
        print("\n[STEP 2/4] Running long-head TFT (strategic: 720 hours @ 1-hour)...")
        long_head_pred = self._predict_long_head(forecast_start, hourly_hist, hourly_weather)
        print(f"          Long-head shape: {long_head_pred.shape}")

        print("          Upsampling long-head to 15-min resolution...")
        long_upsampled = upsample_with_pvlib_shape(long_head_pred, pvlib_15min, method="proportional")
        print(f"          Long upsampled shape: {long_upsampled.shape}")

        # Step 3: Rolling short-head refinement + blending
        print("\n[STEP 3/4] Rolling short-head refinement (30 days × 96 steps)...")
        forecast_15min = np.zeros(2880, dtype=float)
        short_head_daily: list[np.ndarray] = []
        blend_weights_daily: list[dict] = []

        print("[DEBUG] blend_weights passed in:", blend_weights)

        # Normalize / validate RL weights once (if provided)
        rl_w_short = rl_w_long = rl_w_phys = None
        if blend_weights is not None:
            if not all(k in blend_weights for k in ("short", "long", "physics")):
                raise ValueError(f"blend_weights missing required keys: {blend_weights}")
            rl_w_short = float(blend_weights["short"])
            rl_w_long = float(blend_weights["long"])
            rl_w_phys = float(blend_weights["physics"])

        for day in range(30):
            day_start = forecast_start + pd.Timedelta(days=day)
            day_start_idx = day * 96
            day_end_idx = (day + 1) * 96

            if day < 2:
                print(f"[DEBUG] day={day} using_rl_override={blend_weights is not None}")
                if blend_weights is not None:
                    print(f"[DEBUG] rl weights: {blend_weights}")

            short_day_pred = self._predict_short_head_for_day(day_start, day, historical_df, weather_df)
            short_head_daily.append(short_day_pred)

            long_slice = long_upsampled[day_start_idx:day_end_idx]
            pvlib_slice = pvlib_15min[day_start_idx:day_end_idx]

            if blend_weights is not None:
                # RL override blending (no None anywhere)
                w_short = rl_w_short
                w_long = rl_w_long
                w_phys = rl_w_phys

                w_ml = (w_short + w_long)
                if w_ml > 0:
                    alpha_short_norm = w_short / w_ml
                    alpha_long_norm = w_long / w_ml
                else:
                    alpha_short_norm = 0.5
                    alpha_long_norm = 0.5
                    w_ml = 0.0

                day_forecast = blend_hierarchical(
                    short_pred=short_day_pred,
                    long_upsampled=long_slice,
                    pvlib_baseline=pvlib_slice,
                    alpha_short=float(alpha_short_norm),
                    alpha_long=float(alpha_long_norm),
                    alpha_ml=float(w_ml),
                    constraints=True,
                )

                # record weights in a consistent schema
                blend_weights_daily.append(
                    {
                        "alpha_short": float(alpha_short_norm),
                        "alpha_long": float(alpha_long_norm),
                        "alpha_ml": float(w_ml),
                        "rl_short": float(w_short),
                        "rl_long": float(w_long),
                        "rl_physics": float(w_phys),
                    }
                )
                weights_for_print = {
                    "alpha_short": float(alpha_short_norm),
                    "alpha_long": float(alpha_long_norm),
                    "alpha_ml": float(w_ml),
                }
            else:
                # Default scheduled blending
                weights = self.rl_controller.get_blend_weights(day=day)
                day_forecast = blend_hierarchical(
                    short_pred=short_day_pred,
                    long_upsampled=long_slice,
                    pvlib_baseline=pvlib_slice,
                    alpha_short=float(weights["alpha_short"]),
                    alpha_long=float(weights["alpha_long"]),
                    alpha_ml=float(weights["alpha_ml"]),
                    constraints=True,
                )
                blend_weights_daily.append(weights)
                weights_for_print = weights

            forecast_15min[day_start_idx:day_end_idx] = day_forecast

            # Update historical_df with today's prediction for next day's encoder
            day_weather_slice = weather_df[
                (weather_df["timestamp_utc"] >= day_start)
                & (weather_df["timestamp_utc"] < day_start + pd.Timedelta(hours=24))
            ].copy()
            day_weather_slice["power_norm"] = day_forecast
            historical_df = pd.concat([historical_df, day_weather_slice], ignore_index=True)

            if (day + 1) % 5 == 0:
                print(
                    f"          Day {day+1:2d}/30: α_short={weights_for_print['alpha_short']:.2f}, "
                    f"α_long={weights_for_print['alpha_long']:.2f}, α_ml={weights_for_print['alpha_ml']:.2f}"
                )

        print(f"          Final forecast shape: {forecast_15min.shape}")
        print(f"          Final range: [{forecast_15min.min():.3f}, {forecast_15min.max():.3f}]")

        # Step 4: Validation checks
        print("\n[STEP 4/4] Validation...")
        self._validate_forecast(forecast_15min, pvlib_15min)

        print("\n[SUCCESS] Hierarchical 30-day forecast complete!")
        print("          Total TFT calls: 1 long-head + 30 short-head = 31")

        if return_components:
            return {
                "final": forecast_15min,
                "short_head_daily": np.array(short_head_daily),
                "long_head": long_head_pred,
                "pvlib_15min": pvlib_15min,
                "long_upsampled": long_upsampled,
                "blend_weights": blend_weights_daily,
            }

        return forecast_15min

    def _predict_short_head_for_day(
        self,
        day_start: pd.Timestamp,
        day_idx: int,
        historical_df: pd.DataFrame,
        weather_df: pd.DataFrame,
    ) -> np.ndarray:
        encoder_start = day_start - pd.Timedelta(hours=24)
        encoder_end = day_start

        encoder_df = historical_df[
            (historical_df["timestamp_utc"] >= encoder_start) & (historical_df["timestamp_utc"] < encoder_end)
        ].copy()

        decoder_start = day_start
        decoder_end = day_start + pd.Timedelta(hours=24)

        decoder_df = weather_df[
            (weather_df["timestamp_utc"] >= decoder_start) & (weather_df["timestamp_utc"] < decoder_end)
        ].copy()

        validate_inference_window(encoder_df, 96, f"Day {day_idx} encoder")
        validate_inference_window(decoder_df, 96, f"Day {day_idx} decoder")

        inference_df = create_inference_dataframe(
            encoder_df,
            decoder_df,
            self.short_config["roles"],
            plant_id=self.plant_id,
        )

        test_ds = TimeSeriesDataSet.from_dataset(
            self.short_train_ds,
            inference_df,
            predict=True,
            stop_randomization=True,
        )

        test_dl = test_ds.to_dataloader(
            train=False,
            batch_size=1,
            num_workers=0,
            shuffle=False,
        )

        self.short_model.eval()
        with torch.no_grad():
            for x, _y in test_dl:
                x = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in x.items()}
                output = self.short_model(x)
                predictions = extract_q50_prediction(output, self.short_model.loss.quantiles)
                return predictions.squeeze()

        raise RuntimeError(f"No predictions generated for day {day_idx}")

    def _predict_long_head(
        self,
        forecast_start: pd.Timestamp,
        historical_df: pd.DataFrame,
        weather_df: pd.DataFrame,
    ) -> np.ndarray:
        encoder_start = forecast_start - pd.Timedelta(hours=168)
        encoder_end = forecast_start

        encoder_df = historical_df[
            (historical_df["timestamp_utc"] >= encoder_start) & (historical_df["timestamp_utc"] < encoder_end)
        ].copy()

        decoder_start = forecast_start
        decoder_end = forecast_start + pd.Timedelta(hours=720)

        decoder_df = weather_df[
            (weather_df["timestamp_utc"] >= decoder_start) & (weather_df["timestamp_utc"] < decoder_end)
        ].copy()

        validate_inference_window(encoder_df, 168, "Long-head encoder")
        validate_inference_window(decoder_df, 720, "Long-head decoder")

        inference_df = create_inference_dataframe(
            encoder_df,
            decoder_df,
            self.long_config["roles"],
            plant_id=self.plant_id,
        )

        test_ds = TimeSeriesDataSet.from_dataset(
            self.long_train_ds,
            inference_df,
            predict=True,
            stop_randomization=True,
        )

        test_dl = test_ds.to_dataloader(
            train=False,
            batch_size=1,
            num_workers=0,
            shuffle=False,
        )

        self.long_model.eval()
        with torch.no_grad():
            for x, _y in test_dl:
                x = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in x.items()}
                output = self.long_model(x)
                predictions = extract_q50_prediction(output, self.long_model.loss.quantiles)
                return predictions.squeeze()

        raise RuntimeError("No predictions generated for long-head")

    def _validate_forecast(self, forecast: np.ndarray, pvlib: np.ndarray):
        checks = []

        if len(forecast) == 2880:
            checks.append("✓ Shape correct (2880 steps)")
        else:
            checks.append(f"✗ Shape wrong: {len(forecast)} != 2880")

        if forecast.min() >= 0 and forecast.max() <= 1.0:
            checks.append(f"✓ Range valid [{forecast.min():.3f}, {forecast.max():.3f}]")
        else:
            checks.append(f"✗ Range invalid [{forecast.min():.3f}, {forecast.max():.3f}]")

        night_mask = pvlib < 0.01
        night_count = int((forecast[night_mask] > 0.01).sum())
        if night_count == 0:
            checks.append("✓ All night hours zero")
        else:
            checks.append(f"⚠ {night_count} night hours non-zero")

        day_mask = pvlib > 0.1
        if int(day_mask.sum()) > 0:
            day_mean = float(forecast[day_mask].mean())
            checks.append(f"✓ Daylight mean: {day_mean:.3f}")

        for check in checks:
            print(f"         {check}")
