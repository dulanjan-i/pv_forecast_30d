# src/inference/physics_aware_forecaster.py
"""
Physics-Aware Forecaster: Complete 30-day PV Power Forecasting Pipeline.

Integrates:
    - Short-head TFT (96 steps @ 15-min, 24 hours)
    - Long-head TFT (720 steps @ 1-hour, 30 days)
    - PVLib physics baseline
    - Physics-aware upsampling and blending

Architecture:
    1. Load both TFT checkpoints (short + long head)
    2. Generate PVLib baseline from weather forecast
    3. Run short-head inference for Day 1
    4. Run long-head inference for Days 1-30
    5. Upsample long-head using PVLib shape
    6. Blend: Day 1 (short) + Days 2-30 (long upsampled)
    7. Apply physics constraints
    8. Return unified 2880-step forecast @ 15-min

Usage:
    forecaster = PhysicsAwareForecaster(
        short_ckpt="experiments/.../shorthead/best.ckpt",
        long_ckpt="experiments/.../longhead/best.ckpt",
        plant_metadata="data/metadata/germany/plant_03.json"
    )
    
    forecast = forecaster.predict_30d(
        forecast_start="2023-11-01 00:00:00",
        weather_df=weather_forecast,  # 30 days of GHI/DNI/DHI
        historical_df=historical_data  # Recent history for encoder
    )
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TemporalFusionTransformer

from src.inference.pvlib_predictor import PVLibPredictor
from src.inference.physics_glue import (
    blend_hierarchical,
    upsample_with_pvlib_shape,
    apply_physics_constraints
)
from src.inference.rl_controller import RLMetaController
from src.inference.tft_utils import (
    load_tft_config,
    create_training_dataset,
    load_tft_model,
    ensure_time_columns,
    extract_q50_prediction,
    create_inference_dataframe,
    validate_inference_window
)
from pytorch_forecasting import TimeSeriesDataSet


class PhysicsAwareForecaster:
    """
    Complete 30-day PV forecasting system with physics integration.
    
    Attributes:
        short_model: TFT model for Day 1 (96@15min)
        long_model: TFT model for Days 1-30 (720@1hour)
        pvlib_predictor: PVLib physics engine
        device: torch device (cuda/cpu)
    """
    
    def __init__(
        self,
        short_ckpt: str | Path,
        long_ckpt: str | Path,
        plant_metadata: str | Path,
        short_train_parquet: str | Path,
        long_train_parquet: str | Path,
        device: Optional[str] = None
    ):
        """
        Initialize forecaster with model checkpoints and plant configuration.
        
        Args:
            short_ckpt: Path to short-head TFT checkpoint (96 steps @ 15-min)
            long_ckpt: Path to long-head TFT checkpoint (720 steps @ 1-hour)
            plant_metadata: Path to plant JSON metadata
            short_train_parquet: Path to short-head training data (for normalization)
            long_train_parquet: Path to long-head training data (for normalization)
            device: Torch device ('cuda', 'cpu', or None for auto-detect)
        """
        self.short_ckpt = Path(short_ckpt)
        self.long_ckpt = Path(long_ckpt)
        self.plant_metadata_path = Path(plant_metadata)
        self.short_train_parquet = Path(short_train_parquet)
        self.long_train_parquet = Path(long_train_parquet)
        
        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"[INFO] Initializing PhysicsAwareForecaster on {self.device}")
        
        # Load TFT configs
        print("[INFO] Loading TFT configurations...")
        self.short_config = load_tft_config(self.short_ckpt.parent.parent)
        self.long_config = load_tft_config(self.long_ckpt.parent.parent)
        
        # Load training datasets (for normalization inheritance)
        print("[INFO] Loading training datasets for normalization...")
        short_train_df = pd.read_parquet(self.short_train_parquet)
        self.short_train_ds = create_training_dataset(
            short_train_df,
            self.short_config['roles'],
            self.short_config['encoder_len'],
            self.short_config['pred_len']
        )
        print(f"       Short-head dataset: encoder={self.short_config['encoder_len']}, pred={self.short_config['pred_len']}")
        
        long_train_df = pd.read_parquet(self.long_train_parquet)
        self.long_train_ds = create_training_dataset(
            long_train_df,
            self.long_config['roles'],
            self.long_config['encoder_len'],
            self.long_config['pred_len']
        )
        print(f"       Long-head dataset: encoder={self.long_config['encoder_len']}, pred={self.long_config['pred_len']}")
        
        # Load models with weights
        print("[INFO] Loading TFT models...")
        self.short_model = load_tft_model(
            self.short_config,
            self.short_train_ds,
            self.short_ckpt.parent,
            strict=True,
            device=str(self.device)
        )
        print(f"       Short-head model loaded from {self.short_ckpt.parent.name}")
        
        self.long_model = load_tft_model(
            self.long_config,
            self.long_train_ds,
            self.long_ckpt.parent,
            strict=True,
            device=str(self.device)
        )
        print(f"       Long-head model loaded from {self.long_ckpt.parent.name}")
        
        # Initialize PVLib predictor
        self.pvlib_predictor = PVLibPredictor(self.plant_metadata_path)
        self.plant_id = self.pvlib_predictor.plant_id
        print(f"[INFO] PVLib predictor ready for {self.plant_id}")
        
        # Initialize RL meta-controller
        self.rl_controller = RLMetaController(mode="heuristic")
        
        # Load plant metadata
        with open(self.plant_metadata_path, "r") as f:
            self.plant_metadata = json.load(f)
        
        print("[INFO] PhysicsAwareForecaster initialized successfully")
    

    
    def predict_30d(
        self,
        forecast_start: pd.Timestamp | str,
        weather_df: Optional[pd.DataFrame] = None,
        historical_df: Optional[pd.DataFrame] = None,
        use_live_weather: bool = False,
        return_components: bool = False
    ) -> np.ndarray | Dict[str, np.ndarray]:
        """
        Generate 30-day physics-aware forecast with hierarchical refinement.
        
        Architecture (Hierarchical Refinement - "Drone + Fighter Jet"):
            1. Long-head: Predict all 720 hours (rough strategic overview)
            2. Short-head: Refine EACH day to 96×15-min (tactical precision)
            3. Hierarchical blend: short (60%) + long_upsampled (40%) + pvlib (30%)
            Total TFT calls: 1 long + 30 short = 31 calls
        
        Args:
            forecast_start: Start timestamp for forecast (timezone-aware)
            weather_df: Weather forecast DataFrame (Optional if use_live_weather=True)
                Required columns: timestamp_utc, ghi, dni, dhi, temp_air, wind_speed, ...
                Must cover 30 days @ 15-min resolution (2880 steps)
            historical_df: Historical data for encoder window (Optional - uses last known if None)
                Must include recent history before forecast_start
                Same schema as training data (all features + target)
            use_live_weather: If True, fetch real-time weather from OpenMeteo API
                Overrides weather_df parameter
            return_components: If True, return dict with all intermediate predictions
        
        Returns:
            forecast_15min: Final forecast @ 15-min, shape (2880,)
            OR
            components: Dict with keys:
                - 'final': Final blended forecast (2880,)
                - 'short_head_daily': Short predictions per day (30, 96)
                - 'long_head': Long strategic prediction hourly (720,)
                - 'pvlib_15min': PVLib baseline (2880,)
                - 'long_upsampled': Long-head upsampled to 15-min (2880,)
                - 'blend_weights': RL-selected weights per day
        
        Example:
            >>> forecaster = PhysicsAwareForecaster(short_ckpt, long_ckpt, metadata)
            >>> # Option 1: Use provided weather data
            >>> forecast = forecaster.predict_30d(
            ...     forecast_start="2023-11-01 00:00:00",
            ...     weather_df=weather_forecast,
            ...     historical_df=historical_data
            ... )
            >>> # Option 2: Fetch live weather
            >>> forecast = forecaster.predict_30d(
            ...     forecast_start="2023-11-01 00:00:00",
            ...     use_live_weather=True
            ... )
            >>> forecast.shape
            (2880,)  # 30 days @ 15-min
        """
        if isinstance(forecast_start, str):
            forecast_start = pd.Timestamp(forecast_start, tz="UTC")
        
        print(f"\n[INFO] Hierarchical 30-day forecast starting {forecast_start}")
        
        # Fetch live weather if requested
        if use_live_weather:
            print("[INFO] Fetching live weather from OpenMeteo API...")
            from .weather_client import WeatherClient
            
            # Load plant metadata
            with open(self.plant_metadata_path, 'r') as f:
                plant_meta = json.load(f)
            
            client = WeatherClient()
            try:
                # Try 15-day (API maximum)
                weather_df = client.fetch_and_prepare(
                    latitude=plant_meta['latitude'],
                    longitude=plant_meta['longitude'],
                    start_time=forecast_start,
                    days=15,  # OpenMeteo Forecast API max
                    tilt=plant_meta['tilt_deg'],
                    azimuth=plant_meta['azimuth_deg']
                )
                print(f"[WARN] OpenMeteo Forecast API limited to 15 days (fetched {len(weather_df)} steps)")
                print("[WARN] For 30-day forecasts, use ECMWF or Ensemble API in production")
                
                # TODO: For production, extend with climatology or ensemble for days 16-30
                # For now, repeat last day to reach 2880 steps (temporary workaround)
                if len(weather_df) < 2880:
                    steps_needed = 2880 - len(weather_df)
                    last_day = weather_df.iloc[-96:].copy()  # Last 96 steps (1 day)
                    for i in range(steps_needed // 96):
                        repeated = last_day.copy()
                        repeated['timestamp_utc'] = repeated['timestamp_utc'] + pd.Timedelta(days=i+1)
                        weather_df = pd.concat([weather_df, repeated], ignore_index=True)
                    weather_df = weather_df.iloc[:2880]  # Trim to exact 2880
                    print(f"[WARN] Extended with climatology repeat to {len(weather_df)} steps")
                    
            except Exception as e:
                print(f"[ERROR] Live weather fetch failed: {e}")
                print("[ERROR] Falling back to test weather data if available")
                if weather_df is None:
                    raise RuntimeError("Weather data required - live fetch failed and no fallback provided")
        
        # Load historical data if not provided (use training data for encoder window)
        if historical_df is None:
            print("[INFO] No historical data provided - using training data for encoder")
            historical_df = pd.read_parquet(self.short_train_parquet)
            # Trim to last N rows for encoder context (96 + buffer)
            historical_df = historical_df.iloc[-200:].copy()
        
        print("       Architecture: Long-head (strategic) + 30× Short-head (tactical) + Physics")
        
        # Resample to hourly for long-head (if input is 15min)
        print("\n[PREP] Preparing data for multi-resolution inference...")
        if 'timestamp_utc' in historical_df.columns:
            df_freq = pd.infer_freq(historical_df['timestamp_utc'].iloc[:100])
            print(f"       Input frequency detected: {df_freq}")
            
            if df_freq in ['15T', '15min']:
                print("       Resampling to hourly for long-head...")
                
                # Separate numeric and non-numeric columns for EACH dataframe independently
                hist_numeric_cols = historical_df.select_dtypes(include=[np.number]).columns.tolist()
                hist_cat_cols = [c for c in historical_df.columns if c not in hist_numeric_cols + ['timestamp_utc']]
                
                weather_numeric_cols = weather_df.select_dtypes(include=[np.number]).columns.tolist()
                weather_cat_cols = [c for c in weather_df.columns if c not in weather_numeric_cols + ['timestamp_utc']]
                
                # Resample numeric columns (mean aggregation) for each dataframe
                hourly_hist = historical_df[['timestamp_utc'] + hist_numeric_cols].set_index('timestamp_utc').resample('1H').mean().reset_index()
                hourly_weather = weather_df[['timestamp_utc'] + weather_numeric_cols].set_index('timestamp_utc').resample('1H').mean().reset_index()
                
                # Add back categorical columns (copy first value)
                for col in hist_cat_cols:
                    hourly_hist[col] = historical_df[col].iloc[0]
                for col in weather_cat_cols:
                    hourly_weather[col] = weather_df[col].iloc[0]
                
                print(f"       Hourly shape: historical={hourly_hist.shape}, weather={hourly_weather.shape}")
            else:
                # Already hourly or unknown, use as-is
                hourly_hist = historical_df
                hourly_weather = weather_df
        else:
            hourly_hist = historical_df
            hourly_weather = weather_df
        
        # Step 1: Generate PVLib baseline @ 15-min for all 30 days
        print("\n[STEP 1/4] Computing PVLib physics baseline (2880 steps @ 15-min)...")
        # Extract 30-day forecast window from weather_df
        forecast_end = forecast_start + pd.Timedelta(days=30)
        weather_window = weather_df[
            (weather_df['timestamp_utc'] >= forecast_start) & 
            (weather_df['timestamp_utc'] < forecast_end)
        ].copy()
        print(f"          Weather window: {len(weather_window)} steps ({weather_window.timestamp_utc.min()} → {weather_window.timestamp_utc.max()})")
        
        # Check if pre-computed PVLib exists (common in test data)
        if 'pvlib_ac_kw' in weather_window.columns:
            print("          Using pre-computed PVLib baseline (pvlib_ac_kw column)")
            # Normalize pvlib_ac_kw to [0, 1] range
            capacity_kw = self.plant_metadata['installed_capacity_kw']
            pvlib_15min = weather_window['pvlib_ac_kw'].values / capacity_kw
            pvlib_15min = np.clip(pvlib_15min, 0, 1.0)
        else:
            print("          Computing fresh PVLib predictions...")
            pvlib_15min = self.pvlib_predictor.predict_from_weather(weather_window)
        
        print(f"          PVLib shape: {pvlib_15min.shape}, range: [{np.nanmin(pvlib_15min):.3f}, {np.nanmax(pvlib_15min):.3f}]")
        
        # Step 2: Long-head inference (strategic overview: 720 hours)
        print("\n[STEP 2/4] Running long-head TFT (strategic: 720 hours @ 1-hour)...")
        long_head_pred = self._predict_long_head(
            forecast_start, hourly_hist, hourly_weather
        )
        print(f"          Long-head shape: {long_head_pred.shape}")
        
        # Upsample long-head to 15-min using PVLib shape (for hierarchical blending)
        print("          Upsampling long-head to 15-min resolution...")
        long_upsampled = upsample_with_pvlib_shape(
            long_head_pred, pvlib_15min, method="proportional"
        )
        print(f"          Long upsampled shape: {long_upsampled.shape}")
        
        # Step 3: Rolling daily refinement (short-head for each day)
        print("\n[STEP 3/4] Rolling short-head refinement (30 days × 96 steps)...")
        forecast_15min = np.zeros(2880)
        short_head_daily = []  # Store for analysis if requested
        blend_weights_daily = []  # Store RL weights per day
        
        for day in range(30):
            # Extract day's time slice
            day_start = forecast_start + pd.Timedelta(days=day)
            day_start_idx = day * 96
            day_end_idx = (day + 1) * 96
            
            # Get adaptive weights from RL controller
            weights = self.rl_controller.get_blend_weights(day=day)
            blend_weights_daily.append(weights)
            
            # Short-head prediction for this day
            short_day_pred = self._predict_short_head_for_day(
                day_start, day, historical_df, weather_df
            )
            short_head_daily.append(short_day_pred)
            
            # Extract corresponding slices
            long_slice = long_upsampled[day_start_idx:day_end_idx]
            pvlib_slice = pvlib_15min[day_start_idx:day_end_idx]
            
            # Hierarchical 3-way blend: short + long + physics
            day_forecast = blend_hierarchical(
                short_pred=short_day_pred,
                long_upsampled=long_slice,
                pvlib_baseline=pvlib_slice,
                alpha_short=weights['alpha_short'],
                alpha_long=weights['alpha_long'],
                alpha_ml=weights['alpha_ml'],
                constraints=True
            )
            
            forecast_15min[day_start_idx:day_end_idx] = day_forecast
            
            # Update historical_df with today's prediction for next day's encoder
            # Extract today's weather slice and add predicted power_norm
            day_weather_slice = weather_df[
                (weather_df['timestamp_utc'] >= day_start) &
                (weather_df['timestamp_utc'] < day_start + pd.Timedelta(hours=24))
            ].copy()
            day_weather_slice['power_norm'] = day_forecast  # Add predictions
            
            # Append to historical_df for dynamic encoder anchoring
            historical_df = pd.concat([historical_df, day_weather_slice], ignore_index=True)
            
            if (day + 1) % 5 == 0:  # Progress every 5 days
                print(f"          Day {day+1:2d}/30: α_short={weights['alpha_short']:.2f}, "
                      f"α_long={weights['alpha_long']:.2f}, α_ml={weights['alpha_ml']:.2f}")
        
        print(f"          Final forecast shape: {forecast_15min.shape}")
        print(f"          Final range: [{forecast_15min.min():.3f}, {forecast_15min.max():.3f}]")
        
        # Step 4: Validation checks
        print("\n[STEP 4/4] Validation...")
        self._validate_forecast(forecast_15min, pvlib_15min)
        
        print("\n[SUCCESS] Hierarchical 30-day forecast complete!")
        print(f"          Total TFT calls: 1 long-head + 30 short-head = 31")
        
        if return_components:
            return {
                'final': forecast_15min,
                'short_head_daily': np.array(short_head_daily),  # (30, 96)
                'long_head': long_head_pred,  # (720,)
                'pvlib_15min': pvlib_15min,  # (2880,)
                'long_upsampled': long_upsampled,  # (2880,)
                'blend_weights': blend_weights_daily  # List[Dict]
            }
        else:
            return forecast_15min
    
    def _predict_short_head_for_day(
        self,
        day_start: pd.Timestamp,
        day_idx: int,
        historical_df: pd.DataFrame,
        weather_df: pd.DataFrame
    ) -> np.ndarray:
        """
        Run short-head TFT inference for a single day.
        
        Args:
            day_start: Start timestamp for this day
            day_idx: Day index (0-29)
            historical_df: Historical data for encoder window
            weather_df: Weather forecast for decoder window
        
        Returns:
            predictions: shape (96,) @ 15-min resolution
        """
        # Extract encoder window: 96 steps (24 hours) BEFORE day_start
        encoder_start = day_start - pd.Timedelta(hours=24)
        encoder_end = day_start
        
        encoder_df = historical_df[
            (historical_df['timestamp_utc'] >= encoder_start) &
            (historical_df['timestamp_utc'] < encoder_end)
        ].copy()
        
        # Extract decoder window: 96 steps (24 hours) FROM day_start
        decoder_start = day_start
        decoder_end = day_start + pd.Timedelta(hours=24)
        
        decoder_df = weather_df[
            (weather_df['timestamp_utc'] >= decoder_start) &
            (weather_df['timestamp_utc'] < decoder_end)
        ].copy()
        
        # Validate window lengths
        validate_inference_window(encoder_df, 96, f"Day {day_idx} encoder")
        validate_inference_window(decoder_df, 96, f"Day {day_idx} decoder")
        
        # Create inference DataFrame
        inference_df = create_inference_dataframe(
            encoder_df,
            decoder_df,
            self.short_config['roles'],
            plant_id=self.plant_id
        )
        
        # Create TimeSeriesDataSet from inference window (inherits normalization)
        test_ds = TimeSeriesDataSet.from_dataset(
            self.short_train_ds,
            inference_df,
            predict=True,
            stop_randomization=True
        )
        
        # Create DataLoader (single sample)
        test_dl = test_ds.to_dataloader(
            train=False,
            batch_size=1,
            num_workers=0,
            shuffle=False
        )
        
        # Run inference
        self.short_model.eval()
        with torch.no_grad():
            for x, y in test_dl:
                # Move batch to device
                x = {k: v.to(self.device) if torch.is_tensor(v) else v 
                     for k, v in x.items()}
                
                # Forward pass
                output = self.short_model(x)
                
                # Extract q50 quantile: (1, 96, Q) → (1, 96) → (96,)
                predictions = extract_q50_prediction(
                    output,
                    self.short_model.loss.quantiles
                )
                
                return predictions.squeeze()  # (96,)
        
        raise RuntimeError(f"No predictions generated for day {day_idx}")
    
    def _predict_long_head(
        self,
        forecast_start: pd.Timestamp,
        historical_df: pd.DataFrame,
        weather_df: pd.DataFrame
    ) -> np.ndarray:
        """
        Run long-head TFT inference for all 30 days (strategic overview).
        
        Args:
            forecast_start: Start timestamp for forecast
            historical_df: Historical data @ 1-hour for encoder window
            weather_df: Weather forecast @ 1-hour for decoder window
        
        Returns:
            predictions: shape (720,) @ 1-hour resolution
        """
        # Extract encoder window: 168 hours (7 days) BEFORE forecast_start
        encoder_start = forecast_start - pd.Timedelta(hours=168)
        encoder_end = forecast_start
        
        encoder_df = historical_df[
            (historical_df['timestamp_utc'] >= encoder_start) &
            (historical_df['timestamp_utc'] < encoder_end)
        ].copy()
        
        # Extract decoder window: 720 hours (30 days) FROM forecast_start
        decoder_start = forecast_start
        decoder_end = forecast_start + pd.Timedelta(hours=720)
        
        decoder_df = weather_df[
            (weather_df['timestamp_utc'] >= decoder_start) &
            (weather_df['timestamp_utc'] < decoder_end)
        ].copy()
        
        # Validate window lengths
        validate_inference_window(encoder_df, 168, "Long-head encoder")
        validate_inference_window(decoder_df, 720, "Long-head decoder")
        
        # Create inference DataFrame
        inference_df = create_inference_dataframe(
            encoder_df,
            decoder_df,
            self.long_config['roles'],
            plant_id=self.plant_id
        )
        
        # Create TimeSeriesDataSet from inference window
        test_ds = TimeSeriesDataSet.from_dataset(
            self.long_train_ds,
            inference_df,
            predict=True,
            stop_randomization=True
        )
        
        # Create DataLoader (single sample)
        test_dl = test_ds.to_dataloader(
            train=False,
            batch_size=1,
            num_workers=0,
            shuffle=False
        )
        
        # Run inference
        self.long_model.eval()
        with torch.no_grad():
            for x, y in test_dl:
                # Move batch to device
                x = {k: v.to(self.device) if torch.is_tensor(v) else v 
                     for k, v in x.items()}
                
                # Forward pass
                output = self.long_model(x)
                
                # Extract q50 quantile: (1, 720, Q) → (1, 720) → (720,)
                predictions = extract_q50_prediction(
                    output,
                    self.long_model.loss.quantiles
                )
                
                return predictions.squeeze()  # (720,)
        
        raise RuntimeError("No predictions generated for long-head")
    
    def _validate_forecast(self, forecast: np.ndarray, pvlib: np.ndarray):
        """Run sanity checks on final forecast."""
        checks = []
        
        # Check 1: Shape
        if len(forecast) == 2880:
            checks.append("✓ Shape correct (2880 steps)")
        else:
            checks.append(f"✗ Shape wrong: {len(forecast)} != 2880")
        
        # Check 2: Range
        if forecast.min() >= 0 and forecast.max() <= 1.0:
            checks.append(f"✓ Range valid [{forecast.min():.3f}, {forecast.max():.3f}]")
        else:
            checks.append(f"✗ Range invalid [{forecast.min():.3f}, {forecast.max():.3f}]")
        
        # Check 3: Night hours zero
        night_mask = pvlib < 0.01
        night_count = (forecast[night_mask] > 0.01).sum()
        if night_count == 0:
            checks.append("✓ All night hours zero")
        else:
            checks.append(f"⚠ {night_count} night hours non-zero")
        
        # Check 4: Reasonable daylight values
        day_mask = pvlib > 0.1
        if day_mask.sum() > 0:
            day_mean = forecast[day_mask].mean()
            checks.append(f"✓ Daylight mean: {day_mean:.3f}")
        
        for check in checks:
            print(f"         {check}")
    
    def save_forecast(
        self,
        forecast: np.ndarray,
        timestamps: pd.DatetimeIndex,
        output_path: str | Path,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Save forecast to parquet file.
        
        Args:
            forecast: Predictions @ 15-min, shape (2880,)
            timestamps: Corresponding timestamps
            output_path: Output file path
            metadata: Optional metadata dict
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame({
            "timestamp_utc": timestamps,
            "plant_id": self.pvlib_predictor.plant_id,
            "power_norm_pred": forecast,
            "power_kw_pred": forecast * self.pvlib_predictor.capacity_dc
        })
        
        if metadata:
            for key, val in metadata.items():
                df[key] = val
        
        df.to_parquet(output_path, index=False, engine="pyarrow")
        print(f"[INFO] Saved forecast: {output_path}")
        print(f"       Rows: {len(df):,}, Columns: {len(df.columns)}")


# Demo usage
if __name__ == "__main__":
    import sys
    
    print("[INFO] PhysicsAwareForecaster Demo")
    print("=" * 70)
    
    # Example paths (update for your system)
    short_ckpt = "experiments/tft/runs/germany/plant_03/15min/pvlib_warmstart_from_global_noleak/20251229_151100/checkpoints/best.ckpt"
    long_ckpt = "experiments/tft/runs/germany/plant_03/longhead/hourly720/warm/lr8e-4_do0.15_bs64_acc8_seed43/20251231_104405/checkpoints/best.ckpt"
    metadata = "data/metadata/germany/plant_03.json"
    
    # Check if files exist
    if not Path(short_ckpt).exists():
        print(f"[ERROR] Short checkpoint not found: {short_ckpt}")
        print("[INFO] Using placeholder for demo (no real TFT inference)")
    
    if not Path(long_ckpt).exists():
        print(f"[ERROR] Long checkpoint not found: {long_ckpt}")
        print("[INFO] Using placeholder for demo (no real TFT inference)")
    
    if not Path(metadata).exists():
        print(f"[ERROR] Metadata not found: {metadata}")
        sys.exit(1)
    
    # Initialize forecaster
    try:
        forecaster = PhysicsAwareForecaster(
            short_ckpt=short_ckpt,
            long_ckpt=long_ckpt,
            plant_metadata=metadata,
            device="cpu"  # Use CPU for demo
        )
    except Exception as e:
        print(f"[ERROR] Failed to load models: {e}")
        print("[INFO] This is expected if checkpoints don't exist")
        print("[INFO] Skipping demo - implement TFT inference first")
        sys.exit(0)
    
    # Generate synthetic weather for testing
    print("\n[INFO] Generating synthetic weather data for demo...")
    forecast_start = pd.Timestamp("2023-11-01 00:00:00", tz="UTC")
    timestamps_15min = pd.date_range(start=forecast_start, periods=2880, freq="15min")
    
    # Use PVLib clear-sky as synthetic weather
    clearsky_ghi = forecaster.pvlib_predictor.location.get_clearsky(timestamps_15min)
    weather_df = pd.DataFrame({
        "timestamp_utc": timestamps_15min,
        "ghi": clearsky_ghi["ghi"],
        "dni": clearsky_ghi["dni"],
        "dhi": clearsky_ghi["dhi"]
    })
    
    # Create placeholder historical data
    historical_df = pd.DataFrame()  # Empty for demo
    
    # Run forecast
    print("\n[INFO] Running 30-day forecast...")
    components = forecaster.predict_30d(
        forecast_start=forecast_start,
        weather_df=weather_df,
        historical_df=historical_df,
        alpha_day1=0.7,
        alpha_days2_30=0.5,
        return_components=True
    )
    
    # Save forecast
    output_path = "outputs/forecasts/plant_03_30d_demo.parquet"
    forecaster.save_forecast(
        forecast=components['final'],
        timestamps=timestamps_15min,
        output_path=output_path,
        metadata={
            "forecast_start": str(forecast_start),
            "alpha_day1": 0.7,
            "alpha_days2_30": 0.5,
            "model": "PhysicsAwareForecaster_v1"
        }
    )
    
    print("\n[SUCCESS] Demo complete!")
    print(f"Check output: {output_path}")
