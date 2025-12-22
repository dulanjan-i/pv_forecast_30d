# src/training/train_tft_v1.py
"""
Stage 4: Train TFT v1.0 on regional Germany split.

Inputs
- regional_train_tft_full.parquet
- regional_val_tft_full.parquet

Outputs
- experiments/tft/runs/germany/v1_0/<run_id>/
  - best.ckpt
  - last.ckpt
  - metrics.csv

Notes
- This script avoids fragile Lightning internals (no device_parser).
- time_idx is computed from timestamp_utc in 15-minute bins since epoch,
  so train/val share the same time axis.
- We drop:
  - poa_irradiance (GTI proxy used in LSTM stage)
  - plant_01..plant_06 one-hot columns (redundant with plant_id categorical)
  - normalized irradiance duplicates when raw exists (keeps *_raw)
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, UTC
from pathlib import Path

import pandas as pd
import torch

# Lightning import that works for lightning>=2.x
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer

from src.models.tft_model import TFTConfig, build_tft_model, make_dataloaders


KEY_TIME = "timestamp_utc"
KEY_GROUP = "plant_id"
TARGET = "power_norm"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train_parquet", type=str, required=True)
    p.add_argument("--val_parquet", type=str, required=True)
    p.add_argument("--run_root", type=str, default="experiments/tft/runs/germany/v1_0")
    p.add_argument("--max_epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--encoder_len", type=int, default=96)
    p.add_argument("--pred_len", type=int, default=96)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden_size", type=int, default=64)
    p.add_argument("--lstm_layers", type=int, default=2)
    p.add_argument("--attn_heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--precision", type=str, default="16-mixed")
    p.add_argument("--patience", type=int, default=5)
    return p.parse_args()


def _ensure_datetime_utc(df: pd.DataFrame) -> pd.DataFrame:
    df[KEY_TIME] = pd.to_datetime(df[KEY_TIME], utc=True)
    return df


def _add_time_idx(df: pd.DataFrame) -> pd.DataFrame:
    # 15-min bins since epoch
    # int64 nanoseconds -> seconds -> /900
    t = df[KEY_TIME].astype("int64") // 10**9
    df["time_idx"] = (t // 900).astype("int64")
    return df


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = df[KEY_TIME]
    hour = dt.dt.hour + dt.dt.minute / 60.0
    doy = dt.dt.dayofyear.astype("float")

    # cyclic
    df["hour_sin"] = torch.sin(torch.tensor(2.0 * 3.141592653589793 * hour / 24.0)).numpy()
    df["hour_cos"] = torch.cos(torch.tensor(2.0 * 3.141592653589793 * hour / 24.0)).numpy()
    df["doy_sin"] = torch.sin(torch.tensor(2.0 * 3.141592653589793 * doy / 365.25)).numpy()
    df["doy_cos"] = torch.cos(torch.tensor(2.0 * 3.141592653589793 * doy / 365.25)).numpy()
    return df


def _preclean(df: pd.DataFrame) -> pd.DataFrame:
    # Drop GTI proxy and redundant plant one-hot columns
    plant_onehot = [c for c in df.columns if c.startswith("plant_") and c != KEY_GROUP]
    drop_cols = ["poa_irradiance"] + plant_onehot
    drop_cols = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=drop_cols)

    # Prefer raw irradiance columns when both exist
    for base in ["shortwave_radiation_instant", "direct_normal_irradiance_instant", "global_tilted_irradiance_instant"]:
        raw = f"{base}_raw"
        if raw in df.columns and base in df.columns:
            df = df.drop(columns=[base])

    return df


def main() -> None:
    args = parse_args()

    run_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.run_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Stage 4: TFT v1.0 training")
    print(f"train_parquet: {args.train_parquet}")
    print(f"val_parquet:   {args.val_parquet}")
    print(f"run_dir:       {run_dir}")
    print("=" * 80)

    train = pd.read_parquet(args.train_parquet)
    val = pd.read_parquet(args.val_parquet)

    train = _ensure_datetime_utc(train)
    val = _ensure_datetime_utc(val)

    train = _preclean(train)
    val = _preclean(val)

    train = _add_time_idx(train)
    val = _add_time_idx(val)

    train = _add_time_features(train)
    val = _add_time_features(val)

    # Feature lists (derive dynamically so you do not hardcode 90 columns)
    known_time_reals = [
        # time features
        "hour_sin", "hour_cos", "doy_sin", "doy_cos",
        # weather (keep whatever exists)
        "temperature_2m", "relative_humidity_2m", "precipitation", "cloud_cover",
        "wind_speed_10m", "wind_direction_10m", "surface_pressure",
        "weather_code",
        # irradiance
        "shortwave_radiation_instant_raw",
        "direct_normal_irradiance_instant_raw",
        "global_tilted_irradiance_instant_raw",
        "direct_radiation_instant", "diffuse_radiation_instant",
        # pvlib (these are computable from weather + metadata)
        "pvlib_solar_zenith", "pvlib_solar_azimuth",
        "pvlib_poa_global", "pvlib_poa_direct", "pvlib_poa_diffuse", "pvlib_poa_ground_diffuse",
        "pvlib_dc_kw", "pvlib_ac_kw",
    ]
    known_time_reals = [c for c in known_time_reals if c in train.columns]

    # For v1.0 baseline: do NOT include LSTM encodings yet (prevents leakage into decoder horizon).
    # You can add them in v1.1 with "history-only" handling.
    unknown_time_reals = [TARGET]

    # Dataset
    max_encoder_length = args.encoder_len
    max_prediction_length = args.pred_len

    # Ensure sorted (required by TimeSeriesDataSet)
    train = train.sort_values([KEY_GROUP, "time_idx"]).reset_index(drop=True)
    val = val.sort_values([KEY_GROUP, "time_idx"]).reset_index(drop=True)

    train_ds = TimeSeriesDataSet(
        train,
        time_idx="time_idx",
        target=TARGET,
        group_ids=[KEY_GROUP],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=[KEY_GROUP],
        time_varying_known_reals=known_time_reals,
        time_varying_unknown_reals=unknown_time_reals,
        target_normalizer=None,  # target is already power_norm
        add_relative_time_idx=True,
        add_target_scales=False,
        add_encoder_length=True,
        allow_missing_timesteps=True,  # <-- THIS FIXES YOUR CRASH
    )

    val_ds = TimeSeriesDataSet.from_dataset(
        train_ds,
        val,
        predict=True,
        stop_randomization=True,
    )

    cfg = TFTConfig(
        target=TARGET,
        time_idx="time_idx",
        group_ids=[KEY_GROUP],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        learning_rate=args.lr,
        hidden_size=args.hidden_size,
        lstm_layers=args.lstm_layers,
        attention_head_size=args.attn_heads,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
    )

    model = build_tft_model(cfg, train_ds)

    train_loader, val_loader, _ = make_dataloaders(
        train_ds,
        val_ds=val_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Trainer setup (no device_parser)
    use_gpu = torch.cuda.is_available() and args.gpus > 0
    accelerator = "gpu" if use_gpu else "cpu"
    devices = args.gpus if use_gpu else 1

    ckpt = ModelCheckpoint(
        dirpath=str(run_dir),
        filename="best",
        monitor="val_loss",
        mode="min",
        save_last=True,
    )

    callbacks = [
        ckpt,
        EarlyStopping(monitor="val_loss", mode="min", patience=args.patience),
        LearningRateMonitor(logging_interval="step"),
    ]

    logger = CSVLogger(save_dir=str(run_dir), name="logs")

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=accelerator,
        devices=devices,
        strategy="ddp" if (use_gpu and devices > 1) else "auto",
        precision=args.precision,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=50,
        gradient_clip_val=0.1,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    trainer.fit(model, train_loader, val_loader)

    # Save run config for reproducibility
    (run_dir / "run_config.json").write_text(pd.Series(asdict(cfg)).to_json(), encoding="utf-8")

    print("[DONE] TFT training complete.")
    print(f"[INFO] best checkpoint: {ckpt.best_model_path}")
    print(f"[INFO] last checkpoint: {(run_dir / 'last.ckpt') if (run_dir / 'last.ckpt').exists() else 'saved by lightning callback'}")


if __name__ == "__main__":
    main()
