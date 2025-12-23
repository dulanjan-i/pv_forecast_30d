# src/training/train_tft_v1.py
"""
Stage 4: Train TFT v1.0 on regional Germany split.

Inputs
- regional_train_tft_full.parquet
- regional_val_tft_full.parquet

Outputs
- experiments/tft/runs/germany/v1_0/<run_id>/
  - checkpoints/best.ckpt
  - checkpoints/last.ckpt
  - logs/metrics.csv
  - run_config.json
  - dataset_params_train.json
  - dataset_params_val.json
  - column_roles.json

Design notes
- time_idx is computed from timestamp_utc in 15-minute bins since epoch (UTC).
- We drop:
  - poa_irradiance (GTI proxy used in LSTM stage)
  - plant_01..plant_06 one-hot columns (redundant with plant_id categorical)
  - normalized irradiance duplicates when *_raw exists (keeps *_raw)
- Optional LSTM encodings:
  - We DO NOT feed raw lstm_enc_* into the decoder horizon (leakage risk).
  - Instead we add lagged encodings: lstm_enc_*_lagK = shift(lstm_enc_*, K) per plant.
    Default K = pred_len (safe for multi-horizon forecasting).
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger

from pytorch_forecasting import TimeSeriesDataSet

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

    # Encodings
    p.add_argument("--use_lstm_encodings", action="store_true")
    p.add_argument("--enc_lag", type=int, default=None, help="Lag (in 15-min steps) for safe encoding features. Default=pred_len.")
    return p.parse_args()


def _json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)


def _ensure_datetime_utc(df: pd.DataFrame) -> pd.DataFrame:
    df[KEY_TIME] = pd.to_datetime(df[KEY_TIME], utc=True)
    return df


def _add_time_idx(df: pd.DataFrame) -> pd.DataFrame:
    # 15-min bins since epoch (UTC)
    t = df[KEY_TIME].astype("int64") // 10**9
    df["time_idx"] = (t // 900).astype("int64")
    return df


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = df[KEY_TIME]
    hour = (dt.dt.hour.to_numpy(dtype=np.float64) + dt.dt.minute.to_numpy(dtype=np.float64) / 60.0)
    doy = dt.dt.dayofyear.to_numpy(dtype=np.float64)

    df["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0).astype(np.float32)
    df["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0).astype(np.float32)
    df["doy_sin"] = np.sin(2.0 * np.pi * doy / 365.25).astype(np.float32)
    df["doy_cos"] = np.cos(2.0 * np.pi * doy / 365.25).astype(np.float32)
    return df


def _preclean(df: pd.DataFrame) -> pd.DataFrame:
    # Drop GTI proxy and redundant plant one-hot columns
    plant_onehot = [c for c in df.columns if c.startswith("plant_") and c != KEY_GROUP]
    drop_cols = ["poa_irradiance"] + plant_onehot
    drop_cols = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=drop_cols)

    # Prefer raw irradiance columns when both exist
    for base in [
        "shortwave_radiation_instant",
        "direct_normal_irradiance_instant",
        "global_tilted_irradiance_instant",
    ]:
        raw = f"{base}_raw"
        if raw in df.columns and base in df.columns:
            df = df.drop(columns=[base])

    return df


def _find_lstm_enc_cols(df: pd.DataFrame) -> List[str]:
    enc = [c for c in df.columns if c.startswith("lstm_enc_")]
    enc = sorted(enc)
    return enc


def _add_lagged_lstm_encodings(df: pd.DataFrame, lag: int) -> tuple[pd.DataFrame, List[str]]:
    """
    Create safe, lagged encoding features:
      lstm_enc_XXX_lag{lag} = shift(lstm_enc_XXX, lag) per plant_id.
    This prevents decoder leakage because decoder time u uses encoding from u-lag.
    If lag == pred_len, then all decoder steps depend only on encoder-history time.
    """
    enc_cols = _find_lstm_enc_cols(df)
    if not enc_cols:
        raise ValueError("use_lstm_encodings was set but no lstm_enc_* columns were found in dataframe.")

    lagged = df.groupby(KEY_GROUP, sort=False)[enc_cols].shift(lag)
    lagged_cols = [f"{c}_lag{lag}" for c in enc_cols]
    lagged.columns = lagged_cols

    out = pd.concat([df, lagged], axis=1)

    # Drop rows where lagged encodings are missing (startup region per plant)
    before = len(out)
    out = out.dropna(subset=lagged_cols).reset_index(drop=True)
    dropped = before - len(out)
    if dropped > 0:
        print(f"[INFO] Dropped {dropped:,} rows due to lagged encoding NaNs (lag={lag}).")

    # IMPORTANT: drop original encodings so we cannot accidentally feed leaky features
    out = out.drop(columns=enc_cols)

    return out, lagged_cols


def _write_run_metadata(
    run_dir: Path,
    args: argparse.Namespace,
    cfg: TFTConfig,
    train_ds: TimeSeriesDataSet,
    val_ds: TimeSeriesDataSet,
    roles: Dict[str, Any],
) -> None:
    _json_dump(run_dir / "dataset_params_train.json", train_ds.get_parameters())
    _json_dump(run_dir / "dataset_params_val.json", val_ds.get_parameters())
    _json_dump(run_dir / "column_roles.json", roles)

    run_cfg: Dict[str, Any] = {
        "cli_args": vars(args),
        "tft_config": asdict(cfg),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"]).decode().strip()
        run_cfg["git_commit"] = commit
        run_cfg["git_dirty"] = bool(dirty)
    except Exception:
        run_cfg["git_commit"] = None
        run_cfg["git_dirty"] = None

    _json_dump(run_dir / "run_config.json", run_cfg)


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

    # Optional safe LSTM encodings
    lag = args.enc_lag if args.enc_lag is not None else args.pred_len
    lagged_enc_cols: List[str] = []
    if args.use_lstm_encodings:
        print(f"[INFO] Adding safe LSTM encoding features with lag={lag}...")
        train, lagged_enc_cols = _add_lagged_lstm_encodings(train, lag=lag)
        val, _ = _add_lagged_lstm_encodings(val, lag=lag)

    # Known features (dynamic filter against columns present)
    known_time_reals = [
        "hour_sin", "hour_cos", "doy_sin", "doy_cos",
        "temperature_2m", "relative_humidity_2m", "precipitation", "cloud_cover",
        "wind_speed_10m", "wind_direction_10m", "surface_pressure",
        "weather_code",
        "shortwave_radiation_instant_raw",
        "direct_normal_irradiance_instant_raw",
        "global_tilted_irradiance_instant_raw",
        "direct_radiation_instant", "diffuse_radiation_instant",
        "pvlib_solar_zenith", "pvlib_solar_azimuth",
        "pvlib_poa_global", "pvlib_poa_direct", "pvlib_poa_diffuse", "pvlib_poa_ground_diffuse",
        "pvlib_dc_kw", "pvlib_ac_kw",
    ]
    # append safe lagged encodings (if enabled)
    known_time_reals += lagged_enc_cols
    known_time_reals = [c for c in known_time_reals if c in train.columns]

    unknown_time_reals = [TARGET]

    # Sort (required)
    train = train.sort_values([KEY_GROUP, "time_idx"]).reset_index(drop=True)
    val = val.sort_values([KEY_GROUP, "time_idx"]).reset_index(drop=True)

    max_encoder_length = args.encoder_len
    max_prediction_length = args.pred_len

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
        target_normalizer=None,
        add_relative_time_idx=True,
        add_target_scales=False,
        add_encoder_length=True,
        allow_missing_timesteps=True,
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

    # Trainer (force 1 GPU, pytorch-forecasting + ddp is not worth it here)
    use_gpu = torch.cuda.is_available() and args.gpus > 0
    accelerator = "gpu" if use_gpu else "cpu"
    devices = 1

    ckpt = ModelCheckpoint(
        dirpath=str(run_dir / "checkpoints"),
        filename="best",
        monitor="val_loss",
        mode="min",
        save_last=True,
        save_top_k=1,
    )
    callbacks = [
        ckpt,
        EarlyStopping(monitor="val_loss", mode="min", patience=args.patience),
        LearningRateMonitor(logging_interval="step"),
    ]

    # Metrics at run_dir/logs/metrics.csv (no version_0)
    logger = CSVLogger(save_dir=str(run_dir), name="logs", version="")

    roles = {
        "KEY_TIME": KEY_TIME,
        "KEY_GROUP": KEY_GROUP,
        "TARGET": TARGET,
        "static_categoricals": [KEY_GROUP],
        "time_varying_known_reals": known_time_reals,
        "time_varying_unknown_reals": unknown_time_reals,
        "use_lstm_encodings": bool(args.use_lstm_encodings),
        "encoding_mode": ("lagged" if args.use_lstm_encodings else "none"),
        "encoding_lag_steps": (lag if args.use_lstm_encodings else None),
        "allow_missing_timesteps": True,
        "dropped_gtiproxy_and_onehots": True,
        "kept_raw_over_normalized_irradiance": True,
    }

    _write_run_metadata(
        run_dir=run_dir,
        args=args,
        cfg=cfg,
        train_ds=train_ds,
        val_ds=val_ds,
        roles=roles,
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=accelerator,
        devices=devices,
        precision=args.precision,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=50,
        gradient_clip_val=0.1,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    trainer.fit(model, train_loader, val_loader)

    print("[DONE] TFT training complete.")
    print(f"[INFO] best checkpoint: {ckpt.best_model_path}")
    print(f"[INFO] metrics: {run_dir / 'logs' / 'metrics.csv'}")


if __name__ == "__main__":
    main()
