# src/training/pretrain_lstm.py
"""
Pretraining script for LSTMEncoder on global PVDAQ/NSRDB data.
We use the Farm Solar Array (PVDAQ System 2107) dataset for this pretraining.
Data is from: https://data.openei.org/s3_viewer?bucket=oedi-data-lake&prefix=pvdaq%2F2023-solar-data-prize%2F2107_OEDI%2F

Uses:
- experiments/lstm/pretrain_farm2107.yaml


YAML structure expected:

data:
  train_path: "data/processed/pretraining/farm2107_pretrain_train.parquet"
  val_path: "data/processed/pretraining/farm2107_pretrain_val.parquet"
  time_col: "measured_on"
  id_col: null
  feature_cols:
    - pv_power_norm
    - poa_irradiance
    - temperature_2m
    - relative_humidity_2m
    - precipitation
    - cloud_cover
    - wind_speed_10m
    - wind_direction_10m
    - shortwave_radiation_instant
    - direct_radiation_instant
    - diffuse_radiation_instant
    - direct_normal_irradiance_instant
    - global_tilted_irradiance_instant
    - surface_pressure
  target_col: "pv_power_norm"
  window_size: 96
  horizon: 1

model:
  hidden_size: 64
  num_layers: 2
  dropout: 0.1

training:
  batch_size: 256
  max_epochs: 30
  learning_rate: 1e-3
  weight_decay: 1e-4
  gpus: auto         #depends on runtime environment and availability,
                     [M2 Mac: 0, Mira's RTX: 1, DBFZ Calc: 4, HPC: auto] 

This trains LSTMEncoder to do next-step prediction on sliding windows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
from torch.utils.data import DataLoader

from src.features.sequence_generator import SimpleWindowDataset
from src.models.lstm_encoder import LSTMEncoder, LSTMEncoderConfig


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_dataloader(
    csv_path: Path,
    time_col: str,
    id_col: Optional[str],
    feature_cols,
    target_col: str,
    window_size: int,
    horizon: int,
    batch_size: int,
    num_workers: int = 0,
    shuffle: bool = True, 
) -> DataLoader:
    assert csv_path.exists(), f"File not found at {csv_path}"

    ext = csv_path.suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(csv_path)
    elif ext == ".parquet":
        df = pd.read_parquet(csv_path)
    else:
        raise ValueError(f"Unsupported file extension for {csv_path}")

    if id_col in ("", "none", "null", None):
        group_col = None
    else:
        group_col = id_col

    dataset = SimpleWindowDataset(
        df=df,
        time_col=time_col,
        group_col=group_col,
        feature_cols=feature_cols,
        target_col=target_col,
        input_window=window_size,
        forecast_horizon=horizon,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return loader


def main(config_path: str = "experiments/lstm/pretrain_farm2107.yaml") -> None:
    # 1) Load config
    cfg = load_config(config_path)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    train_path = Path(data_cfg["train_path"])
    val_path = Path(data_cfg["val_path"])

    time_col = data_cfg["time_col"]
    id_col = data_cfg.get("id_col", None)
    feature_cols = data_cfg["feature_cols"]
    target_col = data_cfg["target_col"]
    window_size = int(data_cfg["window_size"])
    horizon = int(data_cfg["horizon"])

    batch_size = int(train_cfg["batch_size"])
    max_epochs = int(train_cfg["max_epochs"])
    lr = float(train_cfg["learning_rate"])
    weight_decay = float(train_cfg["weight_decay"])
    gpus = train_cfg.get("gpus", "auto")

    hidden_size = int(model_cfg["hidden_size"])
    num_layers = int(model_cfg["num_layers"])
    dropout = float(model_cfg["dropout"])

    # 2) Build dataloaders
    train_loader = build_dataloader(
        csv_path=train_path,
        time_col=time_col,
        id_col=id_col,
        feature_cols=feature_cols,
        target_col=target_col,
        window_size=window_size,
        horizon=horizon,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True,
    )

    val_loader = build_dataloader(
        csv_path=val_path,
        time_col=time_col,
        id_col=id_col,
        feature_cols=feature_cols,
        target_col=target_col,
        window_size=window_size,
        horizon=horizon,
        batch_size=batch_size,
        num_workers=0,
        shuffle=False,
    )

    # 3) Instantiate LSTMEncoder
    input_size = len(feature_cols)

    enc_cfg = LSTMEncoderConfig(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay,  # Pass weight_decay from config
        aux_predict=True,  # we want next-step prediction head during pretraining
    )

    model = LSTMEncoder(enc_cfg)

    # 4) Set up Trainer (Lightning 1.x)
    if isinstance(gpus, str) and gpus == "auto":
        accelerator = "auto"
        devices = "auto"
    elif isinstance(gpus, int) and gpus > 0:
        accelerator = "gpu"
        devices = gpus
    else:
        accelerator = "cpu"
        devices = None

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        gradient_clip_val=train_cfg.get("gradient_clip_val", 0.0),
        log_every_n_steps=50,
    )

    # 5) Train
    print(f"[Pretrain] Training LSTMEncoder for {max_epochs} epochs on {train_path.name}")
    trainer.fit(model, train_loader, val_loader)

    # 6) Save checkpoints / encoder weights
    ckpt_dir = Path("experiments/lstm/encoders")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    last_ckpt = ckpt_dir / "lstm_encoder_farm2107_last.ckpt"
    state_dict_path = ckpt_dir / "lstm_encoder_farm2107_weights.pt"

    trainer.save_checkpoint(str(last_ckpt))
    torch.save(model.state_dict(), state_dict_path)

    print(f"[Pretrain] Saved final checkpoint to: {last_ckpt}")
    print(f"[Pretrain] Saved encoder weights to: {state_dict_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pretrain LSTM Encoder on Farm Solar Array [PVDAQ System 2107].")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/lstm/pretrain_farm2107.yaml",
        help="Path to YAML config for pretraining.",
    )
    args = parser.parse_args()

    main(config_path=args.config) 