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

import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Add project root to Python path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
from torch.utils.data import DataLoader

from src.features.sequence_generator import SimpleWindowDataset
from src.models.lstm_encoder import LSTMEncoder, LSTMEncoderConfig


def _resolve_config_path(config_path: str) -> Path:
    path = Path(config_path).expanduser()
    if path.is_absolute():
        return path
    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate
    return REPO_ROOT / path


def _resolve_runtime_path(path_like: Optional[str], config_dir: Path) -> Optional[Path]:
    if path_like in (None, ""):
        return None
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path

    repo_candidate = REPO_ROOT / path
    config_candidate = config_dir / path
    if repo_candidate.exists():
        return repo_candidate
    if config_candidate.exists():
        return config_candidate
    return repo_candidate


def load_config(path: Path) -> Dict[str, Any]:
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
    resolved_config_path = _resolve_config_path(config_path)
    cfg = load_config(resolved_config_path)
    config_dir = resolved_config_path.parent

    # Extract experiment tracking info
    exp_cfg = cfg.get("experiment", {})
    paths_cfg = cfg.get("paths", {})
    save_cfg = cfg.get("save", {})
    
    exp_name = exp_cfg.get("name", "pretrain")
    exp_tag = exp_cfg.get("tag", "default")
    output_dir = paths_cfg.get("output_dir", f"experiments/lstm/runs/{exp_tag}")
    init_weights_path = _resolve_runtime_path(paths_cfg.get("init_weights_path", None), config_dir)  # ← Stage 2 support
    encoder_save_path = _resolve_runtime_path(save_cfg.get("encoder_path", None), config_dir)  # ← Custom output path
    
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    train_path = _resolve_runtime_path(data_cfg["train_path"], config_dir)
    val_path = _resolve_runtime_path(data_cfg["val_path"], config_dir)

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
    num_workers = int(train_cfg.get("num_workers", 0))

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
        num_workers=num_workers,
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
        num_workers=num_workers,
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

    # 3.5) Load pretrained weights if specified (Stage 2 transfer learning)
    if init_weights_path:
        if init_weights_path.exists():
            print(f"[Pretrain] Loading pretrained weights from: {init_weights_path}")
            state_dict = torch.load(init_weights_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=True)
            print(f"[Pretrain] ✓ Loaded pretrained encoder (Stage 2 transfer learning)")
        else:
            print(f"[Pretrain] WARNING: init_weights_path specified but not found: {init_weights_path}")
            print(f"[Pretrain] Training from scratch instead.")

    # 4) Set up Logger
    output_dir_path = _resolve_runtime_path(output_dir, config_dir)

    logger = pl.loggers.CSVLogger(
        save_dir=str(output_dir_path),
        name=exp_name,
    )

    # 5) Set up Trainer (Lightning 2.x)
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
        default_root_dir=str(output_dir_path),
        logger=logger,
        gradient_clip_val=train_cfg.get("gradient_clip_val", 0.0),
        log_every_n_steps=50,
    )

    # 6) Train
    print(f"[Pretrain] Experiment: {exp_name} | Tag: {exp_tag}")
    print(f"[Pretrain] Output dir: {output_dir_path}")
    print(f"[Pretrain] Training LSTMEncoder for {max_epochs} epochs on {train_path.name}")
    trainer.fit(model, train_loader, val_loader)

    # 7) Save checkpoints / encoder weights
    ckpt_dir = output_dir_path / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Use config-driven naming instead of hardcoded "farm2107"
    # Default to using exp_tag for backward compatibility
    last_ckpt = ckpt_dir / f"lstm_encoder_{exp_tag}_last.ckpt"
    
    # If encoder_save_path is specified, use it; otherwise use default in checkpoints
    if encoder_save_path:
        state_dict_path = encoder_save_path
        state_dict_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        state_dict_path = ckpt_dir / f"lstm_encoder_{exp_tag}_weights.pt"

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