# src/models/tft_model.py
"""
Temporal Fusion Transformer (TFT) wrapper for MiRACLE.

This file provides:
- TFTConfig: hyperparameters and feature lists (dataset semantics).
- build_tft_model(...): creates a TemporalFusionTransformer from a TimeSeriesDataSet.
- make_dataloaders(...): returns DataLoaders for train/val/test.

Design choices
- Keep this file Lightning-version agnostic.
- Do not reference internal Lightning utilities or fragile hparams keys.
- Let TimeSeriesDataSet define encoders/scalers, then create TFT via from_dataset().
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from torch.utils.data import DataLoader

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss


@dataclass
class TFTConfig:
    """
    Minimal config used to build TFT from a TimeSeriesDataSet.

    Required dataset semantics:
    - target: target column name
    - time_idx: integer time index column name
    - group_ids: list of grouping columns, usually ["plant_id"]

    Sequence lengths:
    - max_encoder_length: history length
    - max_prediction_length: forecast horizon

    Hyperparameters:
    - hidden_size, lstm_layers, attention_head_size, dropout, learning_rate, weight_decay

    Probabilistic output:
    - quantiles: list of quantiles for QuantileLoss
    """
    target: str
    time_idx: str
    group_ids: List[str]

    max_encoder_length: int = 96
    max_prediction_length: int = 96

    hidden_size: int = 64
    lstm_layers: int = 2
    attention_head_size: int = 4
    dropout: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    quantiles: Optional[List[float]] = None

    def __post_init__(self) -> None:
        if self.quantiles is None:
            self.quantiles = [0.1, 0.5, 0.9]


def build_tft_model(
    cfg: TFTConfig,
    train_ds: TimeSeriesDataSet,
) -> TemporalFusionTransformer:
    """
    Construct a TemporalFusionTransformer from a TimeSeriesDataSet.

    Important:
    - We do not assert on model.hparams keys because they vary across versions.
    - TimeSeriesDataSet already encodes categorical variables and normalizes reals.
    """
    loss = QuantileLoss(quantiles=cfg.quantiles)

    model = TemporalFusionTransformer.from_dataset(
        train_ds,
        learning_rate=cfg.learning_rate,
        hidden_size=cfg.hidden_size,
        lstm_layers=cfg.lstm_layers,
        attention_head_size=cfg.attention_head_size,
        dropout=cfg.dropout,
        weight_decay=cfg.weight_decay,
        loss=loss,
        optimizer="adamw",
        reduce_on_plateau_patience=4,
        output_size=len(cfg.quantiles),
        log_interval=50,
    )
    return model


def make_dataloaders(
    train_ds: TimeSeriesDataSet,
    val_ds: Optional[TimeSeriesDataSet] = None,
    test_ds: Optional[TimeSeriesDataSet] = None,
    batch_size: int = 128,
    num_workers: int = 4,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """
    Build PyTorch Forecasting dataloaders with sane performance defaults.
    """
    # These kwargs get forwarded to torch DataLoader in pytorch-forecasting
    dl_kwargs = dict(
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=True,
    )
    # Remove None keys (DataLoader errors on prefetch_factor=None)
    dl_kwargs = {k: v for k, v in dl_kwargs.items() if v is not None}

    train_loader = train_ds.to_dataloader(train=True, batch_size=batch_size, **dl_kwargs)

    val_loader = None
    if val_ds is not None:
        val_loader = val_ds.to_dataloader(train=False, batch_size=batch_size, **dl_kwargs)

    test_loader = None
    if test_ds is not None:
        test_loader = test_ds.to_dataloader(train=False, batch_size=batch_size, **dl_kwargs)

    return train_loader, val_loader, test_loader
