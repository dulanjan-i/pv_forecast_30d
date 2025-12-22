# src/models/tft_model.py
"""
References:
- Lim et al. (2020) Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting.
  NeurIPS. arXiv:1912.09363 https://arxiv.org/abs/1912.09363
- PyTorch Forecasting TFT: https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.models.temporal_fusion_transformer.TemporalFusionTransformer.html
- TimeSeriesDataSet: https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.data.timeseries.TimeSeriesDataSet.html
- LightningModule: https://lightning.ai/docs/pytorch/stable/common/lightning_module.html
"""
#============================================================================#
# ------------ CODE FOR TEMPORAL FUSION TRANSFORMER (TFT) -------------- #
#============================================================================#
"""
Temporal Fusion Transformer (TFT) wrapper for MiRACLE.

This file provides:
- TFTConfig: a dataclass holding all hyperparameters and dataset column names.
- build_tft_model(...): constructs a TemporalFusionTransformer from a TimeSeriesDataSet.
- make_dataloaders(...): returns train/val (and optional test) dataloaders.

Notes
-----
* This uses pytorch-forecasting (which internally uses PyTorch Lightning).
* You train by calling pl.Trainer(...).fit(model, train_loader, val_loader).
* The TemporalFusionTransformer returned here is a LightningModule (already implements training/validation loops).

Minimal usage
-------------
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_lightning import Trainer
from src.models.tft_model import TFTConfig, build_tft_model, make_dataloaders

# 1) Prepare TimeSeriesDataSet (you do this in your data pipeline)
# train_ds = TimeSeriesDataSet(...)
# val_ds   = TimeSeriesDataSet(...)

cfg = TFTConfig(
    target="pv_power",
    time_idx="time_idx",
    group_ids=["site_id"],

    max_encoder_length=96,    # history length (e.g., 24h @ 15-min)
    max_prediction_length=96, # forecast horizon (adjust to your use case)

    # Feature lists must match your TimeSeriesDataSet definition:
    static_categoricals=["site_id"],
    static_reals=["capacity_kw"],
    time_varying_known_categoricals=[],
    time_varying_known_reals=["hour_sin", "hour_cos", "dayofyear_sin", "dayofyear_cos"],
    time_varying_unknown_reals=["pv_power"],  # observed target during encoder
)

model = build_tft_model(cfg, train_ds, val_ds)
train_loader, val_loader, _ = make_dataloaders(train_ds, val_ds, batch_size=128, num_workers=4)

trainer = Trainer(
    max_epochs=30,
    gradient_clip_val=cfg.gradient_clip_val,
    accelerator="auto",
    log_every_n_steps=50,
)
trainer.fit(model, train_loader, val_loader)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

# PyTorch Lightning / Forecasting
import pytorch_lightning as pl
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss, RMSE, MAPE


@dataclass
class TFTConfig:
    """
    Configuration for building a TemporalFusionTransformer from a TimeSeriesDataSet.

    Core dataset columns
    --------------------
    target: str
        Name of the target column in your dataframe (e.g., "pv_power").
    time_idx: str
        Name of the integer time index column (e.g., "time_idx").
    group_ids: List[str]
        Column names identifying each time series entity (e.g., ["site_id"]).

    Sequence lengths
    ----------------
    max_encoder_length: int
        Number of past time steps the encoder should see.
    max_prediction_length: int
        Number of future time steps to forecast.

    Feature definitions (must match your TimeSeriesDataSet)
    -------------------------------------------------------
    static_categoricals: List[str]
    static_reals: List[str]
    time_varying_known_categoricals: List[str]
    time_varying_known_reals: List[str]
    time_varying_unknown_reals: List[str]

    Model hyperparameters
    ---------------------
    hidden_size: int
        Model width for LSTM/GLU/etc.
    lstm_layers: int
        Number of LSTM layers used in TFT.
    attention_head_size: int
        Number of heads for multi-head attention.
    dropout: float
        Dropout rate applied in TFT blocks.
    learning_rate: float
        Optimizer learning rate.
    weight_decay: float
        AdamW weight decay.

    Loss / output
    -------------
    quantiles: Optional[List[float]]
        Quantiles for probabilistic forecasting. If None, defaults to [0.1, 0.5, 0.9].

    Training utils
    --------------
    batch_size: int
        Default batch size for dataloaders (can be overridden in make_dataloaders()).
    num_workers: int
        Default number of workers for dataloaders.
    gradient_clip_val: float
        Gradient clipping value for trainer.
    """
    # dataset keys
    target: str
    time_idx: str
    group_ids: List[str]

    # sequence lengths
    max_encoder_length: int = 96
    max_prediction_length: int = 96

    # feature sets
    static_categoricals: List[str] = None
    static_reals: List[str] = None
    time_varying_known_categoricals: List[str] = None
    time_varying_known_reals: List[str] = None
    time_varying_unknown_reals: List[str] = None

    # model hparams
    hidden_size: int = 64
    lstm_layers: int = 2
    attention_head_size: int = 4
    dropout: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    # loss / output
    quantiles: Optional[List[float]] = None  # defaults inside builder

    # training utilities
    batch_size: int = 128
    num_workers: int = 4
    gradient_clip_val: float = 0.1

    def __post_init__(self):
        # default lists
        self.static_categoricals = self.static_categoricals or []
        self.static_reals = self.static_reals or []
        self.time_varying_known_categoricals = self.time_varying_known_categoricals or []
        self.time_varying_known_reals = self.time_varying_known_reals or []
        self.time_varying_unknown_reals = self.time_varying_unknown_reals or []
        # default quantiles
        if self.quantiles is None:
            self.quantiles = [0.1, 0.5, 0.9]


def build_tft_model(
    cfg: TFTConfig,
    train_ds: TimeSeriesDataSet,
    val_ds: Optional[TimeSeriesDataSet] = None,
) -> TemporalFusionTransformer:
    """
    Construct a TemporalFusionTransformer from a TimeSeriesDataSet and config.

    Parameters
    ----------
    cfg : TFTConfig
        Configuration with model hyperparameters and dataset semantics.
    train_ds : TimeSeriesDataSet
        Prepared training dataset (defines encoders/embeddings and scalers).
    val_ds : Optional[TimeSeriesDataSet]
        Optional validation dataset (used only for validation during fit).

    Returns
    -------
    model : TemporalFusionTransformer (LightningModule)
        A ready-to-train TFT Lightning module.

    Notes
    -----
    * from_dataset() automatically infers embedding sizes for categorical features and
      normalization for continuous features from the provided TimeSeriesDataSet.
    * Loss is set to QuantileLoss for probabilistic forecasts using cfg.quantiles.
    """
    loss = QuantileLoss(quantiles=cfg.quantiles)

    model = TemporalFusionTransformer.from_dataset(
        train_ds,
        learning_rate=cfg.learning_rate,
        hidden_size=cfg.hidden_size,
        lstm_layers=cfg.lstm_layers,
        attention_head_size=cfg.attention_head_size,
        dropout=cfg.dropout,
        loss=loss,
        # Additional common toggles (keep defaults conservative; adjust as needed)
        optimizer="adamw",
        optimizer_params={"weight_decay": cfg.weight_decay},
        reduce_on_plateau_patience=4,
        output_size=len(cfg.quantiles),  # number of quantiles
        log_interval=50,
    )

    # Optional sanity check: make sure the model expects the same forecast horizon
    assert model.hparams.max_prediction_length == cfg.max_prediction_length, (
        f"TFT max_prediction_length ({model.hparams.max_prediction_length}) "
        f"!= cfg.max_prediction_length ({cfg.max_prediction_length}). "
        "Ensure your TimeSeriesDataSet and config use consistent lengths."
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
    Build PyTorch DataLoaders for TFT training/validation/testing.

    Parameters
    ----------
    train_ds : TimeSeriesDataSet
        Prepared training dataset.
    val_ds : Optional[TimeSeriesDataSet]
        Optional validation dataset.
    test_ds : Optional[TimeSeriesDataSet]
        Optional test dataset.
    batch_size : int
        Batch size for all loaders.
    num_workers : int
        Workers per DataLoader.

    Returns
    -------
    train_loader, val_loader, test_loader : Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]
    """
    train_loader = train_ds.to_dataloader(train=True,  batch_size=batch_size, num_workers=num_workers)
    val_loader = None
    test_loader = None

    if val_ds is not None:
        val_loader = val_ds.to_dataloader(train=False, batch_size=batch_size, num_workers=num_workers)

    if test_ds is not None:
        test_loader = test_ds.to_dataloader(train=False, batch_size=batch_size, num_workers=num_workers)

    return train_loader, val_loader, test_loader