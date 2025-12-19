# src/models/lstm_encoder.py
"""
References:
- Hochreiter & Schmidhuber (1997) Long Short-Term Memory. Neural Computation.
  https://doi.org/10.1162/neco.1997.9.8.1735
- PyTorch nn.LSTM docs: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
- PyTorch Lightning LightningModule: https://lightning.ai/docs/pytorch/stable/common/lightning_module.html
"""

from __future__ import annotations              # postpone type-hint evaluation (cleaner imports)

from dataclasses import dataclass               # structured config (YAML -> dataclass)
from typing import Optional, Tuple, Dict, Any   

import torch
from torch import nn
from torch.utils.data import Dataset
import pytorch_lightning as pl                   # Lightning for training loops, logging, checkpoints


# =========================
# 1) Configuration dataclass
# =========================

@dataclass
class LSTMEncoderConfig:
    """
    Hyperparameters and runtime knobs for the LSTM encoder.

    Args:
        input_size: Number of input features per timestep (F).
        hidden_size: Hidden dimension of the LSTM.
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout applied *between* LSTM layers (if num_layers > 1)
                 and on the projection head inputs.
        lr: Learning rate for the optimizer.
        weight_decay: Weight decay coefficient for AdamW (L2).
        aux_predict: If True, enable an auxiliary regression head for
                     next-step PV prediction (stabilizes representation).
        embedding_dim: If set, projects the last hidden state to this dim.
                       If None, the raw hidden_size is used.
        loss_reduction: "mean" or "sum" for training loss aggregation.
    """
    input_size: int
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.0
    lr: float = 1e-3
    weight_decay: float = 0.0
    aux_predict: bool = True
    embedding_dim: Optional[int] = None
    loss_reduction: str = "mean"


# =========================
# 2) LightningModule: LSTM Encoder
# =========================

class LSTMEncoder(pl.LightningModule):
    """
    LSTM-based temporal encoder that turns a fixed-length window of features
    (B, T, F) into a compact embedding (B, D). Optionally learns a small
    auxiliary regression head to predict the next timestep target.

    Typical usage:
    1) Train with windows and (optionally) next-step targets to shape the space.
    2) Call .encode(x) at inference to obtain the encoded learner signal.
    3) Persist weights (.state_dict()) and reuse the frozen encoder downstream.

    Shapes:
        x:  (B, T, F)   — batch of lookback windows
        y:  (B,)        — next-step scalar target (if aux_predict == True)

    Returns (forward):
        {
          "embedding": (B, D),
          "next_pred": (B,) or None
        }
    """

    def __init__(self, cfg: LSTMEncoderConfig):
        super().__init__()
        self.save_hyperparameters()  # Logs config to the checkpoint
        self.cfg = cfg

        # Core LSTM
        self.lstm = nn.LSTM(
            input_size=cfg.input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            bidirectional=False,  # keep simple; can turn this on later if i need bi-directional LSTMs
        )

        # Projection to embedding space (optional)
        emb_in = cfg.hidden_size
        emb_out = cfg.embedding_dim if cfg.embedding_dim is not None else cfg.hidden_size
        self.proj = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(emb_in, emb_out),
        )

        # Auxiliary "next step" regression head (optional)
        if cfg.aux_predict:
            self.next_head = nn.Sequential(
                nn.Dropout(cfg.dropout),
                nn.Linear(emb_out, 1)
            )
        else:
            self.next_head = None

        self.loss_fn = nn.MSELoss(reduction=cfg.loss_reduction)

    # -------------
    # Core forward
    # -------------
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through LSTM -> embedding (-> optional next-step head).

        Args:
            x: Tensor of shape (B, T, F)

        Returns:
            A dict with:
                "embedding": (B, D)
                "next_pred": (B,) or None
        """
        # LSTM returns (output, (h_n, c_n))
        # - output: (B, T, H) all timesteps
        # - h_n:    (num_layers, B, H) last hidden state per layer
        out, (h_n, c_n) = self.lstm(x)
        # Take the final layer's hidden state → (B, H)
        last_h = h_n[-1]  # (B, hidden_size)

        # Project to embedding
        embedding = self.proj(last_h)  # (B, D)

        # Optional next-step head
        next_pred = None
        if self.next_head is not None:
            next_pred = self.next_head(embedding).squeeze(-1)  # (B,)

        return {"embedding": embedding, "next_pred": next_pred}

    # ----------------
    # Training/val/test
    # ----------------
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """
        One training step using the auxiliary regression loss if enabled.
        """
        x, y = batch  # x: (B, T, F), y: (B,)
        out = self.forward(x)
        if self.next_head is None:
            # If no aux head, define a dummy loss (not recommended for training).
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        else:
            loss = self.loss_fn(out["next_pred"], y)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """
        Validation step mirrors training; logs val_loss for early stopping.
        """
        x, y = batch
        out = self.forward(x)
        if self.next_head is None:
            loss = torch.tensor(0.0, device=self.device)
        else:
            loss = self.loss_fn(out["next_pred"], y)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """
        Test step; optional, same metric as validation.
        """
        x, y = batch
        out = self.forward(x)
        if self.next_head is None:
            loss = torch.tensor(0.0, device=self.device)
        else:
            loss = self.loss_fn(out["next_pred"], y)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    # -------------------
    # Optimizer / sched
    # -------------------
    def configure_optimizers(self):
        """
        AdamW optimizer; you can add LR schedulers later.
        """
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay
        )
        return opt

    # ---------------
    # Convenience API
    # ---------------
    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns only the embedding for input windows (no gradient).

        Args:
            x: (B, T, F)

        Returns:
            (B, D) embedding tensor
        """
        self.eval()
        out = self.forward(x)
        return out["embedding"]
    

# =========================
# 3) Minimal Dataset for windows
# =========================

class SimpleWindowDataset(Dataset):
    """
    A tiny dataset wrapper for supervised window learning.

    Assumes fixed-length windows and (optionally) matching next-step targets.

    Args:
        windows: Tensor of shape (N, T, F)
        next_y:  Optional tensor of shape (N,)
    """
    def __init__(self, windows: torch.Tensor, next_y: Optional[torch.Tensor] = None):
        super().__init__()
        assert windows.ndim == 3, "windows must be (N, T, F)"
        self.windows = windows
        self.next_y = next_y

        if self.next_y is not None:
            assert len(self.next_y) == len(self.windows), "length mismatch"

    def __len__(self) -> int:
        return self.windows.shape[0]

    def __getitem__(self, idx: int):
        x = self.windows[idx]
        if self.next_y is None:
            return x
        y = self.next_y[idx]
        return x, y


# =========================
# 4) Trainer factory (Lightning)
# =========================

def make_trainer(
    max_epochs: int = 10,
    gpus: int = 0,
    precision: Optional[str] = None,
    ckpt_path: Optional[str] = None,
    strategy: Optional[str] = None,
) -> pl.Trainer:
    """
    Small convenience wrapper to build a Lightning Trainer.

    Args:
        max_epochs: Number of epochs to train.
        gpus:       Number of GPUs. If 0, uses CPU.
        precision:  Optional mixed precision (e.g. "16-true").
        ckpt_path:  Optional default checkpoint directory.

    Returns:
        pl.Trainer instance
    """
    accelerator = "gpu" if gpus and torch.cuda.is_available() else "cpu"
    if accelerator == "gpu":
        devices = gpus
        ddp_strategy = strategy or ("ddp" if gpus > 1 else "auto")
    else:
        devices = 1
        ddp_strategy = "auto"

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        strategy=ddp_strategy,
        precision=precision,
        log_every_n_steps=10,
        enable_checkpointing=True,
        # You can add callbacks/early stopping later.
    )
    return trainer
