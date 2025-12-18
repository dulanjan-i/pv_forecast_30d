"""
src/training/train_global_lstm_v3.py

Version 3 - Global Forecasting Model training with Rolling Origin CV.

Critical fix:
- WindowDataset must NOT slide across the entire concatenated dataframe.
  It must build windows per plant_id, otherwise windows can contain multiple plants.

This file assumes preprocessing created:
- data/processed/pretraining/germany/global/fold_{k}_train.parquet
- data/processed/pretraining/germany/global/fold_{k}_val.parquet
"""

from __future__ import annotations

import sys
import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.data.schema import (
    GLOBAL_LSTM_INPUT_FEATURES,
    PLANT_ID_COL,
    TARGET_COL,
    TIME_COL,
    TIME_STEP_MINUTES,
)
from src.models.global_lstm_encoder import GlobalLSTMEncoder, transfer_from_farm2107, LSTMEncoderConfig


class GroupedWindowDataset(Dataset):
    """
    Sliding-window dataset built per plant_id.

    Why this exists:
    - If you slide across the full supermatrix you can create a window that starts in plant A
      and ends in plant B. That destroys learning and evaluation.

    Window definition:
    - X: [t - window_size .. t-1]  shape: (window_size, n_features)
    - y: target at time t          shape: scalar

    Gap handling:
    - We only keep windows where timestamps are strictly regular at TIME_STEP_MINUTES.
      This prevents windows from "jumping over" missing rows created by NaN drops.

    Parameters
    ----------
    df : pd.DataFrame
        Must include columns: TIME_COL, PLANT_ID_COL, GLOBAL_LSTM_INPUT_FEATURES, TARGET_COL
    window_size : int
        Number of timesteps in the input window
    stride : int
        Step between candidate windows
    """

    def __init__(self, df: pd.DataFrame, window_size: int = 96, stride: int = 1):
        self.window_size = int(window_size)
        self.stride = int(stride)

        # Defensive copy and sort
        d = df.copy()
        d[TIME_COL] = pd.to_datetime(d[TIME_COL], utc=True)
        d = d.sort_values([PLANT_ID_COL, TIME_COL]).reset_index(drop=True)

        required = set([TIME_COL, PLANT_ID_COL, TARGET_COL] + GLOBAL_LSTM_INPUT_FEATURES)
        missing = sorted(required - set(d.columns))
        if missing:
            raise ValueError(f"GroupedWindowDataset: missing required columns: {missing}")

        # Hard fail on NaNs because model code expects none
        X_all = d[GLOBAL_LSTM_INPUT_FEATURES].to_numpy()
        y_all = d[TARGET_COL].to_numpy()
        if np.isnan(X_all).any() or np.isnan(y_all).any():
            raise ValueError(
                f"Dataset contains NaNs. X NaNs={np.isnan(X_all).sum()}, y NaNs={np.isnan(y_all).sum()}"
            )

        self._by_plant: Dict[str, Dict[str, np.ndarray]] = {}
        self._index: List[Tuple[str, int]] = []  # (plant_id, start_idx)

        freq_s = int(TIME_STEP_MINUTES * 60)

        # Build per-plant arrays and valid window starts
        for plant_id, g in d.groupby(PLANT_ID_COL, sort=True):
            g = g.sort_values(TIME_COL).reset_index(drop=True)

            times = g[TIME_COL].astype("int64").to_numpy() // 10**9  # seconds
            X = g[GLOBAL_LSTM_INPUT_FEATURES].to_numpy(dtype=np.float32)
            y = g[TARGET_COL].to_numpy(dtype=np.float32)

            n = len(g)
            if n <= self.window_size:
                continue

            # Check regularity: diff must equal freq_s
            diffs = np.diff(times)
            good_step = (diffs == freq_s)  # length n-1

            # A window of length window_size requires window_size diffs to be good:
            # from i->i+1 ... i+window_size-1 -> i+window_size
            # That is good_step[i : i+window_size] all True
            max_start = n - self.window_size - 1  # because y uses i+window_size
            valid_starts = []
            for i in range(0, max_start + 1, self.stride):
                if good_step[i : i + self.window_size].all():
                    valid_starts.append(i)

            self._by_plant[plant_id] = {"X": X, "y": y}
            for i in valid_starts:
                self._index.append((plant_id, i))

        if len(self._index) == 0:
            raise ValueError(
                f"No valid windows created. Check window_size={self.window_size}, "
                f"time regularity, and per-plant row counts."
            )

        # Diagnostics
        counts = {}
        for pid, _ in self._index:
            counts[pid] = counts.get(pid, 0) + 1
        print("[INFO] GroupedWindowDataset created")
        print(f"  window_size: {self.window_size}, stride: {self.stride}")
        print(f"  total windows: {len(self._index):,}")
        print("  windows per plant:")
        for pid in sorted(counts):
            print(f"    {pid}: {counts[pid]:,}")

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        plant_id, i = self._index[idx]
        X = self._by_plant[plant_id]["X"][i : i + self.window_size]
        y = self._by_plant[plant_id]["y"][i + self.window_size]
        return torch.from_numpy(X), torch.tensor(y, dtype=torch.float32)


def load_fold_data(fold: int, data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_path = data_dir / f"fold_{fold}_train.parquet"
    val_path = data_dir / f"fold_{fold}_val.parquet"
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Validation data not found: {val_path}")

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)

    print("\n" + "=" * 80)
    print(f"Loading Fold {fold} Data")
    print("=" * 80)
    print(f"Train shape: {train_df.shape}, range: {train_df[TIME_COL].min()} -> {train_df[TIME_COL].max()}")
    print(f"Val   shape: {val_df.shape}, range: {val_df[TIME_COL].min()} -> {val_df[TIME_COL].max()}")

    return train_df, val_df


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    window_size: int = 96,
    batch_size: int = 128,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    print("\n" + "=" * 80)
    print("Creating Dataloaders (Grouped by plant)")
    print("=" * 80)

    train_ds = GroupedWindowDataset(train_df, window_size=window_size, stride=1)
    val_ds = GroupedWindowDataset(val_df, window_size=window_size, stride=1)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"[INFO] Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    return train_loader, val_loader


def create_model_with_transfer(
    farm2107_ckpt: Path,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    lr: float,
) -> pl.LightningModule:
    config = LSTMEncoderConfig(
        input_size=len(GLOBAL_LSTM_INPUT_FEATURES),
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        lr=lr,
    )
    model = GlobalLSTMEncoder(config)

    if farm2107_ckpt.exists():
        model = transfer_from_farm2107(model, str(farm2107_ckpt))
        print(f"[INFO] Loaded transfer weights from: {farm2107_ckpt}")
    else:
        print(f"[WARN] Farm2107 checkpoint not found, training from scratch: {farm2107_ckpt}")

    return model


def setup_trainer(
    fold: int,
    output_dir: Path,
    max_epochs: int,
    patience: int,
    gpus: int,
    precision: str,
) -> pl.Trainer:
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = CSVLogger(save_dir=str(output_dir), name="")

    ckpt = ModelCheckpoint(
        dirpath=str(output_dir),
        filename="best_checkpoint",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    early = EarlyStopping(monitor="val_loss", mode="min", patience=patience)

    accelerator = "gpu" if gpus and torch.cuda.is_available() else "cpu"
    devices = 1 if accelerator == "gpu" else None
    # Normalize precision strings across Lightning versions
    prec = str(precision)
    if prec.lower() in {"16-mixed", "bf16-mixed"}:
    # Most Lightning versions accept these strings, keep as-is.
        pass
    elif prec in {"16", "32", "bf16"}:
        pass
    else:
        print(f"[WARN] Unknown precision='{prec}', defaulting to 32")
        prec = "32"

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        callbacks=[ckpt, early],
        logger=logger,
        enable_progress_bar=True,
        log_every_n_steps=50,
        precision=prec,
    )
    return trainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--window_size", type=int, default=96)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--hidden_size", type=int, default=64)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max_epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--gpus", type=int, default=1)
    # Dataloader + mixed precision controls (kept to match run_stage3_global_training.sh)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--precision", type=str, default="32")
    p.add_argument("--precision_override", type=str, default=None)  # backward compatible; ignored
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.precision_override is not None:
        print(f"[INFO] Ignoring --precision_override={args.precision_override} (kept for backward compatibility).")

    data_dir = REPO_ROOT / "data" / "processed" / "pretraining" / "germany" / "global"
    farm2107_ckpt = REPO_ROOT / "experiments" / "lstm" / "encoders" / "lstm_encoder_farm2107_CANONICAL.pt"
    output_dir = REPO_ROOT / "experiments" / "lstm" / "runs" / "germany" / "global_v3" / f"fold_{args.fold}"

    train_df, val_df = load_fold_data(args.fold, data_dir)
    train_loader, val_loader = create_dataloaders(
        train_df,
        val_df,
        window_size=args.window_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = create_model_with_transfer(
        farm2107_ckpt=farm2107_ckpt,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.lr,
    )

    trainer = setup_trainer(
        fold=args.fold,
        output_dir=output_dir,
        max_epochs=args.max_epochs,
        patience=args.patience,
        gpus=args.gpus,
        precision=args.precision,
    )

    print("\n" + "=" * 80)
    print("Starting Training")
    print("=" * 80 + "\n")
    trainer.fit(model, train_loader, val_loader)

    final_model_path = output_dir / f"lstm_encoder_global_fold_{args.fold}.pt"
    torch.save(model.state_dict(), final_model_path)
    print(f"\n[SUCCESS] Saved final model: {final_model_path}\n")


if __name__ == "__main__":
    main()
