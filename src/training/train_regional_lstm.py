"""
src/training/train_regional_lstm.py

Stage 3.5 (Regional Adaptation): Train ONE canonical Germany-adapted LSTM encoder.

Why this script exists
- Rolling-origin CV (train_global_lstm_v3.py) is for evaluation and debugging.
  It produces fold-specific models and fold-specific scalers.
- For the downstream TFT, we want a single "regional" encoder trained on the
  maximum stable Germany history we intend to use as our regional-adapted
  representation model.
- This script trains that single encoder on:
    - regional_train.parquet (history up to 2023-11-30 23:45 UTC)
    - regional_val.parquet   (tail slice 2023-12-01 .. 2024-02-29 UTC)
  using the same grouped windowing logic that prevents cross-plant windows.

Inputs expected (produced earlier)
- data/processed/pretraining/germany/global/regional_train.parquet
- data/processed/pretraining/germany/global/regional_val.parquet

Outputs (canonical artifacts)
- Training run folder:
    experiments/lstm/runs/germany/global_v3/regional/
  (contains logs/metrics.csv and best_checkpoint.ckpt)
- Canonical encoder weights (BEST checkpoint weights exported):
    experiments/lstm/encoders/lstm_encoder_germany_regional_CANONICAL.pt

Notes
- The model predicts next-step power_norm (MSE loss), consistent with our Stage 3 setup.
- Transfer learning is applied from the Farm2107 canonical encoder checkpoint.
"""

from __future__ import annotations

import sys
import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

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

    Critical contract:
    - Windows must NEVER cross plant boundaries. We build windows per plant_id and then
      concatenate the window index list across plants.

    Gap handling:
    - A candidate window is only kept if timestamps are strictly regular at TIME_STEP_MINUTES.
      This prevents windows from silently "jumping over" missing rows introduced by NaN drops.

    Window definition:
    - X: [t - window_size .. t-1]  shape: (window_size, n_features)
    - y: target at time t          shape: scalar
    """

    def __init__(self, df: pd.DataFrame, window_size: int = 96, stride: int = 1):
        self.window_size = int(window_size)
        self.stride = int(stride)

        d = df.copy()
        d[TIME_COL] = pd.to_datetime(d[TIME_COL], utc=True)
        d = d.sort_values([PLANT_ID_COL, TIME_COL]).reset_index(drop=True)

        required = set([TIME_COL, PLANT_ID_COL, TARGET_COL] + GLOBAL_LSTM_INPUT_FEATURES)
        missing = sorted(required - set(d.columns))
        if missing:
            raise ValueError(f"GroupedWindowDataset: missing required columns: {missing}")

        # Hard fail on NaNs (model + loss assume none)
        X_all = d[GLOBAL_LSTM_INPUT_FEATURES].to_numpy()
        y_all = d[TARGET_COL].to_numpy()
        if np.isnan(X_all).any() or np.isnan(y_all).any():
            raise ValueError(
                f"Dataset contains NaNs. X NaNs={np.isnan(X_all).sum()}, y NaNs={np.isnan(y_all).sum()}"
            )

        self._by_plant: Dict[str, Dict[str, np.ndarray]] = {}
        self._index: List[Tuple[str, int]] = []  # (plant_id, start_idx)

        freq_s = int(TIME_STEP_MINUTES * 60)

        for plant_id, g in d.groupby(PLANT_ID_COL, sort=True):
            g = g.sort_values(TIME_COL).reset_index(drop=True)

            times = g[TIME_COL].astype("int64").to_numpy() // 10**9  # seconds
            X = g[GLOBAL_LSTM_INPUT_FEATURES].to_numpy(dtype=np.float32)
            y = g[TARGET_COL].to_numpy(dtype=np.float32)

            n = len(g)
            if n <= self.window_size:
                continue

            diffs = np.diff(times)
            good_step = (diffs == freq_s)  # length n-1

            max_start = n - self.window_size - 1  # because y uses i+window_size
            valid_starts: List[int] = []
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

        counts: Dict[str, int] = {}
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


def load_regional_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_path = data_dir / "regional_train.parquet"
    val_path = data_dir / "regional_val.parquet"

    if not train_path.exists():
        raise FileNotFoundError(f"Regional training data not found: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Regional validation data not found: {val_path}")

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)

    print("\n" + "=" * 80)
    print("Loading REGIONAL data")
    print("=" * 80)
    print(f"Train shape: {train_df.shape}, range: {train_df[TIME_COL].min()} -> {train_df[TIME_COL].max()}")
    print(f"Val   shape: {val_df.shape}, range: {val_df[TIME_COL].min()} -> {val_df[TIME_COL].max()}")
    print(f"Train plants: {sorted(train_df[PLANT_ID_COL].unique().tolist())}")
    print(f"Val plants:   {sorted(val_df[PLANT_ID_COL].unique().tolist())}")

    return train_df, val_df


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    window_size: int,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader]:
    print("\n" + "=" * 80)
    print("Creating Dataloaders (Grouped by plant)")
    print("=" * 80)

    train_ds = GroupedWindowDataset(train_df, window_size=window_size, stride=1)
    val_ds = GroupedWindowDataset(val_df, window_size=window_size, stride=1)

    pin = bool(torch.cuda.is_available())
    persistent = bool(num_workers and num_workers > 0)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=persistent,
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
    output_dir: Path,
    max_epochs: int,
    patience: int,
    gpus: int,
    precision: str,
) -> pl.Trainer:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Keep metrics at: <output_dir>/logs/metrics.csv (avoid version_0 clutter)
    logger = CSVLogger(save_dir=str(output_dir), name="logs", version="")

    ckpt = ModelCheckpoint(
        dirpath=str(output_dir),
        filename="best_checkpoint",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    early = EarlyStopping(monitor="val_loss", mode="min", patience=patience)

    accelerator = "gpu" if (gpus and torch.cuda.is_available()) else "cpu"
    if accelerator == "gpu":
        n_avail = torch.cuda.device_count()
        devices = min(int(gpus), int(n_avail)) if gpus else 1
    else:
        devices = None

    prec = str(precision)
    if prec.lower() in {"16-mixed", "bf16-mixed"}:
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
    p.add_argument("--window_size", type=int, default=96)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--hidden_size", type=int, default=64)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max_epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--precision", type=str, default="16-mixed")
    p.add_argument("--precision_override", type=str, default=None)  # backward compatible; ignored
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.precision_override is not None:
        print(f"[INFO] Ignoring --precision_override={args.precision_override} (kept for backward compatibility).")

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    data_dir = REPO_ROOT / "data" / "processed" / "pretraining" / "germany" / "global"
    farm2107_ckpt = REPO_ROOT / "experiments" / "lstm" / "encoders" / "lstm_encoder_farm2107_CANONICAL.pt"

    run_dir = REPO_ROOT / "experiments" / "lstm" / "runs" / "germany" / "global_v3" / "regional"
    out_encoder = REPO_ROOT / "experiments" / "lstm" / "encoders" / "lstm_encoder_germany_regional_CANONICAL.pt"
    out_encoder.parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Reproducibility: persist the exact CLI config used for this run
    hparams = vars(args).copy()
    hparams["timestamp_utc"] = pd.Timestamp.now(tz='UTC').isoformat()
    hparams["script"] = str(Path(__file__).resolve())
    hparams["repo_root"] = str(REPO_ROOT)

    # Log git commit + dirty status
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()
        hparams["git_commit"] = commit
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT).decode().strip()
        hparams["git_dirty"] = bool(dirty)
    except Exception:
        hparams["git_commit"] = None
        hparams["git_dirty"] = None

    with open(run_dir / "hparams.json", "w") as f:
        json.dump(hparams, f, indent=2, sort_keys=True)

    with open(run_dir / "hparams.txt", "w") as f:
        for k in sorted(hparams):
            f.write(f"{k}: {hparams[k]}\n")

    train_df, val_df = load_regional_data(data_dir)
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
        output_dir=run_dir,
        max_epochs=args.max_epochs,
        patience=args.patience,
        gpus=args.gpus,
        precision=args.precision,
    )

    print("\n" + "=" * 80)
    print("Starting REGIONAL training")
    print("=" * 80 + "\n")
    trainer.fit(model, train_loader, val_loader)

    # Export BEST checkpoint weights (not last epoch)
    best_path = ""
    if getattr(trainer, "checkpoint_callback", None) is not None:
        best_path = trainer.checkpoint_callback.best_model_path or ""

    if best_path:
        print(f"[INFO] Loading best checkpoint weights before exporting: {best_path}")
        ckpt = torch.load(best_path, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"], strict=True)
    else:
        print("[WARN] No best checkpoint path found. Exporting last-epoch weights.")

    torch.save(model.state_dict(), out_encoder)
    print(f"\n[SUCCESS] Saved canonical regional encoder: {out_encoder}\n")

    run_copy = run_dir / out_encoder.name
    torch.save(model.state_dict(), run_copy)
    print(f"[INFO] Saved run copy: {run_copy}")

    if best_path:
        print(f"[INFO] Best Lightning checkpoint: {best_path}")


if __name__ == "__main__":
    main()
