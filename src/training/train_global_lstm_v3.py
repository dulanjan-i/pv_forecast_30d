"""
src/training/train_global_lstm_v3.py

Version 3 - Global Forecasting Model: Training Script with Rolling Origin CV

PURPOSE
-------
Train GlobalLSTMEncoder on one fold of the rolling origin cross-validation.
This script is called 4 times (once per fold) by run_stage3_global_training.sh.

KEY FEATURES
------------
1. **Transfer Learning**: Loads Farm2107 weights, zero-pads to 20 features
2. **Rolling Origin CV**: Trains on all past data, validates on future season
3. **Window Dataset**: Creates 24-hour sliding windows (96 steps × 15min)
4. **PyTorch Lightning**: Handles training loop, logging, checkpointing
5. **Early Stopping**: Stops if val_loss doesn't improve for 5 epochs

FOLD STRUCTURE
--------------
Fold 1 (Spring): Val = Mar-May 2023, Train = all before Mar 2023
Fold 2 (Summer): Val = Jun-Aug 2023, Train = all before Jun 2023
Fold 3 (Fall): Val = Sep-Nov 2023, Train = all before Sep 2023
Fold 4 (Winter): Val = Dec 2023 - Feb 2024, Train = all before Dec 2023

OUTPUTS
-------
For each fold, saves to: experiments/lstm/runs/germany/global_v3/fold_X/
    - lstm_encoder_global_fold_X.pt  # Final model weights
    - best_checkpoint.ckpt            # Best val_loss checkpoint (Lightning)
    - metrics.csv                     # Train/val loss per epoch
    - hparams.yaml                    # Hyperparameters logged

USAGE
-----
```bash
# Train single fold
python src/training/train_global_lstm_v3.py --fold 1

# Train all folds (via shell wrapper)
./run_stage3_global_training.sh
```

DEPENDENCIES
------------
- data/processed/pretraining/germany/global/fold_X_train.parquet
- data/processed/pretraining/germany/global/fold_X_val.parquet
- experiments/lstm/encoders/lstm_encoder_farm2107_CANONICAL.pt

DESIGN DECISIONS
----------------
1. **Window size = 96**: 24 hours × 4 steps/hour (15-min resolution)
2. **Batch size = 128**: Balance between memory and gradient noise
3. **Learning rate = 1e-4**: Conservative for transfer learning (Farm2107 used 1e-3)
4. **Max epochs = 30**: Early stopping prevents wasted compute
5. **Patience = 5**: Allow 5 epochs for val_loss to improve

Author: PV Forecast Team
Date: December 2024
Version: 3.0 (Global Model with Rolling Origin CV)
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.data.schema import GLOBAL_LSTM_INPUT_FEATURES, TIME_COL, TARGET_COL
from src.models.global_lstm_encoder import GlobalLSTMEncoder, transfer_from_farm2107, LSTMEncoderConfig


class WindowDataset(Dataset):
    """
    Sliding window dataset for time series forecasting.
    
    Creates (X, y) pairs where:
        - X: (window_size, n_features) - historical window
        - y: scalar - target value at next time step
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with columns GLOBAL_LSTM_INPUT_FEATURES + TARGET_COL
    window_size : int
        Number of time steps in each window (default: 96 for 24 hours at 15min)
    stride : int
        Step size for sliding window (default: 1 for dense windows)
        
    Attributes
    ----------
    X : np.ndarray
        Feature array (n_samples, window_size, n_features)
    y : np.ndarray
        Target array (n_samples,)
        
    Examples
    --------
    >>> df = pd.read_parquet("fold_1_train.parquet")
    >>> dataset = WindowDataset(df, window_size=96)
    >>> x, y = dataset[0]
    >>> x.shape  # (96, 20) - 24 hours × 20 features
    >>> y.shape  # () - scalar target
    """
    
    def __init__(self, df: pd.DataFrame, window_size: int = 96, stride: int = 1):
        """Initialize window dataset from dataframe."""
        
        # Extract features and target
        X_data = df[GLOBAL_LSTM_INPUT_FEATURES].values  # (T, 20)
        y_data = df[TARGET_COL].values                   # (T,)
        
        # Validate no NaNs
        if np.isnan(X_data).any() or np.isnan(y_data).any():
            raise ValueError(
                f"Dataset contains NaNs! X: {np.isnan(X_data).sum()}, y: {np.isnan(y_data).sum()}"
            )
        
        # Create sliding windows
        n_samples = len(X_data) - window_size
        if n_samples <= 0:
            raise ValueError(
                f"Dataset too small for window_size={window_size}. "
                f"Need at least {window_size+1} samples, got {len(X_data)}"
            )
        
        # Build window arrays
        X_windows = []
        y_targets = []
        
        for i in range(0, n_samples, stride):
            # Window: [i, i+window_size)
            window = X_data[i:i+window_size]  # (window_size, 20)
            target = y_data[i+window_size]     # Predict next step
            
            X_windows.append(window)
            y_targets.append(target)
        
        self.X = np.array(X_windows, dtype=np.float32)  # (n_windows, window_size, 20)
        self.y = np.array(y_targets, dtype=np.float32)  # (n_windows,)
        
        print(f"[INFO] WindowDataset created:")
        print(f"  Total timesteps: {len(X_data)}")
        print(f"  Window size: {window_size}")
        print(f"  Stride: {stride}")
        print(f"  Number of windows: {len(self.X)}")
        print(f"  X shape: {self.X.shape}")
        print(f"  y shape: {self.y.shape}")
    
    def __len__(self) -> int:
        """Return number of windows."""
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (window, target) pair."""
        return torch.from_numpy(self.X[idx]), torch.from_numpy(np.array(self.y[idx]))


def load_fold_data(fold: int, data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load train and validation data for a specific fold.
    
    Parameters
    ----------
    fold : int
        Fold number (1-4)
    data_dir : Path
        Directory containing fold_X_train.parquet and fold_X_val.parquet
        
    Returns
    -------
    train_df : pd.DataFrame
        Training data
    val_df : pd.DataFrame
        Validation data
        
    Raises
    ------
    FileNotFoundError
        If fold files don't exist
    """
    train_path = data_dir / f"fold_{fold}_train.parquet"
    val_path = data_dir / f"fold_{fold}_val.parquet"
    
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Validation data not found: {val_path}")
    
    print(f"\n{'='*80}")
    print(f"Loading Fold {fold} Data")
    print(f"{'='*80}")
    
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    
    print(f"Train data: {train_path}")
    print(f"  Shape: {train_df.shape}")
    print(f"  Date range: {train_df[TIME_COL].min()} to {train_df[TIME_COL].max()}")
    
    print(f"\nVal data: {val_path}")
    print(f"  Shape: {val_df.shape}")
    print(f"  Date range: {val_df[TIME_COL].min()} to {val_df[TIME_COL].max()}")
    
    # Validate columns
    missing_cols = set(GLOBAL_LSTM_INPUT_FEATURES + [TARGET_COL]) - set(train_df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in train data: {missing_cols}")
    
    return train_df, val_df


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    window_size: int = 96,
    batch_size: int = 256,
    num_workers: int = 2,
    use_ddp: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Parameters
    ----------
    train_df, val_df : pd.DataFrame
        Train and validation dataframes
    window_size : int
        Sliding window size (default: 96 for 24 hours)
    batch_size : int
        Batch size for training (default: 256 for 2 L4 GPUs)
    num_workers : int
        DataLoader workers (default: 2, max with DDP is 2-4)
    use_ddp : bool
        Set True if using DDP to avoid pin_memory issues
        
    Returns
    -------
    train_loader : DataLoader
        Training dataloader (shuffled)
    val_loader : DataLoader
        Validation dataloader (not shuffled)
    """
    print(f"\n{'='*80}")
    print("Creating Dataloaders")
    print(f"{'='*80}")
    print(f"Window size: {window_size} steps (24 hours at 15-min resolution)")
    print(f"Batch size: {batch_size}")
    print(f"Num workers: {num_workers}")
    print(f"DDP mode: {use_ddp}")
    
    train_dataset = WindowDataset(train_df, window_size=window_size, stride=1)
    val_dataset = WindowDataset(val_df, window_size=window_size, stride=1)
    
    # DDP: reduce workers and disable pin_memory to avoid multiprocessing issues
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True if not use_ddp else False,
        num_workers=num_workers,
        pin_memory=not use_ddp,  # Disable for DDP
        persistent_workers=(num_workers > 0) and not use_ddp,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=not use_ddp,  # Disable for DDP
        persistent_workers=(num_workers > 0) and not use_ddp,
    )
    
    print(f"\nTrain loader: {len(train_loader)} batches")
    print(f"Val loader: {len(val_loader)} batches")
    
    return train_loader, val_loader


def create_model_with_transfer(
    farm2107_ckpt_path: Path,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.1,
    lr: float = 1e-4,
) -> GlobalLSTMEncoder:
    """
    Create GlobalLSTMEncoder with Farm2107 transfer learning.
    
    Parameters
    ----------
    farm2107_ckpt_path : Path
        Path to Farm2107 checkpoint
    hidden_size : int
        LSTM hidden size (should match Farm2107: 64)
    num_layers : int
        LSTM layers (should match Farm2107: 2)
    dropout : float
        Dropout rate
    lr : float
        Learning rate (lower than Farm2107 for fine-tuning)
        
    Returns
    -------
    GlobalLSTMEncoder
        Model with transferred weights
    """
    print(f"\n{'='*80}")
    print("Initializing Global LSTM Encoder")
    print(f"{'='*80}")
    
    # Create config
    config = LSTMEncoderConfig(
        input_size=20,         # 15 original + 5 plant IDs
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        lr=lr,
    )
    
    # Create model
    model = GlobalLSTMEncoder(config)
    
    # Transfer Farm2107 weights
    if farm2107_ckpt_path.exists():
        print(f"\n✅ Farm2107 checkpoint found: {farm2107_ckpt_path}")
        model = transfer_from_farm2107(model, str(farm2107_ckpt_path))
    else:
        print(f"\n⚠️ Farm2107 checkpoint NOT found: {farm2107_ckpt_path}")
        print("Training from scratch (no transfer learning)")
    
    return model


def setup_trainer(
    fold: int,
    output_dir: Path,
    max_epochs: int = 30,
    patience: int = 5,
    gpus: int = 2,
    precision: str = "16-mixed",
    precision_override: str = "high",
) -> pl.Trainer:
    """
    Setup PyTorch Lightning Trainer with callbacks and logger.
    
    Parameters
    ----------
    fold : int
        Fold number (for logging)
    output_dir : Path
        Directory to save outputs
    max_epochs : int
        Maximum training epochs
    patience : int
        Early stopping patience
    gpus : int
        Number of GPUs (0 for CPU)
        
    Returns
    -------
    pl.Trainer
        Configured trainer
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename='best_checkpoint',
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        verbose=True,
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        mode='min',
        verbose=True,
    )
    
    # Logger
    csv_logger = CSVLogger(
        save_dir=output_dir,
        name='',
        version='',
    )
    
    # Set tensor core optimization for L4 GPUs
    if gpus > 0 and torch.cuda.is_available():
        torch.set_float32_matmul_precision(precision_override)
        print(f"[INFO] Tensor Core matmul precision: {precision_override}")
    
    # Lightning 2.x expects accelerator/devices instead of deprecated gpus flag
    if gpus > 0 and torch.cuda.is_available():
        accelerator = "gpu"
        devices = gpus
        strategy = "ddp" if gpus > 1 else "auto"
    else:
        accelerator = "cpu"
        devices = 1
        strategy = "auto"

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=csv_logger,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision=precision,
        log_every_n_steps=10,
        enable_progress_bar=True,
    )
    
    print(f"\n{'='*80}")
    print(f"Trainer Configuration")
    print(f"{'='*80}")
    print(f"Max epochs: {max_epochs}")
    print(f"Early stopping patience: {patience}")
    print(f"Output directory: {output_dir}")
    print(f"GPUs requested: {gpus}, accelerator: {accelerator}, devices: {devices}, strategy: {strategy}, precision: {precision}, matmul: {precision_override}")
    
    return trainer


def main():
    """Main training pipeline."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train Global LSTM Encoder (Version 3)")
    parser.add_argument(
        '--fold',
        type=int,
        required=True,
        choices=[1, 2, 3, 4, 5],
        help='Fold number to train (1-4 for CV, 5 for test)'
    )
    parser.add_argument(
        '--window_size',
        type=int,
        default=96,
        help='Sliding window size (default: 96 for 24 hours)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=256,
        help='Batch size (default: 256 for 2 L4 GPUs)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=2,
        help='DataLoader workers (default: 2, max 2-4 with DDP)'
    )
    parser.add_argument(
        '--hidden_size',
        type=int,
        default=64,
        help='LSTM hidden size (default: 64, must match Farm2107)'
    )
    parser.add_argument(
        '--num_layers',
        type=int,
        default=2,
        help='LSTM layers (default: 2, must match Farm2107)'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.1,
        help='Dropout rate (default: 0.1)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Learning rate (default: 1e-4 for fine-tuning)'
    )
    parser.add_argument(
        '--max_epochs',
        type=int,
        default=30,
        help='Max training epochs (default: 30)'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=5,
        help='Early stopping patience (default: 5)'
    )
    parser.add_argument(
        '--gpus',
        type=int,
        default=2,
        help='Number of GPUs (default: 2 L4s)'
    )
    parser.add_argument(
        '--precision',
        type=str,
        default="16-mixed",
        help='Lightning precision (default: 16-mixed for L4, use 32-true for CPU)'
    )
    parser.add_argument(
        '--precision_override',
        type=str,
        default="high",
        help='PyTorch matmul precision (high for L4 tensor cores)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print(f"TRAINING GLOBAL LSTM ENCODER - FOLD {args.fold}")
    print("="*80)
    print(f"Hyperparameters:")
    print(f"  Window size: {args.window_size}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Hidden size: {args.hidden_size}")
    print(f"  Num layers: {args.num_layers}")
    print(f"  Dropout: {args.dropout}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Max epochs: {args.max_epochs}")
    print(f"  Patience: {args.patience}")
    
    # Paths
    data_dir = REPO_ROOT / "data" / "processed" / "pretraining" / "germany" / "global"
    farm2107_ckpt = REPO_ROOT / "experiments" / "lstm" / "encoders" / "lstm_encoder_farm2107_CANONICAL.pt"
    output_dir = REPO_ROOT / "experiments" / "lstm" / "runs" / "germany" / "global_v3" / f"fold_{args.fold}"
    
    # Step 1: Load data
    train_df, val_df = load_fold_data(args.fold, data_dir)
    
    # Step 2: Create dataloaders (with DDP flag if multi-GPU)
    use_ddp = args.gpus > 1
    train_loader, val_loader = create_dataloaders(
        train_df, val_df,
        window_size=args.window_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_ddp=use_ddp,
    )
    

    
    # Step 3: Create model with transfer learning
    model = create_model_with_transfer(
        farm2107_ckpt,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.lr,
    )
    
    # Step 4: Setup trainer
    trainer = setup_trainer(
        fold=args.fold,
        output_dir=output_dir,
        max_epochs=args.max_epochs,
        patience=args.patience,
        gpus=args.gpus,
        precision=args.precision,
        precision_override=args.precision_override,
    )
    
    # Step 5: Train
    print(f"\n{'='*80}")
    print("Starting Training")
    print(f"{'='*80}\n")
    
    trainer.fit(model, train_loader, val_loader)
    
    # Step 6: Save final model
    final_model_path = output_dir / f"lstm_encoder_global_fold_{args.fold}.pt"
    torch.save(model.state_dict(), final_model_path)
    
    print(f"\n{'='*80}")
    print("Training Complete!")
    print(f"{'='*80}")
    print(f"Final model saved: {final_model_path}")
    print(f"Best checkpoint: {output_dir / 'best_checkpoint.ckpt'}")
    print(f"Metrics: {output_dir / 'metrics.csv'}")
    print(f"\n✅ Fold {args.fold} training finished successfully!\n")


if __name__ == "__main__":
    main()
