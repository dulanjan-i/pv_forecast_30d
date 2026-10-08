import argparse
import sys
import time
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from pytorch_forecasting import TimeSeriesDataSet

# Import your existing model logic
from src.models.tft_model import TFTConfig, build_tft_model

def move_to_device(batch, device):
    """
    Safely moves a PyTorch Forecasting batch to the GPU.
    The batch is a tuple: (x, y) where x is a dict of tensors.
    """
    x, y = batch
    # Move inputs (x is a dictionary)
    x_cuda = {k: v.to(device) for k, v in x.items() if isinstance(v, torch.Tensor)}
    # Some parts of x might be lists (leave them alone)
    for k, v in x.items():
        if k not in x_cuda:
            x_cuda[k] = v
            
    # Move targets (y is a tuple or tensor)
    if isinstance(y, (list, tuple)):
        y_cuda = [yi.to(device) for yi in y]
    else:
        y_cuda = y.to(device)
        
    return x_cuda, y_cuda

def main():
    # --- 1. SETUP ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_parquet", type=str, required=True)
    parser.add_argument("--val_parquet", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--max_epochs", type=int, default=30)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training on: {device}")

    # --- 2. LOAD DATA (RAM TRICK) ---
    print("\n[INFO] LOADING DATA TO RAM...")
    
    # Load Parquet
    train_df = pd.read_parquet(args.train_parquet)
    # (We skip Val loading for speed in this test, but you can add it back)
    
    # Convert timestamps (Reuse your existing logic logic roughly)
    if "timestamp_utc" in train_df.columns:
        train_df["timestamp_utc"] = pd.to_datetime(train_df["timestamp_utc"], utc=True)
        train_df["time_idx"] = (train_df["timestamp_utc"].astype(np.int64) // 10**9 // 900).astype(int)
    
    # Minimal cleaning
    if "poa_irradiance" in train_df.columns: train_df.drop(columns=["poa_irradiance"], inplace=True)

    # Setup Dataset (Using your existing parameters)
    # NOTE: I am hardcoding parameters based on your previous logs to ensure it runs.
    # You can make this dynamic later.
    train_ds = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target="power_norm",
        group_ids=["plant_id"],
        max_encoder_length=96,
        max_prediction_length=96,
        static_categoricals=["plant_id"],
        time_varying_known_reals=["shortwave_radiation_instant_raw", "temperature_2m", "hour_sin", "hour_cos"], # Simplified list
        time_varying_unknown_reals=["power_norm"],
        add_relative_time_idx=True,
        add_target_scales=False,
        add_encoder_length=True,
        allow_missing_timesteps=True
    )
    
    # Create Loader (Num workers 0 to avoid Numpy Crash)
    dataloader = train_ds.to_dataloader(train=True, batch_size=args.batch_size, num_workers=0)
    
    # PRE-LOAD TO LIST (The "RAM Trick")
    print("[INFO] Converting to RAM Tensors...")
    ram_loader = list(dataloader)
    print(f"[INFO] Loaded {len(ram_loader)} batches.")

    # --- 3. BUILD MODEL ---
    print("[INFO] Building TFT Model...")
    cfg = TFTConfig(
        target="power_norm",
        time_idx="time_idx",
        group_ids=["plant_id"],
        max_encoder_length=96,
        max_prediction_length=96,
        quantiles=[0.1, 0.5, 0.9]
    )
    model = build_tft_model(cfg, train_ds)
    model.to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # --- 4. THE MANUAL TRAINING LOOP (NO LIGHTNING) ---
    print("\n[INFO] 🏎️  STARTING HIGH-SPEED TRAINING")
    model.train()
    
    for epoch in range(args.max_epochs):
        start_time = time.time()
        total_loss = 0
        steps = 0
        
        for batch in ram_loader:
            optimizer.zero_grad()
            
            # Move to GPU
            x, y = move_to_device(batch, device)
            
            # Forward Pass
            # TFT returns a dictionary output
            output = model(x)
            
            # Calculate Loss (TFT internal loss function)
            # PyTorch Forecasting models have a .loss() method
            loss = model.loss(output, y)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            steps += 1
        
        epoch_time = time.time() - start_time
        avg_loss = total_loss / steps if steps > 0 else 0
        its = steps / epoch_time
        
        print(f"Epoch {epoch+1:02d} | Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s | Speed: {its:.2f} it/s")

if __name__ == "__main__":
    main()