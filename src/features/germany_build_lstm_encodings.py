"""
src/features/germany_build_lstm_encodings.py

Stage 3.6: Generate LSTM encoder representations (encodings) for TFT ingestion.

Why this exists
- Notebooks are for sanity checks only.
- This script produces a deterministic, versioned artifact: per-timestamp LSTM encodings
  aligned with the Germany supermatrix/regional splits.

What it does
1) Load an input parquet (regional_train/val or any Germany parquet that contains:
   TIME_COL, PLANT_ID_COL, GLOBAL_LSTM_INPUT_FEATURES).
2) Build grouped windows per plant_id (no cross-plant leakage, no gap-jumping).
3) Run the trained regional encoder in eval mode to extract a representation vector per window.
4) Save an output parquet with:
   - timestamp_utc (the prediction time t, not the window start)
   - plant_id
   - lstm_enc_000 ... lstm_enc_{H-1}  (H = hidden_size)
   - optional: y_true (power_norm at time t) for debugging

Outputs
- data/processed/pretraining/germany/global/encodings/
    regional_train_lstm_encodings.parquet
    regional_val_lstm_encodings.parquet
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.data.schema import (
    TIME_COL,
    PLANT_ID_COL,
    TARGET_COL,
    TIME_STEP_MINUTES,
    GLOBAL_LSTM_INPUT_FEATURES,
)
from src.models.global_lstm_encoder import GlobalLSTMEncoder, LSTMEncoderConfig


class GroupedWindowWithMetaDataset(Dataset):
    """
    Like your training GroupedWindowDataset, but also returns:
    - pid (string)
    - t (timestamp for y / prediction time)

    Collation note
    - DataLoader cannot collate pandas.Timestamp.
    - We return t as int64 unix seconds.
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
            raise ValueError(f"Missing required columns: {missing}")

        # hard fail on NaNs
        X_all = d[GLOBAL_LSTM_INPUT_FEATURES].to_numpy()
        y_all = d[TARGET_COL].to_numpy()
        if np.isnan(X_all).any() or np.isnan(y_all).any():
            raise ValueError("Input contains NaNs. Fix preprocessing before encoding.")

        self._by_plant: Dict[str, Dict[str, np.ndarray]] = {}
        self._index: List[Tuple[str, int]] = []

        freq_s = int(TIME_STEP_MINUTES * 60)

        for pid, g in d.groupby(PLANT_ID_COL, sort=True):
            g = g.sort_values(TIME_COL).reset_index(drop=True)

            t_sec = (g[TIME_COL].astype("int64").to_numpy() // 10**9).astype(np.int64)
            X = g[GLOBAL_LSTM_INPUT_FEATURES].to_numpy(dtype=np.float32)
            y = g[TARGET_COL].to_numpy(dtype=np.float32)

            n = len(g)
            if n <= self.window_size:
                continue

            diffs = np.diff(t_sec)
            good_step = (diffs == freq_s)

            max_start = n - self.window_size - 1
            for i in range(0, max_start + 1, self.stride):
                if good_step[i : i + self.window_size].all():
                    self._index.append((pid, i))

            self._by_plant[pid] = {"X": X, "y": y, "t": t_sec}

        if not self._index:
            raise ValueError("No valid windows created. Check window_size and gaps.")

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        pid, i = self._index[idx]
        X = self._by_plant[pid]["X"][i : i + self.window_size]
        y = self._by_plant[pid]["y"][i + self.window_size]
        t = self._by_plant[pid]["t"][i + self.window_size]
        return torch.from_numpy(X), torch.tensor(y, dtype=torch.float32), pid, int(t)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input_parquet", type=str, required=True)
    p.add_argument("--encoder_ckpt", type=str, required=True)
    p.add_argument("--output_parquet", type=str, required=True)
    p.add_argument("--window_size", type=int, default=96)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--hidden_size", type=int, default=64)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    inp = Path(args.input_parquet)
    outp = Path(args.output_parquet)
    outp.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(inp)

    ds = GroupedWindowWithMetaDataset(df, window_size=args.window_size, stride=1)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=bool(device == "cuda"),
        persistent_workers=bool(args.num_workers and args.num_workers > 0),
    )

    cfg = LSTMEncoderConfig(
        input_size=len(GLOBAL_LSTM_INPUT_FEATURES),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=1e-4,
    )
    model = GlobalLSTMEncoder(cfg)
    state = torch.load(args.encoder_ckpt, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    rows = []
    for X, y, pid, tsec in loader:
        X = X.to(device)

        # IMPORTANT:
        # We want an encoding vector, not the scalar prediction.
        # Your GlobalLSTMEncoder must expose an encoder output.
        # If it doesn't yet, add a method `encode(X)` that returns last hidden state (B, H).
        enc = model.encode(X)  # shape (B, H)
        enc = enc.detach().cpu().numpy()

        y_np = y.detach().cpu().numpy()
        tsec_np = np.asarray(tsec)

        for i in range(enc.shape[0]):
            rec = {
                TIME_COL: pd.to_datetime(int(tsec_np[i]), unit="s", utc=True),
                PLANT_ID_COL: pid[i],
                TARGET_COL: float(y_np[i]),
            }
            for j in range(enc.shape[1]):
                rec[f"lstm_enc_{j:03d}"] = float(enc[i, j])
            rows.append(rec)

    out_df = pd.DataFrame(rows).sort_values([PLANT_ID_COL, TIME_COL]).reset_index(drop=True)
    out_df.to_parquet(outp, index=False)
    print(f"[SUCCESS] Wrote encodings: {outp} rows={len(out_df):,} cols={out_df.shape[1]}")


if __name__ == "__main__":
    main()
