# src/training/train_tft_v1.py
"""
Stage 4: Train TFT v1.1 on regional Germany split.
(Optimized for H100 with async GPU transfers and multi-worker prefetch)

Version: v1.1
Date: 2025-12-26
Changes from v1.0:
  - Added StreamPrefetcher for async GPU transfers with CUDA streams
  - Persistent workers with prefetch_factor for continuous data pipeline
  - Removed RAM preloading (bottleneck on TimeSeriesDataSet iteration)
  - Optimized thread settings (OMP_NUM_THREADS=1)
  - Increased default num_workers to 8
  - Added gradient accumulation support

Hotfix (2025-12-26):
  - Mixed precision turns on if --precision is 16-mixed or bf16-mixed (no hidden gate)
  - Removed per-step loss.item() GPU sync, only sync when logging and once per epoch
  - Added --disable_prefetcher option
  - Safe DataLoader kwargs: only pass prefetch_factor when num_workers > 0
  - Added throughput log (samples/sec), because it/s alone is misleading
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.models import TemporalFusionTransformer

UTC = timezone.utc

KEY_TIME = "timestamp_utc"
KEY_GROUP = "plant_id"
KEY_TARGET = "power_norm"
KEY_TIME_IDX = "time_idx"


@dataclass
class TFTConfig:
    hidden_size: int = 64
    lstm_layers: int = 2
    attention_head_size: int = 4
    dropout: float = 0.1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train_parquet", type=str, required=True)
    p.add_argument("--val_parquet", type=str, required=True)

    p.add_argument("--run_root", type=str, default="experiments/tft/runs/germany/v1_0")
    p.add_argument("--max_epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--prefetch_factor", type=int, default=4)
    p.add_argument("--disable_prefetcher", action="store_true")

    p.add_argument("--encoder_len", type=int, default=96)
    p.add_argument("--pred_len", type=int, default=96)

    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_accum_steps", type=int, default=1)

    p.add_argument("--hidden_size", type=int, default=64)
    p.add_argument("--lstm_layers", type=int, default=2)
    p.add_argument("--attn_heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)

    p.add_argument("--precision", type=str, default="32-true")
    p.add_argument("--enable_amp", action="store_true")
    p.add_argument("--grad_clip", type=float, default=0.1)

    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--min_delta", type=float, default=1e-5)
    p.add_argument("--log_every_n_steps", type=int, default=50)
    p.add_argument("--progress_every", type=int, default=25)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_lstm_encodings", action="store_true")
    p.add_argument("--enc_lag", type=int, default=None)
    p.add_argument("--init_state_dict", type=str, default="", help="Optional path to a torch-saved model.state_dict() to warm-start from.")
    p.add_argument("--init_strict", action="store_true", help="Strict loading for init_state_dict (default: non-strict).")


    return p.parse_args()


def _json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True


def _ensure_datetime_utc(df: pd.DataFrame) -> pd.DataFrame:
    df[KEY_TIME] = pd.to_datetime(df[KEY_TIME], utc=True, errors="coerce")
    df = df.dropna(subset=[KEY_TIME])
    return df


def _add_time_idx(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values([KEY_GROUP, KEY_TIME]).reset_index(drop=True)
    df[KEY_TIME_IDX] = df.groupby(KEY_GROUP, sort=False).cumcount().astype("int64")
    return df


def _drop_duplicate_raw_vs_norm(df: pd.DataFrame) -> pd.DataFrame:
    for c in list(df.columns):
        if c.endswith("_raw"):
            base = c[:-4]
            if base in df.columns:
                df = df.drop(columns=[base])
    return df


def _preclean(df: pd.DataFrame) -> pd.DataFrame:
    plant_onehot = [c for c in df.columns if c.startswith("plant_") and c != KEY_GROUP]
    drop_cols = ["poa_irradiance"] + plant_onehot
    drop_cols = [c for c in drop_cols if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    df = _drop_duplicate_raw_vs_norm(df)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[KEY_GROUP, KEY_TARGET, KEY_TIME]).reset_index(drop=True)
    return df


def _add_safe_lagged_encodings(df: pd.DataFrame, lag: int) -> Tuple[pd.DataFrame, List[str]]:
    enc_cols = [c for c in df.columns if c.startswith("lstm_enc_")]
    if not enc_cols:
        return df, []

    lagged = df.groupby(KEY_GROUP, sort=False)[enc_cols].shift(lag)
    lagged_cols = [f"{c}_lag{lag}" for c in enc_cols]
    lagged.columns = lagged_cols

    out = pd.concat([df, lagged], axis=1)
    before = len(out)
    out = out.dropna(subset=lagged_cols).reset_index(drop=True)
    dropped = before - len(out)
    if dropped > 0:
        print(f"[INFO] Dropped {dropped:,} rows due to lagged encoding NaNs (lag={lag}).", flush=True)

    out = out.drop(columns=enc_cols)
    return out, lagged_cols


def _write_run_metadata(
    run_dir: Path,
    args: argparse.Namespace,
    cfg: TFTConfig,
    train_ds: TimeSeriesDataSet,
    val_ds: TimeSeriesDataSet,
    roles: Dict[str, Any],
) -> None:
    _json_dump(run_dir / "dataset_params_train.json", train_ds.get_parameters())
    _json_dump(run_dir / "dataset_params_val.json", val_ds.get_parameters())
    _json_dump(run_dir / "column_roles.json", roles)

    run_cfg: Dict[str, Any] = {
        "cli_args": vars(args),
        "tft_config": asdict(cfg),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    _json_dump(run_dir / "run_config.json", run_cfg)


def _gpu_poison_pill() -> None:
    print("\n" + "=" * 70, flush=True)
    print("DEBUG: HARDWARE CHECK INITIATED", flush=True)

    if not torch.cuda.is_available():
        print("CRITICAL: CUDA not available. Exiting.", flush=True)
        sys.exit(1)

    try:
        n = torch.cuda.device_count()
        name0 = torch.cuda.get_device_name(0)
        print(f"CUDA OK. device_count={n}, device0={name0}", flush=True)
        x = torch.randn(1024, 1024, device="cuda")
        y = x @ x
        _ = y.sum().item()
        print("CUDA matmul smoke test OK.", flush=True)
    except Exception as e:
        print(f"CRITICAL FAILURE: GPU detected but allocation FAILED. Error: {e}", flush=True)
        sys.exit(1)

    print("=" * 70 + "\n", flush=True)


class StreamPrefetcher:
    """
    Async GPU transfer with CUDA streams to overlap data movement with compute.
    """
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream()

    def _to_device(self, batch):
        x, y = batch
        x_cuda = {k: v.to(self.device, non_blocking=True) for k, v in x.items() if torch.is_tensor(v)}
        if torch.is_tensor(y):
            y_cuda = y.to(self.device, non_blocking=True)
        elif isinstance(y, (list, tuple)):
            y_cuda = [t.to(self.device, non_blocking=True) if torch.is_tensor(t) else t for t in y]
        else:
            y_cuda = y
        return x_cuda, y_cuda

    def __iter__(self):
        self.loader_iter = iter(self.loader)
        self.preload()
        return self

    def preload(self):
        try:
            self.batch = next(self.loader_iter)
        except StopIteration:
            self.batch = None
            return
        with torch.cuda.stream(self.stream):
            self.batch = self._to_device(self.batch)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch

    def __len__(self):
        return len(self.loader)


def _move_batch_to_device(batch, device: torch.device):
    x, y = batch
    if isinstance(x, dict):
        x = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in x.items()}
    elif torch.is_tensor(x):
        x = x.to(device, non_blocking=True)

    if torch.is_tensor(y):
        y = y.to(device, non_blocking=True)
    elif isinstance(y, (list, tuple)):
        y = [t.to(device, non_blocking=True) if torch.is_tensor(t) else t for t in y]
    return x, y


def _progress_line(prefix: str, epoch: int, step: int, total: int, t0: float) -> None:
    if total <= 0:
        return
    elapsed = time.time() - t0
    it_s = step / max(elapsed, 1e-9)
    remaining = (total - step) / max(it_s, 1e-9)
    pct = 100.0 * step / max(total, 1)
    print(
        f"[{prefix} e{epoch}] {step}/{total} ({pct:.1f}%) it/s={it_s:.2f} ETA={remaining/60:.1f} min",
        flush=True,
    )


def _normalize_precision(p: str) -> str:
    p = str(p).strip().lower()
    if p in {"32", "32-true", "fp32", "float32"}:
        return "32-true"
    if p in {"16", "16-mixed", "fp16", "float16", "mixed"}:
        return "16-mixed"
    if p in {"bf16", "bf16-mixed", "bfloat16"}:
        return "bf16-mixed"
    return "32-true"


def main() -> None:
    args = parse_args()

    omp = int(os.environ.get("OMP_NUM_THREADS", "1"))
    try:
        torch.set_num_threads(omp)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    _seed_everything(args.seed)
    _gpu_poison_pill()

    run_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.run_root) / run_id
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    print("=" * 80, flush=True)
    print("Stage 4: TFT v1.1 training (H100 Optimized)", flush=True)
    print(f"train_parquet: {args.train_parquet}", flush=True)
    print(f"val_parquet:   {args.val_parquet}", flush=True)
    print(f"run_dir:       {run_dir}", flush=True)
    print("=" * 80, flush=True)

    train_df = pd.read_parquet(args.train_parquet)
    val_df = pd.read_parquet(args.val_parquet)

    for df_name, df in [("train", train_df), ("val", val_df)]:
        for col in [KEY_TIME, KEY_GROUP, KEY_TARGET]:
            if col not in df.columns:
                raise ValueError(f"[{df_name}] Missing required column: {col}")

    train_df = _add_time_idx(_preclean(_ensure_datetime_utc(train_df)))
    val_df = _add_time_idx(_preclean(_ensure_datetime_utc(val_df)))

    lagged_cols: List[str] = []
    if args.use_lstm_encodings:
        lag = int(args.enc_lag) if args.enc_lag is not None else int(args.pred_len)
        print(f"[INFO] Adding safe LSTM encoding features with lag={lag}...", flush=True)
        train_df, lagged_cols = _add_safe_lagged_encodings(train_df, lag)
        val_df, _ = _add_safe_lagged_encodings(val_df, lag)

    ignore_cols = {KEY_TIME, KEY_GROUP, KEY_TIME_IDX, KEY_TARGET}
    candidate_reals = [
        c for c in train_df.columns
        if c not in ignore_cols and pd.api.types.is_numeric_dtype(train_df[c])
    ]

    known_time_reals = sorted(candidate_reals)
    unknown_time_reals = [KEY_TARGET]

    roles = {
        "time_col": KEY_TIME,
        "time_idx_col": KEY_TIME_IDX,
        "group_ids": [KEY_GROUP],
        "target": KEY_TARGET,
        "known_time_reals": known_time_reals,
        "unknown_time_reals": unknown_time_reals,
        "lagged_encoding_cols": lagged_cols,
    }

    max_encoder_length = int(args.encoder_len)
    max_prediction_length = int(args.pred_len)

    train_ds = TimeSeriesDataSet(
        train_df,
        time_idx=KEY_TIME_IDX,
        target=KEY_TARGET,
        group_ids=[KEY_GROUP],
        min_encoder_length=max_encoder_length,
        max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length,
        max_prediction_length=max_prediction_length,
        time_varying_known_reals=known_time_reals,
        time_varying_unknown_reals=unknown_time_reals,
        target_normalizer=GroupNormalizer(groups=[KEY_GROUP], transformation="softplus"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )

    val_ds = TimeSeriesDataSet.from_dataset(train_ds, val_df, predict=False, stop_randomization=True)

    cfg = TFTConfig(
        hidden_size=int(args.hidden_size),
        lstm_layers=int(args.lstm_layers),
        attention_head_size=int(args.attn_heads),
        dropout=float(args.dropout),
    )

    model = TemporalFusionTransformer.from_dataset(
        train_ds,
        hidden_size=cfg.hidden_size,
        lstm_layers=cfg.lstm_layers,
        attention_head_size=cfg.attention_head_size,
        dropout=cfg.dropout,
        learning_rate=float(args.lr),
        log_interval=-1,
        reduce_on_plateau_patience=0,
    )

    if args.init_state_dict:
        init_path = Path(args.init_state_dict)
        if not init_path.exists():
            raise FileNotFoundError(f"--init_state_dict not found: {init_path}")
        sd = torch.load(str(init_path), map_location="cpu")
        strict = bool(args.init_strict)
        missing, unexpected = model.load_state_dict(sd, strict=strict)
        print(f"[INFO] Warm-start loaded from {init_path} strict={strict}")
        if missing:
            print(f"[WARN] Missing keys: {missing[:20]}{' ...' if len(missing) > 20 else ''}")
        if unexpected:
            print(f"[WARN] Unexpected keys: {unexpected[:20]}{' ...' if len(unexpected) > 20 else ''}")


    _write_run_metadata(run_dir, args, cfg, train_ds, val_ds, roles)

    nw = int(args.num_workers)
    pf = int(args.prefetch_factor)

    print(f"[INFO] DataLoader: num_workers={nw}, prefetch_factor={(pf if nw > 0 else 'n/a')}, persistent={nw > 0}", flush=True)

    train_kwargs = dict(
        train=True,
        batch_size=int(args.batch_size),
        num_workers=nw,
        shuffle=True,
        pin_memory=True,
        persistent_workers=(nw > 0),
    )
    if nw > 0:
        train_kwargs["prefetch_factor"] = pf

    val_kwargs = dict(
        train=False,
        batch_size=int(args.batch_size),
        num_workers=nw,
        shuffle=False,
        pin_memory=True,
        persistent_workers=(nw > 0),
    )
    if nw > 0:
        val_kwargs["prefetch_factor"] = pf

    train_dl = train_ds.to_dataloader(**train_kwargs)
    val_dl = val_ds.to_dataloader(**val_kwargs)

    device = torch.device("cuda")
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    prec = _normalize_precision(args.precision)

    # FIX: AMP should be enabled when precision requests it,
    # even if --enable_amp was forgotten.
    wants_amp = prec in {"16-mixed", "bf16-mixed"}
    use_amp = bool(device.type == "cuda" and (wants_amp or bool(args.enable_amp)))

    # If user set --enable_amp but left precision fp32, default to bf16 on H100.
    if use_amp and prec == "32-true":
        prec = "bf16-mixed"

    amp_dtype = torch.bfloat16 if (use_amp and prec == "bf16-mixed") else torch.float16
    use_scaler = bool(use_amp and prec == "16-mixed")
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    print(f"[INFO] Mixed precision: {use_amp} (precision={prec}, dtype={amp_dtype}, scaler={use_scaler})", flush=True)

    metrics_path = run_dir / "logs" / "metrics.csv"
    if not metrics_path.exists():
        with metrics_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "epoch",
                    "train_loss",
                    "val_loss",
                    "best_val_loss",
                    "improved",
                    "bad_epochs",
                    "lr",
                    "train_sec",
                    "val_sec",
                    "epoch_sec",
                    "train_it_per_sec",
                    "val_it_per_sec",
                    "gpu_peak_mem_gb",
                    "samples_per_sec",
                ]
            )

    print(f"[INFO] START TRAINING max_epochs={args.max_epochs}", flush=True)

    best_val_loss = float("inf")
    best_epoch = -1
    bad_epochs = 0

    prog_every = int(args.progress_every)
    grad_accum_steps = max(1, int(args.grad_accum_steps))
    if grad_accum_steps > 1:
        print(f"[INFO] Gradient accumulation: {grad_accum_steps} steps", flush=True)

    use_prefetcher = bool(device.type == "cuda" and (not args.disable_prefetcher))

    for epoch in range(int(args.max_epochs)):
        epoch_t0 = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        model.train()
        total_loss_t = torch.zeros((), device=device)
        train_steps = 0
        train_t0 = time.time()

        n_train = len(train_dl)
        optimizer.zero_grad(set_to_none=True)

        train_iter = StreamPrefetcher(train_dl, device) if use_prefetcher else train_dl

        for batch_idx, batch in enumerate(train_iter):
            if use_prefetcher:
                x, y = batch
            else:
                x, y = _move_batch_to_device(batch, device)

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                out = model(x)
                targets = y[0] if isinstance(y, (list, tuple)) else y
                loss = model.loss(out.prediction, targets)
                if grad_accum_steps > 1:
                    loss = loss / grad_accum_steps

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == n_train:
                if use_amp:
                    if args.grad_clip and args.grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if args.grad_clip and args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)

            # FIX: no per-step .item() sync
            loss_unscaled = loss.detach()
            if grad_accum_steps > 1:
                loss_unscaled = loss_unscaled * grad_accum_steps
            total_loss_t += loss_unscaled
            train_steps += 1

            if prog_every > 0 and (train_steps % prog_every == 0):
                _progress_line("train", epoch, train_steps, n_train, train_t0)

            if int(args.log_every_n_steps) > 0 and (train_steps % int(args.log_every_n_steps) == 0):
                # Sync only occasionally
                print(f"  Epoch {epoch} | Step {train_steps} | Loss: {float(loss_unscaled.item()):.4f}", flush=True)

        train_sec = time.time() - train_t0
        train_loss = float((total_loss_t / max(train_steps, 1)).item()) if train_steps > 0 else 0.0
        train_it_s = (train_steps / train_sec) if train_sec > 0 else 0.0

        # More meaningful than it/s
        samples_s = float(int(args.batch_size) * train_it_s)

        model.eval()
        val_loss_sum_t = torch.zeros((), device=device)
        val_steps = 0
        val_t0 = time.time()
        n_val = len(val_dl)

        val_iter = StreamPrefetcher(val_dl, device) if use_prefetcher else val_dl

        with torch.no_grad():
            for batch in val_iter:
                if use_prefetcher:
                    x, y = batch
                else:
                    x, y = _move_batch_to_device(batch, device)

                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    out = model(x)
                    targets = y[0] if isinstance(y, (list, tuple)) else y
                    vloss = model.loss(out.prediction, targets)

                val_loss_sum_t += vloss.detach()
                val_steps += 1

                if prog_every > 0 and (val_steps % prog_every == 0):
                    _progress_line("val", epoch, val_steps, n_val, val_t0)

        val_sec = time.time() - val_t0
        val_loss = float((val_loss_sum_t / max(val_steps, 1)).item()) if val_steps > 0 else 0.0
        val_it_s = (val_steps / val_sec) if val_sec > 0 else 0.0
        epoch_sec = time.time() - epoch_t0

        improved = val_loss < (best_val_loss - float(args.min_delta))
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1

        gpu_mem_gb = float(torch.cuda.max_memory_allocated() / 1e9) if device.type == "cuda" else 0.0

        print(
            f"✅ EPOCH {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} "
            f"| Train: {train_it_s:.2f} it/s | Val: {val_it_s:.2f} it/s | Samples/s: {samples_s:.1f} "
            f"| Epoch: {epoch_sec:.1f}s | GPU: {gpu_mem_gb:.1f}GB",
            flush=True,
        )

        with metrics_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    epoch,
                    f"{train_loss:.8f}",
                    f"{val_loss:.8f}",
                    f"{best_val_loss:.8f}",
                    int(improved),
                    bad_epochs,
                    f"{optimizer.param_groups[0]['lr']:.8e}",
                    f"{train_sec:.3f}",
                    f"{val_sec:.3f}",
                    f"{epoch_sec:.3f}",
                    f"{train_it_s:.3f}",
                    f"{val_it_s:.3f}",
                    f"{gpu_mem_gb:.3f}",
                    f"{samples_s:.3f}",
                ]
            )

        last_path = run_dir / "checkpoints" / "last.ckpt"
        torch.save(model.state_dict(), last_path)

        if improved:
            best_path = run_dir / "checkpoints" / "best.ckpt"
            torch.save(model.state_dict(), best_path)
            print(f"  [INFO] New best saved: val={val_loss:.4f} epoch={epoch}", flush=True)

        if int(args.patience) > 0 and bad_epochs >= int(args.patience):
            print(
                f"[INFO] Early stop. No improvement for {bad_epochs} epochs. Best epoch={best_epoch} val={best_val_loss:.4f}",
                flush=True,
            )
            break

    print(f"[DONE] Training finished. Best epoch={best_epoch} best_val={best_val_loss:.4f}", flush=True)


if __name__ == "__main__":
    main()
