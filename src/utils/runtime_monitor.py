"""
Runtime monitor for training loops.

Purpose:
- Show whether training is dataloader-bound or GPU-bound by measuring:
  - time to fetch next batch (loader time)
  - time to run a forward/backward step (compute time)
- Works with any PyTorch DataLoader.

Usage examples:
  python -m src.utils.runtime_monitor --mode dataloader --steps 200
  python -m src.utils.runtime_monitor --mode dataloader --steps 200 --sleep 0.1

Notes:
- This script does NOT train your model. It probes the dataloader timing pattern.
- If loader_time >> compute_time, your GPU will starve.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class TimingStats:
    loader_s: float = 0.0
    compute_s: float = 0.0
    n: int = 0

    def add(self, loader_s: float, compute_s: float) -> None:
        self.loader_s += loader_s
        self.compute_s += compute_s
        self.n += 1

    def summary(self) -> str:
        if self.n == 0:
            return "No steps recorded."
        l = self.loader_s / self.n
        c = self.compute_s / self.n
        ratio = (l / c) if c > 0 else float("inf")
        return f"avg loader={l:.4f}s, avg compute={c:.4f}s, loader/compute={ratio:.2f}x over {self.n} steps"


def _fake_step_on_gpu(batch, device: torch.device) -> float:
    """
    Do a tiny synthetic compute step on GPU to approximate per-batch overhead.
    We don't know your model here, so we use a simple matmul with batch size scaling.
    """
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    # crude batch-size proxy
    bsz = None
    if isinstance(batch, (list, tuple)) and len(batch) > 0 and torch.is_tensor(batch[0]):
        bsz = batch[0].shape[0]
    elif torch.is_tensor(batch):
        bsz = batch.shape[0]
    else:
        bsz = 256

    x = torch.randn((max(1, bsz), 1024), device=device, dtype=torch.float16)
    w = torch.randn((1024, 1024), device=device, dtype=torch.float16)
    y = x @ w
    y.mean().backward()

    torch.cuda.synchronize()
    return time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dataloader"], default="dataloader")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--sleep", type=float, default=0.0, help="Optional sleep between steps")
    args = ap.parse_args()

    # You need to plug your real dataloader here.
    # For MiRACLE/TFT you already create it in train_tft_v1.py via make_datasets + make_dataloaders.
    # The clean way is to import and reuse those functions.
    #
    # If your train_tft_v1.py exposes a helper to build loaders, use that.
    # If not, create a small helper function there and import it here.

    try:
        from src.training.train_tft_v1 import make_loaders_for_debug  # type: ignore
    except Exception as e:
        raise SystemExit(
            "You need to add a helper in src/training/train_tft_v1.py called "
            "`make_loaders_for_debug()` that returns (train_loader, val_loader).\n"
            f"Import error was: {e}"
        )

    train_loader, _ = make_loaders_for_debug()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))

    stats = TimingStats()

    it = iter(train_loader)
    last = time.perf_counter()

    for i in range(args.steps):
        # loader timing = time waiting for next batch
        t1 = time.perf_counter()
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)
        loader_s = time.perf_counter() - t1

        # compute timing
        compute_s = 0.0
        if device.type == "cuda":
            compute_s = _fake_step_on_gpu(batch, device)

        stats.add(loader_s, compute_s)

        if (i + 1) % 20 == 0:
            print(f"[{i+1:04d}/{args.steps}] {stats.summary()}")

        if args.sleep > 0:
            time.sleep(args.sleep)

    print("Final:", stats.summary())


if __name__ == "__main__":
    main()
