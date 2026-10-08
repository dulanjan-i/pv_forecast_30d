# src/rl/train_ddqn_counterfactual.py
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.rl.training import DQNTrainingConfig, DQNTrainer  # keep your names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--state-dim", type=int, required=True)
    ap.add_argument("--action-dim", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gamma", type=float, default=0.0)
    ap.add_argument("--target-update", type=int, default=10)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    df = pd.read_parquet(args.data)

    cfg = DQNTrainingConfig(
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch,
        num_epochs=args.epochs,
        target_update_freq=args.target_update,
        device=args.device,
    )

    trainer = DQNTrainer(cfg)
    hist = trainer.train(df)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    trainer.save(str(out))

    print("SAVED:", out)
    print("final loss:", hist["loss"][-1] if hist["loss"] else None)


if __name__ == "__main__":
    main()
