"""
Offline DDQN training runner (self-contained).

Uses transitions parquet with columns:
- action, reward, done
- state_0..state_{D-1}
- next_state_0..next_state_{D-1}

Saves checkpoint that is compatible with your inference loader:
- includes 'q_net' state_dict
- also includes 'state_dict' alias for safety
- includes 'config'
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.rl.training import load_transitions


# -----------------------------
# Model
# -----------------------------
class DQNNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: List[int]):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = int(state_dim)
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, int(h)))
            layers.append(nn.ReLU())
            in_dim = int(h)
        layers.append(nn.Linear(in_dim, int(action_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class TrainConfig:
    state_dim: int
    action_dim: int
    hidden_sizes: List[int]
    gamma: float
    lr: float
    batch_size: int
    epochs: int
    target_update: int
    seed: int
    device: str


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ddqn_train(
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    next_states: np.ndarray,
    dones: np.ndarray,
    cfg: TrainConfig,
    outdir: Path,
) -> Path:
    device = torch.device(cfg.device)

    q_net = DQNNetwork(cfg.state_dim, cfg.action_dim, cfg.hidden_sizes).to(device)
    target_net = DQNNetwork(cfg.state_dim, cfg.action_dim, cfg.hidden_sizes).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    opt = optim.Adam(q_net.parameters(), lr=cfg.lr)
    loss_fn = nn.SmoothL1Loss()

    # tensors
    S = torch.tensor(states, dtype=torch.float32, device=device)
    A = torch.tensor(actions, dtype=torch.int64, device=device)
    R = torch.tensor(rewards, dtype=torch.float32, device=device)
    NS = torch.tensor(next_states, dtype=torch.float32, device=device)
    D = torch.tensor(dones.astype(np.float32), dtype=torch.float32, device=device)

    n = S.shape[0]
    steps = 0

    best_loss = float("inf")
    best_ckpt = outdir / "ddqn_best.pt"

    for epoch in range(cfg.epochs):
        # shuffle indices
        idx = torch.randperm(n, device=device)

        epoch_losses = []

        for start in range(0, n, cfg.batch_size):
            batch_idx = idx[start : start + cfg.batch_size]
            if batch_idx.numel() == 0:
                continue

            s = S[batch_idx]
            a = A[batch_idx]
            r = R[batch_idx]
            ns = NS[batch_idx]
            d = D[batch_idx]

            # current Q(s,a)
            q = q_net(s)  # (B, action_dim)
            q_sa = q.gather(1, a.view(-1, 1)).squeeze(1)

            # DDQN target:
            # a* = argmax_a Q_online(ns,a)
            with torch.no_grad():
                q_next_online = q_net(ns)
                a_star = torch.argmax(q_next_online, dim=1)  # (B,)
                q_next_target = target_net(ns)
                q_ns_astar = q_next_target.gather(1, a_star.view(-1, 1)).squeeze(1)
                y = r + cfg.gamma * (1.0 - d) * q_ns_astar

            loss = loss_fn(q_sa, y)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q_net.parameters(), 5.0)
            opt.step()

            steps += 1
            epoch_losses.append(float(loss.detach().cpu().item()))

            if steps % cfg.target_update == 0:
                target_net.load_state_dict(q_net.state_dict())

        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float("inf")
        print(f"[epoch {epoch+1:03d}/{cfg.epochs}] loss={mean_loss:.6f} steps={steps}")

        if mean_loss < best_loss:
            best_loss = mean_loss
            ckpt = {
                "q_net": q_net.state_dict(),
                "target_net": target_net.state_dict(),
                "state_dict": q_net.state_dict(),  # alias for inference loaders
                "config": asdict(cfg),
                "best_loss": best_loss,
                "steps": steps,
            }
            torch.save(ckpt, best_ckpt)

    # save final too
    final_ckpt = outdir / "ddqn_final.pt"
    torch.save(
        {
            "q_net": q_net.state_dict(),
            "target_net": target_net.state_dict(),
            "state_dict": q_net.state_dict(),
            "config": asdict(cfg),
            "best_loss": best_loss,
            "steps": steps,
        },
        final_ckpt,
    )

    print("OK best_ckpt:", best_ckpt)
    print("OK final_ckpt:", final_ckpt)
    return best_ckpt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--state-dim", type=int, required=True)
    ap.add_argument("--action-dim", type=int, required=True)
    ap.add_argument("--hidden-sizes", type=int, nargs="+", default=[128, 64])
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--target-update", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = TrainConfig(
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        hidden_sizes=list(args.hidden_sizes),
        gamma=float(args.gamma),
        lr=float(args.lr),
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        target_update=int(args.target_update),
        seed=int(args.seed),
        device=str(args.device),
    )

    set_seed(cfg.seed)

    states, actions, rewards, next_states, dones = load_transitions(args.data, state_dim=cfg.state_dim)
    print(f"[data] N={len(states)} state_dim={states.shape[1]} actions={len(set(actions.tolist()))}")
    print(f"[data] reward mean={rewards.mean():.6f} std={rewards.std():.6f}")

    # save config for provenance
    (outdir / "train_config.json").write_text(json.dumps(asdict(cfg), indent=2))

    ddqn_train(states, actions, rewards, next_states, dones, cfg, outdir)


if __name__ == "__main__":
    main()
