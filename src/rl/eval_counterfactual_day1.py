# src/rl/eval_counterfactual_day1.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch

from src.rl.training import DQNNetwork  # same net used in training


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="p3_phase1_COUNTERFACTUAL_DAY1.parquet")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--state-dim", type=int, required=True)
    ap.add_argument("--action-dim", type=int, default=3)
    ap.add_argument("--baseline-action", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=None)
    args = ap.parse_args()

    df = pd.read_parquet(args.data).copy()
    df = df.dropna(subset=["forecast_start", "action", "reward"])
    df["forecast_start"] = pd.to_datetime(df["forecast_start"], utc=True)

    # optional cap
    if args.n is not None:
        starts = sorted(df["forecast_start"].unique())[: args.n]
        df = df[df["forecast_start"].isin(starts)].copy()

    state_cols = [c for c in df.columns if c.startswith("state_")]
    if len(state_cols) != args.state_dim:
        raise ValueError(f"state cols={len(state_cols)} but --state-dim={args.state_dim}")

    # load model
    ckpt = torch.load(args.ckpt, map_location="cpu")
    net = DQNNetwork(args.state_dim, args.action_dim, ckpt["config"].hidden_sizes)
    net.load_state_dict(ckpt["q_net"])
    net.eval()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    net.to(device)

    # evaluate per forecast_start
    out_rows: List[dict] = []
    for fs, g in df.groupby("forecast_start"):
        g = g.sort_values("action")
        # state is same for all actions
        st = g.iloc[0][state_cols].to_numpy(dtype=np.float32)
        st_t = torch.from_numpy(st).unsqueeze(0).to(device)

        with torch.no_grad():
            q = net(st_t).squeeze(0).cpu().numpy()
        a_hat = int(np.argmax(q))

        # realized rewards from counterfactual table
        r_bas = float(g.loc[g["action"] == args.baseline_action, "reward"].iloc[0])
        r_pol = float(g.loc[g["action"] == a_hat, "reward"].iloc[0])

        out_rows.append(
            {
                "forecast_start": fs,
                "baseline_action": int(args.baseline_action),
                "policy_action": int(a_hat),
                "baseline_rmse": float(-r_bas),
                "policy_rmse": float(-r_pol),
                "improved": int((-r_pol) < (-r_bas)),
            }
        )

    out = pd.DataFrame(out_rows).sort_values("forecast_start").reset_index(drop=True)
    print("n:", len(out))
    print("mean baseline:", float(out["baseline_rmse"].mean()))
    print("mean policy  :", float(out["policy_rmse"].mean()))
    print("fraction improved:", float(out["improved"].mean()))

    out_path = Path("experiments/rl/plant_03_phase1/eval_counterfactual_day1.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print("WROTE:", out_path)


if __name__ == "__main__":
    main()
