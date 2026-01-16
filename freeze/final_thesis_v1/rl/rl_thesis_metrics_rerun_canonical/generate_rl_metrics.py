#!/usr/bin/env python3
"""
Generate canonical RL metrics for policy checkpoints (v1 and v2).

Outputs:
 - metrics_{tag}.json
 - action_distribution_{tag}.csv
 - per_forecast_actions_{tag}.csv
 - optional: training_dynamics_{tag}.csv (if a train.log is found)
"""
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch


# -----------------------
# Helpers (copied/adapted from phase1_inference)
# -----------------------
def _read_parquet_must_exist(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return pd.read_parquet(path)


def _infer_state_cols(df: pd.DataFrame, qnet_in_dim: int) -> List[str]:
    cols = list(df.columns)
    preferred = []
    for prefix in ("s_", "state_", "feat_", "x_"):
        preferred = [c for c in cols if c.startswith(prefix)]
        if preferred:
            break
    if not preferred:
        preferred = [c for c in cols if (c.startswith("s") and c[1:].isdigit())]
    if not preferred:
        exclude = {
            "action", "a",
            "reward", "r",
            "done", "terminal",
            "forecast_start",
            "blend_short", "blend_long", "blend_physics",
        }
        numeric = []
        for c in cols:
            if c in exclude:
                continue
            if c.startswith("next_") or c.startswith("ns_") or c.startswith("s_next"):
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                numeric.append(c)
        preferred = numeric

    if len(preferred) < qnet_in_dim:
        raise ValueError(
            f"Could not infer enough state columns. Need {qnet_in_dim}, found {len(preferred)}. "
            f"Candidates: {preferred[:50]}"
        )
    return preferred[:qnet_in_dim]


class QNet(torch.nn.Module):
    def __init__(self, layer_sizes: List[Tuple[int, int]]):
        super().__init__()
        layers: List[torch.nn.Module] = []
        for i, (in_f, out_f) in enumerate(layer_sizes):
            layers.append(torch.nn.Linear(in_f, out_f))
            if i < len(layer_sizes) - 1:
                layers.append(torch.nn.ReLU())
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _extract_qnet_state_dict(ckpt: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    for key in ("q_net", "online_net", "policy_net", "model_state_dict", "state_dict"):
        if key in ckpt and isinstance(ckpt[key], dict):
            return ckpt[key]
    if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return ckpt  # type: ignore
    raise ValueError(f"Could not find q-network state_dict keys in checkpoint. Keys: {list(ckpt.keys())}")


def _build_qnet_from_state_dict(sd: Dict[str, torch.Tensor]) -> Tuple[QNet, int, int]:
    weight_keys = [k for k in sd.keys() if k.endswith(".weight") and k.startswith("net.")]
    if not weight_keys:
        raise ValueError(f"State dict does not look like expected MLP with 'net.*.weight'. Keys: {list(sd.keys())[:20]}")
    def _idx(k: str) -> int:
        return int(k.split(".")[1])
    weight_keys = sorted(weight_keys, key=_idx)
    layer_sizes: List[Tuple[int, int]] = []
    for k in weight_keys:
        w = sd[k]
        if w.ndim != 2:
            raise ValueError(f"Unexpected weight tensor shape for {k}: {tuple(w.shape)}")
        out_f, in_f = int(w.shape[0]), int(w.shape[1])
        layer_sizes.append((in_f, out_f))
    state_dim = layer_sizes[0][0]
    action_dim = layer_sizes[-1][1]
    qnet = QNet(layer_sizes)
    qnet.load_state_dict(sd, strict=True)
    return qnet, state_dim, action_dim


# -----------------------
# Core runner
# -----------------------
def inspect_train_log_maybe(checkpoint_path: Path) -> Dict[str, Any]:
    out = {}
    logp = checkpoint_path.parent / "train.log"
    if not logp.exists():
        return out
    try:
        with logp.open() as fh:
            lines = fh.readlines()
        # crude parse: look for "epoch" or "best_loss" or "reward mean"
        for L in lines[-500:]:
            l = L.strip()
            if "best_loss" in l or "best_val" in l or "best loss" in l:
                out.setdefault("tail_lines", []).append(l)
        # return size and first/last lines
        out["train_log_lines"] = len(lines)
        out["train_log_tail"] = lines[-20:]
    except Exception as e:
        out["train_log_error"] = str(e)
    return out


def run_for_variant(tag: str, policy_ckpt: Path, sarns_parquet: Path, out_dir: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    sarns = _read_parquet_must_exist(sarns_parquet).copy()
    if "forecast_start" not in sarns.columns:
        raise ValueError("sarns_norm must contain 'forecast_start' column")
    sarns["forecast_start"] = pd.to_datetime(sarns["forecast_start"], utc=True, errors="coerce")
    sarns = sarns.dropna(subset=["forecast_start"])
    sarns["forecast_start"] = sarns["forecast_start"].dt.floor("D")
    sarns = sarns.sort_values("forecast_start")
    sarns_1 = sarns.drop_duplicates(subset=["forecast_start"], keep="last").set_index("forecast_start")

    ck = torch.load(str(policy_ckpt), map_location="cpu")
    q_sd = _extract_qnet_state_dict(ck)
    qnet, q_state_dim, q_action_dim = _build_qnet_from_state_dict(q_sd)
    qnet.eval()

    # infer state cols
    state_cols = _infer_state_cols(sarns_1.reset_index(), q_state_dim)

    # per-forecast action decisions
    forecast_actions = []
    forecast_rewards = []
    forecasts = list(sarns_1.index)
    action_counts_forecast = {}
    action_counts_timestep = {}
    action_to_weights = None  # not needed here, but could be derived

    for fs in forecasts:
        st = sarns_1.loc[fs, state_cols].to_numpy(dtype=np.float32)
        st_t = torch.from_numpy(st).view(1, -1)
        with torch.no_grad():
            qvals = qnet(st_t)
            a = int(torch.argmax(qvals, dim=1).item())
        forecast_actions.append({"forecast_start": str(fs), "action": int(a)})
        action_counts_forecast[a] = action_counts_forecast.get(a, 0) + 1
        action_counts_timestep[a] = action_counts_timestep.get(a, 0) + 2880  # each forecast = 2880 timesteps

        # find reward for this fs & action
        rows = sarns[(sarns["forecast_start"] == fs) & (sarns.get("action", sarns.get("a", None)) == a)]
        # fallback if above failed or empty:
        if rows is None or len(rows) == 0:
            rows = sarns[(sarns["forecast_start"] == fs) & (sarns.get("action", sarns.get("a", None)) == 0)]
        # pick reward column name
        reward_col = None
        for c in ("reward", "r", "reward_v2", "reward_v1"):
            if c in sarns.columns:
                reward_col = c
                break
        if reward_col is None:
            reward_val = float("nan")
        else:
            if len(rows) == 0:
                reward_val = float("nan")
            else:
                reward_val = float(rows[reward_col].mean())
        forecast_rewards.append({"forecast_start": str(fs), "action": int(a), "reward": reward_val})

    # aggregate reward stats (exclude nan)
    rewards_arr = np.array([r["reward"] for r in forecast_rewards], dtype=float)
    finite_mask = np.isfinite(rewards_arr)
    reward_mean = float(np.nan) if not finite_mask.any() else float(np.mean(rewards_arr[finite_mask]))
    reward_std = float(np.nan) if not finite_mask.any() else float(np.std(rewards_arr[finite_mask]))
    reward_count = int(np.sum(finite_mask))

    # checkpoint metadata
    ck_meta = {}
    if isinstance(ck, dict):
        for k in ("best_loss", "steps", "config"):
            if k in ck:
                val = ck[k]
                try:
                    json.dumps(val)
                    ck_meta[k] = val
                except Exception:
                    # fallback: repr
                    ck_meta[k] = repr(val)[:2000]
    # try train log
    train_log_info = inspect_train_log_maybe(policy_ckpt)

    # write outputs
    out_metrics = {
        "tag": tag,
        "policy_ckpt": str(policy_ckpt),
        "sarns_parquet": str(sarns_parquet),
        "num_forecasts": len(forecasts),
        "action_counts_forecast": action_counts_forecast,
        "action_counts_timestep": action_counts_timestep,
        "reward_mean_per_forecast": reward_mean,
        "reward_std_per_forecast": reward_std,
        "reward_count": reward_count,
        "checkpoint_meta": ck_meta,
        "train_log_info": train_log_info,
    }

    # save files
    out_dir.joinpath(f"metrics_{tag}.json").write_text(json.dumps(out_metrics, indent=2))
    pd.DataFrame.from_records(forecast_actions).to_csv(out_dir.joinpath(f"per_forecast_actions_{tag}.csv"), index=False)
    pd.DataFrame.from_records(forecast_rewards).to_csv(out_dir.joinpath(f"per_forecast_rewards_{tag}.csv"), index=False)
    pd.DataFrame.from_dict(
        [{"action": int(k), "count_forecast": v, "count_timestep": action_counts_timestep.get(k, 0)} for k, v in action_counts_forecast.items()]
    ).to_csv(out_dir.joinpath(f"action_distribution_{tag}.csv"), index=False)

    return out_metrics


def main():
    out_dir = Path("freeze/final_thesis_v1/rl/rl_thesis_metrics_rerun_canonical")
    out_dir.mkdir(parents=True, exist_ok=True)

    # canonical paths (adjust if yours differ)
    v1_ck = Path("freeze/final_thesis_v1/rl/ddqn_minenv_v1/ddqn_best.pt")
    v2_ck = Path("freeze/final_thesis_v1/rl/ddqn_minenv_v2/ddqn_best.pt")
    v1_sarns = Path("freeze/final_thesis_v1/phase1_2024daily_final/processed/p3_phase1_SARNS_MINENV_v1.parquet")
    v2_sarns = Path("freeze/final_thesis_v1/phase1_2024daily_final/processed/p3_phase1_SARNS_MINENV_v2.parquet")

    results = {}
    results["v1"] = run_for_variant("v1", v1_ck, v1_sarns, out_dir)
    results["v2"] = run_for_variant("v2", v2_ck, v2_sarns, out_dir)

    # combined table summary
    rows = []
    for k, v in results.items():
        rows.append({
            "variant": k,
            "policy_ckpt": v["policy_ckpt"],
            "num_forecasts": v["num_forecasts"],
            "action_count_forecast_0": v["action_counts_forecast"].get(0, 0),
            "action_count_forecast_1": v["action_counts_forecast"].get(1, 0),
            "action_count_forecast_2": v["action_counts_forecast"].get(2, 0),
            "action_count_forecast_3": v["action_counts_forecast"].get(3, 0),
            "action_count_forecast_4": v["action_counts_forecast"].get(4, 0),
            "action_count_forecast_5": v["action_counts_forecast"].get(5, 0),
            "action_count_forecast_6": v["action_counts_forecast"].get(6, 0),
            "action_count_forecast_7": v["action_counts_forecast"].get(7, 0),
            "reward_mean_per_forecast": v["reward_mean_per_forecast"],
            "reward_std_per_forecast": v["reward_std_per_forecast"],
            "reward_count": v["reward_count"],
            "best_loss": v["checkpoint_meta"].get("best_loss"),
            "steps": v["checkpoint_meta"].get("steps"),
        })
    pd.DataFrame(rows).to_csv(out_dir.joinpath("summary_table_v1_v2.csv"), index=False)
    Path(out_dir.joinpath("summary_table_v1_v2.json")).write_text(json.dumps(rows, indent=2))

    print("WROTE outputs to", out_dir)


if __name__ == "__main__":
    main()