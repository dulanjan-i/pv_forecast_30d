#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

base = Path("freeze/final_thesis_v1/rl/rl_thesis_metrics_rerun_canonical")
a1 = base / "per_forecast_actions_v1.csv"
a2 = base / "per_forecast_actions_v2.csv"
r1 = base / "per_forecast_rewards_v1.csv"
r2 = base / "per_forecast_rewards_v2.csv"

out_dir = Path("freeze/final_thesis_v1/rl")
out_dir.mkdir(parents=True, exist_ok=True)

df_a1 = pd.read_csv(a1, parse_dates=["forecast_start"])
df_a2 = pd.read_csv(a2, parse_dates=["forecast_start"])
df_r1 = pd.read_csv(r1, parse_dates=["forecast_start"])
df_r2 = pd.read_csv(r2, parse_dates=["forecast_start"])

# alignment checks
align_ok_actions = df_a1["forecast_start"].equals(df_a2["forecast_start"])
align_ok_rewards = df_r1["forecast_start"].equals(df_r2["forecast_start"])

report = {"actions_aligned": bool(align_ok_actions), "rewards_aligned": bool(align_ok_rewards)}

if not align_ok_actions:
    # report mismatches
    m_actions = pd.concat([df_a1[["forecast_start"]].rename(columns={"forecast_start":"fs_v1"}),
                           df_a2[["forecast_start"]].rename(columns={"forecast_start":"fs_v2"})],
                          axis=1)
    m_actions["match"] = m_actions["fs_v1"] == m_actions["fs_v2"]
    m_actions.to_csv(out_dir / "mismatch_actions_forecast_start.csv", index=False)
    report["actions_mismatch_file"] = str(out_dir / "mismatch_actions_forecast_start.csv")

if not align_ok_rewards:
    m_rewards = pd.concat([df_r1[["forecast_start"]].rename(columns={"forecast_start":"fs_v1"}),
                           df_r2[["forecast_start"]].rename(columns={"forecast_start":"fs_v2"})],
                          axis=1)
    m_rewards["match"] = m_rewards["fs_v1"] == m_rewards["fs_v2"]
    m_rewards.to_csv(out_dir / "mismatch_rewards_forecast_start.csv", index=False)
    report["rewards_mismatch_file"] = str(out_dir / "mismatch_rewards_forecast_start.csv")

# If aligned, combine (use left join on forecast_start to be robust)
actions = df_a1.merge(df_a2, on="forecast_start", how="outer", suffixes=("_v1","_v2"))
actions = actions[["forecast_start","action_v1","action_v2"]]
actions.to_csv(out_dir / "combined_per_forecast_actions.csv", index=False)

rewards = df_r1.merge(df_r2, on="forecast_start", how="outer", suffixes=("_v1","_v2"))
# prefer reward columns; keep actions too for sanity
rewards = rewards[["forecast_start","action_v1","reward_v1","action_v2","reward_v2"]]
rewards.to_csv(out_dir / "combined_per_forecast_rewards.csv", index=False)

report["combined_actions"] = str(out_dir / "combined_per_forecast_actions.csv")
report["combined_rewards"] = str(out_dir / "combined_per_forecast_rewards.csv")

print(report)