#!/usr/bin/env python3
"""
Quick status check for running sweep without interfering with training processes.
"""
from pathlib import Path
import pandas as pd

SWEEPS_DIR = Path("experiments/lstm/sweeps")
RUNS_DIR = Path("experiments/lstm/runs")

# Count expected configs
expected_configs = list(SWEEPS_DIR.glob("*.yaml")) if SWEEPS_DIR.exists() else []
print(f"📋 Total configs: {len(expected_configs)}")

# Count started runs
started_runs = list(RUNS_DIR.glob("farm2107_*")) if RUNS_DIR.exists() else []
print(f"🚀 Started runs: {len(started_runs)}")

# Check completion status for each run
completed = []
in_progress = []

for run_dir in started_runs:
    metrics_file = run_dir / "farm2107_pretrain_sweep" / "version_0" / "metrics.csv"
    if metrics_file.exists():
        try:
            df = pd.read_csv(metrics_file)
            if "epoch" in df.columns:
                max_epoch = df["epoch"].max()
                # Assuming 20 epochs total
                if max_epoch >= 19:  # 0-indexed, so 19 = epoch 20
                    completed.append(run_dir.name)
                else:
                    in_progress.append((run_dir.name, int(max_epoch) + 1))
        except:
            in_progress.append((run_dir.name, "?"))

print(f"✅ Completed: {len(completed)}")
print(f"⏳ In progress: {len(in_progress)}")
print(f"⏸️  Not started: {len(expected_configs) - len(started_runs)}")

if in_progress:
    print(f"\n📊 Current progress:")
    for name, epoch in in_progress:
        config_name = name.replace("farm2107_", "")
        print(f"   {config_name}: epoch {epoch}/20")

if completed:
    print(f"\n🎉 Completed runs:")
    for name in completed:
        print(f"   {name.replace('farm2107_', '')}")

print(f"\n💡 Remaining: {len(expected_configs) - len(completed)}/{len(expected_configs)} configs")
