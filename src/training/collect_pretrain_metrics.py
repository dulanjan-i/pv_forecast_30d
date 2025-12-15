"""
Collect pretraining metrics from all hyperparameter sweep runs.

Scans experiments/lstm/runs/farm2107_*/ directories and extracts:
- Hyperparameters (hidden_size, num_layers, learning_rate)
- Final validation loss (MSE)
- Final validation RMSE

Outputs: experiments/lstm/pretrain_hparam_results.csv

Usage:
    python src/training/collect_pretrain_metrics.py
"""

import math
import sys
from pathlib import Path

import pandas as pd
import yaml

# Add project root to Python path
project_root = Path(__file__).resolve().parents[2]  # Go up 2 levels: training -> src -> project_root
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

RUNS_ROOT = Path("experiments/lstm/runs")
SWEEPS_ROOT = Path("experiments/lstm/sweeps")
OUTPUT_CSV = Path("experiments/lstm/pretrain_hparam_results.csv")

def main():
    if not RUNS_ROOT.exists():
        print(f"Error: Runs directory not found at {RUNS_ROOT}")
        print("Have you run the sweep yet? (python scripts/run_pretrain_sweep.py)")
        sys.exit(1)

    rows = []
    found_runs = 0
    skipped_runs = 0

    print(f"Scanning for runs in: {RUNS_ROOT}")
    print(f"{'='*60}\n")

    for run_dir in sorted(RUNS_ROOT.glob("farm2107_h*")):
        found_runs += 1
        print(f"Processing: {run_dir.name}")

        # Load config to get hyperparameters
        cfg_pattern = run_dir.name.replace("farm2107_", "pretrain_farm2107_") + ".yaml"
        cfg_files = list(SWEEPS_ROOT.glob(cfg_pattern))
        
        if not cfg_files:
            print(f"  ⚠ Missing config file: {cfg_pattern}")
            skipped_runs += 1
            continue

        with cfg_files[0].open("r") as f:
            cfg = yaml.safe_load(f)

        # Extract hyperparameters
        h = cfg["model"]["hidden_size"]
        l = cfg["model"]["num_layers"]
        lr = cfg["training"]["learning_rate"]
        max_epochs = cfg["training"]["max_epochs"]
        batch_size = cfg["training"]["batch_size"]
        tag = cfg["experiment"]["tag"]

        # Find Lightning CSV metrics
        # Path should be: experiments/lstm/runs/farm2107_h64_l2_lr1e-3/farm2107_pretrain_sweep/version_0/metrics.csv
        metrics_candidates = [
            run_dir / "farm2107_pretrain_sweep" / "version_0" / "metrics.csv",
            run_dir / "pretrain" / "version_0" / "metrics.csv",
        ]
        
        metrics_path = None
        for candidate in metrics_candidates:
            if candidate.exists():
                metrics_path = candidate
                break
        
        if not metrics_path:
            print(f"  ⚠ Missing metrics.csv (checked {len(metrics_candidates)} locations)")
            skipped_runs += 1
            continue

        # Read metrics
        try:
            df = pd.read_csv(metrics_path)
        except Exception as e:
            print(f"  ✗ Failed to read metrics: {e}")
            skipped_runs += 1
            continue

        if "val_loss" not in df.columns:
            print(f"  ⚠ No val_loss column in metrics")
            skipped_runs += 1
            continue

        val_series = df["val_loss"].dropna()
        if val_series.empty:
            print(f"  ⚠ No val_loss values found")
            skipped_runs += 1
            continue

        # Get final validation loss (MSE)
        final_val_mse = float(val_series.iloc[-1])
        final_val_rmse = math.sqrt(final_val_mse)
        
        # Also get min validation loss
        min_val_mse = float(val_series.min())
        min_val_rmse = math.sqrt(min_val_mse)

        print(f"  ✓ Final val_loss: {final_val_mse:.6f} (RMSE: {final_val_rmse:.6f})")
        print(f"  ✓ Best val_loss:  {min_val_mse:.6f} (RMSE: {min_val_rmse:.6f})")

        row = {
            "tag": tag,
            "hidden_size": h,
            "num_layers": l,
            "learning_rate": lr,
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            "final_val_mse": final_val_mse,
            "final_val_rmse": final_val_rmse,
            "best_val_mse": min_val_mse,
            "best_val_rmse": min_val_rmse,
        }
        rows.append(row)

    print(f"\n{'='*60}")
    print(f"Found {found_runs} run directories")
    print(f"Successfully processed {len(rows)} runs")
    print(f"Skipped {skipped_runs} runs")
    print(f"{'='*60}\n")

    if not rows:
        print("No results to save!")
        sys.exit(1)

    # Create DataFrame and sort by best RMSE
    results = pd.DataFrame(rows)
    results = results.sort_values("best_val_rmse")

    # Save to CSV
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_CSV, index=False)

    print(f"Results saved to: {OUTPUT_CSV}\n")
    print("Top 5 configurations by best validation RMSE:")
    print(results[["tag", "hidden_size", "num_layers", "learning_rate", "best_val_rmse"]].head())
    
    print("\n" + "="*60)
    print("Summary Statistics:")
    print("="*60)
    print(f"Best RMSE:  {results['best_val_rmse'].min():.6f}")
    print(f"Worst RMSE: {results['best_val_rmse'].max():.6f}")
    print(f"Mean RMSE:  {results['best_val_rmse'].mean():.6f}")
    print(f"Std RMSE:   {results['best_val_rmse'].std():.6f}")

if __name__ == "__main__":
    main()
