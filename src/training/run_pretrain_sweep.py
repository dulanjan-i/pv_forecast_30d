"""
Hyperparameter sweep for LSTM pretraining on Farm 2107 dataset.

Generates multiple YAML configs by varying:
- hidden_size: 32, 64, 128
- num_layers: 1, 2
- learning_rate: 5e-4, 1e-3

Grid size: 3 x 2 x 2 = 12 runs

Supports parallel execution across multiple GPUs.

Usage:
    # Sequential (one GPU):
    python src/training/run_pretrain_sweep.py

    # Parallel (4 GPUs):
    python src/training/run_pretrain_sweep.py --parallel --num-gpus 4
"""

import itertools
import subprocess
import sys
import yaml
import argparse
import time
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add project root to Python path
project_root = Path(__file__).resolve().parents[2]  # Go up 2 levels: training -> src -> project_root
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Configuration
BASE_CFG = Path("experiments/lstm/pretrain_farm2107.yaml")
SWEEPS_DIR = Path("experiments/lstm/sweeps")
SWEEPS_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameter grid
hidden_sizes = [32, 64, 128]
num_layers_list = [1, 2]
lrs = [5e-4, 1e-3]

def make_tag(h: int, l: int, lr: float) -> str:
    """Create a readable tag for this config."""
    # Format: h64_l2_lr1e-3
    lr_str = f"{lr:.0e}".replace("-", "m").replace("+", "")
    return f"h{h}_l{l}_lr{lr_str}"


def run_single_experiment(cfg_path: Path, gpu_id: int) -> tuple[str, bool, str]:
    """
    Run a single experiment on a specific GPU.
    
    Returns: (tag, success, error_message)
    """
    tag = cfg_path.stem.replace("pretrain_farm2107_", "")
    
    # Set environment variable to use specific GPU
    env = {"CUDA_VISIBLE_DEVICES": str(gpu_id)}
    
    print(f"[GPU {gpu_id}] Starting: {tag}")
    
    try:
        # Don't capture output - let it stream to terminal
        result = subprocess.run(
            [
                sys.executable,
                "src/training/pretrain_lstm.py",
                "--config",
                str(cfg_path),
            ],
            check=True,
            cwd=project_root,
            env={**os.environ, **env},  # Merge with existing env
        )
        print(f"[GPU {gpu_id}] ✓ Completed: {tag}")
        return (tag, True, "")
    except subprocess.CalledProcessError as e:
        error_msg = f"Exit code {e.returncode}"
        print(f"[GPU {gpu_id}] ✗ Failed: {tag} - {error_msg}")
        return (tag, False, error_msg)


def main():
    parser = argparse.ArgumentParser(description="Run hyperparameter sweep for LSTM pretraining")
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run experiments in parallel across multiple GPUs",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=4,
        help="Number of GPUs to use for parallel execution (default: 4)",
    )
    args = parser.parse_args()
    if not BASE_CFG.exists():
        print(f"Error: Base config not found at {BASE_CFG}")
        sys.exit(1)

    configs_created = []
    
    # Generate all config combinations
    for h, l, lr in itertools.product(hidden_sizes, num_layers_list, lrs):
        tag = make_tag(h, l, lr)

        # Load base config
        with BASE_CFG.open("r") as f:
            cfg = yaml.safe_load(f)

        # Update hyperparameters
        cfg["model"]["hidden_size"] = h
        cfg["model"]["num_layers"] = l
        cfg["training"]["learning_rate"] = lr

        # Update experiment tracking
        cfg.setdefault("experiment", {})
        cfg["experiment"]["name"] = "farm2107_pretrain_sweep"
        cfg["experiment"]["tag"] = tag

        cfg.setdefault("paths", {})
        cfg["paths"]["output_dir"] = f"experiments/lstm/runs/farm2107_{tag}"

        # Save config
        cfg_path = SWEEPS_DIR / f"pretrain_farm2107_{tag}.yaml"
        with cfg_path.open("w") as f:
            yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
        
        configs_created.append((tag, cfg_path))
        print(f"✓ Created config: {cfg_path}")

    print(f"\n{'='*60}")
    print(f"Created {len(configs_created)} configs")
    print(f"Mode: {'PARALLEL' if args.parallel else 'SEQUENTIAL'}")
    if args.parallel:
        print(f"Using {args.num_gpus} GPUs")
    print(f"{'='*60}\n")

    # Track results
    successful = []
    failed = []
    start_time = time.time()

    if args.parallel:
        # Parallel execution across multiple GPUs
        print(f"Running {len(configs_created)} experiments in parallel on {args.num_gpus} GPUs\n")
        
        with ProcessPoolExecutor(max_workers=args.num_gpus) as executor:
            # Submit all jobs with round-robin GPU assignment
            futures = {}
            for i, (tag, cfg_path) in enumerate(configs_created):
                gpu_id = i % args.num_gpus
                future = executor.submit(run_single_experiment, cfg_path, gpu_id)
                futures[future] = (tag, cfg_path, gpu_id)
            
            # Collect results as they complete
            for future in as_completed(futures):
                tag, cfg_path, gpu_id = futures[future]
                result_tag, success, error_msg = future.result()
                
                if success:
                    successful.append(result_tag)
                else:
                    failed.append((result_tag, error_msg))
    else:
        # Sequential execution
        print(f"Running {len(configs_created)} experiments sequentially\n")
        
        for i, (tag, cfg_path) in enumerate(configs_created, 1):
            print(f"\n{'='*60}")
            print(f"[{i}/{len(configs_created)}] Running: {tag}")
            print(f"Config: {cfg_path}")
            print(f"{'='*60}\n")

            try:
                subprocess.run(
                    [
                        sys.executable,
                        "src/training/pretrain_lstm.py",
                        "--config",
                        str(cfg_path),
                    ],
                    check=True,
                    cwd=project_root,
                )
                print(f"\n✓ Completed: {tag}\n")
                successful.append(tag)
            except subprocess.CalledProcessError as e:
                print(f"\n✗ Failed: {tag}")
                print(f"Error: {e}\n")
                failed.append((tag, str(e)))
                continue

    elapsed = time.time() - start_time
    elapsed_min = elapsed / 60

    print(f"\n{'='*60}")
    print("Sweep complete!")
    print(f"{'='*60}")
    print(f"Total time: {elapsed_min:.1f} minutes")
    print(f"Successful: {len(successful)}/{len(configs_created)}")
    print(f"Failed: {len(failed)}/{len(configs_created)}")
    
    if failed:
        print(f"\nFailed runs:")
        for tag, error in failed:
            print(f"  - {tag}")
    
    print(f"\nConfigs saved to: {SWEEPS_DIR}")
    print(f"Results saved to: experiments/lstm/runs/farm2107_*/")
    print(f"\nNext step: Run metrics collection script")
    print(f"  python src/training/collect_pretrain_metrics.py")

if __name__ == "__main__":
    main()
