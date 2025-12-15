# LSTM Hyperparameter Sweep Guide

## Overview

This sweep system runs 12 experiments with different hyperparameter combinations:
- **hidden_size**: 32, 64, 128
- **num_layers**: 1, 2  
- **learning_rate**: 5e-4, 1e-3
- **Total combinations**: 3 × 2 × 2 = 12 runs

Each run trains for 20 epochs on a single GPU.

## Files Structure

```
experiments/lstm/
├── pretrain_farm2107.yaml          # Base template config
├── sweeps/                          # Generated configs (12 files)
│   ├── pretrain_farm2107_h32_l1_lr5em4.yaml
│   ├── pretrain_farm2107_h32_l1_lr1em3.yaml
│   └── ...
├── runs/                            # Training results
│   ├── farm2107_h32_l1_lr5em4/     # Per-config outputs
│   │   ├── checkpoints/
│   │   │   ├── lstm_encoder_farm2107_h32_l1_lr5em4_last.ckpt
│   │   │   └── lstm_encoder_farm2107_h32_l1_lr5em4_weights.pt
│   │   └── farm2107_pretrain_sweep/
│   │       └── version_0/
│   │           └── metrics.csv     # Training metrics
│   └── ...
└── pretrain_hparam_results.csv     # Final comparison table
```

## How to Run

### Option 1: Sequential (One GPU at a time)
Runs all 12 experiments one after another on GPU 0.
**Time**: ~12 × training_time per run

```bash
source ~/.venvs/pvforecast/bin/activate
cd ~/pv_forecast_30d

python src/training/run_pretrain_sweep.py
```

### Option 2: Parallel (All 4 GPUs) ⚡ RECOMMENDED
Runs 4 experiments simultaneously, one per GPU.
**Time**: ~3 × training_time per run (4x speedup!)

```bash
source ~/.venvs/pvforecast/bin/activate
cd ~/pv_forecast_30d

python src/training/run_pretrain_sweep.py --parallel --num-gpus 4
```

**How it works:**
- GPU 0: runs configs 0, 4, 8
- GPU 1: runs configs 1, 5, 9
- GPU 2: runs configs 2, 6, 10
- GPU 3: runs configs 3, 7, 11

Uses `CUDA_VISIBLE_DEVICES` environment variable to isolate each run.

### Monitor Progress

**Check GPU usage:**
```bash
watch -n 1 nvidia-smi
```

**Check specific run logs:**
```bash
# List all runs
ls experiments/lstm/runs/

# View metrics for a specific run
cat experiments/lstm/runs/farm2107_h64_l2_lr1em3/farm2107_pretrain_sweep/version_0/metrics.csv
```

## Collect Results

After sweep completes, aggregate all metrics:

```bash
python src/training/collect_pretrain_metrics.py
```

This creates `experiments/lstm/pretrain_hparam_results.csv` with columns:
- tag
- hidden_size
- num_layers
- learning_rate
- batch_size
- max_epochs
- final_val_mse
- final_val_rmse
- best_val_mse
- best_val_rmse

Results are sorted by **best_val_rmse** (lower is better).

## Modify the Sweep

Edit `src/training/run_pretrain_sweep.py`:

```python
# Change hyperparameter grid
hidden_sizes = [32, 64, 128]        # Try [64, 128, 256]
num_layers_list = [1, 2]            # Try [1, 2, 3]
lrs = [5e-4, 1e-3]                  # Try [1e-4, 5e-4, 1e-3]
```

Or change epochs in `experiments/lstm/pretrain_farm2107.yaml`:
```yaml
training:
  max_epochs: 20  # Change to 30 for longer training
```

## Troubleshooting

### Out of memory on GPU
- Reduce `batch_size` in `pretrain_farm2107.yaml`
- Use fewer parallel workers: `--num-gpus 2`

### Some runs fail
- Check failed run logs in `experiments/lstm/runs/farm2107_<tag>/`
- The sweep continues even if one run fails
- Failed runs are listed at the end

### Clean up and restart
```bash
# Remove all sweep results
rm -rf experiments/lstm/sweeps/
rm -rf experiments/lstm/runs/

# Run again
python scripts/run_pretrain_sweep.py --parallel --num-gpus 4
```

## Expected Timeline (calc02 with 4x L4 GPUs)

**Per run estimates:**
- 20 epochs × ~2-3 min/epoch = ~40-60 min per run

**Parallel execution (4 GPUs):**
- 12 runs ÷ 4 GPUs = 3 batches
- Total time: ~3 × 60 min = **~3 hours** ⚡

**Sequential execution (1 GPU):**
- 12 runs × 60 min = **~12 hours** 🐌

## Next Steps

1. Run the sweep (parallel recommended)
2. Collect metrics
3. Analyze `pretrain_hparam_results.csv`
4. Pick best config for final model
5. Train best config for more epochs if needed
6. Use best encoder weights for downstream tasks
