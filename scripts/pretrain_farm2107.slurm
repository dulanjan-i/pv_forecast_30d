#!/bin/bash
#SBATCH --job-name=miracle_pretrain_2107
#SBATCH --output=logs/slurm/pretrain_2107_%j.out
#SBATCH --error=logs/slurm/pretrain_2107_%j.err

#SBATCH --partition=paula
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00

# 1. Clean environment and load cluster Anaconda
module purge
module load Anaconda3/2024.02-1

# 2. Enable conda and activate YOUR env
source /software/easybuild/el9/amd_zen3/all/Anaconda3/2024.02-1/etc/profile.d/conda.sh
export PYTHONNOUSERSITE=1
conda activate /work/vy17foce-miracle_pv/pvforecast_env

# 3. Go to project directory
cd /work/vy17foce-miracle_pv/pv_forecast_30d

# 4. Sanity check: print GPU status into the log
python - << 'EOF'
import torch
print("=== RUNTIME GPU CHECK ===")
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
print("current device:", torch.cuda.current_device() if torch.cuda.is_available() else None)
EOF

# 5. Run the actual pretraining
python src/training/pretrain_lstm.py --config experiments/lstm/pretrain_farm2107.yaml
