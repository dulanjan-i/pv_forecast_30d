***REMOVED***!/bin/bash
***REMOVED***SBATCH --job-name=miracle_pretrain_2107
***REMOVED***SBATCH --output=logs/slurm/pretrain_2107_%j.out
***REMOVED***SBATCH --error=logs/slurm/pretrain_2107_%j.err

***REMOVED***SBATCH --partition=paula
***REMOVED***SBATCH --gres=gpu:1
***REMOVED***SBATCH --cpus-per-task=4
***REMOVED***SBATCH --mem=16G
***REMOVED***SBATCH --time=12:00:00

***REMOVED*** 1. Clean environment and load cluster Anaconda
module purge
module load Anaconda3/2024.02-1

***REMOVED*** 2. Enable conda and activate YOUR env
source /software/easybuild/el9/amd_zen3/all/Anaconda3/2024.02-1/etc/profile.d/conda.sh
export PYTHONNOUSERSITE=1
conda activate /work/vy17foce-miracle_pv/pvforecast_env

***REMOVED*** 3. Go to project directory
cd /work/vy17foce-miracle_pv/pv_forecast_30d

***REMOVED*** 4. Sanity check: print GPU status into the log
python - << 'EOF'
import torch
print("=== RUNTIME GPU CHECK ===")
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
print("current device:", torch.cuda.current_device() if torch.cuda.is_available() else None)
EOF

***REMOVED*** 5. Run the actual pretraining
python src/training/pretrain_lstm.py --config experiments/lstm/pretrain_farm2107.yaml***REMOVED***!/bin/bash
***REMOVED***SBATCH --job-name=miracle_pretrain_2107
***REMOVED***SBATCH --output=logs/slurm/pretrain_2107_%j.out
***REMOVED***SBATCH --error=logs/slurm/pretrain_2107_%j.err

***REMOVED***SBATCH --partition=paula
***REMOVED***SBATCH --gres=gpu:1
***REMOVED***SBATCH --cpus-per-task=4
***REMOVED***SBATCH --mem=16G
***REMOVED***SBATCH --time=12:00:00

module purge
module load Anaconda3/2024.02-1

cd /work/vy17foce-miracle_pv/pv_forecast_30d

conda run -p /work/vy17foce-miracle_pv/pvforecast_env \
  python src/training/pretrain_lstm.py \
    --config experiments/lstm/pretrain_farm2107.yaml
