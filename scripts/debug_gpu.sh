#!/bin/bash
#SBATCH --partition=gpuh100
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=debug_gpu.out

# DBFZ-Strict Flags
SING_FLAGS="-C --nv --bind /shared/$USER:/shared/$USER,/home/$USER:/home/$USER"
IMG="/shared/$USER/miracle/containers/tft_env_v1.sif"

echo "=== PYTHON CHECK ==="
singularity exec $SING_FLAGS "$IMG" python3 -c "
import torch
print(f'PyTorch Version: {torch.__version__}')
print(f'CUDA Available:  {torch.cuda.is_available()}')
print(f'Device Count:    {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'Current Device:  {torch.cuda.get_device_name(0)}')
else:
    print('ERROR: PyTorch cannot see the GPU!')
"