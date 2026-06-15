# Installation Guide for calc02 VM

## ✅ Installation Complete!

All packages have been successfully installed in `~/.venvs/pvforecast`.

## System Specs
- **GPUs**: 4x NVIDIA L4 (23GB VRAM each)
- **CUDA**: 12.6/12.8 (driver 570.169)
- **Python**: 3.12.3
- **Environment**: Virtual environment at `~/.venvs/pvforecast`

## Installed Packages (194 total)
Key packages:
- **PyTorch**: 2.5.1+cu124 (CUDA 12.4 build, compatible with system CUDA 12.6/12.8)
- **PyTorch Lightning**: 2.4.0
- **torchmetrics**: 1.8.2
- **NumPy**: 2.2.6
- **pandas**: 2.3.3
- **scikit-learn**: 1.7.2
- **xgboost**: 3.1.2
- **lightgbm**: 4.6.0
- **optuna**: 4.6.0
- **pvlib**: 0.13.1
- **shap**: 0.50.0
- **matplotlib**: 3.10.7
- **seaborn**: 0.13.2
- **plotly**: 6.5.0
- **JupyterLab**: 4.5.0
- **streamlit**: 1.51.0

See `requirements_calc02_frozen.txt` for complete list with exact versions.

## Quick Start

### 0. SSH and enter the repo
```bash
ssh <your-user>@calc02
cd ~/pv_forecast_30d
```

### 1. Activate environment
```bash
source ~/.venvs/pvforecast/bin/activate
```

### 2. Verify installation
```bash
python -c "import torch, pytorch_lightning as pl; print(f'PyTorch: {torch.__version__}'); print(f'Lightning: {pl.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPUs: {torch.cuda.device_count()}')"
```

Expected output:
```
PyTorch: 2.5.1+cu124
Lightning: 2.4.0
CUDA: True
GPUs: 4
```

### 3. Test training script
```bash
python src/training/pretrain_lstm.py --help
```

You should see the help message without errors.

## Running Training

### Single GPU training
```bash
python src/training/pretrain_lstm.py --config experiments/lstm/pretrain_farm2107.yaml
```

### Using specific GPU
```bash
CUDA_VISIBLE_DEVICES=0 python src/training/pretrain_lstm.py --config experiments/lstm/pretrain_farm2107.yaml
```

### Multi-GPU training (if script supports it)
```bash
# Use all 4 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/training/pretrain_lstm.py --config experiments/lstm/pretrain_farm2107.yaml
```

## SLURM Jobs
If using SLURM on calc02:
```bash
sbatch scripts/pretrain_farm2107.slurm
```

## Important Notes

### ✅ Installation successful!
All 194 packages installed without errors.

### pytorch-lightning version
- **Installed**: 2.4.0 (not 2.5.6)
- **Reason**: Version 2.5.6 was quarantined by institutional proxy
- **Status**: 2.4.0 works perfectly with PyTorch 2.5.1

### CUDA compatibility
- **System CUDA**: 12.6/12.8
- **PyTorch CUDA**: 12.4
- **Status**: ✅ OK - CUDA is backward compatible

### NumPy version
- Upgraded from 2.3.3 to 2.2.6 for compatibility with scipy/scikit-learn

## Troubleshooting

### If running out of GPU memory
- Check GPU usage: `nvidia-smi`
- Monitor specific GPU: `watch -n 1 nvidia-smi`
- Kill processes if needed

### If you need to reinstall
```bash
source ~/.venvs/pvforecast/bin/activate
pip install -r requirements_calc02.txt
```

### To recreate exact environment elsewhere
```bash
pip install -r requirements_calc02_frozen.txt
```
