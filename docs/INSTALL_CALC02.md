***REMOVED*** Installation Guide for calc02 VM

***REMOVED******REMOVED*** ✅ Installation Complete!

All packages have been successfully installed in `~/.venvs/pvforecast`.

***REMOVED******REMOVED*** System Specs
- **GPUs**: 4x NVIDIA L4 (23GB VRAM each)
- **CUDA**: 12.6/12.8 (driver 570.169)
- **Python**: 3.12.3
- **Environment**: Virtual environment at `~/.venvs/pvforecast`

***REMOVED******REMOVED*** Installed Packages (194 total)
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

***REMOVED******REMOVED*** Quick Start

***REMOVED******REMOVED******REMOVED*** 1. Activate environment
```bash
source ~/.venvs/pvforecast/bin/activate
```

***REMOVED******REMOVED******REMOVED*** 2. Verify installation
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

***REMOVED******REMOVED******REMOVED*** 3. Test training script
```bash
python src/training/pretrain_lstm.py --help
```

You should see the help message without errors.

***REMOVED******REMOVED*** Running Training

***REMOVED******REMOVED******REMOVED*** Single GPU training
```bash
python src/training/pretrain_lstm.py --config experiments/lstm/pretrain_farm2107.yaml
```

***REMOVED******REMOVED******REMOVED*** Using specific GPU
```bash
CUDA_VISIBLE_DEVICES=0 python src/training/pretrain_lstm.py --config experiments/lstm/pretrain_farm2107.yaml
```

***REMOVED******REMOVED******REMOVED*** Multi-GPU training (if script supports it)
```bash
***REMOVED*** Use all 4 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/training/pretrain_lstm.py --config experiments/lstm/pretrain_farm2107.yaml
```

***REMOVED******REMOVED*** SLURM Jobs
If using SLURM on calc02:
```bash
sbatch scripts/pretrain_farm2107.slurm
```

***REMOVED******REMOVED*** Important Notes

***REMOVED******REMOVED******REMOVED*** ✅ Installation successful!
All 194 packages installed without errors.

***REMOVED******REMOVED******REMOVED*** pytorch-lightning version
- **Installed**: 2.4.0 (not 2.5.6)
- **Reason**: Version 2.5.6 was quarantined by institutional proxy
- **Status**: 2.4.0 works perfectly with PyTorch 2.5.1

***REMOVED******REMOVED******REMOVED*** CUDA compatibility
- **System CUDA**: 12.6/12.8
- **PyTorch CUDA**: 12.4
- **Status**: ✅ OK - CUDA is backward compatible

***REMOVED******REMOVED******REMOVED*** NumPy version
- Upgraded from 2.3.3 to 2.2.6 for compatibility with scipy/scikit-learn

***REMOVED******REMOVED*** Troubleshooting

***REMOVED******REMOVED******REMOVED*** If running out of GPU memory
- Check GPU usage: `nvidia-smi`
- Monitor specific GPU: `watch -n 1 nvidia-smi`
- Kill processes if needed

***REMOVED******REMOVED******REMOVED*** If you need to reinstall
```bash
source ~/.venvs/pvforecast/bin/activate
pip install -r requirements_calc02.txt
```

***REMOVED******REMOVED******REMOVED*** To recreate exact environment elsewhere
```bash
pip install -r requirements_calc02_frozen.txt
```
