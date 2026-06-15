# calc02 VM Setup - Quick Reference

## 📁 Files for calc02

1. **`requirements_calc02.txt`** - Human-readable requirements (what to install)
2. **`requirements_calc02_frozen.txt`** - Exact versions (194 packages, frozen snapshot)
3. **`INSTALL_CALC02.md`** - Complete installation guide and usage instructions

## ✅ Status: Installation Complete!

Your `~/.venvs/pvforecast` environment is ready with:
- PyTorch 2.5.1+cu124
- PyTorch Lightning 2.4.0
- All scientific computing packages (pandas, numpy, scipy, sklearn)
- ML packages (xgboost, lightgbm, optuna)
- PV-specific (pvlib)
- Visualization (matplotlib, seaborn, plotly, streamlit)
- JupyterLab 4.5.0

## 🚀 Quick Start

```bash
# SSH into calc02 and move into the repository first
ssh <your-user>@calc02
cd ~/pv_forecast_30d

# Activate environment
source ~/.venvs/pvforecast/bin/activate

# Verify GPU access
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"

# Verify pretraining CLI wiring (safe smoke check)
python src/training/pretrain_lstm.py --help

# Run training
python src/training/pretrain_lstm.py --config experiments/lstm/pretrain_farm2107.yaml
```

## 🖥️ System Info

- **Machine**: calc02 VM
- **GPUs**: 4x NVIDIA L4 (23GB VRAM each)
- **CUDA**: 12.6/12.8 (driver 570.169)
- **Python**: 3.12.3
- **No conda**: Using pip + venv only

## 📝 Important Notes

1. **pytorch-lightning 2.4.0** installed (not 2.5.6 - quarantined by proxy)
2. **CUDA 12.4** PyTorch build works with CUDA 12.6/12.8 system ✅
3. All packages tested and working

See `INSTALL_CALC02.md` for detailed documentation.
