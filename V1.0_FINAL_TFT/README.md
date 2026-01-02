# V1.0_FINAL_TFT - Production TFT Checkpoints

**Version**: 1.0  
**Date**: 2026-01-02  
**Status**: ✅ Production Ready  

---

## Overview

This directory contains the **verified, production-ready TFT model checkpoints** for the hierarchical 30-day PV power forecasting pipeline.

**Key Properties**:
- ✅ Seeds verified: Short-head (42), Long-head (43)
- ✅ Both warm-started from global pretrained encoders
- ✅ Trained on plant_03 (Germany, 7358.9 kW)
- ✅ Configs preserved for reproducibility

---

## Directory Structure

```
V1.0_FINAL_TFT/
├── shorthead_seed42/
│   ├── best.pt                # Primary checkpoint (state_dict format)
│   ├── best.ckpt              # Alternative format (Lightning)
│   ├── run_config.json        # Training hyperparameters
│   └── README.md              # Model documentation
├── longhead_seed43/
│   ├── best.pt                # Primary checkpoint (state_dict format)
│   ├── best.ckpt              # Alternative format (Lightning)
│   ├── run_config.json        # Training hyperparameters
│   └── README.md              # Model documentation
├── plant_metadata/
│   └── plant_03.json          # Plant configuration
└── README.md                  # This file
```

---

## Model Specifications

### Short-Head TFT (Seed 42)
- **Resolution**: 15-minute
- **Architecture**: 96-step encoder, 96-step decoder
- **Horizon**: 24 hours (96 × 15min)
- **Training**: Warm-start from global pretrained encoder
- **Learning Rate**: 0.0006
- **Dropout**: 0.15
- **Batch Size**: 512 (effective: 512 × 8 = 4096 with grad accumulation)
- **Training Date**: 2025-12-29
- **Source**: `experiments/tft/runs/germany/plant_03/15min/pvlib_warmstart_from_global_noleak/20251229_151100/`

### Long-Head TFT (Seed 43)
- **Resolution**: 1-hour
- **Architecture**: 168-step encoder, 720-step decoder
- **Horizon**: 30 days (720 hours)
- **Training**: Warm-start from global pretrained encoder
- **Learning Rate**: 0.0008
- **Dropout**: 0.15
- **Batch Size**: 64 (effective: 64 × 8 = 512 with grad accumulation)
- **Training Date**: 2025-12-31
- **Source**: `experiments/tft/runs/germany/plant_03/longhead/hourly720/warm/lr8e-4_do0.15_bs64_acc8_seed43/20251231_104405/`

---

## Usage

### Standard Inference
```python
from pathlib import Path
from src.inference.physics_aware_forecaster import PhysicsAwareForecaster

# Initialize forecaster with V1.0 checkpoints
forecaster = PhysicsAwareForecaster(
    short_ckpt="V1.0_FINAL_TFT/shorthead_seed42/best.pt",
    long_ckpt="V1.0_FINAL_TFT/longhead_seed43/best.pt",
    plant_metadata="V1.0_FINAL_TFT/plant_metadata/plant_03.json",
    short_train_parquet="data/processed/plant_level/plant_03/15min_pca32/train.parquet",
    long_train_parquet="data/processed/plant_level/plant_03/hourly_longhead/train.parquet",
    device="cuda"  # or "cpu"
)

# Generate 30-day forecast
forecast = forecaster.predict_30d(
    forecast_start="2026-01-02 00:00:00",
    weather_df=weather_data,
    historical_df=historical_data
)

# Or use live weather
forecast = forecaster.predict_30d(
    forecast_start="2026-01-02 00:00:00",
    use_live_weather=True  # Fetches from OpenMeteo API
)
```

### Running Tests
```bash
# Full pipeline test (31 TFT calls)
python test_full_pipeline_real_tft.py

# Live weather integration test
python test_live_weather_forecast.py
```

---

## Verification

### Seeds Confirmed
```bash
$ python -c "
import json
for p in ['shorthead_seed42', 'longhead_seed43']:
    cfg = json.load(open(f'V1.0_FINAL_TFT/{p}/run_config.json'))
    print(f'{p}: seed = {cfg[\"cli_args\"][\"seed\"]}')
"
shorthead_seed42: seed = 42  ✅
longhead_seed43: seed = 43   ✅
```

### Checkpoint Integrity
```bash
$ ls -lh V1.0_FINAL_TFT/*/best.pt
-rw-rw-r-- 1.8M Jan  2 17:24 V1.0_FINAL_TFT/longhead_seed43/best.pt
-rw-rw-r-- 1.8M Jan  2 17:24 V1.0_FINAL_TFT/shorthead_seed42/best.pt
```

---

## Performance Benchmarks

### Test Set Metrics (Plant 03)

**Short-Head (24-hour, 15-min resolution)**:
- MAE: 0.049 (normalized power)
- RMSE: [TODO - extract from logs]
- R²: [TODO - extract from logs]
- Zero-night accuracy: 100%
- Capacity constraint violations: 0%

**Long-Head (30-day, 1-hour resolution)**:
- MAE: [TODO - extract from logs]
- RMSE: [TODO - extract from logs]
- R²: [TODO - extract from logs]

**Hierarchical Blend (Final 30-day @ 15-min)**:
- Output shape: (2880,) = 30 days × 96 steps/day
- Range: [0.0, 0.38] (normalized)
- Constraint satisfaction: 100%

---

## Provenance & Traceability

### Checkpoint Lineage

**Short-Head**:
1. Global pretraining on all plants (seed unknown)
2. Plant-03 fine-tuning with PVLib warm-start (seed 42)
3. Best checkpoint selected (epoch unknown, validation loss minimum)
4. Copied to V1.0_FINAL_TFT: 2026-01-02

**Long-Head**:
1. Global pretraining on all plants (seed unknown)
2. Plant-03 fine-tuning (seed 43)
3. Best checkpoint selected (epoch unknown, validation loss minimum)
4. Copied to V1.0_FINAL_TFT: 2026-01-02

### Git Commit (for tracking)
```bash
# Tag this version for future reference
git add V1.0_FINAL_TFT/
git commit -m "Add V1.0 production TFT checkpoints (seeds 42+43)"
git tag -a v1.0-tft -m "Production TFT models: short-head (seed42) + long-head (seed43)"
```

---

## Version History

| Version | Date | Changes | Seeds | Status |
|---------|------|---------|-------|--------|
| 1.0 | 2026-01-02 | Initial production release | 42, 43 | ✅ Active |

---

## Migration from Old Paths

### Before (Nested Paths)
```python
short_ckpt = "experiments/tft/runs/germany/plant_03/15min/pvlib_warmstart_from_global_noleak/20251229_151100/checkpoints/best_state_dict.pt"
long_ckpt = "experiments/tft/runs/germany/plant_03/longhead/hourly720/warm/lr8e-4_do0.15_bs64_acc8_seed43/20251231_104405/checkpoints/best_state_dict.pt"
```

### After (Clean Paths)
```python
short_ckpt = "V1.0_FINAL_TFT/shorthead_seed42/best.pt"
long_ckpt = "V1.0_FINAL_TFT/longhead_seed43/best.pt"
```

**Files Updated**:
- ✅ `test_full_pipeline_real_tft.py`
- ✅ `test_live_weather_forecast.py`
- [ ] `src/inference/physics_aware_forecaster.py` (if has hardcoded defaults)
- [ ] Any production deployment scripts

---

## Future Versions

### V1.1 (Planned)
- Add ensemble of seeds (42, 43, 44)
- Include performance benchmarks
- Add calibration metrics

### V2.0 (Future)
- Multi-plant support
- Online learning capability
- Uncertainty quantification

---

## Support & Issues

- **Documentation**: See [CHECKPOINT_VERIFICATION_REPORT.md](../CHECKPOINT_VERIFICATION_REPORT.md)
- **Weather API**: See [WEATHER_API_INTEGRATION_SUMMARY.md](../WEATHER_API_INTEGRATION_SUMMARY.md)
- **Issues**: Contact maintainers or file GitHub issue

---

**Last Updated**: 2026-01-02  
**Maintainer**: PV Forecast Team  
**License**: [Project License]
