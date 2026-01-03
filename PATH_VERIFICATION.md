# Path Verification & Reorganization (2026-01-03)

## ✅ COMPLETED REORGANIZATION

### Scripts Moved from scripts/ to src/
- `scripts/collect_rl_data.py` → `src/rl/collect_rl_data.py`
- `scripts/generate_rl_data_from_historical.py` → `src/rl/generate_historical_data.py`
- `scripts/generate_rl_data_simulated.py` → `src/rl/generate_simulated_data.py`
- `scripts/compute_rewards.py` → `src/rl/compute_rewards.py`
- `scripts/train_rl_offline.py` → `src/training/train_rl_offline.py`

### CANONICAL HARDCODED PATHS

All code now uses these **absolute paths**:

#### TFT Model Checkpoints
```
/home/dwijenayake/pv_forecast_30d/V1.0_FINAL_TFT/shorthead_seed42/checkpoints/best.ckpt
/home/dwijenayake/pv_forecast_30d/V1.0_FINAL_TFT/longhead_seed43/checkpoints/best.ckpt
```

#### Plant Metadata
```
/home/dwijenayake/pv_forecast_30d/V1.0_FINAL_TFT/plant_metadata/plant_03.json
```

#### Training Data
```
/home/dwijenayake/pv_forecast_30d/data/processed/plant_level/plant_03/15min_pca32/train.parquet
/home/dwijenayake/pv_forecast_30d/data/processed/plant_level/plant_03/hourly_longhead/train.parquet
```

#### Test/Validation Data
```
/home/dwijenayake/pv_forecast_30d/data/processed/plant_level/plant_03/15min_pca32/test.parquet
/home/dwijenayake/pv_forecast_30d/data/processed/plant_level/plant_03/15min_pca32/val.parquet
/home/dwijenayake/pv_forecast_30d/data/processed/plant_level/plant_03/hourly_longhead/val.parquet
```

#### RL Checkpoints & Logs
```
/home/dwijenayake/pv_forecast_30d/checkpoints/rl/
/home/dwijenayake/pv_forecast_30d/checkpoints/rl/logs/
```

#### RL Transitions Data
```
/home/dwijenayake/pv_forecast_30d/data/rl_transitions/
```

### Files Updated with Canonical Paths
1. ✅ `src/rl/collect_rl_data.py`
2. ✅ `src/rl/generate_historical_data.py`
3. ✅ `src/rl/generate_simulated_data.py`
4. ✅ `src/rl/rl_integrated_forecaster.py`
5. ✅ `src/training/train_rl_offline.py`
6. ✅ `tests/test_rl_integration.py`

### Verification Commands

Test imports:
```bash
python -c "from src.rl.generate_simulated_data import main; print('✓ OK')"
python -c "from src.training.train_rl_offline import OfflineTrainer; print('✓ OK')"
```

Generate RL training data:
```bash
python src/rl/generate_simulated_data.py --num-samples 20
```

Train DDQN:
```bash
python src/training/train_rl_offline.py \
  --data /home/dwijenayake/pv_forecast_30d/data/rl_transitions/historical_batch.parquet \
  --epochs 2000 \
  --batch-size 16 \
  --device cuda
```

### NO MORE RELATIVE PATHS!
All paths are now absolute and hardcoded. No more path jumbles or spaghetti code.

## ✅ VERIFICATION COMPLETE

### All Scripts Moved
```bash
ls src/rl/*.py | grep -E "collect|generate|compute"
```
Output:
- src/rl/collect_rl_data.py
- src/rl/compute_rewards.py
- src/rl/generate_historical_data.py
- src/rl/generate_simulated_data.py

### All Paths Hardcoded
```bash
grep -r "V1.0_FINAL_TFT" src/rl/*.py | wc -l
```
All references use absolute paths starting with `/home/dwijenayake/pv_forecast_30d/`

### Tests Pass
- ✅ Imports work correctly
- ✅ TFT checkpoints found
- ✅ Plant metadata accessible  
- ✅ Data paths valid
- ✅ RL system initializes
- ✅ Training pipeline loads data

### NO MORE PATH ISSUES!
All code now uses CANONICAL ABSOLUTE PATHS. No relative paths, no guessing, no spaghetti.
