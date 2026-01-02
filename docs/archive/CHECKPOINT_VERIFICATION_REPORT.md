# Checkpoint & Configuration Verification Report

**Date**: 2026-01-02  
**Status**: ⚠️ **ISSUES FOUND** - Need Corrections

---

## Issue Summary

| # | Issue | Current State | Required State | Action |
|---|-------|---------------|----------------|--------|
| 1 | ✅ Plant Metadata | plant_03 (correct) | plant_03 | No action |
| 2 | ❌ Short-head Seed | Unknown (timestamp dir) | warm seed 42 | Fix path |
| 3 | ✅ Long-head Seed | warm seed 43 | warm seed 43 | Already correct |
| 4 | 📋 Organization | Scattered in experiments/ | V1.0_FINAL_TFT/ | Reorganize |

---

## Detailed Findings

### 1. Plant Metadata ✅ CORRECT

**Weather Client (`src/inference/weather_client.py`)**:
- Hardcoded test: `latitude=48.694644, longitude=12.597587` ✅
- Uses `plant_03.json` in test code ✅
- In production: accepts plant metadata path as parameter ✅

**plant_03.json content**:
```json
{
  "plant_id": "plant_03",
  "latitude": 48.694644,
  "longitude": 12.597587,
  "tilt_deg": 25.0,
  "azimuth_deg": 180.0,
  "installed_capacity_kw": 7358.9,
  ...
}
```

**Verdict**: ✅ **CORRECT** - All weather API calls use plant_03 coordinates and metadata.

---

### 2. Short-Head Checkpoint ❌ WRONG SEED

**Currently Used**:
```
experiments/tft/runs/germany/plant_03/15min/pvlib_warmstart_from_global_noleak/
└── 20251229_151100/checkpoints/best_state_dict.pt
```

**Problem**: This is a **timestamp-based directory** with no seed identifier. Cannot verify if it's seed 42.

**Available Seed-Specific Directories**:
```
experiments/tft/runs/germany/plant_03/15min/pvlib_warmstart_from_global_noleak/
├── 20251229_151100/      ← Currently used (seed unknown)
├── seed_42/
│   └── 20251229_154617/checkpoints/best.ckpt  ← Correct seed!
├── seed_43/
│   └── 20251229_154617/checkpoints/best.ckpt
└── seed_44/
    └── 20251229_154617/checkpoints/best.ckpt
```

**Required Path**:
```
experiments/tft/runs/germany/plant_03/15min/pvlib_warmstart_from_global_noleak/
└── seed_42/20251229_154617/checkpoints/best.ckpt
```

**File Info**:
- Size: 1.7 MB
- Date: Dec 29 16:02
- Format: `.ckpt` (Lightning format)

**Verdict**: ❌ **WRONG** - Need to switch to `seed_42/20251229_154617/checkpoints/best.ckpt`

---

### 3. Long-Head Checkpoint ✅ CORRECT SEED

**Currently Used**:
```
experiments/tft/runs/germany/plant_03/longhead/hourly720/warm/
└── lr8e-4_do0.15_bs64_acc8_seed43/20251231_104405/checkpoints/best_state_dict.pt
```

**Verified Seed**: `seed43` ✅ (in directory name)

**Available Seeds in warm/**:
```
warm/
├── lr8e-4_do0.15_bs64_acc8_seed42/  ← Alternative
├── lr8e-4_do0.15_bs64_acc8_seed43/  ← Currently used ✅
└── lr8e-4_do0.15_bs64_acc8_seed44/
```

**File Info**:
- Size: 1.8 MB
- Date: Dec 31 14:26
- Format: `.pt` (state_dict format)

**Verdict**: ✅ **CORRECT** - Already using seed 43 as required.

---

### 4. Checkpoint Organization 📋 RECOMMENDATION

**Current State**: Checkpoints scattered across:
```
experiments/tft/runs/germany/plant_03/
├── 15min/pvlib_warmstart_from_global_noleak/seed_42/20251229_154617/...
└── longhead/hourly720/warm/lr8e-4_do0.15_bs64_acc8_seed43/20251231_104405/...
```

**Problems**:
1. Long paths (error-prone)
2. Mixed with ablation experiments
3. No clear "production" designation
4. Multiple seeds/configs nearby (confusion risk)

**Proposed Structure**:
```
V1.0_FINAL_TFT/
├── shorthead_seed42/
│   ├── best.ckpt                   # Canonical checkpoint
│   ├── config.yaml                 # Model hyperparams
│   ├── train_stats.json            # Normalization stats
│   └── README.md                   # Provenance info
├── longhead_seed43/
│   ├── best_state_dict.pt
│   ├── config.yaml
│   ├── train_stats.json
│   └── README.md
└── plant_metadata/
    └── plant_03.json               # Symlink or copy
```

**Benefits**:
- ✅ Clear production designation
- ✅ Short, memorable paths
- ✅ Isolated from experiments
- ✅ Easy version control (V1.0 → V1.1 → ...)
- ✅ Self-documenting structure

---

## Required Actions

### Action 1: Fix Short-Head Checkpoint Path (HIGH PRIORITY)

**Files to Update**:
1. `test_full_pipeline_real_tft.py`
2. `test_live_weather_forecast.py`
3. Any inference scripts using short-head

**Change**:
```python
# OLD (WRONG - seed unknown)
short_ckpt = Path("experiments/tft/runs/germany/plant_03/15min/pvlib_warmstart_from_global_noleak/20251229_151100/checkpoints/best_state_dict.pt")

# NEW (CORRECT - seed 42)
short_ckpt = Path("experiments/tft/runs/germany/plant_03/15min/pvlib_warmstart_from_global_noleak/seed_42/20251229_154617/checkpoints/best.ckpt")
```

**Note**: File format changes from `.pt` → `.ckpt` (both are valid Lightning checkpoints)

---

### Action 2: Create V1.0_FINAL_TFT Structure (RECOMMENDED)

**Step 1: Create Directory**
```bash
mkdir -p V1.0_FINAL_TFT/{shorthead_seed42,longhead_seed43,plant_metadata}
```

**Step 2: Copy Checkpoints**
```bash
# Short-head (seed 42)
cp experiments/tft/runs/germany/plant_03/15min/pvlib_warmstart_from_global_noleak/seed_42/20251229_154617/checkpoints/best.ckpt \
   V1.0_FINAL_TFT/shorthead_seed42/best.ckpt

# Long-head (seed 43)
cp experiments/tft/runs/germany/plant_03/longhead/hourly720/warm/lr8e-4_do0.15_bs64_acc8_seed43/20251231_104405/checkpoints/best_state_dict.pt \
   V1.0_FINAL_TFT/longhead_seed43/best_state_dict.pt

# Plant metadata
cp data/metadata/germany/plant_03.json \
   V1.0_FINAL_TFT/plant_metadata/plant_03.json
```

**Step 3: Create Provenance Files**
```bash
# V1.0_FINAL_TFT/shorthead_seed42/README.md
cat > V1.0_FINAL_TFT/shorthead_seed42/README.md << 'EOF'
# Short-Head TFT Checkpoint (Seed 42)

**Model**: Temporal Fusion Transformer  
**Architecture**: 96-step encoder, 96-step decoder (15-min resolution)  
**Training**: Warm-start from global pretrained encoder  
**Seed**: 42  
**Training Date**: 2025-12-29  
**Best Checkpoint**: epoch unknown, step unknown  

## Provenance
- Source: experiments/tft/runs/germany/plant_03/15min/pvlib_warmstart_from_global_noleak/seed_42/20251229_154617/
- Plant: plant_03 (Germany, 7358.9 kW)
- Training Data: data/processed/plant_level/plant_03/15min_pca32/train.parquet
- Config: pvlib_warmstart_from_global_noleak

## Performance
- Test MAE: [TODO - add from logs]
- Test RMSE: [TODO - add from logs]
- Test R²: [TODO - add from logs]

## Usage
```python
from src.inference.physics_aware_forecaster import PhysicsAwareForecaster

forecaster = PhysicsAwareForecaster(
    short_ckpt="V1.0_FINAL_TFT/shorthead_seed42/best.ckpt",
    long_ckpt="V1.0_FINAL_TFT/longhead_seed43/best_state_dict.pt",
    plant_metadata="V1.0_FINAL_TFT/plant_metadata/plant_03.json",
    ...
)
```
EOF
```

**Step 4: Update All Inference Paths**
```python
# Standard paths (after reorganization)
SHORT_CKPT = Path("V1.0_FINAL_TFT/shorthead_seed42/best.ckpt")
LONG_CKPT = Path("V1.0_FINAL_TFT/longhead_seed43/best_state_dict.pt")
PLANT_META = Path("V1.0_FINAL_TFT/plant_metadata/plant_03.json")
```

---

## Verification Checklist

Before deploying:
- [ ] Verify short-head uses seed 42 checkpoint
- [ ] Verify long-head uses seed 43 checkpoint
- [ ] Test both checkpoints load correctly
- [ ] Run full pipeline test (31 TFT calls)
- [ ] Validate forecast output matches previous results
- [ ] Create V1.0_FINAL_TFT structure
- [ ] Update all paths in inference code
- [ ] Document performance metrics
- [ ] Add provenance tracking
- [ ] Create backup of current working state

---

## Comparison: Current vs Proposed

### Current (BEFORE)
```
test_full_pipeline_real_tft.py:
short_ckpt = "experiments/tft/.../pvlib_warmstart.../20251229_151100/checkpoints/best_state_dict.pt"  # ❌ seed unknown
long_ckpt = "experiments/tft/.../warm/lr8e-4_do0.15_bs64_acc8_seed43/.../best_state_dict.pt"  # ✅ seed 43

test_live_weather_forecast.py:
short_ckpt = (same as above)  # ❌ seed unknown
long_ckpt = (same as above)   # ✅ seed 43
```

### Proposed (AFTER)
```
test_full_pipeline_real_tft.py:
short_ckpt = "V1.0_FINAL_TFT/shorthead_seed42/best.ckpt"  # ✅ seed 42
long_ckpt = "V1.0_FINAL_TFT/longhead_seed43/best_state_dict.pt"  # ✅ seed 43

test_live_weather_forecast.py:
short_ckpt = (same as above)  # ✅ seed 42
long_ckpt = (same as above)   # ✅ seed 43
```

---

## Recommendation: PROCEED WITH BOTH FIXES

**My Take on the 3 Issues**:

1. **Plant Metadata** ✅: Already correct, no action needed.

2. **Checkpoint Seeds** ⚠️:
   - Long-head: ✅ Already correct (seed 43)
   - Short-head: ❌ Must fix to seed 42
   - **Action**: Update paths to use `seed_42/20251229_154617/checkpoints/best.ckpt`

3. **V1.0_FINAL_TFT Organization** 👍:
   - **Strongly agree** with this idea
   - Benefits: clarity, version control, production readiness
   - Cost: ~5 minutes setup, ~2 minutes to update paths
   - **Recommendation**: Do it now to avoid confusion later

**Proposed Workflow**:
1. Create V1.0_FINAL_TFT structure (5 min)
2. Copy correct checkpoints (seed 42 + seed 43) (2 min)
3. Update all test/inference paths (3 min)
4. Run verification test (2 min)
5. Document in README (2 min)

**Total Time**: ~15 minutes  
**Benefit**: Clean, production-ready structure with correct seeds ✅

---

## Next Steps

**Option A (Quick Fix)**:
- Just update short_ckpt path to seed_42
- Keep long paths as-is
- Time: 2 minutes

**Option B (Recommended - Full Refactor)**:
- Create V1.0_FINAL_TFT structure
- Update all paths
- Add documentation
- Time: 15 minutes

**My Recommendation**: **Option B** - The upfront investment pays off in:
- Reduced confusion (no more "which seed is this?")
- Easier version upgrades (V1.0 → V1.1 → V2.0)
- Cleaner codebase
- Production-ready structure

Let me know if you want me to proceed with Option B (full refactor)!
