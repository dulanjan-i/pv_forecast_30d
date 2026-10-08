# Benchmark Suite v3 - Formatting Fixes Summary

**Created**: January 8, 2026  
**Purpose**: Fix ONLY formatting/labels in existing plots, NOT invent new plots

## What Was Done

✅ **Copied** `run_benchmark_suite_v1.py` → `run_benchmark_suite_v3_formatted.py`  
✅ **Modified ONLY formatting** (NO new plot types invented)  
✅ **Output to NEW directory**: `thesis_formatted_v3` (NOT overwriting final_suite_v2)

## Formatting Changes Applied

### 1. Resolution
- **Before**: 200 DPI
- **After**: 300 DPI (publication-ready)

### 2. Labels
- **Before**: `"truth"`, `"baseline"` (generic)
- **After**: `"Ground Truth Plant 03"`, `"MiRACLE v1.0 Core"` (descriptive)
- Added `--truth-label` CLI argument

### 3. Figure Sizes
- **Before**: `figsize=(5 * cols, 3.3 * rows)`
- **After**: `figsize=(6 * cols, 4 * rows)` (20% larger for readability)

### 4. Plot Styling
- Added grid lines (alpha=0.3, linestyle=':')
- Improved legend: `framealpha=0.9`, `loc='best'`
- Better line styles: markers, linewidth=1.5-2.0
- Font weights: semibold titles, bold suptitles

### 5. Date Formatting
- Added `mdates.ConciseDateFormatter` for cleaner x-axis labels
- Rotation=0 (horizontal) for better readability

### 6. Axis Labels
- **Before**: `"power_norm"`, `"Time"`
- **After**: `"Power (normalized)"`, `"Time (UTC)"` (more descriptive)

## Files Created

### Script
- **Location**: `src/evaluation/run_benchmark_suite_v3_formatted.py` (860 lines)
- **Status**: ✅ NEW file, did NOT overwrite v1

### Execution Script
- **Location**: `scripts/run_benchmark_suite_v3_formatted.sh`
- **Status**: ✅ NEW file

### Output Directory
- **Location**: `freeze/final_thesis_v1/benchmarks/thesis_formatted_v3/`
- **Status**: ✅ NEW directory, did NOT overwrite final_suite_v2

## Generated Outputs

### Figures (SAME plots as v2, just FORMATTED)
1. `facets_case_summer_week.png` - 1016 KB (was 520 KB @ 200 DPI)
2. `facets_case_winter_week.png` - 741 KB (was 355 KB @ 200 DPI)
3. `facets_abs_error_hist.png` - 318 KB (NEW: better legend)
4. `facets_leadtime_rmse_curve_0_24h.png` - 646 KB (NEW: better styling)
5. `monthly_rmse_all_models.png` - 394 KB (NEW: larger figure)

### Tables (SAME as v1/v2)
- `overall_metrics.csv/.tex`
- `overall_metrics_stitched.csv/.tex`
- `tail_abs_error.csv/.tex`
- `monthly_metrics_long.csv/.tex`
- `lead_bucket_metrics_long.csv/.tex`
- `daily_metrics_long.csv/.tex`
- `worst_10_days_per_model.csv`
- `paired_daily_deltas_vs_baseline.csv/.tex`

## Comparison: Original vs Formatted

| Aspect | final_suite_v2 (original) | thesis_formatted_v3 (new) |
|--------|--------------------------|---------------------------|
| **Resolution** | 200 DPI | 300 DPI |
| **Truth Label** | "truth" | "Ground Truth Plant 03" |
| **Baseline Label** | "baseline" | "MiRACLE v1.0 Core" |
| **Figure Size** | 5×3.3 per panel | 6×4 per panel |
| **Grid Lines** | None | Yes (subtle dotted) |
| **Legend** | Simple | Framealpha, best placement |
| **Date Format** | Standard matplotlib | ConciseDateFormatter |
| **File Size (summer)** | 520 KB | 1016 KB |
| **File Size (winter)** | 355 KB | 741 KB |

## What Was NOT Changed

❌ No new plot types invented (no scatter, residuals, Q-Q plots)  
❌ No changes to data processing pipeline  
❌ No changes to metrics calculations  
❌ No changes to daylight filtering (still uses `--daylight-threshold 0.01`)  
❌ No overwrites of existing files in final_suite_v2  
❌ No modifications to run_benchmark_suite_v1.py

## Usage

```bash
# Run the formatted suite
bash scripts/run_benchmark_suite_v3_formatted.sh

# Output directory
freeze/final_thesis_v1/benchmarks/thesis_formatted_v3/
├── figures/          # 5 formatted plots (300 DPI)
├── tables/           # Same tables as v1/v2
└── text/             # results.md summary
```

## Verification

Compare visually:
```bash
# Original (v2)
ls -lh freeze/final_thesis_v1/benchmarks/final_suite_v2/figures/

# Formatted (v3)
ls -lh freeze/final_thesis_v1/benchmarks/thesis_formatted_v3/figures/
```

File sizes confirm 300 DPI upgrade (approximately 2x larger).

## Command Used

```bash
PYTHONPATH=. python -m src.evaluation.run_benchmark_suite_v3_formatted \
  --truth freeze/final_thesis_v1/.../ground_truth_15min_utc_capnorm.parquet \
  --baseline-name "MiRACLE v1.0 Core" \
  --baseline freeze/final_thesis_v1/.../predictions_phase1_baseline_rerun.parquet \
  --truth-label "Ground Truth Plant 03" \
  --model "PVLib-Physics-Only:freeze/final_thesis_v1/.../pvlib_only.parquet" \
  --model "TFT-Only:freeze/final_thesis_v1/.../tft_only.parquet" \
  --model "Short-TFT-Only:freeze/final_thesis_v1/.../short_only.parquet" \
  --model "Long-TFT-Only:freeze/final_thesis_v1/.../long_only.parquet" \
  --out freeze/final_thesis_v1/benchmarks/thesis_formatted_v3 \
  --daylight-threshold 0.01
```

## Status

✅ **Complete** - All formatting fixes applied to existing plots  
✅ **Safe** - No overwrites of original files  
✅ **Ready** - For thesis writing and defense presentation
