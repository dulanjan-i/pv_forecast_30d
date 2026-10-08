# Thesis Color Scheme - Applied Across All Figures

## Summary
All evaluation figures have been regenerated with a **consistent, thesis-ready color scheme** emphasizing visual hierarchy:

### Color Mapping (Applied to ALL plots)

1. **Ground Truth**: Light grey `#888888`
   - Linewidth: 1.5
   - Alpha: 0.7
   - Purpose: Subtle reference line, not the focal point

2. **Model A / Baseline / MiRACLE v1.0 Core**: Bold green `#00AA00`
   - Linewidth: 2.5
   - Alpha: 1.0
   - Purpose: **HIGHLIGHTED** - stands out as the main result

3. **Model B / Comparison Models**: Light blue `#6BA3D8`
   - Linewidth: 1.5 (time series) or 1.0 (facets)
   - Alpha: 0.9
   - Purpose: De-emphasized comparison

4. **Multi-model monthly RMSE**: Separate colors per model
   - MiRACLE Core: Bold green `#00AA00` (linewidth 2.5)
   - Comparisons: Color palette [`#6BA3D8`, `#FAA43A`, `#B276B2`, `#F17CB0`, `#60BD68`] (linewidth 1.5)

### Resolution & Quality
- **DPI**: 300 (up from 200) for thesis-quality printing
- **Figure sizes**: Increased for better readability
- **Grid lines**: Subtle dotted grid (alpha=0.3)
- **Legends**: Improved with `framealpha=0.9`, better positioning

## Updated Files

### Scripts Updated
1. **src/evaluation/run_benchmark_suite_v3_formatted.py**
   - Facet grid comparisons (summer/winter case studies)
   - Error histograms (comparison model behind, MiRACLE in front)
   - Lead-time RMSE curves
   - Monthly RMSE with distinct colors per model

2. **src/evaluation/run_pair_eval.py**
   - All RQ1 and RQ2 evaluations
   - Monthly RMSE, error histograms, cumulative error, case studies

3. **src/evaluation/run_full_eval.py**
   - RQ4 baseline vs policy evaluation
   - Includes policy action distribution plot

### Evaluation Directories Regenerated
All plots regenerated with consistent colors at 300 DPI:

- **RQ1 (Ablation studies)**:
  - `freeze/final_thesis_v1/eval/rq1_warm_vs_pvlib/` - Core vs PVLib-Physics-Only
  - `freeze/final_thesis_v1/eval/rq1_warm_vs_tft/` - Core vs TFT-Only
  - `freeze/final_thesis_v1/eval/rq1_warm_vs_short/` - Core vs Short-TFT-Only
  - `freeze/final_thesis_v1/eval/rq1_warm_vs_long/` - Core vs Long-TFT-Only

- **RQ2 (Cold-start)**:
  - `freeze/final_thesis_v1/eval/rq2_warm_vs_cold/` - Warm vs Cold-start

- **RQ4 (Policy vs Baseline)**:
  - `freeze/final_thesis_v1/eval/rq4_baseline_vs_policy/` - Core vs Full (with RL policy)

- **Benchmark Suite**:
  - `freeze/final_thesis_v1/benchmarks/thesis_formatted_v3/` - Multi-model comparison

## Plot Types Covered

Each RQ directory contains the following plots (all with consistent colors):
1. **case_summer_week.png** - Time series for summer week
2. **case_winter_week.png** - Time series for winter week
3. **abs_error_hist.png** - Histogram of absolute errors
4. **cumulative_abs_error.png** - Cumulative error over time
5. **daily_mae_scatter.png** - Daily MAE scatter plot
6. **monthly_rmse.png** - Monthly RMSE comparison

RQ4 additionally includes:
7. **policy_action_distribution.png** - RL policy action frequencies

Benchmark suite includes:
1. **facets_case_summer_week.png** - Facet grid (all models)
2. **facets_case_winter_week.png** - Facet grid (winter)
3. **facets_abs_error_hist.png** - Error histograms (all models)
4. **facets_leadtime_rmse_curve_0_24h.png** - RMSE vs lead time
5. **monthly_rmse_all_models.png** - Monthly RMSE (all models, distinct colors)

## Visual Hierarchy Rationale

The color scheme was designed to:
1. **Highlight MiRACLE Core/Baseline** as the main contribution (bold green, thicker)
2. **De-emphasize comparisons** (light blue, thinner) - they support the story but aren't the focus
3. **Make ground truth subtle** (light grey) - it's a reference, not the result
4. **Maintain consistency** across all thesis figures for professional appearance
5. **Improve readability** with higher DPI and better styling

## Regeneration Commands

To regenerate all evaluations:
```bash
bash scripts/regenerate_all_rq_evals_thesis_colors.sh
```

To regenerate benchmark suite only:
```bash
bash scripts/run_benchmark_suite_v3_formatted.sh
```

## File Size Comparison
- **Old (200 DPI)**: ~150-250KB per plot
- **New (300 DPI)**: ~300-500KB per plot (2x larger, better quality)

## Next Steps
All figures are now ready for:
- ✅ Thesis document inclusion
- ✅ Defense presentation slides
- ✅ Publication-quality printing
- ✅ Consistent visual narrative across all research questions
