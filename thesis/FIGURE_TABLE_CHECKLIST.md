# Thesis Figures & Tables Checklist (MiRACLE)

This file maps the “must-have” checklist to **existing canonical artifacts** in the repo (prefer `freeze/final_thesis_v1/` for thesis headline results).

## Figures

### Architecture

- MiRACLE system architecture diagram (high-level)
  - Figure: [figures/architecture/miracle_high_level.png](figures/architecture/miracle_high_level.png)
  - Vector/PDF: [figures/architecture/miracle_high_level.pdf](figures/architecture/miracle_high_level.pdf)
  - Source (PlantUML): [architecture diagrams/miracle_high_level.puml](../architecture%20diagrams/miracle_high_level.puml)
- Stage-by-stage pipeline flow (under the hood)
  - Figure: [figures/architecture/miracle_full_data_pipeline.png](figures/architecture/miracle_full_data_pipeline.png)
  - Vector/PDF: [figures/architecture/miracle_full_data_pipeline.pdf](figures/architecture/miracle_full_data_pipeline.pdf)
  - Source (PlantUML): [architecture diagrams/miracle_full_data_pipeline.puml](../architecture%20diagrams/miracle_full_data_pipeline.puml)

### Benchmarks / Results (canonical)

- Error vs. forecast horizon plot
  - `freeze/final_thesis_v1/benchmarks/thesis_formatted_v3/figures/facets_leadtime_rmse_curve_0_24h.png`
- Sample 30-day forecast visualizations
  - `freeze/final_thesis_v1/benchmarks/thesis_formatted_v3/figures/facets_case_summer_week.png`
  - `freeze/final_thesis_v1/benchmarks/thesis_formatted_v3/figures/facets_case_winter_week.png`
- Monthly RMSE plot (all models)
  - `freeze/final_thesis_v1/benchmarks/thesis_formatted_v3/figures/monthly_rmse_all_models.png`

### RL

- Baseline vs policy evaluation results
  - `freeze/final_thesis_v1/eval/rq4_baseline_vs_policy/text/results.md`

## Tables

### Canonical benchmark tables (thesis headline)

- Overall metrics
  - CSV: `freeze/final_thesis_v1/benchmarks/thesis_formatted_v3/tables/overall_metrics.csv`
  - LaTeX: `freeze/final_thesis_v1/benchmarks/thesis_formatted_v3/tables/overall_metrics.tex`
- Lead bucket metrics
  - CSV: `freeze/final_thesis_v1/benchmarks/thesis_formatted_v3/tables/lead_bucket_metrics_long.csv`
  - LaTeX: `freeze/final_thesis_v1/benchmarks/thesis_formatted_v3/tables/lead_bucket_metrics_long.tex`
- Monthly metrics
  - CSV: `freeze/final_thesis_v1/benchmarks/thesis_formatted_v3/tables/monthly_metrics_long.csv`
  - LaTeX: `freeze/final_thesis_v1/benchmarks/thesis_formatted_v3/tables/monthly_metrics_long.tex`
- Paired daily deltas vs baseline
  - CSV: `freeze/final_thesis_v1/benchmarks/thesis_formatted_v3/tables/paired_daily_deltas_vs_baseline.csv`
  - LaTeX: `freeze/final_thesis_v1/benchmarks/thesis_formatted_v3/tables/paired_daily_deltas_vs_baseline.tex`

### RQ evaluation tables

- RQ1 warm vs baselines
  - `freeze/final_thesis_v1/eval/rq1_warm_vs_tft/`
  - `freeze/final_thesis_v1/eval/rq1_warm_vs_short/`
  - `freeze/final_thesis_v1/eval/rq1_warm_vs_long/`
  - `freeze/final_thesis_v1/eval/rq1_warm_vs_pvlib/`
- RQ2 warm vs cold
  - `freeze/final_thesis_v1/eval/rq2_warm_vs_cold/`
- RQ4 baseline vs policy
  - `freeze/final_thesis_v1/eval/rq4_baseline_vs_policy/`

## Chapter drafts

- Chapter 3: `thesis/chapters/CH03_Methodology_MiRACLE.md`
- Chapter 4: `thesis/chapters/CH04_Experimental_Design_Ablations.md`
- Chapter 5: `thesis/chapters/CH05_Results_Performance_Analysis.md`
- Chapter 6: `thesis/chapters/CH06_Discussion.md`
- Chapter 7: `thesis/chapters/CH07_Conclusions_Future_Work.md`
