Summary of TFT ablation provenance

This folder contains sweep summaries and provenance for the TFT ablation that informed thesis model selection.

- `sweep_summary.csv`: per-sweep-run hyperparameters and best validation loss (extracted from run `logs/metrics.csv`).
- `provenance_table.csv`: maps the ablation snapshot (training-run selection) to the sweep-run winners and the short-head RMSE used for final selection, plus canonical freeze RMSE for reference.

Notes:
- Ablation snapshot `best_val_loss` values are taken from `experiments/tft/runs/germany/ablations/ablation_summary_extended.csv`.
- Short-head RMSE entries come from `experiments/tft/notes/short_head_eval.csv` and reference the exact checkpoint paths used.
- Canonical freeze RMSEs (for `Short-TFT-Only` and `PVLib-Physics-Only`) are taken from `freeze/final_thesis_v1/benchmarks/thesis_formatted_v3/tables/overall_metrics.csv`.

Provenance gap:
- Some run metadata do not include an explicit git commit hash; include repository commit in thesis notes if required for reproducibility.
# TFT Ablation Artifacts

This folder contains LaTeX and metadata exports derived from the ablation experiments located at `experiments/tft/runs/germany/ablations/`.

Files included:

- `ablation_summary_extended.tex` — full extended ablation table converted from `experiments/tft/runs/germany/ablations/ablation_summary_extended.csv`.
- `ablation_summary.tex` — compact ablation table converted from `experiments/tft/runs/germany/ablations/ablation_summary.csv`.

Source CSVs:

- `experiments/tft/runs/germany/ablations/ablation_summary_extended.csv`
- `experiments/tft/runs/germany/ablations/ablation_summary.csv`

Usage:

- Include the generated tables in your LaTeX document with `\input{freeze/final_thesis_v1/tft_ablation/ablation_summary.tex}` and `\input{freeze/final_thesis_v1/tft_ablation/ablation_summary_extended.tex}`.
- The `.tex` files are simple `tabular` environments and can be edited if you want to change column formatting or captions.

Notes:

- These tables were generated directly from the CSV content in the repository; they intentionally preserve the values observed at the time of archiving. If you regenerate ablation CSVs, re-run this conversion step to update the `.tex` files.
- If you prefer a different LaTeX style (e.g., `tabularx` or `longtable`), I can produce alternate versions.
