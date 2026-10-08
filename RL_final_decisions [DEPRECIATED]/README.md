# RL Final Decisions

This folder collects the RL-related artifacts produced during the final evaluation and audit run.

Contents:
- `tables/` — CSV and LaTeX tables summarizing metrics and action selections.
- `figures/` — thesis-ready figures comparing MiRACLE Core, Heuristic RL (v1), and Full RL (v2).
- `scripts/` — audit script used to produce these artifacts (`audit_rl_modes.py`).

Provenance:
- All per-timestep RL predictions were synthesized from per-forecast summary weights and component per-timestep predictions (short/long/physics) and stored under `freeze_corrected/...`.
- The v1 summary contained an incorrect action→blend mapping; v2 was re-run with corrected mapping and shows improved performance in the summer case.

Key findings (summer & winter weeks):

Summer (2024-07-01 → 2024-07-08), N=673:

- MiRACLE Core — MAE 0.08540, RMSE 0.16559, MBE +0.05255, MeanAbsDiff 0.01149
- Heuristic RL (v1) — MAE 0.09596, RMSE 0.17702, MBE +0.07069, MeanAbsDiff 0.01118
- Full RL (v2) — MAE 0.08110, RMSE 0.14056, MBE +0.01856, MeanAbsDiff 0.00852

Winter (2024-01-10 → 2024-01-17), N=673:

- MiRACLE Core — MAE 0.02196, RMSE 0.05201, MBE −0.00320, MeanAbsDiff 0.00489
- Heuristic RL (v1) — MAE 0.02142, RMSE 0.05297, MBE −0.00847, MeanAbsDiff 0.00429
- Full RL (v2) — MAE 0.02419, RMSE 0.06263, MBE −0.01327, MeanAbsDiff 0.00377

Interpretation (scientific grounding):

- In higher-variance conditions (summer), the Full RL policy (v2) both smooths high-frequency jitter and improves point-wise accuracy (lower RMSE/MAE), indicating effective selective blending of component predictors that reduces variance while maintaining unbiasedness.
- In low-variance conditions (winter), the deterministic MiRACLE Core is already very close to ground truth; the RL policies behave as conservative controllers that emphasize stability (lower short-term variability) at the cost of peak alignment — in particular v2 shows extra smoothing and a slight negative bias, which can increase RMSE in these calm periods.

For reproducibility and inclusion in the thesis, see the LaTeX tables in `tables/` and the figures in `figures/`.

If you want, I can:
- Copy these corrected figures into the canonical `freeze/final_thesis_v1/benchmarks/thesis_formatted_v3/figures/` (with backups). 
- Produce a one-paragraph LaTeX snippet for Chapter 3/4 incorporating the numeric results above.
