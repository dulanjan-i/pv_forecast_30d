# RL Modes Audit: Heuristic RL (v1) vs Full RL (v2)

## Summary
This audit compares two synthesized RL prediction modes:
- **Heuristic RL (v1)**: synthesized from the v1 summary weights (earlier run).
- **Full RL (v2)**: synthesized from the v2 summary weights (re-run with corrected action→blend mapping).

## What was tested
- Both modes were evaluated on the same processed backtest dataset using the thesis-ready evaluation pipeline.
- Predictions were synthesized by applying per-forecast weights onto component per-timestep predictions (short/long/pvlib).

## Metrics (stitched, most-recent selection)
- Baseline RMSE: 0.124802
- Heuristic RL (v1) RMSE: 0.134679
- Full RL (v2) RMSE: 0.123669

## Per-forecast win rates
- Heuristic RL (v1) win rate vs baseline: 4.17%
- Full RL (v2) win rate vs baseline: 41.32%

## Action selection differences
- Action selection table saved as `tables/action_selection_v1_vs_v2.csv`

## Observations
- The Full RL (v2) policy improves RMSE by 0.91% relative to the baseline.

## Provenance & Notes
- Per-timestep predictions for RL modes were synthesized from per-forecast summary weights and the component predictions (short, long, physics).
- v1 contained an incorrect action→blend mapping; v2 was re-run with corrected mapping. This explains the performance gap between v1 and v2.
