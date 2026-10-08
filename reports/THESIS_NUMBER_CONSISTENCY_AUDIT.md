# Thesis Number Consistency Audit (auto)

**Goal**: Identify places where reported headline metrics in `docs/` and `reports/` disagree with the canonical thesis tables/figures generated under `freeze/final_thesis_v1/`.

## Canon rule (timestamp-based)

To resolve inconsistencies, treat **the latest-generated evaluation artifacts** (by filesystem modification time) as canonical for thesis headline numbers.

As of **2026-01-08** (workspace timestamps), the newest “headline” artifacts are:

- Multi-model benchmark suite summary: `freeze/final_thesis_v1/benchmarks/thesis_formatted_v3/text/results.md` (mtime: 2026-01-08 14:40 UTC)
- Multi-model benchmark suite table: `freeze/final_thesis_v1/benchmarks/thesis_formatted_v3/tables/overall_metrics.csv` (mtime: 2026-01-08 14:40 UTC)
- RQ4 baseline vs policy summary: `freeze/final_thesis_v1/eval/rq4_baseline_vs_policy/text/results.md` (mtime: 2026-01-08 15:45 UTC)

Re-run the timestamp audit anytime via: `scripts/audit_thesis_canonical_by_timestamp.sh`.

## Canonical sources (use these for thesis headline numbers)

1) **2024 full-pipeline inference benchmark suite (multi-model)**
- Summary: `freeze/final_thesis_v1/benchmarks/thesis_formatted_v3/text/results.md`
- Tables: `freeze/final_thesis_v1/benchmarks/thesis_formatted_v3/tables/overall_metrics.csv`
- Key configuration note in summary: **Night filtering ON** (`y_true >= 0.01`), $N=394{,}008$.

2) **RQ4 baseline vs policy evaluation**
- Summary: `freeze/final_thesis_v1/eval/rq4_baseline_vs_policy/text/results.md`
- Key configuration note: **Night filtering ON** (`y_true >= 0.01`).

## Findings (mismatch ledger)

### 1) Ground-truth verification doc reports different RMSE than canonical benchmark suite
- File: `docs/MiRACLE_V1.0_ground_truth_verification.md`
- Reported:
  - Overall RMSE (normalized): **0.13886**
  - “Daylight (06:00–18:00)” RMSE: **0.18828**
  - Sample count: **124,032** comparisons
- Canonical (benchmark suite):
  - MiRACLE v1.0 Core RMSE: **0.11713** (night-filtered, `y_true >= 0.01`, $N=394{,}008$)
- Status: **Mismatch**
- Likely reasons (needs confirmation):
  - Different evaluation protocol (time-based daylight window vs threshold-based night filtering)
  - Different dataset join / deduping logic (doc shows prediction timesteps > 1 year of 15-min points)
  - Potentially different ground-truth source/normalization pathway
- Recommendation:
  - Add a banner to the doc: “Not thesis headline; uses a different evaluation protocol than `thesis_formatted_v3`.”
  - If this is intended to validate the *same* predictions+truth, rerun it using the benchmark suite’s join + filtering rules and update the numbers.

### 2) Evidence Bible frames training-validation RMSE as deployed-system performance
- File: `reports/THESIS_EVIDENCE_BIBLE.md`
- Reported (as headline / deployed performance):
  - “achieves RMSE **0.087** (short-head) and **0.076** (long-head)”
- Canonical (deployment/inference backtest headline):
  - Full pipeline 2024 backtest (MiRACLE v1.0 Core) RMSE: **0.11713** (night-filtered, $N=394{,}008$)
- Status: **Needs reframing (not necessarily wrong numbers, but wrong level-of-claim)**
- Recommendation:
  - Rewrite wording to explicitly label 0.087/0.076 as **model-level training evaluation / test-on-windows**, not full end-to-end backtest.
  - Point readers to the canonical inference benchmark for final performance.

### 3) RL policy audit “0–24h win” contradicts canonical RQ4 evaluation outputs
- File: `reports/RL_POLICY_AUDIT_THESIS_DEFENSE.md`
- Reported:
  - “0.65% RMSE improvement in Day-1 trading window”
  - Baseline RMSE (Day 1): **0.127**, Policy RMSE (Day 1): **0.126**
- Canonical (RQ4): `freeze/final_thesis_v1/eval/rq4_baseline_vs_policy/text/results.md`
  - Lead bucket **0–24h**: baseline RMSE **0.118582**, policy RMSE **0.119484** (policy worse by ~0.000902)
  - Overall: baseline RMSE **0.11713**, policy RMSE **0.117161** (essentially equal)
- Status: **Mismatch**
- Recommendation:
  - Update the audit’s “Day-1 win” section to match the canonical results OR rerun policy evaluation if the current RQ4 outputs are not the intended final artifacts.
  - If you keep the “specialist” narrative, rebase it on whichever slice actually improves in the final evaluation (if any), and cite the specific table.

## Notes (not treated as mismatches)

- Files like `reports/VERIFICATION_SUMMARY_v1.md` reporting RMSE **0.04855** appear to refer to **training/validation ablation metrics**, not 2024 inference backtest metrics. They are fine as long as labeled unambiguously.
- Files like `reports/RL_IMPLEMENTATION_SUMMARY.md` contain integration-test placeholder RMSE values (e.g., **0.0500**) and should not be cited as empirical results.
