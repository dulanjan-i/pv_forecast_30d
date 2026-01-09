# Chapter 6 — Discussion

## 6.1 Key findings summary (mapped to RQs)

### RQ1 — Hybrid physics + deep learning improves PV forecasting

Evidence:

- MiRACLE v1.0 Core vs TFT-only and PVLib-only in the canonical 2024 backtest:
  - benchmark suite: `freeze/final_thesis_v1/benchmarks/thesis_formatted_v3/`
  - RQ1 eval summaries: `freeze/final_thesis_v1/eval/rq1_warm_vs_*`

Interpretation:

- Physics baselines alone provide a meaningful prior but are insufficient for high accuracy.
- The combination of learned models with physics-informed constraints yields strong gains and improves plausibility.

### RQ2 — Transfer learning improves robustness across domains

Evidence:

- warm-start vs cold-start comparison:
  - `freeze/final_thesis_v1/eval/rq2_warm_vs_cold/text/results.md`

Interpretation:

- Warm initialization reduces error across near and long horizons.
- The effect is strongest in the near-term bucket where calibration and local regime alignment matter.

### RQ3 — Long-horizon stability via dual-head + physics glue

Evidence:

- dual-head vs single-head baselines:
  - `freeze/final_thesis_v1/eval/rq1_warm_vs_short/text/results.md`
  - `freeze/final_thesis_v1/eval/rq1_warm_vs_long/text/results.md`

Interpretation:

- Neither short-only nor long-only forecasts are competitive alone.
- Hierarchical blending stabilizes forecasts across the full 30-day horizon.

### RQ4 — Self-adaptation via an RL meta-controller

Evidence:

- baseline vs policy evaluation:
  - `freeze/final_thesis_v1/eval/rq4_baseline_vs_policy/text/results.md`

Interpretation:

- The current policy is near-neutral overall and slightly worse in the 0–24h bucket under the canonical run.
- This does not invalidate the architectural contribution; rather, it positions the RL controller as a reproducible, measurable adaptive layer that requires further environment tuning and additional training data to deliver consistent improvements.


## 6.2 Scientific contributions

### 6.2.1 Methodological innovations

1. **Hybrid ensemble architecture** combining:
   - PVLib physics features and constraints,
   - LSTM-based temporal representation learning,
   - TFT-based multi-horizon forecasting.

2. **Physics-glue hierarchical inference**:
   - PVLib-shaped upsampling of hourly forecasts to 15-minute resolution,
   - multi-stage blending and hard physical constraints.

3. **RL-supervised forecasting pipeline**:
   - DDQN meta-controller scaffold with prioritized replay and soft target updates,
   - local rule-based advisors producing interpretable state signals.

### 6.2.2 Empirical insights

- Hybridization is not only a modeling choice but a stability mechanism for long-horizon forecasting.
- Transfer learning works best when the intermediate domain is regionally aligned (Germany regional pretraining as a bridge).
- Multi-resolution forecasting is necessary for simultaneously capturing near-term ramps and long-term structure.


## 6.3 Comparison with state-of-the-art

This section should position MiRACLE against recent PV forecasting work along dimensions:

- horizon length (30 days vs typical <7 days),
- temporal resolution (15-min),
- physics integration,
- transfer learning,
- adaptive meta-control.

A thesis-ready comparison table can be built once the target comparison papers are selected.


## 6.4 Practical implications

- Operational value: 30-day forecasts support maintenance scheduling, grid integration planning, and trading/dispatch decisions.
- System design: MiRACLE’s architecture is organized to support real-time ingestion, feature generation, forecasting, and monitoring.


## 6.5 Limitations

- Single target plant evaluation limits generalization claims.
- Weather API dependence can be a bottleneck and a source of shift.
- RL controller performance is currently near-neutral under the canonical backtest; further training and environment realism are required.
- Database/dashboard integration is not fully implemented as a production system (positioned as future work).


## 6.6 Lessons from exploratory experiments

A key strength of the methodology is that it explicitly tracks exploratory steps and removes them from headline claims when they are not representative.

- Farm2107 pretraining served as an exploratory stage to identify a stable encoder configuration and initialization.
- Early Germany transfer variants were deprecated when validation methodology flaws or leakage risks were discovered.

Audit trail:

- `docs/archive/AUDIT_LSTM_PRETRAIN.md`

This iterative elimination process is presented not as “failure”, but as the scientific mechanism that led to a robust final design.


## 6.7 Summary

MiRACLE’s evidence base supports the core thesis claims about hybridization, transfer, and hierarchical inference, while honestly characterizing the RL controller as a promising adaptive layer with measurable but not yet consistently positive impact under the canonical evaluation.
