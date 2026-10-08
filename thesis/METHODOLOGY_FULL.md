# MiRACLE — Comprehensive Methodology (Consolidated)

This document consolidates the full, step-by-step methodology used to develop, evaluate, and deploy the MiRACLE forecasting system. It is intended as a canonical reference that (i) lists each stage of the experimental workflow, (ii) documents the rationale and procedures used, (iii) cites repository artifacts and code that implement the stage, and (iv) records the design decisions and empirical outcomes that shaped the final MiRACLE v1.0 core.

Summary of the 10-stage workflow
- Stage 1: LSTM Farm2107 Pretrain (exploratory; deprecated but archived)
- Stage 2: Germany Regional Pretrain (audited)
- Stage 3: Plant Fine-Tune (target-plant adaptation)
- Stage 4: Ablation study (component isolation → TFT+PVLib selected)
- Stage 5: TFT-PVLIB Selection (rigorous sweep and seed-based verification)
- Stage 6: Global Pretrain Germany (no-leak validation)
- Stage 7: Long-Head Addition (hourly TFT for long horizons)
- Stage 8: Hierarchical Inference (physics-glue reconciliation)
- Stage 9: RL Meta-Controller (offline RL for blending decisions)
- Stage 10: Real-time inference and weather API router (deployment layer)

For each stage below: Purpose, protocol, code/data artifacts, evaluation criteria, and outcome/decision.

---

## Stage 1 — LSTM Farm2107 Pretrain (exploratory)
Purpose
- Use a geographically distant, high-quality dataset (Farm2107) to explore representation learning via an LSTM encoder and to run large-scale encoder hyperparameter sweeps.
Protocol and artifacts
- Preprocessing pipeline and canonical capacity normalization: `src/preprocessing/farm2107_preprocess.py`.
- Pretraining experiments and canonical configs under `experiments/lstm/pretrain_farm2107*.yaml`.
- Encoder code: `src/models/lstm_encoder.py` and lightweight baseline `src/models/lstm_model.py`.
Notes and outcome
- Stage 1 was exploratory and later deprecated for the final pipeline due to domain mismatch; results and audit notes are preserved in `docs/archive/AUDIT_LSTM_PRETRAIN.md`.
- The experimental record is retained for transparency; it influenced subsequent regional pretraining but is not part of the final production pipeline.

---

## Stage 2 — Germany Regional Pretrain
Purpose
- Pretrain encoder representations on regionally relevant plants to reduce domain gap and improve transfer to the target plant.
Protocol and artifacts
- Germany pretraining preprocess and scaler generation: `src/preprocessing/germany_pretrain_normalize_split.py`.
- Per-plant interim parquets and scalers produced under `data/processed/pretraining/germany/{plant_id}/` (train/val/test + `scaler.json`).
- Audit and notes: `docs/archive/AUDIT_LSTM_PRETRAIN.md` (documents pivot from Stage 1).
Evaluation
- Fit encoder on regional pretraining splits (train-only scalers) and validate embeddings' stability on held-out folds.
Outcome
- Regional pretraining selected over Farm2107 pretraining due to better domain alignment and lower leakage risk.

---

## Stage 3 — Plant Fine-Tune (target plant)
Purpose
- Fine-tune pre-trained encoders and forecasters on plant-specific data (plant_03 target) using strictly temporal isolation.
Protocol and artifacts
- Plant-level fine-tuning scripts and training configs for warm-start vs cold-start comparisons appear in training scripts and `experiments/`.
- Target-plant backtest and ground-truth anchoring discussed in chapter: `thesis/chapters/CH04_Experimental_Design_Ablations.md` and `thesis/chapters/CH03_Methodology_MiRACLE.md`.
Evaluation
- Compare warm-start (pretrained encoders) vs cold-start models under the sealed 2024 test year for plant_03.
Outcome
- Warm-start provides modest but consistent improvements (see `thesis/chapters/CH05_Results_Performance_Analysis.md`).

---

## Stage 4 — Ablation Study (component isolation)
Purpose
- Attribute performance gains to specific components by isolating/removing them while holding the evaluation protocol fixed.
Protocol and artifacts
- Ablation logic and evaluation described in `thesis/chapters/CH04_Experimental_Design_Ablations.md` and the extended ablation CSVs referenced in that chapter (e.g., `ablation_summary_extended.csv` under the `freeze/` benchmark artifacts).
- Implemented ablations include: TFT only, PVLib-only, short-head only, long-head only, TFT+PVLib, TFT+LSTM+PVLib (full MiRACLE core).
Evaluation
- Use strictly held-out future period metrics (RMSE/MAE) and horizon-bucketed evaluations.
Key decision and outcome
- Empirically, the TFT+PVLib configuration consistently outperformed variants that included the LSTM encoder as an active forecasting head. Consequently, the LSTM as a parallel forecast head was removed from the final operational ensemble and retained only as an optional encoder/warm-start artifact. This decision and its motivation MUST be documented in the thesis; see `thesis/chapters/CH04_Experimental_Design_Ablations.md` and `thesis/chapters/CH05_Results_Performance_Analysis.md` for results.

---

## Stage 5 — TFT+PVLib Selection (short-head model selection)
Purpose
- Select the short-head (high-resolution) model configuration via rigorous hyperparameter sweeps and seed-based verification.
Protocol and artifacts
- Short-head sweep notes and winner summary: `experiments/tft/notes/short_head_model_selection.md` (documents enc_len=96, pred_len=96, hidden_size=64, layers=2, dropout choices, and winning run path).
- Candidate configurations and run artifacts stored under `experiments/tft/runs/` and `freeze/` benchmarks.
Evaluation
- Compare RMSE/MAE across seeds and hyperparameters; choose winner by validation RMSE, then verify on test backtest year.
Outcome
- Selected TFT short-head configuration (`tft_pvlib` winner) is documented in `experiments/tft/notes/short_head_model_selection.md` and incorporated into MiRACLE as the canonical short-head.

---

## Stage 6 — Global Pretrain Germany (no-leak validation)
Purpose
- Train global models and preprocessors on Germany regional data while strictly avoiding leakage to the target backtest year.
Protocol and artifacts
- Preprocessing and split utilities: `src/preprocessing/germany_pretrain_normalize_split.py` (stratified temporal split), `src/data/fix_germany_pv_scaling.py` (unit/scale fixes), and metadata under `data/metadata/germany`.
- Training harness and dataset manifests under `src/` and `experiments/`.
Evaluation
- Confirm scalers are fitted on train-only splits and that test-year data are never used for estimating preprocessing parameters.
Outcome
- Global pretraining stabilized TFT training and formed the basis for subsequent long-head addition and hierarchical inference.

---

## Stage 7 — Long-Head Addition (hourly resolution)
Purpose
- Add a long-head forecasting component at hourly resolution to capture long-term structure (30-day horizon) with improved stability.
Motivation
- A single 2880-step 15-min predictor is unstable and data-hungry; an hourly long-head reduces variance and complements the short-head's high-resolution detail.
Protocol and artifacts
- Hourly aggregation utilities: `src/data/make_hourly_from_15min_parquets.py`.
- Long-head training configs and runs under `experiments/tft/` (hourly training uses `max_prediction_length=720`, encoder=168 (7 days), see `src/models/tft_model.py` defaults and references in `src/rl/build_counterfactual_day1.py`).
Evaluation
- Compare long-head-only forecasts and combined hierarchical outcomes on 30-day backtests (hourly → upsample to 15-min via PVLib shape during inference).
Outcome
- Long-head added to MiRACLE as an additional, complementary component; its hourly outputs are upsampled to 15-min and reconciled with the short-head.

---

## Stage 8 — Hierarchical Inference (physics-glue)
Purpose
- Reconcile multi-resolution outputs (short-head 15-min, long-head hourly) and a physics-based baseline (`pvlib`) into a single 15-minute operational forecast.
Why hierarchical glue rather than one 2880-step model
- Hierarchical design separates variability scales: short-term stochasticity vs long-term structural trends. It reduces variance, enables physically plausible within-hour shapes (via PVLib), and provides robustness against horizon-dependent error accumulation.
Protocol and artifacts
- Implementation: `src/inference/physics_glue.py` (upsampling, convex blending, constraints) and orchestration in `src/inference/physics_aware_forecaster.py` and `src/inference/phase1_inference_pipeline_v3.py`.
- Default blend weights and RL override hooks are implemented; the blending procedure is: (1) inner-day short+long convex blend, (2) ML vs PVLib convex blend with `alpha_ml`, (3) apply physical constraints (clip, night forcing, capacity multiplier).
Evaluation
- Ablation and backtesting show that physics-glue reduces physically implausible excursions and improves day-1 RMSE and long-horizon stability. See `thesis/chapters/CH03_Methodology_MiRACLE.md` Section 3.7 and `thesis/chapters/CH04_Experimental_Design_Ablations.md`.
Outcome
- Hierarchical physics-glue established as a central methodological contribution.

---

## Stage 9 — RL Meta-Controller (adaptive blending control)
Purpose
- Implement a discrete-action meta-controller (double DQN) that observes recent performance and data-quality signals and selects operational actions that modify blending/routing decisions.
Protocol and artifacts
- RL code: `src/rl/` (controller implementation `src/rl/rl_meta_controller.py` and training harness), counterfactual builders: `src/rl/build_counterfactual_day1.py`.
- Action-space design: originally 8 discrete actions; empirical training and evaluation led the effective policy to select a smaller subset (3 actions) in practice.
Evaluation and honest assessment
- The policy was trained offline on historical transitions. Under the canonical evaluation, the RL controller produced performance similar to the baseline MiRACLE core (near-neutral overall).
- Observed issues: Day-1 degradation in some counterfactuals, conservative behavior from controller (prefers safe actions); limitations are traced to limited offline action coverage, reward shaping, and the restricted state representation.
Recorded conclusion (transparent)
- Present truth: RL blending control implemented and evaluated; it did not produce consistent improvements in the canonical backtest. The thesis must record this honestly: "RL blending control implemented and evaluated; performance similar overall, with Day-1 degradation observed and neutral long-horizon effect." See `thesis/chapters/CH06_Discussion.md` and RL artifacts under `src/rl/`.
Recommendations to fix
- Improve state representation (horizon-conditioned uncertainty estimates), redesign reward to reflect asymmetric operational costs, expand action space carefully with additional offline data or simulation, and consider online fine-tuning under careful safety constraints.

---

## Stage 10 — Real-time inference and weather API router (deployment)
Purpose
- Provide a production-ready online inference layer that routes weather sources and serves the MiRACLE forecast outputs.
Protocol and artifacts
- Weather API router and deployment notes: `WEATHER_API_DECISION.md` (deployment decision logic), `src/inference/` scripts, and `freeze/` benchmark artifacts demonstrating the inference pipeline.
Constraints and outcome
- For the thesis, a live demo could not be conducted due to missing PV ground-truth for 2026; the system is nonetheless runnable given live weather feeds and ground-truth measurement access.

---

## Cross-stage notes: normalization, leakage prevention, and reproducibility
- Canonical capacity normalization: `power_norm = power_kw / installed_capacity_kw`. Implementations: `src/preprocessing/farm2107_preprocess.py` and `src/preprocessing/germany_pretrain_normalize_split.py`. Inference fallback synthesizes `power_norm` from `pvlib_ac_kw / capacity` (see `src/inference/phase1_inference_pipeline_v3.py`).
- Leakage prevention: scalers/normalizers fitted on TRAIN only; `germany_pretrain_normalize_split.py` fits stats on train split and applies to val/test. Preprocessing fixes (unit rescale) are implemented in `src/data/fix_germany_pv_scaling.py`.
- Reproducibility: experiments and canonical run artifacts are organized under `experiments/` and `freeze/` with manifests pointing to checkpoints used for thesis results. Figures and architecture diagrams are under `thesis/figures/` and `thesis/diagrams/`.

---

## Recommendation and action items for the thesis
- This consolidated methodology file (`thesis/METHODOLOGY_FULL.md`) should be referenced from `thesis/chapters/CH03_Methodology_MiRACLE.md` and `PROGRESS_TRACKER.md` as the single canonical procedural source.
- Required explicit statements to include in the thesis: (i) the Stage-4 empirical decision to drop the LSTM from the final forecasting ensemble and the supporting metrics, (ii) the honest RL assessment (near-neutral), and (iii) the exact preprocessing normalization rules with file references.

---

## Repository pointers (examples)
- Encoder & LSTM: `src/models/lstm_encoder.py`, `src/models/lstm_model.py`
- TFT model and dataset: `src/models/tft_model.py`, `experiments/tft/` (sweeps and notes)
- Physics-glue: `src/inference/physics_glue.py`, `src/inference/physics_aware_forecaster.py`
- RL controller: `src/rl/` (DDQN implementation and counterfactual builders)
- Preprocessing and normalization: `src/preprocessing/farm2107_preprocess.py`, `src/preprocessing/germany_pretrain_normalize_split.py`, `src/data/fix_germany_pv_scaling.py`
- Audits and archives: `docs/archive/AUDIT_LSTM_PRETRAIN.md`, `claude_copilot_prompt.md` (authoring prompt & experimental plan)

---

If you want, I will:
- (A) commit this file and update `thesis/chapters/CH03_Methodology_MiRACLE.md` to reference it, and
- (B) extract specific figures/tables and append a short appendix listing all ablation CSVs and run directories used to make Stage-4 and Stage-5 decisions.

File created: `thesis/METHODOLOGY_FULL.md` (consolidated canonical methodology).