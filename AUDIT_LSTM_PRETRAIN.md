# LSTM Pretrain Audit (Farm2107 → Germany → TFT)

- **Scope:** Full encoder journey from PVDAQ Farm2107 canonical pretrain through Germany transfer, global V3 iterations, and regional encoder for TFT.
- **Sources used:** `git log --oneline`, root scripts (`run_stage2_transfer_learning.sh`, `run_stage3_global_training.sh`), training scripts in `src/training`, reports in `reports/`, config summary in `TRAINING_CONFIG_SUMMARY.md`.

## Timeline & Versions
- [x] **Farm2107 canonical pretrain (Stage 1)** — goal: strong single-site baseline; selected h64_l2_lr1e-3 (RMSE≈0.0404) for transfer; artifact `experiments/lstm/encoders/lstm_encoder_farm2107_CANONICAL.pt` (`git f44ef90`, `reports/lstm_results.md`).
- [x] **Stage 2 Germany transfer V01 (❌ deprecated)** — naive chronological 70/15/15 split produced seasonal bias/zeros (winter-heavy val); documented failure in `reports/stage2_version01_failed_chronological_split.md`; commit `f151469`.
- [x] **Stage 2 Germany transfer V02 (❌ deprecated)** — stratified temporal split + per-plant fine-tune configs `experiments/lstm/germany/pretrain_plant_0X.yaml`; orchestrated by `run_stage2_transfer_learning.sh` (waves over 2×L4); outputs per-plant checkpoints under `experiments/lstm/runs/germany/`; superseded by Stage 3 global pooling (V3) to avoid per-plant drift and reduce overfitting.
- [x] **Stage 3 Global V3.0 (❌ deprecated)** — initial supermatrix + rolling-origin CV; later flagged for cross-plant window leakage and target-scaling risk; commit `c165ddf`.
- [x] **Stage 3 Global V3.1 (✅ fix)** — plant-aware window dataset, target-safe scaling, fold audit; commit `ab437e0`; training script `src/training/train_global_lstm_v3.py`; preprocessing `src/preprocessing/germany_build_global_supermatrix.py`, `germany_global_rolling_origin_split.py`; wrapper `run_stage3_global_training.sh`.
- [x] **Stage 3 Global V3.1.1 (maintenance)** — logging directory cleanup, no behavioral change; commit `3953de0`.
- [x] **Stage 3.5 Regional encoder (✅ current)** — single Germany-adapted encoder for TFT base; script `src/training/train_regional_lstm.py` (`dd93ca9`); data split `regional_{train,val}.parquet`; outputs `experiments/lstm/encoders/lstm_encoder_germany_regional_CANONICAL.pt` + run copy under `.../global_v3/regional/`.

## Why Each Step
- [x] **Transfer seed:** Farm2107 canonical provides stable initialization for scarce German data; zero-padding preserves first 15 features, frees plant IDs (see `src/models/global_lstm_encoder.py`).
- [x] **Stage 2 per-plant:** Needed plant-specific adaptation before pooling; exposed validation methodology flaw → V02 stratification.
- [x] **Global V3:** Reduce overfitting by pooling plants + rolling-origin CV; V3.1 fixes ensured windows never cross plants and scalers fit on train only.
- [x] **Regional encoder:** Single canonical Germany encoder for TFT downstream (stable embeddings, avoid fold-specific scalers).

## Artifacts & Paths (Current)
- [x] Farm seed: `experiments/lstm/encoders/lstm_encoder_farm2107_CANONICAL.pt`
- [x] Plant transfer runs: `experiments/lstm/runs/germany/logs/plant_0X.log` + checkpoints in per-plant run dirs
- [x] Global V3.1.* runs: `experiments/lstm/runs/germany/global_v3/fold_{1..4}/`
- [x] Regional encoder: `experiments/lstm/encoders/lstm_encoder_germany_regional_CANONICAL.pt` (+ run copy in `.../global_v3/regional/`)
- [x] Preprocessing outputs: `data/processed/pretraining/germany/global/supermatrix_base.parquet`, `fold_{k}_{train,val}.parquet`, `regional_{train,val}.parquet`

## Deprecations / Warnings
- [x] Stage 2 V01 results invalid — do not reuse metrics or checkpoints (seasonal bias).
- [x] Stage 2 V02 superseded by Global V3 — per-plant checkpoints not used downstream; prefer V3.1+ global/regional artifacts.
- [x] Global V3.0 outputs superseded by V3.1 — discard any fold weights produced before commit `ab437e0`.
- [x] Any runs without Farm2107 checkpoint present will train from scratch; verify existence before reuse (`run_stage3_global_training.sh` warning).

## Follow-Ups / Checks
- [ ] Ensure `TRAINING_CONFIG_SUMMARY.md` suggested dataloader/DDP tweaks are applied to `train_global_lstm_v3.py`.
- [ ] Centralize best checkpoints per fold + regional in a manifest for TFT (paths + metrics).
- [ ] Confirm no lingering V3.0 artifacts are referenced in downstream scripts/notebooks.
