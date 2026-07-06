# RL Meta-Controller Deployment Notes

## Canonical Thesis Checkpoints

All intermediate and candidate checkpoints were removed during cleanup (2026-07-06).
The thesis-canonical checkpoints are in `freeze/final_thesis_v1/rl/` and are
verified by SHA256 manifest at `freeze/CHECKPOINT_MANIFEST.sha256`.

| Checkpoint | Path | SHA256 (first 16) | Notes |
|---|---|---|---|
| RL v1 best | `freeze/final_thesis_v1/rl/ddqn_minenv_v1/ddqn_best.pt` | `0fd9f14bf01342f6` | Thesis RL v1 result (638 steps) |
| RL v2 best | `freeze/final_thesis_v1/rl/ddqn_minenv_v2/ddqn_best.pt` | `bcc32ceb87765fe5` | Thesis RL v2 result (660 steps) |
| Phase1 policy | `freeze/final_thesis_v1/phase1_2024daily_final/rl/ddqn_phase1_daily_norm.pt` | `777319ebaceceeae` | RQ4 offline evaluation policy |

Verify integrity before any deployment:
```bash
shasum -a 256 -c freeze/CHECKPOINT_MANIFEST.sha256
```

## Usage Example (inference)
```bash
# Use v2 best (recommended for production)
python -m src.inference.phase1_inference_with_policy \
  --policy-ckpt freeze/final_thesis_v1/rl/ddqn_minenv_v2/ddqn_best.pt \
  --start-date 2024-01-01 --end-date 2024-12-31 \
  --phase-dir freeze/final_thesis_v1/phase1_2024daily_final \
  --plant-meta V1.0_FINAL_TFT/plant_metadata/plant_03.json \
  --short-ckpt V1.0_FINAL_TFT/shorthead_seed42/checkpoints/best.ckpt \
  --long-ckpt V1.0_FINAL_TFT/longhead_seed43/checkpoints/best.ckpt \
  --short-train data/processed/plant_level/plant_03/15min_pca32/train.parquet \
  --long-train data/processed/plant_level/plant_03/hourly_longhead/train.parquet \
  --sarns-norm freeze/final_thesis_v1/phase1_2024daily_final/rl/sarns_norm_stats.json
```

## Notes
- State dim: 35, Action dim: 8 (MiRACLE action space)
- Architecture: MLP [35 → 128 → 64 → 8], DDQN with soft target updates (τ=0.005)
- Trained with Prioritized Experience Replay (PER) on H100 GPU
- See `freeze/CHECKPOINT_MANIFEST.json` for full provenance metadata
