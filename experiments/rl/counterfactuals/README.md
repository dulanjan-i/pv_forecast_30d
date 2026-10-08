# RL counterfactual experiments

This folder contains tools and outputs for RL counterfactual / stress‑test experiments.

Key items:
- `src/rl/perturb_weather.py`: create perturbed 15-min weather parquet variants (per-date, per-magnitude) and a `manifest.json`.
- `experiments/rl/counterfactuals/<plant_id>/`: per-plant experiment outputs (parquets, manifest, logs).
- Use `src/rl/build_counterfactual_day1.py` with `--weather_15min` pointing to a perturbed parquet and `--gt` pointing to the canonical ground truth parquet.

Minimal workflow:
1. Create variants for `plant_03`:
   ```bash
   python src/rl/perturb_weather.py \
     --weather_in data/processed/plant_level/plant_03/weather_15min_parquet.parquet \
     --out_dir experiments/rl/counterfactuals/plant_03/weather_variants \
     --plant_id plant_03 \
     --sample_per_season 2 \
     --magnitudes 0.8,0.6,0.4
   ```
2. For each generated parquet, run `build_counterfactual_day1.py` with the same `--gt` (ground truth) path and record outputs.

Recording: update `experiments/rl/counterfactuals/plant_03/experiment_log.md` with the variant details and the command used (include git commit id and ckpt paths).
