#!/usr/bin/env python3
"""
Run build_counterfactual_day1.py for every variant listed in a manifest.json
and append provenance + result path to the experiment log.

Usage:
  python src/rl/run_counterfactuals_from_manifest.py --manifest experiments/rl/counterfactuals/plant_03/weather_variants/manifest.json

The script assumes the repository layout and default artifact paths used by the project.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from datetime import datetime


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    root = Path(".")
    with open(manifest_path) as fh:
        manifest = json.load(fh)

    variants = manifest.get("variants", [])
    out_dir = Path(manifest_path).parent.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Default artifact paths (project conventions)
    short_ckpt = Path("V1.0_FINAL_TFT/shorthead_seed42/checkpoints/best.pt")
    long_ckpt = Path("V1.0_FINAL_TFT/longhead_seed43/checkpoints/best.pt")
    plant_meta = Path("data/metadata/germany/plant_03.json")
    short_train = Path("data/processed/plant_level/plant_03/15min_pca32/train.parquet")
    long_train = Path("data/processed/plant_level/plant_03/hourly_longhead/train.parquet")
    sarns = Path("freeze/final_thesis_v1/phase1_2024daily_final/rl/sarns_norm_with_blends.parquet")
    # Use the canonical processed backtest weather (15-min, includes pvlib if present)
    hist_weather = Path("data/processed/plant_level/plant_03/hist_weather_gt_15min_utc.parquet")
    gt = Path("data/processed/plant_level/plant_03/ground_truth_from_sheet_15min_utc_capnorm.parquet")

    log_path = Path("experiments/rl/counterfactuals/plant_03/experiment_log.md")

    for v in variants:
        date = v.get("date")
        mag = v.get("magnitude")
        weather_file = Path(v.get("file"))
        variant_id = f"{date}_mag{mag}"
        out_file = out_dir / f"results_{variant_id}.parquet"

        cmd = [
            "python3", "src/rl/build_counterfactual_day1.py",
            "--out", str(out_file),
            "--plant_meta", str(plant_meta),
            "--short_ckpt", str(short_ckpt),
            "--long_ckpt", str(long_ckpt),
            "--short_train_parquet", str(short_train),
            "--long_train_parquet", str(long_train),
            "--sarns_norm", str(sarns),
            "--hist_weather_gt", str(hist_weather),
            "--weather_15min", str(weather_file),
            "--gt", str(gt),
        ]

        started = datetime.utcnow().isoformat() + "Z"
        print(f"Running variant {variant_id} -> {out_file}")
        # Ensure project root is on PYTHONPATH so `src` imports resolve
        import os
        env = os.environ.copy()
        env["PYTHONPATH"] = str(root.resolve())
        res = subprocess.run(cmd, env=env)
        finished = datetime.utcnow().isoformat() + "Z"

        status = "OK" if res.returncode == 0 else f"ERROR({res.returncode})"

        # Append record to experiment log
        with open(log_path, "a") as logf:
            logf.write(f"| {variant_id} | {date} | {mag} | {weather_file} | {out_file} | {status} | {started} | {finished} |\n")

    print("All variants processed. See experiment_log.md for details")


if __name__ == "__main__":
    main()
