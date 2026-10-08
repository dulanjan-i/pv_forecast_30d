"""
Fix per-plant scaling issues in German PV interim parquets.

DO:
- Use this script only on data/interim/germany/plant_*_pv_15min.parquet files.
- Treat "power_kw" as the primary signal that might be in the wrong unit.
- Compare installed_capacity_kW to max(power_kw) to flag obviously broken plants.
- Rescale only plants where the ratio capacity_kW / max_power_kw is extremely large.
- After rescaling, recompute:
    - power_kw (scaled)
    - power_w  (scaled in the same way)
    - power_norm = power_kw / installed_capacity_kw
- Overwrite the existing parquet in place, so there are no duplicate versions.

DO NOT:
- Do not run this on final processed feature tables.
- Do not use this to "force" every plant to reach its installed capacity.
- Do not apply the scaling to plants that are already in a realistic range
  where max power is around 60 to 90 percent of capacity.
- Do not assume that all broken plants share the same scale factor like x1000.
  Each plant is checked and scaled individually.
- Do not change the timestamp_utc column or the daily curve shape.
  This script only fixes magnitude, not time or shape.

IMPORTANT - what the rescaling really does:
- scale = capacity / max(power_kw), so after rescaling max(power_kw) == installed capacity and
  power_norm peaks at exactly 1.0. The normalisation of the rescaled plants is therefore
  "relative to the observed peak, which is set equal to installed capacity", and it depends on one
  extreme sample over the whole file. Every run writes a provenance record next to the parquet
  (<plant>_scaling.json) with the factor, the pre-scaling maximum and an outlier indicator, so the
  normalisation can be audited later. Use --dry-run to inspect without writing.

Concept:
- Plants 01, 02, and 05 already have realistic magnitudes.
  Their max(power_kw) is at a sensible fraction of capacity, so they are left untouched.
- Plants 03, 04, and 06 are clearly in the wrong units or have a huge scale mismatch,
  for example max(power_kw) in the range of micro kW for a multi MW plant.
  For these, we treat the mismatch as a unit error and rescale them in place.
"""

from datetime import datetime, timezone
from pathlib import Path
import argparse
import json

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
PV_DIR = REPO_ROOT / "data" / "interim" / "germany"
META_DIR = REPO_ROOT / "data" / "metadata" / "germany"

PLANT_IDS = ["plant_01", "plant_02", "plant_03", "plant_04", "plant_05", "plant_06"]

# If capacity_kW / max_power_kw is below or equal to this threshold,
# we assume the plant magnitude is physically plausible and do NOT rescale.
# Example:
#   capacity = 746 kW, max = 560 kW  -> ratio ~1.33  -> OK
#   capacity = 7358 kW, max = 6.9e-06 kW -> ratio ~1e9 -> clearly broken

RATIO_THRESHOLD = 5.0       # 20% threshold


def load_meta(pid: str) -> dict:
    meta_path = META_DIR / f"{pid}.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found for {pid}: {meta_path}")
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_provenance(pid: str, record: dict) -> None:
    path = PV_DIR / f"{pid}_scaling.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")


def fix_plant_if_needed(pid: str, dry_run: bool = False) -> None:
    parquet_path = PV_DIR / f"{pid}_pv_15min.parquet"
    if not parquet_path.exists():
        print(f"[WARN] parquet not found for {pid}, skipping")
        return

    meta = load_meta(pid)
    cap_kw = float(meta["installed_capacity_kw"])

    df = pd.read_parquet(parquet_path)

    if "power_kw" not in df.columns:
        print(f"[WARN] {pid}: power_kw column missing, skipping")
        return

    max_kw = df["power_kw"].max()
    if max_kw is None or max_kw <= 0:
        print(f"[WARN] {pid}: max_kw <= 0, skipping")
        return

    ratio = cap_kw / max_kw
    p999 = float(df["power_kw"].quantile(0.999))
    # max / p99.9 far above 1 means the peak is a single outlier, so a max-based factor is unreliable
    outlier_ratio = float(max_kw / p999) if p999 > 0 else float("inf")
    print(f"[INFO] {pid}: cap={cap_kw:.3f}  max_kw={max_kw:.6g}  cap/max={ratio:.3g}  max/p99.9={outlier_ratio:.3g}")
    record = {
        "plant_id": pid,
        "installed_capacity_kw": cap_kw,
        "max_power_kw_before": float(max_kw),
        "p999_power_kw_before": p999,
        "max_over_p999": outlier_ratio,
        "cap_over_max": float(ratio),
        "ratio_threshold": RATIO_THRESHOLD,
        "rule": "scale = installed_capacity_kw / max(power_kw); max(power_kw) becomes installed capacity",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    if outlier_ratio > 1.5:
        print(f"[WARN] {pid}: max is {outlier_ratio:.2f}x the 99.9th percentile; the scale factor may be "
              f"driven by a single outlier. Inspect the series before trusting the normalisation.")

    # If ratio is within a reasonable range, plant is assumed to be fine.
    # We do NOT try to "normalize" plants to exactly reach capacity.
    # This respects physical factors like losses, temperature, snow, shading.
    if ratio <= RATIO_THRESHOLD:
        print(f"[INFO] {pid}: ratio <= {RATIO_THRESHOLD}, assumed OK, no scaling applied")
        if not dry_run:
            # Do not overwrite an earlier record that documents an applied scale factor.
            if not (PV_DIR / f"{pid}_scaling.json").exists():
                _write_provenance(pid, {**record, "applied": False, "scale": 1.0})
        return

    # At this point the plant is clearly broken in magnitude.
    # We interpret this as a unit or scale mismatch, not as a physical phenomenon.
    # We compute a scale factor that maps the current max power close to capacity.
    scale = ratio
    print(f"[INFO] {pid}: applying scale factor {scale:.6g}")
    if dry_run:
        print(f"[DRY-RUN] {pid}: nothing written")
        return

    df = df.copy()

    # This is the core scaling logic:
    # - We rescale power_kw and power_w by the same factor.
    # - We do NOT touch timestamp_utc or the temporal shape.
    # - We recompute power_norm from capacity, so it stays consistent.
    df["power_kw"] = df["power_kw"] * scale

    if "power_w" in df.columns:
        df["power_w"] = df["power_w"] * scale
    else:
        # If power_w does not exist (edge case), we derive it from power_kw.
        df["power_w"] = df["power_kw"] * 1000.0

    if cap_kw > 0:
        df["power_norm"] = df["power_kw"] / cap_kw
    else:
        df["power_norm"] = pd.NA

    df.to_parquet(parquet_path, index=False)
    _write_provenance(pid, {**record, "applied": True, "scale": float(scale)})
    print(f"[INFO] {pid}: wrote fixed parquet and {pid}_scaling.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="report ratios and outliers, write nothing")
    args = ap.parse_args()
    for pid in PLANT_IDS:
        fix_plant_if_needed(pid, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
