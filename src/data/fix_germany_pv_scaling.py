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

Concept:
- Plants 01, 02, and 05 already have realistic magnitudes.
  Their max(power_kw) is at a sensible fraction of capacity, so they are left untouched.
- Plants 03, 04, and 06 are clearly in the wrong units or have a huge scale mismatch,
  for example max(power_kw) in the range of micro kW for a multi MW plant.
  For these, we treat the mismatch as a unit error and rescale them in place.
"""

from pathlib import Path
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


def fix_plant_if_needed(pid: str) -> None:
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
    print(f"[INFO] {pid}: cap={cap_kw:.3f}  max_kw={max_kw:.6g}  cap/max={ratio:.3g}")

    # If ratio is within a reasonable range, plant is assumed to be fine.
    # We do NOT try to "normalize" plants to exactly reach capacity.
    # This respects physical factors like losses, temperature, snow, shading.
    if ratio <= RATIO_THRESHOLD:
        print(f"[INFO] {pid}: ratio <= {RATIO_THRESHOLD}, assumed OK, no scaling applied")
        return

    # At this point the plant is clearly broken in magnitude.
    # We interpret this as a unit or scale mismatch, not as a physical phenomenon.
    # We compute a scale factor that maps the current max power close to capacity.
    scale = ratio
    print(f"[INFO] {pid}: applying scale factor {scale:.6g}")

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
    print(f"[INFO] {pid}: wrote fixed parquet")


def main() -> None:
    for pid in PLANT_IDS:
        fix_plant_if_needed(pid)


if __name__ == "__main__":
    main()
