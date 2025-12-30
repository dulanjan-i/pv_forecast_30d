"""
src/validation/inspect_regional_tft_full_inputs.py

Quick sanity checks for the final TFT input parquets:
- Loads train/val full parquets
- Prints shape, time range, plant set, duplicate key counts, NaN counts
- Prints full column lists
- Checks train vs val column set equality
- Flags suspicious merge artifacts (_x/_y suffixes)
- Writes a small JSON report + two text files with column names

Usage:
  python src/validation/inspect_regional_tft_full_inputs.py

Notes:
- This is complementary to validate_regional_tft_inputs.py, which validates base/weather/pvlib merges. :contentReference[oaicite:1]{index=1}
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

TIME_COL = "timestamp_utc"
PLANT_COL = "plant_id"
KEY = [PLANT_COL, TIME_COL]

BASE_DIR = REPO_ROOT / "data" / "processed" / "pretraining" / "germany" / "global"
IN_DIR = BASE_DIR / "tft_inputs"
OUT_DIR = BASE_DIR / "validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FULL = IN_DIR / "regional_train_tft_full.parquet"
VAL_FULL = IN_DIR / "regional_val_tft_full.parquet"


def must_exist(p: Path) -> None:
    if not p.exists():
        raise FileNotFoundError(str(p))


def summarize_df(name: str, df: pd.DataFrame) -> dict:
    d = df.copy()
    d[TIME_COL] = pd.to_datetime(d[TIME_COL], utc=True)
    d[PLANT_COL] = d[PLANT_COL].astype(str)

    out: dict = {}
    out["name"] = name
    out["rows"] = int(len(d))
    out["cols"] = int(d.shape[1])
    out["time_min"] = str(d[TIME_COL].min())
    out["time_max"] = str(d[TIME_COL].max())
    out["plants"] = sorted(d[PLANT_COL].unique().tolist())
    out["has_plant_04"] = "plant_04" in set(out["plants"])
    out["dup_keys"] = int(d.duplicated(KEY).sum())
    out["nan_total"] = int(d.isna().sum().sum())

    # Merge artifact flags
    cols = list(d.columns)
    out["has_suffix_x"] = any(c.endswith("_x") for c in cols)
    out["has_suffix_y"] = any(c.endswith("_y") for c in cols)

    # Helpful: which columns are objects (often accidental)
    obj_cols = [c for c in cols if d[c].dtype == "object"]
    out["object_cols"] = obj_cols

    return out


def main() -> None:
    must_exist(TRAIN_FULL)
    must_exist(VAL_FULL)

    train = pd.read_parquet(TRAIN_FULL)
    val = pd.read_parquet(VAL_FULL)

    rep = {
        "train": summarize_df("train", train),
        "val": summarize_df("val", val),
    }

    train_cols = list(train.columns)
    val_cols = list(val.columns)

    rep["train_only_cols"] = sorted(set(train_cols) - set(val_cols))
    rep["val_only_cols"] = sorted(set(val_cols) - set(train_cols))
    rep["same_colset"] = (set(train_cols) == set(val_cols))

    # Write artifacts
    (OUT_DIR / "regional_tft_full_cols_train.txt").write_text("\n".join(train_cols) + "\n")
    (OUT_DIR / "regional_tft_full_cols_val.txt").write_text("\n".join(val_cols) + "\n")
    (OUT_DIR / "regional_tft_full_report.json").write_text(json.dumps(rep, indent=2) + "\n")

    # Console output (fast skim)
    print("== FULL TFT INPUTS ==")
    print(f"TRAIN: {TRAIN_FULL}")
    print(f"VAL:   {VAL_FULL}\n")

    for split in ("train", "val"):
        s = rep[split]
        print(f"== {split} ==")
        print(f"shape: ({s['rows']}, {s['cols']})")
        print(f"time:  {s['time_min']} -> {s['time_max']}")
        print(f"plants: {s['plants']}")
        print(f"dup keys: {s['dup_keys']}")
        print(f"nan total: {s['nan_total']}")
        print(f"has plant_04: {s['has_plant_04']}")
        print(f"has _x: {s['has_suffix_x']} | has _y: {s['has_suffix_y']}")
        if s["object_cols"]:
            print(f"object cols (check): {s['object_cols']}")
        print("")

    print("== column set diff ==")
    print("same_colset:", rep["same_colset"])
    print("train_only_cols:", rep["train_only_cols"])
    print("val_only_cols:", rep["val_only_cols"])
    print(f"\nWrote:\n  {OUT_DIR / 'regional_tft_full_report.json'}")
    print(f"  {OUT_DIR / 'regional_tft_full_cols_train.txt'}")
    print(f"  {OUT_DIR / 'regional_tft_full_cols_val.txt'}")


if __name__ == "__main__":
    main()
