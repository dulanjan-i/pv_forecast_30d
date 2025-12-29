"""
Build "no-leak" global parquets by excluding a target plant from the global pretrain data.

Why:
- We discovered contamination when using global pretraining data that included the target plant
  while also evaluating on that plant's held-out test split.
- This script creates a global train/val set that EXCLUDES a specified plant_id, so the resulting
  global model can be used safely for warm-start on that target plant.

Inputs:
- src_train parquet (global train)
- src_val parquet (global val)
- exclude_plant_id (e.g., "plant_03")

Outputs (written to out_dir):
- train.parquet
- val.parquet
- manifest.json (counts + time ranges)

Notes:
- We do NOT add time_idx here. train_tft_v1.py will create time_idx internally from timestamp_utc.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


KEY_T = "timestamp_utc"
KEY_G = "plant_id"


def _must_have(df: pd.DataFrame, cols: list[str], name: str) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise KeyError(f"[{name}] Missing columns: {miss}")


def _time_range(df: pd.DataFrame) -> dict:
    t = pd.to_datetime(df[KEY_T], utc=True, errors="coerce")
    t = t.dropna()
    if len(t) == 0:
        return {"start": None, "end": None}
    return {"start": str(t.min()), "end": str(t.max())}


def _filter_out(df: pd.DataFrame, exclude_plant_id: str) -> pd.DataFrame:
    df = df.copy()
    df[KEY_G] = df[KEY_G].astype(str)
    return df[df[KEY_G] != str(exclude_plant_id)].reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src_train", type=str, required=True)
    p.add_argument("--src_val", type=str, required=True)
    p.add_argument("--exclude_plant_id", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    src_train = Path(args.src_train)
    src_val = Path(args.src_val)
    if not src_train.exists():
        raise FileNotFoundError(f"--src_train not found: {src_train}")
    if not src_val.exists():
        raise FileNotFoundError(f"--src_val not found: {src_val}")

    train_df = pd.read_parquet(src_train)
    val_df = pd.read_parquet(src_val)

    _must_have(train_df, [KEY_T, KEY_G], "train")
    _must_have(val_df, [KEY_T, KEY_G], "val")

    before_train = len(train_df)
    before_val = len(val_df)

    train_out = _filter_out(train_df, args.exclude_plant_id)
    val_out = _filter_out(val_df, args.exclude_plant_id)

    after_train = len(train_out)
    after_val = len(val_out)

    if after_train == 0 or after_val == 0:
        raise ValueError(
            f"After excluding {args.exclude_plant_id}, got empty split: "
            f"train={after_train}, val={after_val}. Check plant_id strings."
        )

    (out_dir / "train.parquet").write_bytes(b"")  # fail-fast on perms
    (out_dir / "val.parquet").write_bytes(b"")
    (out_dir / "train.parquet").unlink()
    (out_dir / "val.parquet").unlink()

    train_out.to_parquet(out_dir / "train.parquet", index=False)
    val_out.to_parquet(out_dir / "val.parquet", index=False)

    manifest = {
        "exclude_plant_id": args.exclude_plant_id,
        "src_train": str(src_train),
        "src_val": str(src_val),
        "rows": {
            "train_before": before_train,
            "train_after": after_train,
            "val_before": before_val,
            "val_after": after_val,
        },
        "time_range": {
            "train_after": _time_range(train_out),
            "val_after": _time_range(val_out),
        },
        "unique_plants": {
            "train_after": sorted(train_out[KEY_G].astype(str).unique().tolist()),
            "val_after": sorted(val_out[KEY_G].astype(str).unique().tolist()),
        },
    }

    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    print("[OK] Wrote:")
    print("  ", out_dir / "train.parquet")
    print("  ", out_dir / "val.parquet")
    print("  ", out_dir / "manifest.json")


if __name__ == "__main__":
    main()
