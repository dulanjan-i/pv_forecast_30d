"""
Check timestamp overlap between:
1) plant-level splits (train/val/test)
2) global pretrain splits (train/val) vs plant test split (for warm-start contamination check)

Assumptions:
- timestamp column: timestamp_utc
- plant id column: plant_id (string like "plant_03")
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


KEY_T = "timestamp_utc"
KEY_P = "plant_id"


def _load_times(p: Path, plant_id: str | None) -> pd.Series:
    df = pd.read_parquet(p, columns=[KEY_T, KEY_P])
    if plant_id is not None:
        df[KEY_P] = df[KEY_P].astype(str).str.strip()
        df = df[df[KEY_P] == plant_id]

    ts = pd.to_datetime(df[KEY_T], utc=True, errors="coerce")
    ts = ts.dropna()
    ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)  # tz-aware -> tz-naive (still UTC)
    return ts.sort_values().reset_index(drop=True)

def _summ(name: str, s: pd.Series) -> None:
    if len(s) == 0:
        print(f"[{name}] EMPTY")
        return
    print(f"[{name}] n={len(s)} start={s.iloc[0]} end={s.iloc[-1]}")


def _overlap(a: pd.Series, b: pd.Series) -> int:
    # convert to int64 ns for fast set intersection
    aset = set(a.view("int64").tolist())
    bset = set(b.view("int64").tolist())
    return len(aset.intersection(bset))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plant_id", required=True, help="e.g. plant_03")

    ap.add_argument("--plant_train", required=True)
    ap.add_argument("--plant_val", required=True)
    ap.add_argument("--plant_test", required=True)

    ap.add_argument("--global_train", default=None)
    ap.add_argument("--global_val", default=None)

    args = ap.parse_args()

    pid = str(args.plant_id).strip()

    plant_train = _load_times(Path(args.plant_train), pid)
    plant_val   = _load_times(Path(args.plant_val), pid)
    plant_test  = _load_times(Path(args.plant_test), pid)

    print("\n=== Plant splits summary ===")
    _summ("plant_train", plant_train)
    _summ("plant_val  ", plant_val)
    _summ("plant_test ", plant_test)

    print("\n=== Plant split overlaps (should be 0) ===")
    print("train ∩ val  =", _overlap(plant_train, plant_val))
    print("train ∩ test =", _overlap(plant_train, plant_test))
    print("val   ∩ test =", _overlap(plant_val, plant_test))

    if args.global_train and args.global_val:
        gtrain = _load_times(Path(args.global_train), pid)
        gval   = _load_times(Path(args.global_val), pid)

        print("\n=== Global pretrain (filtered to plant_id) ===")
        _summ("global_train", gtrain)
        _summ("global_val  ", gval)

        print("\n=== Warm-start contamination check ===")
        print("(global_train ∪ global_val) ∩ plant_test should be 0")
        contam = _overlap(pd.concat([gtrain, gval], ignore_index=True), plant_test)
        print("contamination_count =", contam)


if __name__ == "__main__":
    main()
