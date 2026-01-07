# src/evaluation/inspect_parquets.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

print("[BOOT] inspect_parquets.py loaded", file=sys.stderr)

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception as e:
    print("[FATAL] failed to import pyarrow:", repr(e), file=sys.stderr)
    raise


def read_parquet_fast(path: Path, columns: Optional[Sequence[str]] = None) -> "pa.Table":
    if not path.exists():
        raise FileNotFoundError(f"Parquet not found: {path}")
    return pq.read_table(path.as_posix(), columns=list(columns) if columns else None)


def print_schema_and_preview(t: "pa.Table", n: int = 3) -> None:
    print("\n--- schema ---")
    print(t.schema)

    print("\n--- columns ---")
    for c in t.column_names:
        print(c)

    print(f"\n--- preview (first {n} rows) ---")
    df = t.slice(0, n).to_pandas()
    print(df.to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", type=str, required=True)
    ap.add_argument("--baseline", type=str, required=True)
    ap.add_argument("--policy", type=str, required=True)
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()

    print("[INFO] args parsed OK", file=sys.stderr)

    paths = {
        "truth": Path(args.truth),
        "baseline": Path(args.baseline),
        "policy": Path(args.policy),
    }

    for name, p in paths.items():
        print(f"\n==================== {name.upper()} ====================")
        print(f"path: {p}")
        t = read_parquet_fast(p)
        print(f"rows: {t.num_rows:,}  cols: {t.num_columns}")
        print_schema_and_preview(t, n=args.n)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[FATAL] exception:", repr(e), file=sys.stderr)
        raise
