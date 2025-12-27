"""
Compress LSTM encoding columns in Germany TFT input parquets using PCA.

What it does
- Finds columns that start with "lstm_enc_" (original 64-d encodings)
- Fits PCA on TRAIN ONLY (no leakage)
- Rewrites new TRAIN and VAL parquets:
  - Drops original lstm_enc_* columns
  - Adds lstm_enc_pca_000.. columns (n_components)
  - Casts numeric columns to float32 to reduce memory and speed up collation

Outputs
- <out_dir>/train_pca{K}.parquet
- <out_dir>/val_pca{K}.parquet
- <out_dir>/pca.pkl
- <out_dir>/manifest.json
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception as e:
    raise RuntimeError("pyarrow is required for this script. Install pyarrow in the container/env.") from e

try:
    from sklearn.decomposition import IncrementalPCA
except Exception:
    IncrementalPCA = None


UTC = timezone.utc


@dataclass
class PCAInfo:
    n_components: int
    enc_cols: List[str]
    out_cols: List[str]
    train_rows_seen_for_fit: int
    explained_variance_ratio_sum: float | None
    timestamp_utc: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train_parquet", type=str, required=True)
    p.add_argument("--val_parquet", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--n_components", type=int, default=32)
    p.add_argument("--row_batch", type=int, default=200_000, help="Rows per partial_fit batch")
    p.add_argument("--compression", type=str, default="zstd")
    p.add_argument("--float32", action="store_true", help="Cast numeric columns to float32")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _list_enc_cols(pq_path: Path) -> List[str]:
    pf = pq.ParquetFile(str(pq_path))
    cols = pf.schema.names
    enc = [c for c in cols if c.startswith("lstm_enc_")]
    enc = [c for c in enc if not c.startswith("lstm_enc_pca_")]
    if not enc:
        raise ValueError(f"No encoding columns found in {pq_path}. Expected columns starting with 'lstm_enc_'.")
    return sorted(enc)


def _iter_row_groups(pq_path: Path, columns: List[str] | None = None):
    pf = pq.ParquetFile(str(pq_path))
    for rg in range(pf.num_row_groups):
        yield pf.read_row_group(rg, columns=columns)


def _cast_numeric_float32(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]) or pd.api.types.is_integer_dtype(df[c]):
            if df[c].dtype != np.float32 and df[c].dtype != np.int64:
                # keep int64 as is (time_idx etc), cast floats to float32
                if pd.api.types.is_float_dtype(df[c]):
                    df[c] = df[c].astype(np.float32, copy=False)
    return df


def _fit_pca_incremental(train_path: Path, enc_cols: List[str], n_components: int, row_batch: int, seed: int) -> Tuple[object, int]:
    if IncrementalPCA is None:
        raise RuntimeError("scikit-learn is missing, cannot use IncrementalPCA. Install scikit-learn in the env.")

    ipca = IncrementalPCA(n_components=n_components, batch_size=row_batch)
    rows_seen = 0

    for table in _iter_row_groups(train_path, columns=enc_cols):
        df = table.to_pandas()
        X = df[enc_cols].to_numpy(dtype=np.float32, copy=False)

        if np.isnan(X).any():
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        n = X.shape[0]
        start = 0
        while start < n:
            end = min(start + row_batch, n)
            ipca.partial_fit(X[start:end])
            rows_seen += (end - start)
            start = end

    return ipca, rows_seen


def _rewrite_with_pca(
    in_path: Path,
    out_path: Path,
    pca: object,
    enc_cols: List[str],
    out_cols: List[str],
    compression: str,
    cast_float32: bool,
) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    total_rows = 0

    for table in _iter_row_groups(in_path, columns=None):
        df = table.to_pandas()

        # Keep original columns, but cast floats if requested
        if cast_float32:
            df = _cast_numeric_float32(df)

        X = df[enc_cols].to_numpy(dtype=np.float32, copy=False)
        if np.isnan(X).any():
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        Z = pca.transform(X).astype(np.float32, copy=False)

        # Drop original encodings and add PCA columns
        df = df.drop(columns=enc_cols)
        for j, c in enumerate(out_cols):
            df[c] = Z[:, j]

        pa_table = pa.Table.from_pandas(df, preserve_index=False)

        if writer is None:
            writer = pq.ParquetWriter(
                str(out_path),
                pa_table.schema,
                compression=compression,
                use_dictionary=True,
            )
        writer.write_table(pa_table)
        total_rows += len(df)

    if writer is not None:
        writer.close()

    return total_rows


def main() -> int:
    args = parse_args()
    np.random.seed(int(args.seed))

    train_path = Path(args.train_parquet)
    val_path = Path(args.val_parquet)
    out_dir = Path(args.out_dir)

    if not train_path.exists():
        raise FileNotFoundError(str(train_path))
    if not val_path.exists():
        raise FileNotFoundError(str(val_path))

    enc_cols = _list_enc_cols(train_path)
    k = int(args.n_components)

    out_cols = [f"lstm_enc_pca_{i:03d}" for i in range(k)]

    print(f"[INFO] Found {len(enc_cols)} encoding cols.", flush=True)
    print(f"[INFO] Fitting PCA on TRAIN only, n_components={k}", flush=True)

    pca, rows_seen = _fit_pca_incremental(
        train_path=train_path,
        enc_cols=enc_cols,
        n_components=k,
        row_batch=int(args.row_batch),
        seed=int(args.seed),
    )

    evr_sum = None
    if hasattr(pca, "explained_variance_ratio_"):
        evr = getattr(pca, "explained_variance_ratio_")
        evr_sum = float(np.sum(evr))

    out_train = out_dir / f"train_pca{k}.parquet"
    out_val = out_dir / f"val_pca{k}.parquet"
    out_pkl = out_dir / "pca.pkl"
    out_manifest = out_dir / "manifest.json"

    print(f"[INFO] Rewriting TRAIN -> {out_train}", flush=True)
    train_rows = _rewrite_with_pca(
        in_path=train_path,
        out_path=out_train,
        pca=pca,
        enc_cols=enc_cols,
        out_cols=out_cols,
        compression=str(args.compression),
        cast_float32=bool(args.float32),
    )

    print(f"[INFO] Rewriting VAL   -> {out_val}", flush=True)
    val_rows = _rewrite_with_pca(
        in_path=val_path,
        out_path=out_val,
        pca=pca,
        enc_cols=enc_cols,
        out_cols=out_cols,
        compression=str(args.compression),
        cast_float32=bool(args.float32),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    with out_pkl.open("wb") as f:
        pickle.dump(pca, f)

    info = PCAInfo(
        n_components=k,
        enc_cols=enc_cols,
        out_cols=out_cols,
        train_rows_seen_for_fit=rows_seen,
        explained_variance_ratio_sum=evr_sum,
        timestamp_utc=datetime.now(UTC).isoformat(),
    )

    manifest = {
        "train_in": str(train_path),
        "val_in": str(val_path),
        "train_out": str(out_train),
        "val_out": str(out_val),
        "pca_pkl": str(out_pkl),
        "train_rows_out": int(train_rows),
        "val_rows_out": int(val_rows),
        "pca_info": asdict(info),
    }

    with out_manifest.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[DONE] Wrote outputs to {out_dir}", flush=True)
    print(f"[DONE] explained_variance_ratio_sum={evr_sum}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
