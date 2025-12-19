"""
src/features/run_build_regional_lstm_encodings.py

Stage 3.6: Build LSTM embeddings for TFT.

What it does:
- Runs germany_build_lstm_encodings.py twice:
  1) regional_train.parquet -> regional_train_lstm_encodings.parquet
  2) regional_val.parquet   -> regional_val_lstm_encodings.parquet

Why:
- Keeps pipeline actions in scripts (not notebooks).
- Produces stable, versionable artifacts for TFT training.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def run_one(repo_root: Path, input_parquet: Path, encoder_ckpt: Path, output_parquet: Path) -> None:
    cmd = [
        "python",
        str(repo_root / "src" / "features" / "germany_build_lstm_encodings.py"),
        "--input_parquet", str(input_parquet),
        "--encoder_ckpt", str(encoder_ckpt),
        "--output_parquet", str(output_parquet),
    ]
    print("\n[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo_root", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--data_dir", type=Path, default=None)
    p.add_argument("--encoder_ckpt", type=Path, default=None)
    args = p.parse_args()

    repo_root = args.repo_root

    data_dir = args.data_dir or (repo_root / "data" / "processed" / "pretraining" / "germany" / "global")
    enc_dir = data_dir / "encodings"
    enc_dir.mkdir(parents=True, exist_ok=True)

    encoder_ckpt = args.encoder_ckpt or (
        repo_root / "experiments" / "lstm" / "encoders" / "lstm_encoder_germany_regional_CANONICAL.pt"
    )

    train_in = data_dir / "regional_train.parquet"
    val_in = data_dir / "regional_val.parquet"

    train_out = enc_dir / "regional_train_lstm_encodings.parquet"
    val_out = enc_dir / "regional_val_lstm_encodings.parquet"

    for f in [train_in, val_in, encoder_ckpt]:
        if not f.exists():
            raise FileNotFoundError(f"Missing required file: {f}")

    run_one(repo_root, train_in, encoder_ckpt, train_out)
    run_one(repo_root, val_in, encoder_ckpt, val_out)

    print("\n[SUCCESS] Encodings written:")
    print(" ", train_out)
    print(" ", val_out)


if __name__ == "__main__":
    main()
