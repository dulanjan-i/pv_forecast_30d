from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--components", required=True, type=str)
    ap.add_argument("--out-dir", required=True, type=str)
    args = ap.parse_args()

    inp = Path(args.components)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(inp)

    need = ["pred_pvlib_norm", "pred_short_norm", "pred_long_norm"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"Missing component column: {c}. Re-run inference with --save-components 1")

    def write(name: str, series_col: str) -> None:
        out = df.copy()
        out["predicted_power_norm"] = out[series_col].astype("float32")
        keep_cols = [c for c in out.columns if not c.startswith("pred_") or c == "predicted_power_norm"]
        out = out[keep_cols]
        out.to_parquet(out_dir / f"{name}.parquet", index=False)

    write("pvlib_only", "pred_pvlib_norm")
    write("short_only", "pred_short_norm")
    write("long_only", "pred_long_norm")

    # TFT-only: short+long equal weights, or use your evaluator-side weighting later
    out = df.copy()
    out["predicted_power_norm"] = (0.5 * out["pred_short_norm"] + 0.5 * out["pred_long_norm"]).astype("float32")
    keep_cols = [c for c in out.columns if not c.startswith("pred_") or c == "predicted_power_norm"]
    out = out[keep_cols]
    out.to_parquet(out_dir / "tft_only.parquet", index=False)

    print("WROTE:", out_dir)


if __name__ == "__main__":
    main()
