#!/usr/bin/env python3
"""Generate a thesis-styled ablation RMSE bar chart from canonical freeze tables.

Input (canonical):
- freeze/final_thesis_v1/benchmarks/thesis_formatted_v3/tables/overall_metrics.csv

Output:
- thesis/figures/ablations/ablation_rmse_overall.png
- thesis/figures/ablations/ablation_rmse_overall.pdf

Styling:
- Reuses the established thesis palette from `src/evaluation/run_benchmark_suite_v3_formatted.py`:
  - MiRACLE highlighted: #00AA00
  - Comparisons: #6BA3D8
"""

from __future__ import annotations

import argparse
import pathlib

import matplotlib.pyplot as plt
import pandas as pd


MIRACLE_COLOR = "#00AA00"
COMPARISON_COLOR = "#6BA3D8"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="freeze/final_thesis_v1/benchmarks/thesis_formatted_v3/tables/overall_metrics.csv",
        help="Path to overall_metrics.csv",
    )
    parser.add_argument(
        "--out-dir",
        default="thesis/figures/ablations",
        help="Output directory",
    )
    parser.add_argument(
        "--title",
        default="Overall RMSE (Canonical Backtest, 2024)",
        help="Plot title",
    )
    args = parser.parse_args()

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    input_path = repo_root / args.input
    out_dir = repo_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    if "model" not in df.columns or "RMSE" not in df.columns:
        raise ValueError(f"Expected columns {{'model','RMSE'}} in {input_path}, got {list(df.columns)}")

    df = df[["model", "RMSE"]].copy()
    df["RMSE"] = pd.to_numeric(df["RMSE"], errors="raise")

    # Sort best->worst
    df = df.sort_values("RMSE", ascending=True).reset_index(drop=True)

    labels = df["model"].tolist()
    values = df["RMSE"].tolist()

    colors = [MIRACLE_COLOR if "MiRACLE" in name else COMPARISON_COLOR for name in labels]

    # Figure size tuned for thesis readability.
    fig, ax = plt.subplots(figsize=(9.5, 4.8))

    bars = ax.bar(labels, values, color=colors, alpha=0.9, edgecolor="black", linewidth=0.6)

    # Annotate values.
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="semibold" if v == min(values) else "normal",
        )

    ax.set_title(args.title, fontsize=13, fontweight="bold")
    ax.set_ylabel("RMSE (normalized power)", fontsize=11)
    ax.set_xlabel("Model / ablation", fontsize=11)

    ax.grid(True, axis="y", alpha=0.3, linestyle=":")
    ax.set_axisbelow(True)

    # Improve label readability
    ax.tick_params(axis="x", labelrotation=20)

    fig.tight_layout()

    out_png = out_dir / "ablation_rmse_overall.png"
    out_pdf = out_dir / "ablation_rmse_overall.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote: {out_png.relative_to(repo_root)}")
    print(f"Wrote: {out_pdf.relative_to(repo_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
