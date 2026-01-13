#!/usr/bin/env python3
"""
Thesis-style plots: Ground Truth vs MiRACLE v1.0 Core vs RL policy.

Style contract (matches your thesis colors):
- Ground truth: light grey (#888888), lw=1.5, alpha=0.7
- MiRACLE v1.0 Core: bold green (#00AA00), lw=2.5, alpha=1.0
- RL comparison: light blue (#6BA3D8), lw=1.8, alpha=0.9

Outputs:
- facets_case_winter_week_core_vs_rl.png
- facets_case_summer_week_core_vs_rl.png
- monthly_rmse_core_vs_rl.png
- leadtime_rmse_curve_0_24h_core_vs_rl.png
- tails_abs_error_hist_core_vs_rl.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


COL_TRUTH = "#888888"
COL_CORE = "#00AA00"
COL_COMP = "#6BA3D8"

LW_TRUTH = 1.5
LW_CORE = 2.5
LW_COMP = 1.8

ALPHA_TRUTH = 0.7
ALPHA_CORE = 1.0
ALPHA_COMP = 0.9


def _to_utc_dt(s: pd.Series) -> pd.Series:
    # Accept tz-aware or naive strings. Force UTC tz-aware.
    ts = pd.to_datetime(s, errors="coerce", utc=True)
    return ts


def _require_cols(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing cols {missing}. Has: {list(df.columns)[:40]} ...")


def load_truth(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    _require_cols(df, ["timestamp_utc", "power_norm"], "ground_truth")
    df = df.copy()
    df["timestamp_utc"] = _to_utc_dt(df["timestamp_utc"])
    df = df.dropna(subset=["timestamp_utc", "power_norm"]).sort_values("timestamp_utc")
    return df[["timestamp_utc", "power_norm"]]


def load_preds(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # expected from your phase1 pipelines
    _require_cols(df, ["timestamp_utc", "forecast_start", "predicted_power_norm"], path.name)

    df = df.copy()
    df["timestamp_utc"] = _to_utc_dt(df["timestamp_utc"])
    df["forecast_start"] = _to_utc_dt(df["forecast_start"])

    # hours_ahead may be missing in some variants, derive it if possible
    if "hours_ahead" not in df.columns:
        if "step_ahead" in df.columns:
            # infer resolution from step spacing if possible, fallback: assume 15-min
            # step_ahead usually counts 15-min steps in your pipeline
            df["hours_ahead"] = df["step_ahead"].astype(float) * 0.25
        else:
            # last resort: compute from timestamps
            df["hours_ahead"] = (df["timestamp_utc"] - df["forecast_start"]).dt.total_seconds() / 3600.0

    df = df.dropna(subset=["timestamp_utc", "forecast_start", "predicted_power_norm", "hours_ahead"])
    df = df.sort_values(["timestamp_utc", "hours_ahead"])
    return df


def stitch_operator_view(preds: pd.DataFrame) -> pd.DataFrame:
    """
    For each timestamp, take the smallest hours_ahead prediction.
    That corresponds to the most recent available forecast for that timestamp.
    """
    # Assure sorting so idxmin works correctly
    preds = preds.sort_values(["timestamp_utc", "hours_ahead"])
    idx = preds.groupby("timestamp_utc", sort=False)["hours_ahead"].idxmin()
    stitched = preds.loc[idx, ["timestamp_utc", "predicted_power_norm"]].sort_values("timestamp_utc")
    return stitched.reset_index(drop=True)


def join_on_time(truth: pd.DataFrame, series: pd.DataFrame, value_col: str) -> pd.DataFrame:
    df = truth.merge(series.rename(columns={"predicted_power_norm": value_col}), on="timestamp_utc", how="inner")
    return df


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    return float(np.sqrt(np.mean(d * d)))


def plot_week(
    outpath: Path,
    title: str,
    truth_week: pd.DataFrame,
    core_week: pd.DataFrame,
    rl_week: pd.DataFrame,
) -> None:
    plt.figure(figsize=(14, 5))
    plt.plot(truth_week["timestamp_utc"], truth_week["power_norm"], label="Ground Truth Plant 03",
             linewidth=LW_TRUTH, alpha=ALPHA_TRUTH, color=COL_TRUTH)
    plt.plot(core_week["timestamp_utc"], core_week["core"], label="MiRACLE v1.0 Core",
             linewidth=LW_CORE, alpha=ALPHA_CORE, color=COL_CORE)
    plt.plot(rl_week["timestamp_utc"], rl_week["rl"], label="RL Policy (minenv_v1)",
             linewidth=LW_COMP, alpha=ALPHA_COMP, color=COL_COMP)

    plt.title(title)
    plt.xlabel("Time (UTC)")
    plt.ylabel("Power (normalized)")
    plt.legend()
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_monthly_rmse(outpath: Path, joined: pd.DataFrame) -> None:
    df = joined.copy()
    df["month"] = df["timestamp_utc"].dt.to_period("M").dt.to_timestamp()

    rows = []
    for m, g in df.groupby("month"):
        y = g["power_norm"].to_numpy()
        core = g["core"].to_numpy()
        rl = g["rl"].to_numpy()
        rows.append((m, rmse(core, y), rmse(rl, y)))

    mm = pd.DataFrame(rows, columns=["month", "rmse_core", "rmse_rl"]).sort_values("month")

    plt.figure(figsize=(14, 6))
    plt.plot(mm["month"], mm["rmse_core"], marker="o", linewidth=LW_CORE, color=COL_CORE, label="MiRACLE v1.0 Core")
    plt.plot(mm["month"], mm["rmse_rl"], marker="s", linewidth=LW_COMP, color=COL_COMP, label="RL Policy (minenv_v1)")
    plt.title("Monthly RMSE (normalized power)")
    plt.xlabel("Month")
    plt.ylabel("RMSE (normalized power)")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_leadtime_rmse_0_24h(outpath: Path, truth: pd.DataFrame, core_preds: pd.DataFrame, rl_preds: pd.DataFrame) -> None:
    """
    Compute RMSE by lead-time hour bucket for 0–24h.
    Uses raw prediction tables merged to truth at timestamp_utc.
    """
    def lead_rmse(preds: pd.DataFrame, label: str) -> pd.DataFrame:
        df = preds.merge(truth, on="timestamp_utc", how="inner")
        df = df[(df["hours_ahead"] >= 0.0) & (df["hours_ahead"] <= 24.0)].copy()
        # bucket to integer hours (operator-ish)
        df["h"] = np.floor(df["hours_ahead"]).astype(int)
        rows = []
        for h, g in df.groupby("h"):
            y = g["power_norm"].to_numpy()
            p = g["predicted_power_norm"].to_numpy()
            rows.append((h, rmse(p, y)))
        out = pd.DataFrame(rows, columns=["h", f"rmse_{label}"]).sort_values("h")
        return out

    a = lead_rmse(core_preds, "core")
    b = lead_rmse(rl_preds, "rl")
    m = a.merge(b, on="h", how="outer").sort_values("h")

    plt.figure(figsize=(14, 6))
    plt.plot(m["h"], m["rmse_core"], marker="o", linewidth=LW_CORE, color=COL_CORE, label="MiRACLE v1.0 Core")
    plt.plot(m["h"], m["rmse_rl"], marker="s", linewidth=LW_COMP, color=COL_COMP, label="RL Policy (minenv_v1)")
    plt.title("Lead-Time RMSE Curve (0–24h)")
    plt.xlabel("Hours ahead")
    plt.ylabel("RMSE")
    plt.legend()
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_tail_abs_error_hist(outpath: Path, joined: pd.DataFrame, clip: float = 0.9) -> None:
    df = joined.copy()
    df["abs_err_core"] = np.abs(df["core"] - df["power_norm"])
    df["abs_err_rl"] = np.abs(df["rl"] - df["power_norm"])

    core = np.clip(df["abs_err_core"].to_numpy(), 0, clip)
    rl = np.clip(df["abs_err_rl"].to_numpy(), 0, clip)

    plt.figure(figsize=(12, 6))
    plt.hist(core, bins=60, alpha=0.6, label="MiRACLE v1.0 Core", color=COL_CORE)
    plt.hist(rl, bins=60, alpha=0.6, label="RL Policy (minenv_v1)", color=COL_COMP)
    plt.title("Absolute Error Histogram (clipped)")
    plt.xlabel("Abs error (clipped)")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=300)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True, type=Path)
    ap.add_argument("--core", required=True, type=Path)
    ap.add_argument("--rl", required=True, type=Path)
    ap.add_argument("--outdir", required=True, type=Path)

    # fixed thesis windows (match your earlier plots)
    ap.add_argument("--winter-start", default="2024-01-10", type=str)
    ap.add_argument("--winter-end", default="2024-01-17", type=str)
    ap.add_argument("--summer-start", default="2024-07-01", type=str)
    ap.add_argument("--summer-end", default="2024-07-08", type=str)

    args = ap.parse_args()

    truth = load_truth(args.truth)
    core_preds = load_preds(args.core)
    rl_preds = load_preds(args.rl)

    core_stitched = stitch_operator_view(core_preds)
    rl_stitched = stitch_operator_view(rl_preds)

    core_join = join_on_time(truth, core_stitched, "core")
    rl_join = join_on_time(truth, rl_stitched, "rl")
    joined = core_join.merge(rl_join[["timestamp_utc", "rl"]], on="timestamp_utc", how="inner")

    # weekly windows
    w0 = pd.to_datetime(args.winter_start, utc=True)
    w1 = pd.to_datetime(args.winter_end, utc=True)
    s0 = pd.to_datetime(args.summer_start, utc=True)
    s1 = pd.to_datetime(args.summer_end, utc=True)

    def cut(df: pd.DataFrame, t0: pd.Timestamp, t1: pd.Timestamp) -> pd.DataFrame:
        return df[(df["timestamp_utc"] >= t0) & (df["timestamp_utc"] < t1)].copy()

    winter = cut(joined, w0, w1)
    summer = cut(joined, s0, s1)

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    # Week plots
    plot_week(
        outdir / "facets_case_winter_week_core_vs_rl.png",
        "Case Study: Winter Week (Core vs RL)",
        winter[["timestamp_utc", "power_norm"]],
        winter[["timestamp_utc", "core"]],
        winter[["timestamp_utc", "rl"]],
    )
    plot_week(
        outdir / "facets_case_summer_week_core_vs_rl.png",
        "Case Study: Summer Week (Core vs RL)",
        summer[["timestamp_utc", "power_norm"]],
        summer[["timestamp_utc", "core"]],
        summer[["timestamp_utc", "rl"]],
    )

    # Monthly RMSE
    plot_monthly_rmse(outdir / "monthly_rmse_core_vs_rl.png", joined)

    # Lead-time curve
    plot_leadtime_rmse_0_24h(outdir / "facets_leadtime_rmse_curve_0_24h_core_vs_rl.png", truth, core_preds, rl_preds)

    # Tail histogram
    plot_tail_abs_error_hist(outdir / "tails_abs_error_hist_core_vs_rl.png", joined)

    # quick sanity stats
    y = joined["power_norm"].to_numpy()
    core = joined["core"].to_numpy()
    rl = joined["rl"].to_numpy()
    stats = {
        "n_points": int(len(joined)),
        "rmse_core": rmse(core, y),
        "rmse_rl": rmse(rl, y),
        "core_minus_rl": rmse(core, y) - rmse(rl, y),
    }
    (outdir / "core_vs_rl_quick_stats.json").write_text(pd.Series(stats).to_json(indent=2))


if __name__ == "__main__":
    main()
