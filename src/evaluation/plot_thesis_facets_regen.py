#!/usr/bin/env python3
"""
Canonical thesis plot regenerator for MiRACLE.

Produces (matching your shown figures):
1) Case Study: Winter Week (4 panels)
2) Case Study: Summer Week (4 panels)
3) Absolute Error Histogram (4 panels)
4) Monthly RMSE (all models on one plot)
5) Lead-Time RMSE Curve (0–24h) (4 panels)

Key guarantees:
- Uses timestamp_utc as canonical time key
- Converts to tz-aware UTC
- "Most recent" stitching is deterministic:
  - If hours_ahead exists: smallest hours_ahead per timestamp_utc
  - Else: newest forecast_start per timestamp_utc
- Color mapping is fixed and explicit

Assumptions (based on your freeze artifacts):
- Truth has: timestamp_utc, power_norm (optionally is_daylight)
- Predictions have: timestamp_utc, predicted_power_norm
  (optionally hours_ahead, forecast_start)

Run example is provided below.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------
# Thesis color constants
# -----------------------
COL_TRUTH = "#888888"   # grey
COL_CORE  = "#00AA00"   # green
COL_COMP  = "#6BA3D8"   # light blue

DPI = 300


# -----------------------
# Helpers
# -----------------------
def _must_exist(p: Path) -> Path:
    if not p.exists():
        raise FileNotFoundError(str(p))
    return p


def _read_parquet(p: Path) -> pd.DataFrame:
    df = pd.read_parquet(p)
    if "timestamp_utc" not in df.columns:
        raise ValueError(f"{p} missing timestamp_utc. cols={list(df.columns)[:30]}")
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp_utc"])
    return df


def _stitch_most_recent(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deterministic single-value per timestamp_utc.
    Prefer smallest hours_ahead if available, else newest forecast_start if available.
    """
    if "predicted_power_norm" not in df.columns:
        raise ValueError(f"missing predicted_power_norm. cols={list(df.columns)[:30]}")

    cols = df.columns

    if "hours_ahead" in cols:
        d = df.copy()
        d["hours_ahead"] = pd.to_numeric(d["hours_ahead"], errors="coerce")
        d = d.dropna(subset=["hours_ahead"])
        d = d.sort_values(["timestamp_utc", "hours_ahead"], ascending=[True, True])
        out = d.groupby("timestamp_utc", as_index=False).first()
        return out

    if "forecast_start" in cols:
        d = df.copy()
        d["forecast_start"] = pd.to_datetime(d["forecast_start"], utc=True, errors="coerce")
        d = d.dropna(subset=["forecast_start"])
        d = d.sort_values(["timestamp_utc", "forecast_start"], ascending=[True, False])
        out = d.groupby("timestamp_utc", as_index=False).first()
        return out

    # Fallback: if neither exists, just keep first occurrence per timestamp_utc
    d = df.sort_values(["timestamp_utc"])
    return d.groupby("timestamp_utc", as_index=False).first()


def _align(pred_stitched: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    if "power_norm" not in truth.columns:
        raise ValueError(f"truth missing power_norm. cols={list(truth.columns)[:30]}")
    t = truth[["timestamp_utc", "power_norm"]].rename(columns={"power_norm": "y_true"})
    j = pred_stitched.merge(t, on="timestamp_utc", how="inner")
    return j


def _filter_window(j: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    m = (j["timestamp_utc"] >= start) & (j["timestamp_utc"] <= end)
    return j.loc[m].sort_values("timestamp_utc")


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _month_key(ts: pd.Series) -> pd.Series:
    # ts must be datetime64[ns, UTC]
    return ts.dt.strftime("%Y-%m")


@dataclass
class ModelSpec:
    name: str
    path: Path
    color: str


# -----------------------
# Plot builders
# -----------------------
def plot_case_study_4panel(
    outpath: Path,
    truth: pd.DataFrame,
    core_stitched: pd.DataFrame,
    comps: List[Tuple[str, pd.DataFrame]],
    start: pd.Timestamp,
    end: pd.Timestamp,
    title: str,
) -> None:
    # Prepare truth series
    truth_small = truth[["timestamp_utc", "power_norm"]].rename(columns={"power_norm": "y_true"})
    truth_win = truth_small[(truth_small["timestamp_utc"] >= start) & (truth_small["timestamp_utc"] <= end)].sort_values("timestamp_utc")

    core_j = _align(core_stitched, truth).pipe(_filter_window, start, end)

    fig = plt.figure(figsize=(18, 9), dpi=DPI)
    fig.suptitle(title, fontsize=20, fontweight="bold")

    for i, (comp_name, comp_stitched) in enumerate(comps):
        ax = fig.add_subplot(2, 2, i + 1)
        comp_j = _align(comp_stitched, truth).pipe(_filter_window, start, end)

        ax.plot(truth_win["timestamp_utc"], truth_win["y_true"], color=COL_TRUTH, linewidth=1.8, alpha=0.7, label="Ground Truth Plant 03")
        ax.plot(core_j["timestamp_utc"], core_j["predicted_power_norm"], color=COL_CORE, linewidth=3.0, alpha=1.0, label="MiRACLE v1.0 Core")
        ax.plot(comp_j["timestamp_utc"], comp_j["predicted_power_norm"], color=COL_COMP, linewidth=1.8, alpha=0.95, label=comp_name)

        ax.set_title(comp_name, fontsize=13, fontweight="bold")
        ax.set_xlabel("Time (UTC)")
        ax.set_ylabel("Power (normalized)")
        ax.grid(alpha=0.3, linestyle=":")
        ax.legend(framealpha=0.9)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(outpath, dpi=DPI)
    plt.close()


def plot_abs_error_hist_4panel(
    outpath: Path,
    truth: pd.DataFrame,
    core_stitched: pd.DataFrame,
    comps: List[Tuple[str, pd.DataFrame]],
    title: str,
    clip_max: float = 1.0,
) -> None:
    core_j = _align(core_stitched, truth)
    core_abs = np.abs(core_j["y_true"].values - core_j["predicted_power_norm"].values)

    fig = plt.figure(figsize=(18, 9), dpi=DPI)
    fig.suptitle(title, fontsize=20, fontweight="bold")

    bins = np.linspace(0, clip_max, 80)

    for i, (comp_name, comp_stitched) in enumerate(comps):
        ax = fig.add_subplot(2, 2, i + 1)
        comp_j = _align(comp_stitched, truth)
        comp_abs = np.abs(comp_j["y_true"].values - comp_j["predicted_power_norm"].values)

        ax.hist(comp_abs.clip(0, clip_max), bins=bins, alpha=0.6, edgecolor="black", linewidth=0.5, label=comp_name)
        ax.hist(core_abs.clip(0, clip_max), bins=bins, alpha=0.7, edgecolor="black", linewidth=0.5, label="MiRACLE v1.0 Core")

        ax.set_title(comp_name, fontsize=13, fontweight="bold")
        ax.set_xlabel("Abs error (clipped)")
        ax.set_ylabel("Count")
        ax.grid(alpha=0.3, linestyle=":")
        ax.legend(framealpha=0.9)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(outpath, dpi=DPI)
    plt.close()


def plot_monthly_rmse_all(
    outpath: Path,
    truth: pd.DataFrame,
    stitched_models: List[ModelSpec],
) -> None:
    truth_small = truth[["timestamp_utc", "power_norm"]].rename(columns={"power_norm": "y_true"})
    fig = plt.figure(figsize=(16, 6), dpi=DPI)
    ax = fig.add_subplot(1, 1, 1)

    # Build month grid from truth
    months = sorted(truth_small["timestamp_utc"].dt.strftime("%Y-%m").unique().tolist())

    for ms in stitched_models:
        j = _align(ms_df_map[ms.name], truth)  # uses global map filled in main
        j["month"] = _month_key(j["timestamp_utc"])

        rmse_by_m = []
        for m in months:
            s = j[j["month"] == m]
            if len(s) == 0:
                rmse_by_m.append(np.nan)
            else:
                rmse_by_m.append(_rmse(s["y_true"].values, s["predicted_power_norm"].values))

        ax.plot(months, rmse_by_m, marker="o", linewidth=2.0, label=ms.name)

    ax.set_title("Monthly RMSE (normalized power)", fontsize=18, fontweight="bold")
    ax.set_xlabel("Month", fontsize=14, fontweight="bold")
    ax.set_ylabel("RMSE (normalized power)", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.3, linestyle=":")
    ax.legend(framealpha=0.9)
    plt.xticks(rotation=45, ha="right")

    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=DPI)
    plt.close()


def plot_leadtime_rmse_4panel(
    outpath: Path,
    truth_with_sun: pd.DataFrame,
    core_full: pd.DataFrame,
    comps_full: List[Tuple[str, pd.DataFrame]],
    title: str,
    hours_max: float = 24.0,
    daylight_only: bool = True,
) -> None:
    """
    Lead-time RMSE curve using non-stitched predictions, grouped by hours_ahead.
    Requires hours_ahead in prediction parquet.
    Optionally filters to is_daylight==1 from truth_with_sun.
    """
    if "is_daylight" not in truth_with_sun.columns:
        daylight_only = False

    truth_small = truth_with_sun[["timestamp_utc", "power_norm"] + (["is_daylight"] if "is_daylight" in truth_with_sun.columns else [])].copy()
    truth_small = truth_small.rename(columns={"power_norm": "y_true"})

    def curve(df_pred: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if "hours_ahead" not in df_pred.columns:
            raise ValueError("lead-time plot requires hours_ahead column in prediction parquet")
        d = df_pred.copy()
        d["hours_ahead"] = pd.to_numeric(d["hours_ahead"], errors="coerce")
        d = d.dropna(subset=["hours_ahead"])
        d = d[(d["hours_ahead"] >= 0.0) & (d["hours_ahead"] <= hours_max)]
        j = d.merge(truth_small, on="timestamp_utc", how="inner")

        if daylight_only:
            j = j[j["is_daylight"] == 1]

        xs = np.sort(j["hours_ahead"].unique())
        ys = []
        for x in xs:
            s = j[j["hours_ahead"] == x]
            ys.append(_rmse(s["y_true"].values, s["predicted_power_norm"].values))
        return xs, np.asarray(ys, dtype=float)

    fig = plt.figure(figsize=(18, 9), dpi=DPI)
    fig.suptitle(title, fontsize=20, fontweight="bold")

    # Each panel is Core vs one comparison
    for i, (comp_name, comp_full) in enumerate(comps_full):
        ax = fig.add_subplot(2, 2, i + 1)

        x_core, y_core = curve(core_full)
        x_cmp, y_cmp = curve(comp_full)

        ax.plot(x_cmp, y_cmp, marker="s", linewidth=2.0, alpha=0.75, label=comp_name)
        ax.plot(x_core, y_core, marker="o", linewidth=3.0, alpha=1.0, label="MiRACLE v1.0 Core")

        ax.set_title(comp_name, fontsize=13, fontweight="bold")
        ax.set_xlabel("Hours ahead")
        ax.set_ylabel("RMSE")
        ax.grid(alpha=0.3, linestyle=":")
        ax.legend(framealpha=0.9)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(outpath, dpi=DPI)
    plt.close()


# Global map for monthly plot convenience (filled in main)
ms_df_map: Dict[str, pd.DataFrame] = {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True, help="Truth parquet with timestamp_utc,power_norm")
    ap.add_argument("--truth-with-sun", required=True, help="Truth parquet with timestamp_utc,power_norm,is_daylight")
    ap.add_argument("--core", required=True, help="Core predictions parquet")
    ap.add_argument("--pvlib", required=True, help="PVLib-only predictions parquet")
    ap.add_argument("--tft", required=True, help="TFT-only predictions parquet")
    ap.add_argument("--short", required=True, help="Short-only predictions parquet")
    ap.add_argument("--long", required=True, help="Long-only predictions parquet")
    ap.add_argument("--outdir", required=True, help="Output directory for figures")

    # Fixed windows to match your shown plots
    ap.add_argument("--winter-start", default="2024-01-10T00:00:00Z")
    ap.add_argument("--winter-end",   default="2024-01-17T00:00:00Z")
    ap.add_argument("--summer-start", default="2024-07-01T00:00:00Z")
    ap.add_argument("--summer-end",   default="2024-07-08T00:00:00Z")

    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    truth = _read_parquet(_must_exist(Path(args.truth)))
    truth_sun = _read_parquet(_must_exist(Path(args.truth_with_sun)))

    core_full  = _read_parquet(_must_exist(Path(args.core)))
    pvlib_full = _read_parquet(_must_exist(Path(args.pvlib)))
    tft_full   = _read_parquet(_must_exist(Path(args.tft)))
    short_full = _read_parquet(_must_exist(Path(args.short)))
    long_full  = _read_parquet(_must_exist(Path(args.long)))

    core_st  = _stitch_most_recent(core_full)
    pvlib_st = _stitch_most_recent(pvlib_full)
    tft_st   = _stitch_most_recent(tft_full)
    short_st = _stitch_most_recent(short_full)
    long_st  = _stitch_most_recent(long_full)

    # Fill monthly map
    global ms_df_map
    ms_df_map = {
        "MiRACLE v1.0 Core": core_st,
        "PVLib-Physics-Only": pvlib_st,
        "TFT-Only": tft_st,
        "Short-TFT-Only": short_st,
        "Long-TFT-Only": long_st,
    }

    # 4-panel comparisons list in the exact order of your figure
    comps_st = [
        ("PVLib-Physics-Only", pvlib_st),
        ("TFT-Only", tft_st),
        ("Short-TFT-Only", short_st),
        ("Long-TFT-Only", long_st),
    ]
    comps_full = [
        ("PVLib-Physics-Only", pvlib_full),
        ("TFT-Only", tft_full),
        ("Short-TFT-Only", short_full),
        ("Long-TFT-Only", long_full),
    ]

    winter_start = pd.to_datetime(args.winter_start, utc=True)
    winter_end   = pd.to_datetime(args.winter_end, utc=True)
    summer_start = pd.to_datetime(args.summer_start, utc=True)
    summer_end   = pd.to_datetime(args.summer_end, utc=True)

    plot_case_study_4panel(
        outpath=outdir / "facets_case_winter_week.png",
        truth=truth,
        core_stitched=core_st,
        comps=comps_st,
        start=winter_start,
        end=winter_end,
        title="Case Study: Winter Week",
    )

    plot_case_study_4panel(
        outpath=outdir / "facets_case_summer_week.png",
        truth=truth,
        core_stitched=core_st,
        comps=comps_st,
        start=summer_start,
        end=summer_end,
        title="Case Study: Summer Week",
    )

    plot_abs_error_hist_4panel(
        outpath=outdir / "facets_abs_error_hist.png",
        truth=truth,
        core_stitched=core_st,
        comps=comps_st,
        title="Absolute Error Histogram",
        clip_max=1.0,
    )

    plot_monthly_rmse_all(
        outpath=outdir / "monthly_rmse_all_models.png",
        truth=truth,
        stitched_models=[
            ModelSpec("MiRACLE v1.0 Core", Path(args.core), COL_CORE),
            ModelSpec("PVLib-Physics-Only", Path(args.pvlib), COL_COMP),
            ModelSpec("TFT-Only", Path(args.tft), COL_COMP),
            ModelSpec("Short-TFT-Only", Path(args.short), COL_COMP),
            ModelSpec("Long-TFT-Only", Path(args.long), COL_COMP),
        ],
    )

    plot_leadtime_rmse_4panel(
        outpath=outdir / "facets_leadtime_rmse_curve_0_24h.png",
        truth_with_sun=truth_sun,
        core_full=core_full,
        comps_full=comps_full,
        title="Lead-Time RMSE Curve (0–24h)",
        hours_max=24.0,
        daylight_only=True,
    )

    print(f"[OK] wrote figures to: {outdir}")


if __name__ == "__main__":
    main()
