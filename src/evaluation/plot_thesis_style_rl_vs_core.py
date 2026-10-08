#!/usr/bin/env python3
"""
Generate thesis-style RL vs MiRACLE Core vs Ground Truth plots
WITHOUT touching the existing figure generators.

Outputs go to: <outdir>/rl_vs_core/

Figures:
- case_winter_week_core_vs_rl.png
- case_summer_week_core_vs_rl.png
- monthly_rmse_core_vs_rl.png
- leadtime_rmse_curve_core_vs_rl_0_24h.png
- tails_abs_error_hist_core_vs_rl.png
- tails_abs_error_quantiles_core_vs_rl.png

Robustness:
- Uses timestamp_utc as join key (tz-aware UTC)
- For stitching: smallest hours_ahead per timestamp_utc, else newest forecast_start
- For lead-time curve: uses hours_ahead if available, else derives from step_ahead (15-min => *0.25h)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Match your thesis style
COL_TRUTH = "#888888"   # grey
COL_CORE  = "#00AA00"   # green (MiRACLE Core)
COL_RL    = "#6BA3D8"   # same light-blue family you used in other comparisons

DPI = 300


def _must_exist(p: Path) -> Path:
    if not p.exists():
        raise FileNotFoundError(str(p))
    return p


def _read(p: Path) -> pd.DataFrame:
    df = pd.read_parquet(p)
    if "timestamp_utc" not in df.columns:
        raise ValueError(f"{p} missing timestamp_utc. cols={list(df.columns)[:30]}")
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp_utc"])
    return df


def _ensure_hours_ahead(df: pd.DataFrame) -> pd.DataFrame:
    if "hours_ahead" in df.columns:
        d = df.copy()
        d["hours_ahead"] = pd.to_numeric(d["hours_ahead"], errors="coerce")
        return d
    if "step_ahead" in df.columns:
        d = df.copy()
        d["step_ahead"] = pd.to_numeric(d["step_ahead"], errors="coerce")
        # assume 15-min steps for step_ahead in phase1 outputs
        d["hours_ahead"] = d["step_ahead"] * 0.25
        return d
    return df


def _stitch(df: pd.DataFrame) -> pd.DataFrame:
    if "predicted_power_norm" not in df.columns:
        raise ValueError(f"missing predicted_power_norm. cols={list(df.columns)[:30]}")

    d = df.copy()

    if "hours_ahead" in d.columns:
        d["hours_ahead"] = pd.to_numeric(d["hours_ahead"], errors="coerce")
        d = d.dropna(subset=["hours_ahead"])
        d = d.sort_values(["timestamp_utc", "hours_ahead"], ascending=[True, True])
        return d.groupby("timestamp_utc", as_index=False).first()

    if "forecast_start" in d.columns:
        d["forecast_start"] = pd.to_datetime(d["forecast_start"], utc=True, errors="coerce")
        d = d.dropna(subset=["forecast_start"])
        d = d.sort_values(["timestamp_utc", "forecast_start"], ascending=[True, False])
        return d.groupby("timestamp_utc", as_index=False).first()

    d = d.sort_values("timestamp_utc")
    return d.groupby("timestamp_utc", as_index=False).first()


def _align(pred_stitched: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    if "power_norm" not in truth.columns:
        raise ValueError(f"truth missing power_norm. cols={list(truth.columns)[:30]}")
    t = truth[["timestamp_utc", "power_norm"]].rename(columns={"power_norm": "y_true"})
    j = pred_stitched.merge(t, on="timestamp_utc", how="inner")
    return j


def _rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def _month_key(ts: pd.Series) -> pd.Series:
    return ts.dt.strftime("%Y-%m")


def _window(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    m = (df["timestamp_utc"] >= start) & (df["timestamp_utc"] <= end)
    return df.loc[m].sort_values("timestamp_utc")


def plot_case_week(outpath: Path, truth: pd.DataFrame, core_st: pd.DataFrame, rl_st: pd.DataFrame,
                   start: pd.Timestamp, end: pd.Timestamp, title: str) -> None:
    truth_small = truth[["timestamp_utc", "power_norm"]].rename(columns={"power_norm": "y_true"})
    t = _window(truth_small, start, end)

    core_j = _window(_align(core_st, truth), start, end)
    rl_j   = _window(_align(rl_st, truth), start, end)

    fig = plt.figure(figsize=(18, 6), dpi=DPI)
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(t["timestamp_utc"], t["y_true"], color=COL_TRUTH, linewidth=2.0, alpha=0.7, label="Ground Truth Plant 03")
    ax.plot(core_j["timestamp_utc"], core_j["predicted_power_norm"], color=COL_CORE, linewidth=3.0, label="MiRACLE v1.0 Core")
    ax.plot(rl_j["timestamp_utc"], rl_j["predicted_power_norm"], color=COL_RL, linewidth=2.0, label="RL Policy")

    ax.set_title(title, fontsize=18, fontweight="bold")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Power (normalized)")
    ax.grid(alpha=0.3, linestyle=":")
    ax.legend(framealpha=0.9)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=DPI)
    plt.close()


def plot_monthly_rmse(outpath: Path, truth: pd.DataFrame, core_st: pd.DataFrame, rl_st: pd.DataFrame) -> None:
    core_j = _align(core_st, truth)
    rl_j   = _align(rl_st, truth)

    core_j["month"] = _month_key(core_j["timestamp_utc"])
    rl_j["month"]   = _month_key(rl_j["timestamp_utc"])

    months = sorted(pd.concat([core_j["month"], rl_j["month"]]).unique().tolist())

    core_rmse = []
    rl_rmse = []
    for m in months:
        c = core_j[core_j["month"] == m]
        r = rl_j[rl_j["month"] == m]
        core_rmse.append(_rmse(c["y_true"].values, c["predicted_power_norm"].values) if len(c) else np.nan)
        rl_rmse.append(_rmse(r["y_true"].values, r["predicted_power_norm"].values) if len(r) else np.nan)

    fig = plt.figure(figsize=(16, 6), dpi=DPI)
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(months, core_rmse, marker="o", linewidth=3.0, color=COL_CORE, label="MiRACLE v1.0 Core")
    ax.plot(months, rl_rmse, marker="s", linewidth=2.0, color=COL_RL, alpha=0.9, label="RL Policy")

    ax.set_title("Monthly RMSE (MiRACLE Core vs RL)", fontsize=18, fontweight="bold")
    ax.set_xlabel("Month", fontsize=14, fontweight="bold")
    ax.set_ylabel("RMSE (normalized power)", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.3, linestyle=":")
    ax.legend(framealpha=0.9)
    plt.xticks(rotation=45, ha="right")

    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=DPI)
    plt.close()


def plot_leadtime_rmse(outpath: Path, truth_with_sun: pd.DataFrame,
                      core_full: pd.DataFrame, rl_full: pd.DataFrame,
                      hours_max: float = 24.0, daylight_only: bool = True) -> None:
    if "is_daylight" not in truth_with_sun.columns:
        daylight_only = False

    truth_small = truth_with_sun[["timestamp_utc", "power_norm"] + (["is_daylight"] if "is_daylight" in truth_with_sun.columns else [])].copy()
    truth_small = truth_small.rename(columns={"power_norm": "y_true"})

    def curve(df_pred: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        d = _ensure_hours_ahead(df_pred)
        if "hours_ahead" not in d.columns:
            raise ValueError("Need hours_ahead or step_ahead in prediction parquet to build lead-time curve.")
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

    x_c, y_c = curve(core_full)
    x_r, y_r = curve(rl_full)

    fig = plt.figure(figsize=(14, 6), dpi=DPI)
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(x_c, y_c, marker="o", linewidth=3.0, color=COL_CORE, label="MiRACLE v1.0 Core")
    ax.plot(x_r, y_r, marker="s", linewidth=2.0, color=COL_RL, alpha=0.9, label="RL Policy")

    ax.set_title("Lead-Time RMSE Curve (0–24h), Core vs RL", fontsize=18, fontweight="bold")
    ax.set_xlabel("Hours ahead")
    ax.set_ylabel("RMSE")
    ax.grid(alpha=0.3, linestyle=":")
    ax.legend(framealpha=0.9)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=DPI)
    plt.close()


def plot_tails(outdir: Path, truth: pd.DataFrame, core_st: pd.DataFrame, rl_st: pd.DataFrame,
              truth_with_sun: Optional[pd.DataFrame] = None, clip_max: float = 1.0) -> None:
    # Full-sample tails
    core_j = _align(core_st, truth)
    rl_j   = _align(rl_st, truth)

    core_abs = np.abs(core_j["y_true"].values - core_j["predicted_power_norm"].values)
    rl_abs   = np.abs(rl_j["y_true"].values - rl_j["predicted_power_norm"].values)

    # Histogram
    fig = plt.figure(figsize=(14, 6), dpi=DPI)
    ax = fig.add_subplot(1, 1, 1)
    bins = np.linspace(0, clip_max, 80)

    ax.hist(core_abs.clip(0, clip_max), bins=bins, alpha=0.75, edgecolor="black", linewidth=0.5, color=COL_CORE, label="MiRACLE v1.0 Core")
    ax.hist(rl_abs.clip(0, clip_max), bins=bins, alpha=0.45, edgecolor="black", linewidth=0.5, color=COL_RL, label="RL Policy")

    ax.set_title("Tail View: Absolute Error Histogram (clipped)", fontsize=18, fontweight="bold")
    ax.set_xlabel("Abs error (clipped)")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.3, linestyle=":")
    ax.legend(framealpha=0.9)

    outdir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outdir / "tails_abs_error_hist_core_vs_rl.png", dpi=DPI)
    plt.close()

    # Quantiles (optionally daylight only)
    def qstats(abs_err: np.ndarray) -> Tuple[float, float, float]:
        return (float(np.quantile(abs_err, 0.90)),
                float(np.quantile(abs_err, 0.95)),
                float(np.quantile(abs_err, 0.99)))

    core_q_all = qstats(core_abs)
    rl_q_all   = qstats(rl_abs)

    # daylight quantiles if available
    core_q_day = None
    rl_q_day = None
    if truth_with_sun is not None and "is_daylight" in truth_with_sun.columns:
        tday = truth_with_sun[["timestamp_utc", "is_daylight"]].copy()
        tday["timestamp_utc"] = pd.to_datetime(tday["timestamp_utc"], utc=True, errors="coerce")
        tday = tday.dropna(subset=["timestamp_utc"])

        core_day = core_j.merge(tday, on="timestamp_utc", how="inner")
        rl_day   = rl_j.merge(tday, on="timestamp_utc", how="inner")

        core_abs_day = np.abs(core_day.loc[core_day["is_daylight"] == 1, "y_true"].values -
                              core_day.loc[core_day["is_daylight"] == 1, "predicted_power_norm"].values)
        rl_abs_day = np.abs(rl_day.loc[rl_day["is_daylight"] == 1, "y_true"].values -
                            rl_day.loc[rl_day["is_daylight"] == 1, "predicted_power_norm"].values)

        if len(core_abs_day) > 0 and len(rl_abs_day) > 0:
            core_q_day = qstats(core_abs_day)
            rl_q_day   = qstats(rl_abs_day)

    # Bar plot quantiles
    fig = plt.figure(figsize=(14, 6), dpi=DPI)
    ax = fig.add_subplot(1, 1, 1)

    labels = ["P90", "P95", "P99"]
    x = np.arange(len(labels))
    w = 0.35

    ax.bar(x - w/2, core_q_all, width=w, color=COL_CORE, alpha=0.85, label="Core (all)")
    ax.bar(x + w/2, rl_q_all, width=w, color=COL_RL, alpha=0.85, label="RL (all)")

    if core_q_day is not None and rl_q_day is not None:
        # overlay as points so we do not clutter
        ax.scatter(x - w/2, core_q_day, marker="o", s=60, color="black", label="Core (daylight)")
        ax.scatter(x + w/2, rl_q_day, marker="s", s=60, color="black", label="RL (daylight)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Tail View: Absolute Error Quantiles (Core vs RL)", fontsize=18, fontweight="bold")
    ax.set_ylabel("Abs error (normalized)")
    ax.grid(alpha=0.3, linestyle=":")
    ax.legend(framealpha=0.9)

    plt.tight_layout()
    plt.savefig(outdir / "tails_abs_error_quantiles_core_vs_rl.png", dpi=DPI)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True)
    ap.add_argument("--truth-with-sun", required=False, default=None)
    ap.add_argument("--core", required=True)
    ap.add_argument("--rl", required=True)
    ap.add_argument("--outdir", required=True)

    # Lock the same weeks you already used
    ap.add_argument("--winter-start", default="2024-01-10T00:00:00Z")
    ap.add_argument("--winter-end",   default="2024-01-17T00:00:00Z")
    ap.add_argument("--summer-start", default="2024-07-01T00:00:00Z")
    ap.add_argument("--summer-end",   default="2024-07-08T00:00:00Z")

    args = ap.parse_args()

    outroot = Path(args.outdir)
    out = outroot / "rl_vs_core"
    out.mkdir(parents=True, exist_ok=True)

    truth = _read(_must_exist(Path(args.truth)))
    truth_sun = None
    if args.truth_with_sun:
        truth_sun = _read(_must_exist(Path(args.truth_with_sun)))

    core_full = _read(_must_exist(Path(args.core)))
    rl_full   = _read(_must_exist(Path(args.rl)))

    core_full = _ensure_hours_ahead(core_full)
    rl_full   = _ensure_hours_ahead(rl_full)

    core_st = _stitch(core_full)
    rl_st   = _stitch(rl_full)

    winter_start = pd.to_datetime(args.winter_start, utc=True)
    winter_end   = pd.to_datetime(args.winter_end, utc=True)
    summer_start = pd.to_datetime(args.summer_start, utc=True)
    summer_end   = pd.to_datetime(args.summer_end, utc=True)

    plot_case_week(
        outpath=out / "case_winter_week_core_vs_rl.png",
        truth=truth,
        core_st=core_st,
        rl_st=rl_st,
        start=winter_start,
        end=winter_end,
        title="Case Study: Winter Week (Ground Truth vs Core vs RL)",
    )

    plot_case_week(
        outpath=out / "case_summer_week_core_vs_rl.png",
        truth=truth,
        core_st=core_st,
        rl_st=rl_st,
        start=summer_start,
        end=summer_end,
        title="Case Study: Summer Week (Ground Truth vs Core vs RL)",
    )

    plot_monthly_rmse(
        outpath=out / "monthly_rmse_core_vs_rl.png",
        truth=truth,
        core_st=core_st,
        rl_st=rl_st,
    )

    if truth_sun is not None:
        plot_leadtime_rmse(
            outpath=out / "leadtime_rmse_curve_core_vs_rl_0_24h.png",
            truth_with_sun=truth_sun,
            core_full=core_full,
            rl_full=rl_full,
            hours_max=24.0,
            daylight_only=True,
        )
    else:
        # Still try without daylight filter
        truth_sun_fallback = truth.copy()
        plot_leadtime_rmse(
            outpath=out / "leadtime_rmse_curve_core_vs_rl_0_24h.png",
            truth_with_sun=truth_sun_fallback,
            core_full=core_full,
            rl_full=rl_full,
            hours_max=24.0,
            daylight_only=False,
        )

    plot_tails(
        outdir=out,
        truth=truth,
        core_st=core_st,
        rl_st=rl_st,
        truth_with_sun=truth_sun,
        clip_max=1.0,
    )

    print(f"[OK] wrote RL vs Core figures to: {out}")


if __name__ == "__main__":
    main()
