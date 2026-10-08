from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Helpers
# -----------------------------
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true[m] - y_pred[m]) ** 2)))


def most_recent_stitch(preds: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only 1 prediction per timestamp_utc by selecting the row with the latest forecast_start.
    Assumes preds has: timestamp_utc, forecast_start, predicted_power_norm.
    """
    df = preds.copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df["forecast_start"] = pd.to_datetime(df["forecast_start"], utc=True)

    # Sort so latest forecast_start comes last, then drop duplicates keeping last
    df = df.sort_values(["timestamp_utc", "forecast_start"])
    df = df.drop_duplicates(subset=["timestamp_utc"], keep="last")
    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    return df


def load_preds(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    need = {"timestamp_utc", "forecast_start", "predicted_power_norm"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}. Has: {list(df.columns)[:30]}")
    df = df[["timestamp_utc", "forecast_start", "predicted_power_norm"]].copy()
    return df


def load_preds_with_step(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    need = {"timestamp_utc", "forecast_start", "predicted_power_norm", "step_ahead"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}. Has: {list(df.columns)[:30]}")
    df = df[["timestamp_utc", "forecast_start", "predicted_power_norm", "step_ahead"]].copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df["forecast_start"] = pd.to_datetime(df["forecast_start"], utc=True)
    return df


def join_gt(stitched: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    """
    Join stitched predictions with ground truth on timestamp_utc.
    """
    g = gt.copy()
    if "timestamp_utc" not in g.columns or "power_norm" not in g.columns:
        raise ValueError(f"Ground truth must have ['timestamp_utc','power_norm'], has: {list(g.columns)}")
    g["timestamp_utc"] = pd.to_datetime(g["timestamp_utc"], utc=True)
    g = g[["timestamp_utc", "power_norm"]].copy()

    out = stitched.merge(g, on="timestamp_utc", how="inner")
    return out


def savefig(outdir: Path, name: str):
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / name
    plt.tight_layout()
    plt.savefig(p, dpi=200)
    plt.close()
    print("[OK] wrote", p)


# -----------------------------
# Plotters
# -----------------------------
def plot_case_week(df: pd.DataFrame, outdir: Path, week_start_utc: str, title: str, fname: str,
                   colors: dict):
    ws = pd.Timestamp(week_start_utc, tz="UTC")
    we = ws + pd.Timedelta(days=7)

    w = df[(df["timestamp_utc"] >= ws) & (df["timestamp_utc"] < we)].copy()
    if len(w) == 0:
        print(f"[WARN] no rows for week {week_start_utc}")
        return

    plt.figure(figsize=(14, 4.5))
    plt.plot(w["timestamp_utc"], w["power_norm"], label="Ground truth", color=colors["gt"], linewidth=1.2)
    plt.plot(w["timestamp_utc"], w["core"], label="MiRACLE Core", color=colors["core"], linewidth=1.6)
    plt.plot(w["timestamp_utc"], w["v1"], label="RL v1", color=colors["v1"], linewidth=1.3)
    plt.plot(w["timestamp_utc"], w["v2"], label="RL v2", color=colors["v2"], linewidth=1.3)

    plt.title(title)
    plt.xlabel("timestamp_utc")
    plt.ylabel("power_norm")
    plt.legend()
    savefig(outdir, fname)


def plot_monthly_rmse(df: pd.DataFrame, outdir: Path, colors: dict):
    d = df.copy()
    d["month"] = d["timestamp_utc"].dt.to_period("M").dt.to_timestamp()

    rows = []
    for m, g in d.groupby("month"):
        rows.append({
            "month": m,
            "rmse_core": rmse(g["power_norm"], g["core"]),
            "rmse_v1": rmse(g["power_norm"], g["v1"]),
            "rmse_v2": rmse(g["power_norm"], g["v2"]),
        })
    mm = pd.DataFrame(rows).sort_values("month")

    plt.figure(figsize=(12, 4.5))
    plt.plot(mm["month"], mm["rmse_core"], label="MiRACLE Core", color=colors["core"], linewidth=1.2)
    plt.plot(mm["month"], mm["rmse_v1"], label="RL v1", color=colors["v1"], linewidth=1.6)
    plt.plot(mm["month"], mm["rmse_v2"], label="RL v2", color=colors["v2"], linewidth=1.6)

    plt.title("Monthly RMSE (stitched, most-recent)")
    plt.xlabel("month")
    plt.ylabel("RMSE")
    plt.legend()
    savefig(outdir, "monthly_rmse_core_vs_rl_v1_v2.png")

    mm.to_csv(outdir / "monthly_rmse_core_vs_rl_v1_v2.csv", index=False)
    print("[OK] wrote", outdir / "monthly_rmse_core_vs_rl_v1_v2.csv")


def plot_leadtime_rmse_curve(preds_core_path: str, preds_v1_path: str, preds_v2_path: str,
                             gt_path: str, outdir: Path, colors: dict):
    """
    Lead-time curve for Day-1 only (step_ahead 0..95). We compute RMSE per step using
    the same timestamp_utc join against ground truth, but WITHOUT stitched collapsing
    since step_ahead is defined per forecast.
    """
    gt = pd.read_parquet(gt_path)[["timestamp_utc", "power_norm"]].copy()
    gt["timestamp_utc"] = pd.to_datetime(gt["timestamp_utc"], utc=True)

    core = load_preds_with_step(preds_core_path)
    v1 = load_preds_with_step(preds_v1_path)
    v2 = load_preds_with_step(preds_v2_path)

    # join each with gt on timestamp_utc
    core = core.merge(gt, on="timestamp_utc", how="inner")
    v1 = v1.merge(gt, on="timestamp_utc", how="inner")
    v2 = v2.merge(gt, on="timestamp_utc", how="inner")

    # keep only day1 steps 0..95 (if file contains more)
    core = core[(core["step_ahead"] >= 0) & (core["step_ahead"] <= 95)]
    v1 = v1[(v1["step_ahead"] >= 0) & (v1["step_ahead"] <= 95)]
    v2 = v2[(v2["step_ahead"] >= 0) & (v2["step_ahead"] <= 95)]

    steps = list(range(96))
    rmse_core = []
    rmse_v1 = []
    rmse_v2 = []

    for s in steps:
        gc = core[core["step_ahead"] == s]
        gv1 = v1[v1["step_ahead"] == s]
        gv2 = v2[v2["step_ahead"] == s]
        rmse_core.append(rmse(gc["power_norm"], gc["predicted_power_norm"]))
        rmse_v1.append(rmse(gv1["power_norm"], gv1["predicted_power_norm"]))
        rmse_v2.append(rmse(gv2["power_norm"], gv2["predicted_power_norm"]))

    hours = np.array(steps) * 0.25

    plt.figure(figsize=(12, 4.5))
    plt.plot(hours, rmse_core, label="MiRACLE Core", color=colors["core"], linewidth=2.0)
    plt.plot(hours, rmse_v1, label="RL v1", color=colors["v1"], linewidth=1.6)
    plt.plot(hours, rmse_v2, label="RL v2", color=colors["v2"], linewidth=1.6)

    plt.title("Lead-time RMSE curve (0 to 24h, 15-min steps)")
    plt.xlabel("hours ahead")
    plt.ylabel("RMSE")
    plt.legend()
    savefig(outdir, "leadtime_rmse_curve_core_vs_rl_v1_v2_0_24h.png")

    pd.DataFrame({
        "step_ahead": steps,
        "hours_ahead": hours,
        "rmse_core": rmse_core,
        "rmse_v1": rmse_v1,
        "rmse_v2": rmse_v2,
    }).to_csv(outdir / "leadtime_rmse_curve_core_vs_rl_v1_v2_0_24h.csv", index=False)
    print("[OK] wrote", outdir / "leadtime_rmse_curve_core_vs_rl_v1_v2_0_24h.csv")


def plot_tails(df: pd.DataFrame, outdir: Path, colors: dict):
    d = df.copy()
    d["abs_err_core"] = (d["core"] - d["power_norm"]).abs()
    d["abs_err_v1"] = (d["v1"] - d["power_norm"]).abs()
    d["abs_err_v2"] = (d["v2"] - d["power_norm"]).abs()
    # Robust histogram: clip x-axis (abs error), use denser bins, emphasize MiRACLE Core on top
    core_vals = d["abs_err_core"].to_numpy()
    v1_vals = d["abs_err_v1"].to_numpy()
    v2_vals = d["abs_err_v2"].to_numpy()

    all_vals = np.concatenate([core_vals, v1_vals, v2_vals])
    # choose a clipping threshold similar to example: min(0.8, 99.5th percentile)
    try:
        clip_x = float(min(0.8, np.quantile(all_vals[np.isfinite(all_vals)], 0.995)))
        if clip_x <= 0 or not np.isfinite(clip_x):
            clip_x = 0.8
    except Exception:
        clip_x = 0.8

    bins = np.linspace(0.0, clip_x, 80)

    plt.figure(figsize=(12, 4.5))

    # plot RL variants behind
    plt.hist(np.clip(v1_vals, 0, clip_x), bins=bins, alpha=0.6, label="RL v1", color=colors["v1"], zorder=1)
    plt.hist(np.clip(v2_vals, 0, clip_x), bins=bins, alpha=0.6, label="RL v2", color=colors["v2"], zorder=1)

    # plot core on top with black edge for clarity
    plt.hist(np.clip(core_vals, 0, clip_x), bins=bins, alpha=0.9, label="MiRACLE Core",
             color=colors["core"], zorder=3, edgecolor="black", linewidth=0.4)

    plt.title("Tail behavior: absolute error histogram (stitched, most-recent)")
    plt.xlabel("Abs error (clipped)")
    plt.ylabel("Count")
    plt.xlim(0, clip_x)
    plt.grid(axis="y", linestyle="--", alpha=0.35)

    lg = plt.legend(frameon=True, framealpha=0.95)
    for lh in getattr(lg, "legend_handles", getattr(lg, "legendHandles", [])):
        try:
            lh.set_alpha(1.0)
        except Exception:
            pass

    savefig(outdir, "tails_abs_error_hist_core_vs_rl_v1_v2.png")

    # quantiles table plot
    qs = [0.50, 0.90, 0.95, 0.99]
    qtab = pd.DataFrame({
        "q": qs,
        "core": [float(d["abs_err_core"].quantile(q)) for q in qs],
        "v1": [float(d["abs_err_v1"].quantile(q)) for q in qs],
        "v2": [float(d["abs_err_v2"].quantile(q)) for q in qs],
    })

    plt.figure(figsize=(10, 4.2))
    plt.plot(qtab["q"], qtab["core"], marker="o", label="MiRACLE Core", color=colors["core"], linewidth=1.2)
    plt.plot(qtab["q"], qtab["v1"], marker="o", label="RL v1", color=colors["v1"], linewidth=1.6)
    plt.plot(qtab["q"], qtab["v2"], marker="o", label="RL v2", color=colors["v2"], linewidth=1.6)
    plt.title("Tail quantiles of absolute error (stitched, most-recent)")
    plt.xlabel("quantile")
    plt.ylabel("|error|")
    plt.legend()
    savefig(outdir, "tails_abs_error_quantiles_core_vs_rl_v1_v2.png")

    qtab.to_csv(outdir / "tails_abs_error_quantiles_core_vs_rl_v1_v2.csv", index=False)
    print("[OK] wrote", outdir / "tails_abs_error_quantiles_core_vs_rl_v1_v2.csv")


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", required=True, help="Core predictions parquet (baseline)")
    ap.add_argument("--v1", required=True, help="RL v1 predictions parquet")
    ap.add_argument("--v2", required=True, help="RL v2 predictions parquet")
    ap.add_argument("--gt", required=True, help="Ground truth parquet")
    ap.add_argument("--outdir", required=True, help="Output folder")
    ap.add_argument("--summer-week-start-utc", required=True)
    ap.add_argument("--winter-week-start-utc", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Color rules 
    colors = {
        "gt": "grey",
        "core": "green",
        "v1": "#7EC8E3",   # light blue
        "v2": "orange",
    }

    # Load and stitch (most-recent)
    gt = pd.read_parquet(args.gt)[["timestamp_utc", "power_norm"]].copy()
    gt["timestamp_utc"] = pd.to_datetime(gt["timestamp_utc"], utc=True)

    core_st = join_gt(most_recent_stitch(load_preds(args.core)), gt).rename(columns={"predicted_power_norm": "core"})
    v1_st = join_gt(most_recent_stitch(load_preds(args.v1)), gt).rename(columns={"predicted_power_norm": "v1"})
    v2_st = join_gt(most_recent_stitch(load_preds(args.v2)), gt).rename(columns={"predicted_power_norm": "v2"})

    # Merge into a single stitched frame
    df = core_st[["timestamp_utc", "power_norm", "core"]].merge(
        v1_st[["timestamp_utc", "v1"]], on="timestamp_utc", how="inner"
    ).merge(
        v2_st[["timestamp_utc", "v2"]], on="timestamp_utc", how="inner"
    ).sort_values("timestamp_utc").reset_index(drop=True)

    print("[OK] stitched rows:", len(df))
    print("[OK] overall RMSE:",
          "core=", rmse(df["power_norm"], df["core"]),
          "v1=", rmse(df["power_norm"], df["v1"]),
          "v2=", rmse(df["power_norm"], df["v2"]))

    df.to_parquet(outdir / "stitched_core_vs_rl_v1_v2.parquet", index=False)
    print("[OK] wrote", outdir / "stitched_core_vs_rl_v1_v2.parquet")

    # Plots
    plot_tails(df, outdir, colors)
    plot_case_week(df, outdir, args.summer_week_start_utc, "Case study: summer week", "case_summer_week_core_vs_rl_v1_v2.png", colors)
    plot_case_week(df, outdir, args.winter_week_start_utc, "Case study: winter week", "case_winter_week_core_vs_rl_v1_v2.png", colors)
    plot_monthly_rmse(df, outdir, colors)

    # Lead-time curve from raw forecast rows
    plot_leadtime_rmse_curve(args.core, args.v1, args.v2, args.gt, outdir, colors)

    print("[OK] wrote plots to:", outdir)


if __name__ == "__main__":
    main()
