#src/evaluation/run_full_eval.py

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt


# -----------------------------
# IO
# -----------------------------
def read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing parquet: {path}")
    t = pq.read_table(path.as_posix())
    return t.to_pandas()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Metrics
# -----------------------------
def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def mbe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_pred - y_true))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y = y_true
    yp = y_pred
    ss_res = np.sum((y - yp) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def nrmse_capacity(y_true: np.ndarray, y_pred: np.ndarray, cap: float = 1.0) -> float:
    # power_norm should already be cap-normalized, so cap=1 is fine
    if cap == 0:
        return float("nan")
    return rmse(y_true, y_pred) / cap


def metrics_row(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "nRMSE": nrmse_capacity(y_true, y_pred, cap=1.0),
        "MBE": mbe(y_true, y_pred),
        "R2": r2(y_true, y_pred),
        "N": int(len(y_true)),
    }


# -----------------------------
# Buckets + helpers
# -----------------------------
def lead_bucket(hours_ahead: float) -> str:
    # Adjust if you want different bins
    if hours_ahead <= 24:
        return "0-24h"
    if hours_ahead <= 24 * 7:
        return "2-7d"
    return "8-30d"


def to_utc_datetime(s: pd.Series) -> pd.Series:
    # Keep timezone-aware UTC
    return pd.to_datetime(s, utc=True)


def bootstrap_mean_ci(deltas: np.ndarray, n_boot: int = 5000, seed: int = 42) -> Tuple[float, float, float]:
    """
    Bootstrap CI for mean(delta). Returns (mean, lo, hi).
    """
    rng = np.random.default_rng(seed)
    deltas = deltas[~np.isnan(deltas)]
    if len(deltas) == 0:
        return float("nan"), float("nan"), float("nan")

    mean0 = float(np.mean(deltas))
    idx = rng.integers(0, len(deltas), size=(n_boot, len(deltas)))
    boot_means = np.mean(deltas[idx], axis=1)
    lo = float(np.quantile(boot_means, 0.025))
    hi = float(np.quantile(boot_means, 0.975))
    return mean0, lo, hi


# -----------------------------
# Plotting
# -----------------------------
def save_fig(path: Path, dpi: int = 200) -> None:
    plt.tight_layout()
    plt.savefig(path.as_posix(), dpi=dpi, bbox_inches='tight')
    plt.close()


def plot_monthly_rmse(monthly: pd.DataFrame, out: Path) -> None:
    # monthly columns: month, RMSE_baseline, RMSE_policy
    x = monthly["month"].astype(str).tolist()
    yb = monthly["RMSE_baseline"].values
    yp = monthly["RMSE_policy"].values

    plt.figure(figsize=(10, 4))
    # Baseline (MiRACLE Core) = BOLD GREEN (HIGHLIGHTED)
    plt.plot(x, yb, marker="o", label="MiRACLE v1.0 (Core)", color='#00AA00', linewidth=2.5, markersize=7, alpha=1.0)
    # Policy (MiRACLE Full) = LIGHT BLUE (de-emphasized)
    plt.plot(x, yp, marker="s", label="MiRACLE v1.0 (Meta-control)", color='#6BA3D8', linewidth=1.5, markersize=5, alpha=0.9)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Month", fontsize=11)
    plt.ylabel("RMSE (power_norm)", fontsize=11)
    plt.legend(fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle=':')
    plt.tight_layout()
    save_fig(out, dpi=300)


def plot_error_hist(df: pd.DataFrame, out: Path, max_abs: float = 1.0) -> None:
    # histogram of absolute error
    plt.figure(figsize=(8, 5))
    # Policy first (behind, de-emphasized) = LIGHT BLUE
    plt.hist(df["abs_err_policy"].clip(0, max_abs), bins=80, alpha=0.6, label="MiRACLE v1.0 (Meta-control)", color='#6BA3D8', edgecolor='black', linewidth=0.5)
    # Baseline on top (HIGHLIGHTED) = BOLD GREEN
    plt.hist(df["abs_err_baseline"].clip(0, max_abs), bins=80, alpha=0.7, label="MiRACLE v1.0 (Core)", color='#00AA00', edgecolor='black', linewidth=0.5)
    plt.xlabel("Absolute error (clipped)", fontsize=11)
    plt.ylabel("Count", fontsize=11)
    plt.legend(fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle=':')
    plt.tight_layout()
    save_fig(out, dpi=300)


def plot_cumulative_abs_error(df: pd.DataFrame, out: Path) -> None:
    dd = df.sort_values("timestamp_utc")
    cum_b = np.cumsum(dd["abs_err_baseline"].values)
    cum_p = np.cumsum(dd["abs_err_policy"].values)

    plt.figure(figsize=(10, 4))
    # Baseline (MiRACLE Core) = BOLD GREEN (HIGHLIGHTED)
    plt.plot(dd["timestamp_utc"].values, cum_b, label="MiRACLE v1.0 (Core)", color='#00AA00', linewidth=2.5, alpha=1.0)
    # Policy (MiRACLE Full) = LIGHT BLUE (de-emphasized)
    plt.plot(dd["timestamp_utc"].values, cum_p, label="MiRACLE v1.0 (Meta-control)", color='#6BA3D8', linewidth=1.5, alpha=0.9)
    plt.xlabel("Time", fontsize=11)
    plt.ylabel("Cumulative absolute error", fontsize=11)
    plt.legend(fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle=':')
    plt.tight_layout()
    save_fig(out, dpi=300)


def plot_daily_scatter(daily: pd.DataFrame, out: Path) -> None:
    # daily columns: day, MAE_baseline, MAE_policy
    x = daily["MAE_baseline"].values
    y = daily["MAE_policy"].values

    plt.figure()
    plt.scatter(x, y, s=12)
    lim = max(np.nanmax(x), np.nanmax(y))
    plt.plot([0, lim], [0, lim])
    plt.xlabel("Daily MAE MiRACLE v1.0 (Core)")
    plt.ylabel("Daily MAE MiRACLE v1.0 (Meta-control)")
    save_fig(out)


def plot_action_distribution(policy_df: pd.DataFrame, out: Path) -> None:
    counts = policy_df["policy_action"].value_counts().sort_index()
    plt.figure()
    plt.bar(counts.index.astype(str), counts.values)
    plt.xlabel("policy_action")
    plt.ylabel("count")
    save_fig(out)


def plot_case_study_stitched(
    df_all: pd.DataFrame,
    start: str,
    end: str,
    out: Path,
) -> None:
    """
    Build a continuous series by selecting, for each timestamp, the prediction with the smallest hours_ahead.
    This approximates "most recent forecast available".
    """
    s = pd.to_datetime(start, utc=True)
    e = pd.to_datetime(end, utc=True)

    dd = df_all[(df_all["timestamp_utc"] >= s) & (df_all["timestamp_utc"] <= e)].copy()
    if dd.empty:
        return

    # pick smallest lead for each timestamp
    dd = dd.sort_values(["timestamp_utc", "hours_ahead"])
    dd = dd.groupby("timestamp_utc", as_index=False).first()

    plt.figure(figsize=(12, 5))
    # Ground truth = LIGHT GREY (subtle reference)
    plt.plot(dd["timestamp_utc"].values, dd["y_true"].values, label="Ground Truth", color='#888888', linewidth=1.5, alpha=0.7)
    # Baseline (MiRACLE Core) = BOLD GREEN (HIGHLIGHTED)
    plt.plot(dd["timestamp_utc"].values, dd["y_baseline"].values, label="MiRACLE v1.0 (Core)", color='#00AA00', linewidth=2.5, alpha=1.0)
    # Policy (MiRACLE Full) = LIGHT BLUE (de-emphasized)
    plt.plot(dd["timestamp_utc"].values, dd["y_policy"].values, label="MiRACLE v1.0 (Meta-control)", color='#6BA3D8', linewidth=1.5, alpha=0.9)
    plt.xlabel("Time (UTC)", fontsize=11)
    plt.ylabel("Power (normalized)", fontsize=11)
    plt.legend(fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle=':')
    plt.tight_layout()
    save_fig(out, dpi=300)


# -----------------------------
# Main eval pipeline
# -----------------------------
@dataclass
class Paths:
    out_dir: Path
    tables_dir: Path
    figures_dir: Path
    text_dir: Path


def write_csv_and_tex(df: pd.DataFrame, csv_path: Path, tex_path: Path, index: bool = False) -> None:
    df.to_csv(csv_path.as_posix(), index=index)
    try:
        tex = df.to_latex(index=index, float_format="%.6f")
        tex_path.write_text(tex)
    except Exception:
        # latex is optional
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True, type=str)
    ap.add_argument("--baseline", required=True, type=str)
    ap.add_argument("--policy", required=True, type=str)
    ap.add_argument("--out", required=True, type=str, help="Output folder, e.g. freeze/.../eval_outputs")
    ap.add_argument("--daylight-threshold", type=float, default=0.01, help="Keep rows where y_true >= threshold")
    ap.add_argument("--include-night", action="store_true", help="If set, do not filter nighttime")
    ap.add_argument("--case-summer-start", type=str, default="2024-07-01T00:00:00Z")
    ap.add_argument("--case-summer-end", type=str, default="2024-07-08T00:00:00Z")
    ap.add_argument("--case-winter-start", type=str, default="2024-01-10T00:00:00Z")
    ap.add_argument("--case-winter-end", type=str, default="2024-01-17T00:00:00Z")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    truth_path = Path(args.truth)
    base_path = Path(args.baseline)
    pol_path = Path(args.policy)

    out_dir = Path(args.out)
    paths = Paths(
        out_dir=out_dir,
        tables_dir=out_dir / "tables",
        figures_dir=out_dir / "figures",
        text_dir=out_dir / "text",
    )
    ensure_dir(paths.tables_dir)
    ensure_dir(paths.figures_dir)
    ensure_dir(paths.text_dir)

    # Load
    truth = read_parquet(truth_path)
    base = read_parquet(base_path)
    pol = read_parquet(pol_path)

    # Standardize timestamps
    truth["timestamp_utc"] = to_utc_datetime(truth["timestamp_utc"])
    base["timestamp_utc"] = to_utc_datetime(base["timestamp_utc"])
    base["forecast_start"] = to_utc_datetime(base["forecast_start"])
    pol["timestamp_utc"] = to_utc_datetime(pol["timestamp_utc"])
    pol["forecast_start"] = to_utc_datetime(pol["forecast_start"])

    # Rename truth target
    if "power_norm" not in truth.columns:
        raise ValueError("Truth parquet must contain power_norm")

    truth = truth[["timestamp_utc", "power_norm"]].rename(columns={"power_norm": "y_true"})

    # Join baseline/policy with truth on timestamp
    base_join = base.merge(truth, on="timestamp_utc", how="inner")
    pol_join = pol.merge(truth, on="timestamp_utc", how="inner")

    # Align baseline and policy rows by (forecast_start, step_ahead, timestamp_utc)
    key = ["forecast_start", "step_ahead", "timestamp_utc"]
    base_keep = base_join[key + ["hours_ahead", "predicted_power_norm", "y_true"]].rename(
        columns={"predicted_power_norm": "y_baseline"}
    )
    pol_keep = pol_join[key + ["hours_ahead", "predicted_power_norm", "y_true", "policy_action", "blend_short", "blend_long", "blend_physics"]].rename(
        columns={"predicted_power_norm": "y_policy"}
    )

    df = base_keep.merge(pol_keep, on=key + ["hours_ahead", "y_true"], how="inner")

    # Optional daylight filtering
    if not args.include_night:
        df = df[df["y_true"] >= float(args.daylight_threshold)].copy()

    # Derived columns
    df["abs_err_baseline"] = (df["y_baseline"] - df["y_true"]).abs()
    df["abs_err_policy"] = (df["y_policy"] - df["y_true"]).abs()
    df["sq_err_baseline"] = (df["y_baseline"] - df["y_true"]) ** 2
    df["sq_err_policy"] = (df["y_policy"] - df["y_true"]) ** 2
    df["month"] = df["timestamp_utc"].dt.to_period("M").astype(str)
    df["day"] = df["timestamp_utc"].dt.date.astype(str)
    df["lead_bucket"] = df["hours_ahead"].apply(lead_bucket)

    def tail_stats(x: np.ndarray) -> dict:
        return {
            "P50": float(np.quantile(x, 0.50)),
            "P90": float(np.quantile(x, 0.90)),
            "P95": float(np.quantile(x, 0.95)),
            "P99": float(np.quantile(x, 0.99)),
            "mean": float(np.mean(x)),
        }

    tail_baseline = tail_stats(df["abs_err_baseline"].values)
    tail_policy = tail_stats(df["abs_err_policy"].values)

    tail_tbl = pd.DataFrame([
        {"model": "baseline", **tail_baseline},
        {"model": "policy", **tail_policy},
    ])

    tail_tbl.to_csv((paths.tables_dir / "tail_abs_error.csv").as_posix(), index=False)

    # -----------------------------
    # Overall metrics
    # -----------------------------
    overall = pd.DataFrame(
        [
            {"model": "baseline", **metrics_row(df["y_true"].values, df["y_baseline"].values)},
            {"model": "policy", **metrics_row(df["y_true"].values, df["y_policy"].values)},
        ]
    )
    write_csv_and_tex(
        overall,
        paths.tables_dir / "overall_metrics.csv",
        paths.tables_dir / "overall_metrics.tex",
        index=False,
    )

    stitched = df.sort_values(["timestamp_utc", "hours_ahead"]).groupby("timestamp_utc", as_index=False).first()

    stitched_overall = pd.DataFrame([
        {"model": "baseline", **metrics_row(stitched["y_true"].values, stitched["y_baseline"].values)},
        {"model": "policy", **metrics_row(stitched["y_true"].values, stitched["y_policy"].values)},
    ])

    write_csv_and_tex(
        stitched_overall,
        paths.tables_dir / "overall_metrics_stitched.csv",
        paths.tables_dir / "overall_metrics_stitched.tex",
        index=False,
    )

    # -----------------------------
    # Monthly metrics
    # -----------------------------
    rows = []
    for m, g in df.groupby("month"):
        mb = metrics_row(g["y_true"].values, g["y_baseline"].values)
        mp = metrics_row(g["y_true"].values, g["y_policy"].values)
        rows.append(
            {
                "month": m,
                "MAE_baseline": mb["MAE"],
                "MAE_policy": mp["MAE"],
                "RMSE_baseline": mb["RMSE"],
                "RMSE_policy": mp["RMSE"],
                "delta_RMSE": mp["RMSE"] - mb["RMSE"],
                "delta_MAE": mp["MAE"] - mb["MAE"],
                "N": int(mb["N"]),
            }
        )
    monthly = pd.DataFrame(rows).sort_values("month")
    write_csv_and_tex(
        monthly,
        paths.tables_dir / "monthly_metrics.csv",
        paths.tables_dir / "monthly_metrics.tex",
        index=False,
    )

    # -----------------------------
    # Lead bucket metrics
    # -----------------------------
    rows = []
    for b, g in df.groupby("lead_bucket"):
        mb = metrics_row(g["y_true"].values, g["y_baseline"].values)
        mp = metrics_row(g["y_true"].values, g["y_policy"].values)
        rows.append(
            {
                "lead_bucket": b,
                "MAE_baseline": mb["MAE"],
                "MAE_policy": mp["MAE"],
                "RMSE_baseline": mb["RMSE"],
                "RMSE_policy": mp["RMSE"],
                "delta_RMSE": mp["RMSE"] - mb["RMSE"],
                "delta_MAE": mp["MAE"] - mb["MAE"],
                "N": int(mb["N"]),
            }
        )
    lead_tbl = pd.DataFrame(rows)
    lead_order = {"0-24h": 0, "2-7d": 1, "8-30d": 2}
    lead_tbl["__ord"] = lead_tbl["lead_bucket"].map(lead_order).fillna(99).astype(int)
    lead_tbl = lead_tbl.sort_values("__ord").drop(columns="__ord")
    write_csv_and_tex(
        lead_tbl,
        paths.tables_dir / "lead_bucket_metrics.csv",
        paths.tables_dir / "lead_bucket_metrics.tex",
        index=False,
    )

    # -----------------------------
    # Daily metrics (paired comparison)
    # -----------------------------
    daily_rows = []
    for d, g in df.groupby("day"):
        mb = metrics_row(g["y_true"].values, g["y_baseline"].values)
        mp = metrics_row(g["y_true"].values, g["y_policy"].values)
        daily_rows.append(
            {
                "day": d,
                "MAE_baseline": mb["MAE"],
                "MAE_policy": mp["MAE"],
                "RMSE_baseline": mb["RMSE"],
                "RMSE_policy": mp["RMSE"],
                "delta_MAE": mp["MAE"] - mb["MAE"],
                "delta_RMSE": mp["RMSE"] - mb["RMSE"],
                "N": int(mb["N"]),
            }
        )
    daily = pd.DataFrame(daily_rows).sort_values("day")
    write_csv_and_tex(
        daily,
        paths.tables_dir / "daily_metrics.csv",
        paths.tables_dir / "daily_metrics.tex",
        index=False,
    )

    # Worst days by baseline RMSE
    worst = daily.sort_values("RMSE_baseline", ascending=False).head(10)
    worst.to_csv((paths.tables_dir / "worst_10_days.csv").as_posix(), index=False)

    # Significance style summary via bootstrap on daily delta_MAE
    deltas = daily["delta_MAE"].values
    mean_delta, lo, hi = bootstrap_mean_ci(deltas, n_boot=5000, seed=int(args.seed))
    frac_improved = float(np.mean(deltas < 0))

    # -----------------------------
    # RL diagnostics
    # -----------------------------
    action_counts = df["policy_action"].value_counts().sort_index()
    action_tbl = action_counts.reset_index()
    action_tbl.columns = ["policy_action", "count"]
    action_tbl.to_csv((paths.tables_dir / "policy_action_distribution.csv").as_posix(), index=False)

    # -----------------------------
    # Plots
    # -----------------------------
    plot_monthly_rmse(monthly, paths.figures_dir / "monthly_rmse.png")
    plot_error_hist(df, paths.figures_dir / "abs_error_hist.png", max_abs=1.0)
    plot_cumulative_abs_error(df, paths.figures_dir / "cumulative_abs_error.png")
    plot_daily_scatter(daily, paths.figures_dir / "daily_mae_scatter.png")
    plot_action_distribution(df, paths.figures_dir / "policy_action_distribution.png")

    # Case studies (stitched from most-recent forecast)
    plot_case_study_stitched(df, args.case_summer_start, args.case_summer_end, paths.figures_dir / "case_summer_week.png")
    plot_case_study_stitched(df, args.case_winter_start, args.case_winter_end, paths.figures_dir / "case_winter_week.png")

    # -----------------------------
    # Results markdown
    # -----------------------------
    md = []
    md.append("# Evaluation summary\n")
    md.append(f"- Night filtering: {'OFF' if args.include_night else 'ON'} (threshold y_true >= {args.daylight_threshold})\n")
    md.append("## Overall\n")
    md.append(overall.to_markdown(index=False))
    md.append("\n\n## Lead buckets\n")
    md.append(lead_tbl.to_markdown(index=False))
    md.append("\n\n## Monthly\n")
    md.append(monthly.to_markdown(index=False))
    md.append("\n\n## Paired daily comparison\n")
    md.append(f"- Mean daily delta MAE (policy - baseline): {mean_delta:.6f}\n")
    md.append(f"- 95% bootstrap CI for mean delta: [{lo:.6f}, {hi:.6f}]\n")
    md.append(f"- Fraction of days improved (delta_MAE < 0): {frac_improved:.3f}\n")
    md.append("\n\n## RL actions\n")
    md.append(action_tbl.to_markdown(index=False))
    md.append("\n\n## Tail absolute error\n")
    md.append(tail_tbl.to_markdown(index=False))
    (paths.text_dir / "results.md").write_text("\n".join(md))

    # Save the joined eval frame for reuse
    df.to_parquet((paths.out_dir / "eval_joined.parquet").as_posix(), index=False)

    print(f"[OK] Wrote outputs to: {paths.out_dir}")


if __name__ == "__main__":
    main()
