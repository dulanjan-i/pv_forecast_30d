from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# -----------------------------
# IO
# -----------------------------
def read_parquet(path: Path, columns: Optional[List[str]] = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing parquet: {path}")
    table = pq.read_table(path.as_posix(), columns=columns)
    return table.to_pandas()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def to_utc_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce")


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


def tail_stats_abs(err_abs: np.ndarray) -> Dict[str, float]:
    x = err_abs[~np.isnan(err_abs)]
    if len(x) == 0:
        return {"P50": float("nan"), "P90": float("nan"), "P95": float("nan"), "P99": float("nan"), "mean": float("nan")}
    return {
        "P50": float(np.quantile(x, 0.50)),
        "P90": float(np.quantile(x, 0.90)),
        "P95": float(np.quantile(x, 0.95)),
        "P99": float(np.quantile(x, 0.99)),
        "mean": float(np.mean(x)),
    }


def lead_bucket(hours_ahead: float) -> str:
    if hours_ahead <= 24:
        return "0-24h"
    if hours_ahead <= 24 * 7:
        return "2-7d"
    return "8-30d"


def bootstrap_mean_ci(deltas: np.ndarray, n_boot: int = 5000, seed: int = 42) -> Tuple[float, float, float]:
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
# Pred schema helpers
# -----------------------------
REQUIRED_KEY_COLS = ["timestamp_utc", "forecast_start", "step_ahead", "hours_ahead"]


def detect_pred_col(df: pd.DataFrame) -> str:
    # Preferred
    for c in ["predicted_power_norm", "y_pred", "pred", "prediction", "power_pred", "power_norm_pred"]:
        if c in df.columns:
            return c
    # Fallback: choose a float-like column that is not an obvious feature
    bad = set(REQUIRED_KEY_COLS + ["policy_action", "blend_short", "blend_long", "blend_physics"])
    candidates = [c for c in df.columns if c not in bad]
    numeric = []
    for c in candidates:
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric.append(c)
    if len(numeric) == 0:
        raise ValueError(f"Could not detect prediction column. Columns={list(df.columns)[:80]}")
    # Prefer something with "pred" in name
    predish = [c for c in numeric if "pred" in c.lower()]
    return predish[0] if predish else numeric[0]


def standardize_preds(df: pd.DataFrame, name: str) -> pd.DataFrame:
    # Validate keys
    for c in REQUIRED_KEY_COLS:
        if c not in df.columns:
            raise ValueError(f"[{name}] Missing required col: {c}. Has={list(df.columns)[:80]}")

    df = df.copy()
    df["timestamp_utc"] = to_utc_datetime(df["timestamp_utc"])
    df["forecast_start"] = to_utc_datetime(df["forecast_start"])

    pred_col = detect_pred_col(df)
    if pred_col != "predicted_power_norm":
        df = df.rename(columns={pred_col: "predicted_power_norm"})

    # Ensure types
    df["step_ahead"] = pd.to_numeric(df["step_ahead"], errors="coerce").astype("Int64")
    df["hours_ahead"] = pd.to_numeric(df["hours_ahead"], errors="coerce")

    # Drop rows with broken keys
    df = df.dropna(subset=["timestamp_utc", "forecast_start", "step_ahead", "hours_ahead", "predicted_power_norm"])
    df["step_ahead"] = df["step_ahead"].astype(int)

    return df


def standardize_truth(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp_utc" not in df.columns:
        raise ValueError("Truth parquet must contain timestamp_utc")
    if "power_norm" not in df.columns:
        raise ValueError("Truth parquet must contain power_norm")
    out = df[["timestamp_utc", "power_norm"]].copy()
    out["timestamp_utc"] = to_utc_datetime(out["timestamp_utc"])
    out = out.dropna(subset=["timestamp_utc", "power_norm"])
    out = out.rename(columns={"power_norm": "y_true"})
    return out


# -----------------------------
# Joining and stitched series
# -----------------------------
def join_with_truth(preds: pd.DataFrame, truth: pd.DataFrame, name: str) -> pd.DataFrame:
    # join only on timestamp_utc, truth is single series
    out = preds.merge(truth, on="timestamp_utc", how="inner")
    if out.empty:
        raise RuntimeError(f"[{name}] Join with truth produced 0 rows. Check timestamp overlap and tz.")
    return out


def daylight_filter(df: pd.DataFrame, include_night: bool, threshold: float) -> pd.DataFrame:
    if include_night:
        return df
    return df[df["y_true"] >= float(threshold)].copy()


def make_stitched(df_joined: pd.DataFrame) -> pd.DataFrame:
    # choose smallest lead for each timestamp, approximates "most recent forecast available"
    dd = df_joined.sort_values(["timestamp_utc", "hours_ahead"]).groupby("timestamp_utc", as_index=False).first()
    return dd


# -----------------------------
# Output helpers
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
        tex = df.to_latex(index=index, float_format="%.6f", escape=False)
        tex_path.write_text(tex)
    except Exception:
        pass


def save_fig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path.as_posix(), dpi=300)  # FORMATTING FIX: 200 -> 300 DPI
    plt.close()


# -----------------------------
# Plotting: facet grids
# -----------------------------
def grid_shape(n: int) -> Tuple[int, int]:
    if n <= 0:
        return 1, 1
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    return rows, cols


def plot_facets_case_week(
    stitched_truth: pd.DataFrame,
    stitched_baseline: pd.DataFrame,
    stitched_models: Dict[str, pd.DataFrame],
    start: str,
    end: str,
    out: Path,
    title: str,
    truth_label: str = "Ground Truth Plant 03",  # FORMATTING FIX: configurable truth label
    baseline_label: str = "MiRACLE v1.0 Core",  # FORMATTING FIX: configurable baseline label
) -> None:
    s = pd.to_datetime(start, utc=True)
    e = pd.to_datetime(end, utc=True)

    bt = stitched_baseline[(stitched_baseline["timestamp_utc"] >= s) & (stitched_baseline["timestamp_utc"] <= e)].copy()
    tt = stitched_truth[(stitched_truth["timestamp_utc"] >= s) & (stitched_truth["timestamp_utc"] <= e)].copy()
    if bt.empty or tt.empty:
        return

    names = list(stitched_models.keys())
    rows, cols = grid_shape(len(names))

    # FORMATTING FIX: larger figure for better readability
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)
    fig.suptitle(title, fontsize=14, fontweight='bold')

    for i, name in enumerate(names):
        r = i // cols
        c = i % cols
        ax = axes[r][c]

        mm = stitched_models[name]
        mm = mm[(mm["timestamp_utc"] >= s) & (mm["timestamp_utc"] <= e)].copy()
        if mm.empty:
            ax.set_title(f"{name} (no data)")
            ax.axis("off")
            continue

        # FORMATTING FIX: Consistent color mapping across all graphs
        # Ground truth = LIGHT GREY (subtle reference, not focal point)
        ax.plot(tt["timestamp_utc"].values, tt["y_true"].values, label=truth_label, 
                linewidth=1.5, alpha=0.7, color='#888888')
        # Baseline (MiRACLE Core) = BOLD GREEN, thicker to HIGHLIGHT final result
        ax.plot(bt["timestamp_utc"].values, bt["y_pred"].values, label=baseline_label, 
                linewidth=2.5, alpha=1.0, color='#00AA00')
        # Comparison models = LIGHT BLUE, thinner, de-emphasized
        ax.plot(mm["timestamp_utc"].values, mm["y_pred"].values, label=name, 
                linewidth=1.0, alpha=0.9, color='#6BA3D8')

        ax.set_title(name, fontsize=11, fontweight='semibold')
        ax.set_xlabel("Time (UTC)", fontsize=10)
        ax.set_ylabel("Power (normalized)", fontsize=10)
        
        # FORMATTING FIX: Better date formatting
        locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        ax.tick_params(axis='x', rotation=0, labelsize=9)
        
        # FORMATTING FIX: Improved legend
        ax.legend(fontsize=9, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle=':')

    # Turn off unused axes
    for j in range(len(names), rows * cols):
        r = j // cols
        c = j % cols
        axes[r][c].axis("off")

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(out.as_posix(), dpi=300, bbox_inches='tight')  # FORMATTING FIX: 200 -> 300 DPI
    plt.close()


def plot_facets_abs_error_hist(
    joined_baseline: pd.DataFrame,
    joined_models: Dict[str, pd.DataFrame],
    out: Path,
    title: str,
    max_abs: float = 1.0,
    baseline_label: str = "MiRACLE v1.0 Core",  # FORMATTING FIX: configurable baseline label
) -> None:
    names = list(joined_models.keys())
    rows, cols = grid_shape(len(names))
    # FORMATTING FIX: larger figure
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)
    fig.suptitle(title, fontsize=14, fontweight='bold')

    b_abs = (joined_baseline["y_pred"] - joined_baseline["y_true"]).abs().clip(0, max_abs).values

    for i, name in enumerate(names):
        r = i // cols
        c = i % cols
        ax = axes[r][c]

        mm = joined_models[name]
        if mm.empty:
            ax.set_title(f"{name} (no data)")
            ax.axis("off")
            continue

        m_abs = (mm["y_pred"] - mm["y_true"]).abs().clip(0, max_abs).values

        # FORMATTING FIX: Consistent colors, MiRACLE HIGHLIGHTED
        # Comparison model first (behind, de-emphasized) = LIGHT BLUE
        ax.hist(m_abs, bins=80, alpha=0.6, label=name, edgecolor='black', linewidth=0.5, color='#6BA3D8')
        # Baseline (MiRACLE Core) on top (HIGHLIGHTED) = BOLD GREEN
        ax.hist(b_abs, bins=80, alpha=0.7, label=baseline_label, edgecolor='black', linewidth=0.5, color='#00AA00')
        ax.set_title(name, fontsize=11, fontweight='semibold')
        ax.set_xlabel("Abs error (clipped)", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        
        # FORMATTING FIX: Improved legend
        ax.legend(fontsize=9, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle=':', axis='y')

    for j in range(len(names), rows * cols):
        r = j // cols
        c = j % cols
        axes[r][c].axis("off")

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(out.as_posix(), dpi=300, bbox_inches='tight')  # FORMATTING FIX: 200 -> 300 DPI
    plt.close()


def rmse_by_hours(df: pd.DataFrame, max_h: float = 24.0) -> pd.DataFrame:
    dd = df[df["hours_ahead"] <= max_h].copy()
    if dd.empty:
        return pd.DataFrame(columns=["hours_ahead", "RMSE"])
    rows = []
    for h, g in dd.groupby("hours_ahead"):
        rows.append({"hours_ahead": float(h), "RMSE": rmse(g["y_true"].values, g["y_pred"].values), "N": int(len(g))})
    out = pd.DataFrame(rows).sort_values("hours_ahead")
    return out


def plot_facets_leadtime_rmse_curve(
    joined_baseline: pd.DataFrame,
    joined_models: Dict[str, pd.DataFrame],
    out: Path,
    title: str,
    max_h: float = 24.0,
    baseline_label: str = "MiRACLE v1.0 Core",  # FORMATTING FIX: configurable baseline label
) -> None:
    names = list(joined_models.keys())
    rows, cols = grid_shape(len(names))
    # FORMATTING FIX: larger figure
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)
    fig.suptitle(title, fontsize=14, fontweight='bold')

    b_curve = rmse_by_hours(joined_baseline, max_h=max_h)

    for i, name in enumerate(names):
        r = i // cols
        c = i % cols
        ax = axes[r][c]

        mm = joined_models[name]
        if mm.empty:
            ax.set_title(f"{name} (no data)")
            ax.axis("off")
            continue

        m_curve = rmse_by_hours(mm, max_h=max_h)
        if b_curve.empty or m_curve.empty:
            ax.set_title(f"{name} (no data <= {max_h}h)")
            ax.axis("off")
            continue

        # FORMATTING FIX: Consistent color mapping
        # Baseline (MiRACLE Core) = BOLD GREEN, thicker to HIGHLIGHT
        ax.plot(b_curve["hours_ahead"].values, b_curve["RMSE"].values, label=baseline_label, 
                linewidth=2.5, marker='o', markersize=6, alpha=1.0, color='#00AA00')
        # Comparison model = LIGHT BLUE, thinner, de-emphasized
        ax.plot(m_curve["hours_ahead"].values, m_curve["RMSE"].values, label=name, 
                linewidth=1.0, marker='s', markersize=5, alpha=0.8, color='#6BA3D8')
        ax.set_title(name, fontsize=11, fontweight='semibold')
        ax.set_xlabel("Hours ahead", fontsize=10)
        ax.set_ylabel("RMSE", fontsize=10)
        
        # FORMATTING FIX: Improved legend
        ax.legend(fontsize=9, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle=':')

    for j in range(len(names), rows * cols):
        r = j // cols
        c = j % cols
        axes[r][c].axis("off")

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(out.as_posix(), dpi=300, bbox_inches='tight')  # FORMATTING FIX: 200 -> 300 DPI
    plt.close()


def plot_monthly_rmse_all(monthly_long: pd.DataFrame, out: Path, baseline_label: str = "MiRACLE v1.0 Core") -> None:
    # monthly_long: columns = month, model, RMSE
    # FORMATTING FIX: larger figure, better styling, SEPARATE colors per model
    plt.figure(figsize=(12, 5))
    # Color palette for comparison models (light blue, orange, purple, pink, teal)
    comparison_colors = ['#6BA3D8', '#FAA43A', '#B276B2', '#F17CB0', '#60BD68']
    comparison_idx = 0
    
    for model, g in monthly_long.groupby("model"):
        x = g["month"].astype(str).tolist()
        y = g["RMSE"].values
        # MiRACLE gets BOLD GREEN to HIGHLIGHT, others get distinct colors for clarity
        if model == baseline_label:
            plt.plot(x, y, marker="o", label=model, linewidth=2.5, markersize=7, alpha=1.0, color='#00AA00')
        else:
            color = comparison_colors[comparison_idx % len(comparison_colors)]
            comparison_idx += 1
            plt.plot(x, y, marker="s", label=model, linewidth=1.5, markersize=5, alpha=0.9, color=color)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.xlabel("Month", fontsize=11, fontweight='semibold')
    plt.ylabel("RMSE (normalized power)", fontsize=11, fontweight='semibold')
    plt.legend(fontsize=10, loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle=':')
    plt.tight_layout()
    plt.savefig(out.as_posix(), dpi=300, bbox_inches='tight')  # FORMATTING FIX: 200 -> 300 DPI
    plt.close()


# -----------------------------
# Main
# -----------------------------
def parse_models(args_models: List[str]) -> Dict[str, Path]:
    """
    --model name:/abs/or/rel/path.parquet   (repeatable)
    """
    out: Dict[str, Path] = {}
    for s in args_models:
        if ":" not in s:
            raise ValueError(f"Bad --model '{s}'. Use name:/path/to/file.parquet")
        name, path = s.split(":", 1)
        name = name.strip()
        path = path.strip()
        if not name:
            raise ValueError(f"Bad --model '{s}', empty name")
        out[name] = Path(path)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark suite v3 - FORMATTED for thesis (300 DPI, clear labels)")
    ap.add_argument("--truth", required=True, type=str)
    ap.add_argument("--baseline-name", required=True, type=str, help="Label for baseline (e.g., 'MiRACLE v1.0 Core')")
    ap.add_argument("--baseline", required=True, type=str)
    ap.add_argument("--model", action="append", default=[], help="Repeat: name:/path/to/preds.parquet")
    ap.add_argument("--out", required=True, type=str, help="Output directory (MUST be NEW, not final_suite_v2)")

    ap.add_argument("--daylight-threshold", type=float, default=0.01, help="Filter out y_true < threshold (daytime only)")
    ap.add_argument("--include-night", action="store_true", help="Include nighttime data")

    # FORMATTING FIX: Add truth-label argument
    ap.add_argument("--truth-label", type=str, default="Ground Truth Plant 03", help="Label for ground truth in plots")

    ap.add_argument("--case-summer-start", type=str, default="2024-07-01T00:00:00Z")
    ap.add_argument("--case-summer-end", type=str, default="2024-07-08T00:00:00Z")
    ap.add_argument("--case-winter-start", type=str, default="2024-01-10T00:00:00Z")
    ap.add_argument("--case-winter-end", type=str, default="2024-01-17T00:00:00Z")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-hours-curve", type=float, default=24.0)
    args = ap.parse_args()

    out_dir = Path(args.out)
    paths = Paths(out_dir, out_dir / "tables", out_dir / "figures", out_dir / "text")
    ensure_dir(paths.tables_dir)
    ensure_dir(paths.figures_dir)
    ensure_dir(paths.text_dir)

    truth = standardize_truth(read_parquet(Path(args.truth)))
    baseline = standardize_preds(read_parquet(Path(args.baseline)), args.baseline_name)
    models = parse_models(args.model)

    # Load and standardize contenders
    contenders: Dict[str, pd.DataFrame] = {}
    for name, p in models.items():
        contenders[name] = standardize_preds(read_parquet(p), name)

    # Join baseline with truth, filter
    base_join = join_with_truth(baseline, truth, args.baseline_name)
    base_join = daylight_filter(base_join, include_night=args.include_night, threshold=args.daylight_threshold)
    base_join = base_join.rename(columns={"predicted_power_norm": "y_pred"})

    # Join contenders with truth, filter, then align to baseline keys for fair paired comparisons
    key = ["forecast_start", "step_ahead", "timestamp_utc", "hours_ahead"]

    base_keys = base_join[key + ["y_true", "y_pred"]].copy()

    joined_models: Dict[str, pd.DataFrame] = {}
    for name, dfp in contenders.items():
        jj = join_with_truth(dfp, truth, name)
        jj = daylight_filter(jj, include_night=args.include_night, threshold=args.daylight_threshold)
        jj = jj.rename(columns={"predicted_power_norm": "y_pred"})
        # Align to baseline sample set
        jj2 = jj.merge(base_keys[key + ["y_true"]], on=key + ["y_true"], how="inner")
        joined_models[name] = jj2

    # Also align baseline to itself (already filtered), for safety
    base_aligned = base_join.merge(base_keys[key + ["y_true"]], on=key + ["y_true"], how="inner")
    base_aligned = base_aligned[key + ["y_true", "y_pred"]].copy()

    # -----------------------------
    # Overall metrics table (all models including baseline)
    # -----------------------------
    overall_rows = []
    tail_rows = []
    stitched_rows = []

    # baseline overall
    overall_rows.append({"model": args.baseline_name, **metrics_row(base_aligned["y_true"].values, base_aligned["y_pred"].values)})
    tail_rows.append({"model": args.baseline_name, **tail_stats_abs((base_aligned["y_pred"] - base_aligned["y_true"]).abs().values)})

    # contenders overall
    for name, jj in joined_models.items():
        overall_rows.append({"model": name, **metrics_row(jj["y_true"].values, jj["y_pred"].values)})
        tail_rows.append({"model": name, **tail_stats_abs((jj["y_pred"] - jj["y_true"]).abs().values)})

    overall = pd.DataFrame(overall_rows).sort_values("model")
    tail_tbl = pd.DataFrame(tail_rows).sort_values("model")

    write_csv_and_tex(overall, paths.tables_dir / "overall_metrics.csv", paths.tables_dir / "overall_metrics.tex", index=False)
    write_csv_and_tex(tail_tbl, paths.tables_dir / "tail_abs_error.csv", paths.tables_dir / "tail_abs_error.tex", index=False)

    # -----------------------------
    # Stitched overall (all models)
    # -----------------------------
    stitched_truth = truth.sort_values("timestamp_utc").copy()
    stitched_baseline = make_stitched(base_join.rename(columns={"predicted_power_norm": "y_pred"}))
    stitched_baseline = stitched_baseline[["timestamp_utc", "y_true", "y_pred"]].copy()

    stitched_models: Dict[str, pd.DataFrame] = {}
    for name, jj in joined_models.items():
        # Need original hours_ahead series in jj for stitching, it exists
        st = make_stitched(jj)
        st = st[["timestamp_utc", "y_true", "y_pred"]].copy()
        stitched_models[name] = st

    stitched_rows.append({"model": args.baseline_name, **metrics_row(stitched_baseline["y_true"].values, stitched_baseline["y_pred"].values)})
    for name, st in stitched_models.items():
        stitched_rows.append({"model": name, **metrics_row(st["y_true"].values, st["y_pred"].values)})

    stitched_overall = pd.DataFrame(stitched_rows).sort_values("model")
    write_csv_and_tex(
        stitched_overall,
        paths.tables_dir / "overall_metrics_stitched.csv",
        paths.tables_dir / "overall_metrics_stitched.tex",
        index=False,
    )

    # -----------------------------
    # Monthly, lead buckets, daily (all models)
    # -----------------------------
    def add_time_cols(df: pd.DataFrame) -> pd.DataFrame:
        dd = df.copy()
        dd["month"] = dd["timestamp_utc"].dt.to_period("M").astype(str)
        dd["day"] = dd["timestamp_utc"].dt.date.astype(str)
        dd["lead_bucket"] = dd["hours_ahead"].apply(lead_bucket)
        return dd

    base2 = add_time_cols(base_aligned.copy())
    models2: Dict[str, pd.DataFrame] = {name: add_time_cols(jj.copy()) for name, jj in joined_models.items()}

    # Monthly metrics long form
    monthly_long_rows = []
    for m, g in base2.groupby("month"):
        monthly_long_rows.append({"month": m, "model": args.baseline_name, **metrics_row(g["y_true"].values, g["y_pred"].values)})
    for name, dfm in models2.items():
        for m, g in dfm.groupby("month"):
            monthly_long_rows.append({"month": m, "model": name, **metrics_row(g["y_true"].values, g["y_pred"].values)})

    monthly_long = pd.DataFrame(monthly_long_rows).sort_values(["month", "model"])
    write_csv_and_tex(monthly_long, paths.tables_dir / "monthly_metrics_long.csv", paths.tables_dir / "monthly_metrics_long.tex", index=False)

    # Lead bucket metrics long form
    lead_long_rows = []
    for b, g in base2.groupby("lead_bucket"):
        lead_long_rows.append({"lead_bucket": b, "model": args.baseline_name, **metrics_row(g["y_true"].values, g["y_pred"].values)})
    for name, dfm in models2.items():
        for b, g in dfm.groupby("lead_bucket"):
            lead_long_rows.append({"lead_bucket": b, "model": name, **metrics_row(g["y_true"].values, g["y_pred"].values)})

    lead_long = pd.DataFrame(lead_long_rows)
    order = {"0-24h": 0, "2-7d": 1, "8-30d": 2}
    lead_long["__ord"] = lead_long["lead_bucket"].map(order).fillna(99).astype(int)
    lead_long = lead_long.sort_values(["__ord", "model"]).drop(columns="__ord")
    write_csv_and_tex(lead_long, paths.tables_dir / "lead_bucket_metrics_long.csv", paths.tables_dir / "lead_bucket_metrics_long.tex", index=False)

    # Daily metrics, used for worst days and paired deltas
    daily_rows = []
    for d, g in base2.groupby("day"):
        daily_rows.append({"day": d, "model": args.baseline_name, **metrics_row(g["y_true"].values, g["y_pred"].values)})
    for name, dfm in models2.items():
        for d, g in dfm.groupby("day"):
            daily_rows.append({"day": d, "model": name, **metrics_row(g["y_true"].values, g["y_pred"].values)})

    daily_long = pd.DataFrame(daily_rows).sort_values(["day", "model"])
    write_csv_and_tex(daily_long, paths.tables_dir / "daily_metrics_long.csv", paths.tables_dir / "daily_metrics_long.tex", index=False)

    # Worst days per model (top 10 by RMSE)
    worst_rows = []
    for model in daily_long["model"].unique():
        dm = daily_long[daily_long["model"] == model].copy()
        dm = dm.sort_values("RMSE", ascending=False).head(10)
        dm = dm.assign(rank=np.arange(1, len(dm) + 1))
        worst_rows.append(dm)
    worst_tbl = pd.concat(worst_rows, axis=0, ignore_index=True)
    worst_tbl.to_csv((paths.tables_dir / "worst_10_days_per_model.csv").as_posix(), index=False)

    # Paired deltas vs baseline (daily MAE and RMSE)
    base_daily = daily_long[daily_long["model"] == args.baseline_name][["day", "MAE", "RMSE"]].rename(
        columns={"MAE": "MAE_baseline", "RMSE": "RMSE_baseline"}
    )
    paired_rows = []
    for name in models2.keys():
        md = daily_long[daily_long["model"] == name][["day", "MAE", "RMSE"]].rename(columns={"MAE": "MAE_model", "RMSE": "RMSE_model"})
        pp = md.merge(base_daily, on="day", how="inner")
        pp["delta_MAE"] = pp["MAE_model"] - pp["MAE_baseline"]
        pp["delta_RMSE"] = pp["RMSE_model"] - pp["RMSE_baseline"]

        mean_d, lo, hi = bootstrap_mean_ci(pp["delta_MAE"].values, n_boot=5000, seed=int(args.seed))
        frac_improved = float(np.mean(pp["delta_MAE"].values < 0))

        paired_rows.append(
            {
                "model": name,
                "mean_daily_delta_MAE": mean_d,
                "ci95_lo": lo,
                "ci95_hi": hi,
                "frac_days_improved_MAE": frac_improved,
                "N_days": int(len(pp)),
            }
        )

    paired_tbl = pd.DataFrame(paired_rows).sort_values("model")
    write_csv_and_tex(paired_tbl, paths.tables_dir / "paired_daily_deltas_vs_baseline.csv", paths.tables_dir / "paired_daily_deltas_vs_baseline.tex", index=False)

    # -----------------------------
    # Plots (FORMATTED)
    # -----------------------------
    # Monthly RMSE (all models)
    plot_monthly_rmse_all(monthly_long[["month", "model", "RMSE"]], paths.figures_dir / "monthly_rmse_all_models.png", baseline_label=args.baseline_name)

    # Facet grids: case studies
    stitched_truth2 = stitched_truth.rename(columns={"y_true": "y_true"}).copy()
    stitched_truth2 = stitched_truth2[["timestamp_utc", "y_true"]].copy()

    # FORMATTING FIX: Pass truth_label and baseline_label
    plot_facets_case_week(
        stitched_truth=stitched_truth2,
        stitched_baseline=stitched_baseline,
        stitched_models=stitched_models,
        start=args.case_summer_start,
        end=args.case_summer_end,
        out=paths.figures_dir / "facets_case_summer_week.png",
        title=f"Case Study: Summer Week",
        truth_label=args.truth_label,
        baseline_label=args.baseline_name,
    )

    plot_facets_case_week(
        stitched_truth=stitched_truth2,
        stitched_baseline=stitched_baseline,
        stitched_models=stitched_models,
        start=args.case_winter_start,
        end=args.case_winter_end,
        out=paths.figures_dir / "facets_case_winter_week.png",
        title=f"Case Study: Winter Week",
        truth_label=args.truth_label,
        baseline_label=args.baseline_name,
    )

    # Facet grid: abs error hist baseline vs each model
    joined_models_simple = {name: dfm[["timestamp_utc", "hours_ahead", "y_true", "y_pred"]].copy() for name, dfm in joined_models.items()}
    base_simple = base_aligned[["timestamp_utc", "hours_ahead", "y_true", "y_pred"]].copy()

    # FORMATTING FIX: Pass baseline_label
    plot_facets_abs_error_hist(
        joined_baseline=base_simple,
        joined_models=joined_models_simple,
        out=paths.figures_dir / "facets_abs_error_hist.png",
        title=f"Absolute Error Histogram",
        max_abs=1.0,
        baseline_label=args.baseline_name,
    )

    # Facet grid: lead-time RMSE curve up to 24h
    # FORMATTING FIX: Pass baseline_label
    plot_facets_leadtime_rmse_curve(
        joined_baseline=base_simple,
        joined_models=joined_models_simple,
        out=paths.figures_dir / "facets_leadtime_rmse_curve_0_24h.png",
        title=f"Lead-Time RMSE Curve (0–{args.max_hours_curve:.0f}h)",
        max_h=float(args.max_hours_curve),
        baseline_label=args.baseline_name,
    )

    # -----------------------------
    # Results markdown summary
    # -----------------------------
    md = []
    md.append("# Benchmark suite summary (v3 - FORMATTED)\n")
    md.append(f"- Truth: {Path(args.truth)}\n")
    md.append(f"- Baseline: {args.baseline_name} = {Path(args.baseline)}\n")
    md.append(f"- Models: {', '.join(models2.keys()) if len(models2) else '(none)'}\n")
    md.append(f"- Night filtering: {'OFF' if args.include_night else 'ON'} (y_true >= {args.daylight_threshold})\n")
    md.append(f"- Truth label: {args.truth_label}\n")
    md.append("\n## Overall metrics\n")
    md.append(overall.to_markdown(index=False))
    md.append("\n\n## Stitched overall metrics\n")
    md.append(stitched_overall.to_markdown(index=False))
    md.append("\n\n## Tail abs error\n")
    md.append(tail_tbl.to_markdown(index=False))
    md.append("\n\n## Paired daily deltas vs baseline\n")
    md.append(paired_tbl.to_markdown(index=False))
    (paths.text_dir / "results.md").write_text("\n".join(md))

    # Save eval joins for reuse
    base_aligned.to_parquet((paths.out_dir / "baseline_eval_joined.parquet").as_posix(), index=False)
    for name, jj in joined_models.items():
        safe_name = name.replace("/", "_").replace(" ", "_")
        jj.to_parquet((paths.out_dir / f"{safe_name}_eval_joined.parquet").as_posix(), index=False)

    print(f"[OK] Wrote FORMATTED benchmark outputs to: {paths.out_dir}")
    print(f"     - Figures: 300 DPI, clear labels")
    print(f"     - Truth label: {args.truth_label}")
    print(f"     - Baseline label: {args.baseline_name}")


if __name__ == "__main__":
    main()
