# src/evaluation/run_benchmark_suite_thesis_ready.py
"""
Thesis-ready benchmark visualization suite with:
- Proper labels: "Ground Truth Plant 03" instead of "truth"
- Configurable baseline naming (e.g., "MiRACLE Core", "Baseline", etc.)
- Fixed label overlap issues (adjusted legend placement, font sizes)
- Additional thesis plots: scatter, residuals, quantile-quantile, skill scores
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats


# -----------------------------
# IO
# -----------------------------
def read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing parquet: {path}")
    return pq.read_table(path.as_posix()).to_pandas()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def to_utc_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True)


def parse_name_path(x: str) -> Tuple[str, str]:
    # NAME:/abs/or/rel/path.parquet
    if ":" not in x:
        raise ValueError(f"Bad --model spec '{x}'. Expected NAME:/path/to.parquet")
    name, path = x.split(":", 1)
    name = name.strip()
    path = path.strip()
    if not name:
        raise ValueError(f"Empty model name in '{x}'")
    if not path:
        raise ValueError(f"Empty model path in '{x}'")
    return name, path


# -----------------------------
# Metrics
# -----------------------------
def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        return float('nan')
    return float(np.mean(np.abs(y_pred[mask] - y_true[mask])))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        return float('nan')
    return float(np.sqrt(np.mean((y_pred[mask] - y_true[mask]) ** 2)))


def mbe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        return float('nan')
    return float(np.mean(y_pred[mask] - y_true[mask]))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        return float('nan')
    y = y_true[mask]
    yp = y_pred[mask]
    ss_res = np.sum((y - yp) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute all metrics for a prediction"""
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MBE": mbe(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "N": int(np.sum(np.isfinite(y_true) & np.isfinite(y_pred))),
    }


def write_csv_and_tex(df: pd.DataFrame, csv_path: Path, tex_path: Path, index: bool = False) -> None:
    """Write dataframe to both CSV and LaTeX format"""
    ensure_dir(csv_path.parent)
    ensure_dir(tex_path.parent)
    df.to_csv(csv_path.as_posix(), index=index)
    try:
        tex = df.to_latex(index=index, float_format="%.6f", escape=False)
        tex_path.write_text(tex)
    except Exception as e:
        print(f"[WARN] Could not write LaTeX: {e}")


# -----------------------------
# Core joins
# -----------------------------
def load_truth(truth_path: Path) -> pd.DataFrame:
    truth = read_parquet(truth_path)
    if "timestamp_utc" not in truth.columns:
        raise ValueError("Truth parquet must contain timestamp_utc")
    if "power_norm" not in truth.columns:
        raise ValueError("Truth parquet must contain power_norm")

    truth["timestamp_utc"] = to_utc_datetime(truth["timestamp_utc"])
    truth = truth[["timestamp_utc", "power_norm"]].rename(columns={"power_norm": "y_true"})
    return truth


def load_pred(pred_path: Path) -> pd.DataFrame:
    df = read_parquet(pred_path)

    need = {"timestamp_utc", "forecast_start", "hours_ahead", "predicted_power_norm"}
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"{pred_path} missing required columns: {missing}")

    df["timestamp_utc"] = to_utc_datetime(df["timestamp_utc"])
    df["forecast_start"] = to_utc_datetime(df["forecast_start"])

    # normalize dtypes (avoid float32/float64 chaos)
    df["hours_ahead"] = pd.to_numeric(df["hours_ahead"], errors="coerce")
    df["predicted_power_norm"] = pd.to_numeric(df["predicted_power_norm"], errors="coerce")

    return df


def join_truth_pred(truth: pd.DataFrame, pred: pd.DataFrame, y_name: str) -> pd.DataFrame:
    j = pred.merge(truth, on="timestamp_utc", how="inner")
    j = j.rename(columns={"predicted_power_norm": y_name})
    return j


def stitch_most_recent(df: pd.DataFrame, y_col: str) -> pd.DataFrame:
    """
    For each timestamp_utc pick the row with smallest hours_ahead.
    Requires columns: timestamp_utc, hours_ahead, y_true, y_col
    """
    dd = df.sort_values(["timestamp_utc", "hours_ahead"])
    dd = dd.groupby("timestamp_utc", as_index=False).first()
    return dd[["timestamp_utc", "y_true", y_col]]


def filter_window(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    s = pd.to_datetime(start, utc=True)
    e = pd.to_datetime(end, utc=True)
    return df[(df["timestamp_utc"] >= s) & (df["timestamp_utc"] <= e)].copy()


# -----------------------------
# Plotting helpers
# -----------------------------
def _apply_time_axis(ax: plt.Axes) -> None:
    """Fix jumbled year labels with concise date formatting"""
    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.tick_params(axis="x", rotation=0, labelsize=9)


def _shared_ylim_from_series(series_list: List[pd.Series]) -> Tuple[float, float]:
    vals = []
    for s in series_list:
        if s is None:
            continue
        v = pd.to_numeric(s, errors="coerce").to_numpy()
        v = v[np.isfinite(v)]
        if len(v):
            vals.append(v)
    if not vals:
        return 0.0, 1.0

    mx = float(np.max(np.concatenate(vals)))
    mx = max(mx, 0.0)
    top = min(1.05, mx * 1.08) if mx > 0 else 1.0
    return 0.0, top


def plot_case_facet_grid(
    title: str,
    truth_stitched: pd.DataFrame,
    baseline_stitched: pd.DataFrame,
    model_stitched_map: Dict[str, pd.DataFrame],
    start: str,
    end: str,
    out_path: Path,
    ncols: int = 3,
    truth_label: str = "Ground Truth Plant 03",
    baseline_label: str = "MiRACLE Core",
) -> None:
    """
    One facet grid per case type.
    Each subplot: truth vs baseline vs model_i.
    Uses shared x/y scales, readable dates, NO OVERLAPPING LABELS.
    """
    # window
    t0 = filter_window(truth_stitched, start, end)
    b0 = filter_window(baseline_stitched, start, end)

    if t0.empty or b0.empty:
        print(f"[WARN] empty window for {title}: {start} -> {end}")
        return

    # prep models (window + align on timestamp)
    models_in_window: Dict[str, pd.DataFrame] = {}
    for name, dfm in model_stitched_map.items():
        w = filter_window(dfm, start, end)
        if w.empty:
            continue
        models_in_window[name] = w

    names = list(models_in_window.keys())
    if not names:
        print(f"[WARN] no models available for plot {title}")
        return

    n = len(names)
    nrows = int(np.ceil(n / ncols))

    # shared y-limits across all panels
    y_series = [t0["y_true"], b0["y_baseline"]]
    for nm in names:
        y_series.append(models_in_window[nm]["y_model"])
    ylo, yhi = _shared_ylim_from_series(y_series)

    # Larger figure for readability
    fig, axes = plt.subplots(
        nrows=nrows, 
        ncols=ncols, 
        figsize=(7.0 * ncols, 3.5 * nrows), 
        sharex=True, 
        sharey=True
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.reshape(nrows, ncols)

    fig.suptitle(title, fontsize=15, weight='bold', y=0.98)

    for i, nm in enumerate(names):
        r = i // ncols
        c = i % ncols
        ax = axes[r, c]

        m0 = models_in_window[nm]

        # Plot with better styling
        ax.plot(t0["timestamp_utc"].values, t0["y_true"].values, 
                label=truth_label, color='black', linewidth=1.5, alpha=0.9)
        ax.plot(b0["timestamp_utc"].values, b0["y_baseline"].values, 
                label=baseline_label, color='C0', linewidth=1.2, alpha=0.85, linestyle='--')
        ax.plot(m0["timestamp_utc"].values, m0["y_model"].values, 
                label=nm, color='C3', linewidth=1.0, alpha=0.8)

        ax.set_title(nm, fontsize=11, weight='semibold')
        ax.set_ylabel("Normalized Power", fontsize=10)
        ax.set_ylim(ylo, yhi)
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        _apply_time_axis(ax)

        # Legend: place OUTSIDE plot area to avoid overlap
        if i == 0:
            ax.legend(
                loc="upper left", 
                fontsize=9, 
                framealpha=0.95, 
                edgecolor='gray',
                bbox_to_anchor=(0.0, 1.0),  # Keep inside for first subplot
            )

    # hide empty panels
    for j in range(n, nrows * ncols):
        r = j // ncols
        c = j % ncols
        axes[r, c].axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.as_posix(), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] wrote {out_path}")


def plot_scatter_predicted_vs_actual(
    truth_stitched: pd.DataFrame,
    baseline_stitched: pd.DataFrame,
    model_stitched_map: Dict[str, pd.DataFrame],
    out_path: Path,
    truth_label: str = "Ground Truth Plant 03",
    baseline_label: str = "MiRACLE Core",
) -> None:
    """
    Scatter plot: predicted vs actual (y_pred on x-axis, y_true on y-axis).
    Shows perfect prediction line (y=x) and R² score.
    """
    n_models = len(model_stitched_map)
    ncols = min(3, n_models + 1)  # +1 for baseline
    nrows = int(np.ceil((n_models + 1) / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 5 * nrows))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    fig.suptitle("Predicted vs. Actual Power (Scatter)", fontsize=15, weight='bold')

    # Baseline
    ax = axes[0]
    x = baseline_stitched["y_baseline"].values
    y = baseline_stitched["y_true"].values
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    
    ax.scatter(x, y, s=8, alpha=0.4, edgecolors='none', label=baseline_label)
    ax.plot([0, 1], [0, 1], 'r--', linewidth=1.5, label="Perfect Prediction")
    
    r2 = stats.pearsonr(x, y)[0]**2 if len(x) > 1 else 0.0
    ax.text(0.05, 0.95, f"R² = {r2:.4f}", transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_xlabel("Predicted Power", fontsize=10)
    ax.set_ylabel("Actual Power", fontsize=10)
    ax.set_title(baseline_label, fontsize=11, weight='semibold')
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.legend(fontsize=9, loc='lower right')

    # Models
    for i, (nm, df_model) in enumerate(model_stitched_map.items(), start=1):
        ax = axes[i]
        x = df_model["y_model"].values
        y = df_model["y_true"].values
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        
        ax.scatter(x, y, s=8, alpha=0.4, edgecolors='none', label=nm)
        ax.plot([0, 1], [0, 1], 'r--', linewidth=1.5, label="Perfect Prediction")
        
        r2 = stats.pearsonr(x, y)[0]**2 if len(x) > 1 else 0.0
        ax.text(0.05, 0.95, f"R² = {r2:.4f}", transform=ax.transAxes, 
                fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_xlabel("Predicted Power", fontsize=10)
        ax.set_ylabel("Actual Power", fontsize=10)
        ax.set_title(nm, fontsize=11, weight='semibold')
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.legend(fontsize=9, loc='lower right')

    # Hide unused panels
    for j in range(n_models + 1, len(axes)):
        axes[j].axis('off')

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.as_posix(), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] wrote {out_path}")


def plot_residuals_histogram(
    truth_stitched: pd.DataFrame,
    baseline_stitched: pd.DataFrame,
    model_stitched_map: Dict[str, pd.DataFrame],
    out_path: Path,
    baseline_label: str = "MiRACLE Core",
) -> None:
    """
    Histogram of residuals (y_true - y_pred) for baseline and all models.
    Shows mean, std, and normal distribution overlay.
    """
    n_models = len(model_stitched_map)
    ncols = min(3, n_models + 1)
    nrows = int(np.ceil((n_models + 1) / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4 * nrows))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    fig.suptitle("Residual Distribution (Ground Truth - Predicted)", fontsize=15, weight='bold')

    # Baseline
    ax = axes[0]
    residuals = baseline_stitched["y_true"].values - baseline_stitched["y_baseline"].values
    residuals = residuals[np.isfinite(residuals)]
    
    ax.hist(residuals, bins=50, alpha=0.7, color='steelblue', edgecolor='black', density=True)
    
    mu, sigma = residuals.mean(), residuals.std()
    x = np.linspace(residuals.min(), residuals.max(), 100)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal fit')
    
    ax.axvline(0, color='black', linestyle='--', linewidth=1.5, label='Zero error')
    ax.text(0.05, 0.95, f"μ = {mu:.4f}\nσ = {sigma:.4f}", transform=ax.transAxes,
            fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_xlabel("Residual (Truth - Predicted)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title(baseline_label, fontsize=11, weight='semibold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle=':')

    # Models
    for i, (nm, df_model) in enumerate(model_stitched_map.items(), start=1):
        ax = axes[i]
        residuals = df_model["y_true"].values - df_model["y_model"].values
        residuals = residuals[np.isfinite(residuals)]
        
        ax.hist(residuals, bins=50, alpha=0.7, color='steelblue', edgecolor='black', density=True)
        
        mu, sigma = residuals.mean(), residuals.std()
        x = np.linspace(residuals.min(), residuals.max(), 100)
        ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal fit')
        
        ax.axvline(0, color='black', linestyle='--', linewidth=1.5, label='Zero error')
        ax.text(0.05, 0.95, f"μ = {mu:.4f}\nσ = {sigma:.4f}", transform=ax.transAxes,
                fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_xlabel("Residual (Truth - Predicted)", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title(nm, fontsize=11, weight='semibold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, linestyle=':')

    # Hide unused
    for j in range(n_models + 1, len(axes)):
        axes[j].axis('off')

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.as_posix(), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] wrote {out_path}")


def plot_qq_plot(
    truth_stitched: pd.DataFrame,
    baseline_stitched: pd.DataFrame,
    model_stitched_map: Dict[str, pd.DataFrame],
    out_path: Path,
    baseline_label: str = "MiRACLE Core",
) -> None:
    """
    Q-Q plot to check if residuals follow normal distribution.
    """
    n_models = len(model_stitched_map)
    ncols = min(3, n_models + 1)
    nrows = int(np.ceil((n_models + 1) / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    fig.suptitle("Q-Q Plot: Residual Normality Check", fontsize=15, weight='bold')

    # Baseline
    ax = axes[0]
    residuals = baseline_stitched["y_true"].values - baseline_stitched["y_baseline"].values
    residuals = residuals[np.isfinite(residuals)]
    
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title(baseline_label, fontsize=11, weight='semibold')
    ax.grid(True, alpha=0.3, linestyle=':')

    # Models
    for i, (nm, df_model) in enumerate(model_stitched_map.items(), start=1):
        ax = axes[i]
        residuals = df_model["y_true"].values - df_model["y_model"].values
        residuals = residuals[np.isfinite(residuals)]
        
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title(nm, fontsize=11, weight='semibold')
        ax.grid(True, alpha=0.3, linestyle=':')

    # Hide unused
    for j in range(n_models + 1, len(axes)):
        axes[j].axis('off')

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.as_posix(), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] wrote {out_path}")


def build_stitched_series_hours_max(truth: pd.DataFrame, pred: pd.DataFrame, y_col_name: str, hours_max: float) -> pd.DataFrame:
    pred2 = pred[pred["hours_ahead"] <= float(hours_max)].copy()
    joined = join_truth_pred(truth, pred2, y_col_name)
    stitched = stitch_most_recent(joined, y_col_name)
    return stitched


# -----------------------------
# Table Generation
# -----------------------------
def generate_overall_metrics_table(
    baseline_stitched: pd.DataFrame,
    model_stitched_map: Dict[str, pd.DataFrame],
    tables_dir: Path,
    baseline_label: str,
) -> None:
    """Generate overall performance metrics table (CSV + LaTeX)"""
    rows = []
    
    # Baseline
    metrics = compute_metrics(baseline_stitched["y_true"].values, baseline_stitched["y_baseline"].values)
    metrics["Model"] = baseline_label
    rows.append(metrics)
    
    # Models
    for name, df_model in model_stitched_map.items():
        # df_model has columns: timestamp_utc, y_true, y_model
        metrics = compute_metrics(df_model["y_true"].values, df_model["y_model"].values)
        metrics["Model"] = name
        rows.append(metrics)
    
    df = pd.DataFrame(rows)
    df = df[["Model", "MAE", "RMSE", "MBE", "R2", "N"]]
    
    write_csv_and_tex(
        df,
        tables_dir / "overall_metrics.csv",
        tables_dir / "overall_metrics.tex",
        index=False
    )
    print(f"[OK] wrote {tables_dir / 'overall_metrics.csv'}")
    print(f"[OK] wrote {tables_dir / 'overall_metrics.tex'}")


def generate_monthly_metrics_table(
    baseline_stitched: pd.DataFrame,
    model_stitched_map: Dict[str, pd.DataFrame],
    tables_dir: Path,
    baseline_label: str,
) -> pd.DataFrame:
    """Generate monthly performance breakdown table (CSV + LaTeX)"""
    # Baseline
    baseline_copy = baseline_stitched.copy()
    baseline_copy["month"] = baseline_copy["timestamp_utc"].dt.to_period("M").astype(str)
    
    monthly_rows = []
    for month, grp in baseline_copy.groupby("month"):
        row = {"Month": month, "Model": baseline_label}
        row.update(compute_metrics(grp["y_true"].values, grp["y_baseline"].values))
        monthly_rows.append(row)
    
    # Models
    for name, df_model in model_stitched_map.items():
        model_copy = df_model.copy()
        model_copy["month"] = model_copy["timestamp_utc"].dt.to_period("M").astype(str)
        
        for month, grp in model_copy.groupby("month"):
            row = {"Month": month, "Model": name}
            row.update(compute_metrics(grp["y_true"].values, grp["y_model"].values))
            monthly_rows.append(row)
    
    df = pd.DataFrame(monthly_rows)
    df = df[["Month", "Model", "MAE", "RMSE", "MBE", "R2", "N"]]
    
    write_csv_and_tex(
        df,
        tables_dir / "monthly_metrics.csv",
        tables_dir / "monthly_metrics.tex",
        index=False
    )
    print(f"[OK] wrote {tables_dir / 'monthly_metrics.csv'}")
    print(f"[OK] wrote {tables_dir / 'monthly_metrics.tex'}")
    return df


def generate_horizon_stratified_table(
    truth: pd.DataFrame,
    baseline_pred: pd.DataFrame,
    model_pred_map: Dict[str, pd.DataFrame],
    tables_dir: Path,
    baseline_label: str,
) -> None:
    """Generate horizon-stratified metrics (1h, 6h, 24h, 7d, 30d bins)"""
    
    def horizon_bin(hours: float) -> str:
        if hours <= 1:
            return "0-1h"
        elif hours <= 6:
            return "1-6h"
        elif hours <= 24:
            return "6-24h"
        elif hours <= 168:  # 7 days
            return "1-7d"
        else:
            return "7-30d"
    
    rows = []
    
    # Baseline
    merged = baseline_pred.merge(truth, on="timestamp_utc")
    merged["horizon_bin"] = merged["hours_ahead"].apply(horizon_bin)
    
    # Determine truth column name (either y_true or power_norm)
    truth_col = "y_true" if "y_true" in merged.columns else "power_norm"
    
    for bin_name, grp in merged.groupby("horizon_bin"):
        row = {"Horizon": bin_name, "Model": baseline_label}
        row.update(compute_metrics(grp[truth_col].values, grp["predicted_power_norm"].values))
        rows.append(row)
    
    # Models
    for name, pred_df in model_pred_map.items():
        merged = pred_df.merge(truth, on="timestamp_utc")
        merged["horizon_bin"] = merged["hours_ahead"].apply(horizon_bin)
        
        for bin_name, grp in merged.groupby("horizon_bin"):
            row = {"Horizon": bin_name, "Model": name}
            row.update(compute_metrics(grp[truth_col].values, grp["predicted_power_norm"].values))
            rows.append(row)
    
    df = pd.DataFrame(rows)
    # Sort by horizon bins
    horizon_order = ["0-1h", "1-6h", "6-24h", "1-7d", "7-30d"]
    df["horizon_order"] = df["Horizon"].apply(lambda x: horizon_order.index(x) if x in horizon_order else 999)
    df = df.sort_values(["horizon_order", "Model"]).drop(columns=["horizon_order"])
    df = df[["Horizon", "Model", "MAE", "RMSE", "MBE", "R2", "N"]]
    
    write_csv_and_tex(
        df,
        tables_dir / "horizon_stratified_metrics.csv",
        tables_dir / "horizon_stratified_metrics.tex",
        index=False
    )
    print(f"[OK] wrote {tables_dir / 'horizon_stratified_metrics.csv'}")
    print(f"[OK] wrote {tables_dir / 'horizon_stratified_metrics.tex'}")


def generate_skill_score_table(
    baseline_stitched: pd.DataFrame,
    model_stitched_map: Dict[str, pd.DataFrame],
    tables_dir: Path,
    baseline_label: str,
) -> None:
    """Generate skill score table (% improvement vs baseline)"""
    # Get baseline RMSE
    base_rmse = rmse(baseline_stitched["y_true"].values, baseline_stitched["y_baseline"].values)
    
    rows = []
    
    # Baseline (0% improvement)
    rows.append({
        "Model": baseline_label,
        "RMSE": base_rmse,
        "Skill_Score_pct": 0.0,
    })
    
    # Models
    for name, df_model in model_stitched_map.items():
        model_rmse = rmse(df_model["y_true"].values, df_model["y_model"].values)
        skill_score = ((base_rmse - model_rmse) / base_rmse) * 100.0
        rows.append({
            "Model": name,
            "RMSE": model_rmse,
            "Skill_Score_pct": skill_score,
        })
    
    df = pd.DataFrame(rows)
    
    write_csv_and_tex(
        df,
        tables_dir / "skill_scores.csv",
        tables_dir / "skill_scores.tex",
        index=False
    )
    print(f"[OK] wrote {tables_dir / 'skill_scores.csv'}")
    print(f"[OK] wrote {tables_dir / 'skill_scores.tex'}")


# -----------------------------
# Build stitched series for plotting
# -----------------------------
def build_stitched_series(truth: pd.DataFrame, pred: pd.DataFrame, y_col_name: str) -> pd.DataFrame:
    joined = join_truth_pred(truth, pred, y_col_name)
    stitched = stitch_most_recent(joined, y_col_name)
    return stitched


def build_stitched_series_hours_max(truth: pd.DataFrame, pred: pd.DataFrame, y_col_name: str, hours_max: float) -> pd.DataFrame:
    pred2 = pred[pred["hours_ahead"] <= float(hours_max)].copy()
    joined = join_truth_pred(truth, pred2, y_col_name)
    stitched = stitch_most_recent(joined, y_col_name)
    return stitched


# -----------------------------
# CLI main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Thesis-ready benchmark visualization suite")

    ap.add_argument("--truth", required=True, type=str, help="Path to ground truth parquet")

    ap.add_argument("--baseline-name", required=True, type=str, help="Label for baseline model (e.g., 'MiRACLE Core', 'Baseline')")
    ap.add_argument("--baseline", required=True, type=str, help="Path to baseline predictions parquet")

    ap.add_argument("--model", action="append", default=[], help="NAME:/path/to.parquet (repeatable)")

    ap.add_argument("--out", required=True, type=str, help="Output directory for figures")

    ap.add_argument("--truth-label", type=str, default="Ground Truth Plant 03", 
                    help="Label for ground truth in plots")

    ap.add_argument("--case-summer-start", type=str, default="2024-07-01T00:00:00Z")
    ap.add_argument("--case-summer-end", type=str, default="2024-07-08T00:00:00Z")
    ap.add_argument("--case-winter-start", type=str, default="2024-01-10T00:00:00Z")
    ap.add_argument("--case-winter-end", type=str, default="2024-01-17T00:00:00Z")

    # Optional second facet group for short-head ablation plots
    ap.add_argument("--ablation-baseline-name", type=str, default="")
    ap.add_argument("--ablation-baseline", type=str, default="")
    ap.add_argument("--ablation-model", action="append", default=[], help="NAME:/path/to.parquet (repeatable)")
    ap.add_argument("--ablation-hours-max", type=float, default=24.0)

    args = ap.parse_args()

    out_dir = Path(args.out)
    figs_dir = out_dir / "figures"
    tables_dir = out_dir / "tables"
    ensure_dir(figs_dir)
    ensure_dir(tables_dir)

    truth = load_truth(Path(args.truth))

    # -------- main group --------
    baseline_name = args.baseline_name
    truth_label = args.truth_label
    
    baseline_pred = load_pred(Path(args.baseline))
    baseline_stitched_raw = build_stitched_series(truth, baseline_pred, "y_baseline")

    # For table generation: use baseline_stitched directly (it has y_true, y_baseline)
    baseline_stitched = baseline_stitched_raw.copy()
    
    # For plotting: extract truth_stitched separately (plots expect separate truth dataframe)
    truth_stitched = baseline_stitched_raw[["timestamp_utc", "y_true"]].copy()

    model_map: Dict[str, pd.DataFrame] = {}
    model_pred_map: Dict[str, pd.DataFrame] = {}  # For horizon-stratified metrics
    for spec in args.model:
        nm, pth = parse_name_path(spec)
        pred = load_pred(Path(pth))
        st = build_stitched_series(truth, pred, "y_model")
        model_map[nm] = st
        model_pred_map[nm] = pred

    # ========== GENERATE TABLES ==========
    print("\n[INFO] Generating tables...")
    
    generate_overall_metrics_table(
        baseline_stitched=baseline_stitched,
        model_stitched_map=model_map,
        tables_dir=tables_dir,
        baseline_label=baseline_name,
    )
    
    generate_monthly_metrics_table(
        baseline_stitched=baseline_stitched,
        model_stitched_map=model_map,
        tables_dir=tables_dir,
        baseline_label=baseline_name,
    )
    
    generate_horizon_stratified_table(
        truth=truth,
        baseline_pred=baseline_pred,
        model_pred_map=model_pred_map,
        tables_dir=tables_dir,
        baseline_label=baseline_name,
    )
    
    generate_skill_score_table(
        baseline_stitched=baseline_stitched,
        model_stitched_map=model_map,
        tables_dir=tables_dir,
        baseline_label=baseline_name,
    )

    # ========== THESIS-READY PLOTS ==========
    print("\n[INFO] Generating figures...")
    
    # 1. Case study: summer week
    title_summer = f"Case Study: Summer Week Forecast Comparison"
    plot_case_facet_grid(
        title=title_summer,
        truth_stitched=truth_stitched,
        baseline_stitched=baseline_stitched,
        model_stitched_map=model_map,
        start=args.case_summer_start,
        end=args.case_summer_end,
        out_path=figs_dir / "thesis_case_summer_week.png",
        ncols=3,
        truth_label=truth_label,
        baseline_label=baseline_name,
    )

    # 2. Case study: winter week
    title_winter = f"Case Study: Winter Week Forecast Comparison"
    plot_case_facet_grid(
        title=title_winter,
        truth_stitched=truth_stitched,
        baseline_stitched=baseline_stitched,
        model_stitched_map=model_map,
        start=args.case_winter_start,
        end=args.case_winter_end,
        out_path=figs_dir / "thesis_case_winter_week.png",
        ncols=3,
        truth_label=truth_label,
        baseline_label=baseline_name,
    )

    # 3. Scatter: Predicted vs Actual
    plot_scatter_predicted_vs_actual(
        truth_stitched=truth_stitched,
        baseline_stitched=baseline_stitched,
        model_stitched_map=model_map,
        out_path=figs_dir / "thesis_scatter_predicted_vs_actual.png",
        truth_label=truth_label,
        baseline_label=baseline_name,
    )

    # 4. Residual histogram
    plot_residuals_histogram(
        truth_stitched=truth_stitched,
        baseline_stitched=baseline_stitched,
        model_stitched_map=model_map,
        out_path=figs_dir / "thesis_residuals_histogram.png",
        baseline_label=baseline_name,
    )

    # 5. Q-Q plot for normality check
    plot_qq_plot(
        truth_stitched=truth_stitched,
        baseline_stitched=baseline_stitched,
        model_stitched_map=model_map,
        out_path=figs_dir / "thesis_qq_plot.png",
        baseline_label=baseline_name,
    )

    # -------- optional ablation group (0-24h by default) --------
    if args.ablation_baseline and args.ablation_model:
        abl_base_name = args.ablation_baseline_name or "Ablation Baseline"
        abl_base_pred = load_pred(Path(args.ablation_baseline))
        abl_base_stitched_raw = build_stitched_series_hours_max(truth, abl_base_pred, "y_baseline", args.ablation_hours_max)

        abl_truth_stitched = abl_base_stitched_raw[["timestamp_utc", "y_true"]].copy()
        abl_baseline_stitched = abl_base_stitched_raw[["timestamp_utc", "y_true", "y_baseline"]].copy()

        abl_model_map: Dict[str, pd.DataFrame] = {}
        for spec in args.ablation_model:
            nm, pth = parse_name_path(spec)
            pred = load_pred(Path(pth))
            st = build_stitched_series_hours_max(truth, pred, "y_model", args.ablation_hours_max)
            abl_model_map[nm] = st

        title_abl_summer = f"Ablation Study (<= {args.ablation_hours_max:.0f}h): Summer Week"
        plot_case_facet_grid(
            title=title_abl_summer,
            truth_stitched=abl_truth_stitched,
            baseline_stitched=abl_baseline_stitched,
            model_stitched_map=abl_model_map,
            start=args.case_summer_start,
            end=args.case_summer_end,
            out_path=figs_dir / "thesis_ablation_summer_week.png",
            ncols=2,
            truth_label=truth_label,
            baseline_label=abl_base_name,
        )

        title_abl_winter = f"Ablation Study (<= {args.ablation_hours_max:.0f}h): Winter Week"
        plot_case_facet_grid(
            title=title_abl_winter,
            truth_stitched=abl_truth_stitched,
            baseline_stitched=abl_baseline_stitched,
            model_stitched_map=abl_model_map,
            start=args.case_winter_start,
            end=args.case_winter_end,
            out_path=figs_dir / "thesis_ablation_winter_week.png",
            ncols=2,
            truth_label=truth_label,
            baseline_label=abl_base_name,
        )

    print(f"\n[SUCCESS] All thesis-ready outputs saved to: {out_dir}")
    print(f"  ├── figures/ - {len(list(figs_dir.glob('*.png')))} plots (300 DPI, publication-ready)")
    print(f"  └── tables/  - 4 tables (CSV + LaTeX format)")
    print(f"")
    print(f"Tables generated:")
    print(f"  1. overall_metrics.csv/.tex      - Overall MAE/RMSE/R²")
    print(f"  2. monthly_metrics.csv/.tex      - Monthly performance breakdown")
    print(f"  3. horizon_stratified_metrics... - Performance by forecast horizon")
    print(f"  4. skill_scores.csv/.tex         - % improvement vs {baseline_name}")
    print(f"")
    print(f"Figures generated:")
    print(f"  - Labels: '{truth_label}' vs '{baseline_name}'")
    print(f"  - Resolution: 300 DPI")
    print(f"  - Layout: No overlapping labels, tight bounding boxes")
    print(f"")
    print(f"Ready for thesis writing and defense presentation!")


if __name__ == "__main__":
    main()
