# src/evaluation/run_benchmark_suite_v2.py
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

    # step_ahead may exist, but is optional for plotting
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
    # This fixes the “jumbled year labels”
    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.tick_params(axis="x", rotation=0)


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
) -> None:
    """
    One facet grid per case type.
    Each subplot: truth vs baseline vs model_i.
    Uses shared x/y scales, readable dates.
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

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6.6 * ncols, 3.2 * nrows), sharex=True, sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.reshape(nrows, ncols)

    fig.suptitle(title, fontsize=14)

    for i, nm in enumerate(names):
        r = i // ncols
        c = i % ncols
        ax = axes[r, c]

        m0 = models_in_window[nm]

        ax.plot(t0["timestamp_utc"].values, t0["y_true"].values, label="truth")
        ax.plot(b0["timestamp_utc"].values, b0["y_baseline"].values, label="baseline")
        ax.plot(m0["timestamp_utc"].values, m0["y_model"].values, label=nm)

        ax.set_title(nm)
        ax.set_ylabel("power_norm")
        ax.set_ylim(ylo, yhi)
        _apply_time_axis(ax)

        # legend only once to avoid clutter
        if i == 0:
            ax.legend(loc="upper right")

    # hide empty panels
    for j in range(n, nrows * ncols):
        r = j // ncols
        c = j % ncols
        axes[r, c].axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.as_posix(), dpi=200)
    plt.close(fig)
    print(f"[OK] wrote {out_path}")


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
    ap = argparse.ArgumentParser()

    ap.add_argument("--truth", required=True, type=str)

    ap.add_argument("--baseline-name", required=True, type=str)
    ap.add_argument("--baseline", required=True, type=str)

    ap.add_argument("--model", action="append", default=[], help="NAME:/path/to.parquet (repeatable)")

    ap.add_argument("--out", required=True, type=str)

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
    ensure_dir(figs_dir)

    truth = load_truth(Path(args.truth))

    # -------- main group --------
    baseline_name = args.baseline_name
    baseline_pred = load_pred(Path(args.baseline))
    baseline_stitched_raw = build_stitched_series(truth, baseline_pred, "y_baseline")

    # rename to canonical columns for plotting
    truth_stitched = baseline_stitched_raw[["timestamp_utc", "y_true"]].copy()
    baseline_stitched = baseline_stitched_raw[["timestamp_utc", "y_true", "y_baseline"]].copy()

    model_map: Dict[str, pd.DataFrame] = {}
    for spec in args.model:
        nm, pth = parse_name_path(spec)
        pred = load_pred(Path(pth))
        st = build_stitched_series(truth, pred, "y_model")
        model_map[nm] = st

    title_summer = f"Case study summer week, truth vs {baseline_name} vs each model"
    plot_case_facet_grid(
        title=title_summer,
        truth_stitched=truth_stitched,
        baseline_stitched=baseline_stitched,
        model_stitched_map=model_map,
        start=args.case_summer_start,
        end=args.case_summer_end,
        out_path=figs_dir / "facets_case_summer_week.png",
        ncols=3,
    )

    title_winter = f"Case study winter week, truth vs {baseline_name} vs each model"
    plot_case_facet_grid(
        title=title_winter,
        truth_stitched=truth_stitched,
        baseline_stitched=baseline_stitched,
        model_stitched_map=model_map,
        start=args.case_winter_start,
        end=args.case_winter_end,
        out_path=figs_dir / "facets_case_winter_week.png",
        ncols=3,
    )

    # -------- optional ablation group (0-24h by default) --------
    if args.ablation_baseline and args.ablation_model:
        abl_base_name = args.ablation_baseline_name or "ablation_baseline"
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

        title_abl_summer = f"Short-head ablation (<= {args.ablation_hours_max:.0f}h), truth vs {abl_base_name} vs each model"
        plot_case_facet_grid(
            title=title_abl_summer,
            truth_stitched=abl_truth_stitched,
            baseline_stitched=abl_baseline_stitched,
            model_stitched_map=abl_model_map,
            start=args.case_summer_start,
            end=args.case_summer_end,
            out_path=figs_dir / "facets_short_ablation_summer_week.png",
            ncols=2,  # usually 4 models, 2x2 looks clean
        )

        title_abl_winter = f"Short-head ablation (<= {args.ablation_hours_max:.0f}h), truth vs {abl_base_name} vs each model"
        plot_case_facet_grid(
            title=title_abl_winter,
            truth_stitched=abl_truth_stitched,
            baseline_stitched=abl_baseline_stitched,
            model_stitched_map=abl_model_map,
            start=args.case_winter_start,
            end=args.case_winter_end,
            out_path=figs_dir / "facets_short_ablation_winter_week.png",
            ncols=2,
        )

    print(f"[OK] done. figures in: {figs_dir}")


if __name__ == "__main__":
    main()
