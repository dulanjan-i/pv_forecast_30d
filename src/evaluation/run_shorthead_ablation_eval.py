from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt


def read_parquet(p: Path) -> pd.DataFrame:
    t = pq.read_table(p.as_posix())
    return t.to_pandas()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def rmse(y: np.ndarray, yp: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - yp) ** 2)))


def mae(y: np.ndarray, yp: np.ndarray) -> float:
    return float(np.mean(np.abs(y - yp)))


def mbe(y: np.ndarray, yp: np.ndarray) -> float:
    return float(np.mean(yp - y))


def r2(y: np.ndarray, yp: np.ndarray) -> float:
    ss_res = np.sum((y - yp) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float("nan") if ss_tot == 0 else float(1.0 - ss_res / ss_tot)


def metrics_row(y: np.ndarray, yp: np.ndarray) -> Dict[str, float]:
    return {
        "MAE": mae(y, yp),
        "RMSE": rmse(y, yp),
        "nRMSE": rmse(y, yp),  # power_norm already cap-normalized
        "MBE": mbe(y, yp),
        "R2": r2(y, yp),
        "N": int(len(y)),
    }


def tail_row(abs_err: np.ndarray) -> Dict[str, float]:
    return {
        "P90": float(np.quantile(abs_err, 0.90)),
        "P95": float(np.quantile(abs_err, 0.95)),
        "P99": float(np.quantile(abs_err, 0.99)),
        "mean_abs": float(np.mean(abs_err)),
    }


def stitched(df: pd.DataFrame) -> pd.DataFrame:
    # Choose the smallest lead per timestamp (most recent forecast proxy)
    df = df.sort_values(["timestamp_utc", "hours_ahead"])
    return df.groupby("timestamp_utc", as_index=False).first()


def lead_rmse_curve(df: pd.DataFrame, max_hours: float) -> pd.DataFrame:
    dd = df[df["hours_ahead"] <= max_hours].copy()
    rows = []
    for h, g in dd.groupby("hours_ahead"):
        rows.append({"hours_ahead": float(h), "RMSE": rmse(g["y_true"].values, g["y_pred"].values), "N": int(len(g))})
    out = pd.DataFrame(rows).sort_values("hours_ahead")
    return out


def plot_case(ax, st: pd.DataFrame, start: str, end: str, title: str) -> None:
    s = pd.to_datetime(start, utc=True)
    e = pd.to_datetime(end, utc=True)
    d = st[(st["timestamp_utc"] >= s) & (st["timestamp_utc"] <= e)]
    ax.plot(d["timestamp_utc"].values, d["y_true"].values, label="truth")
    ax.plot(d["timestamp_utc"].values, d["y_pred"].values, label="pred")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)


def plot_hist(ax, st: pd.DataFrame, title: str) -> None:
    ax.hist(st["abs_err"].values, bins=80)
    ax.set_title(title)


def plot_lead_curve(ax, curve: pd.DataFrame, title: str) -> None:
    ax.plot(curve["hours_ahead"].values, curve["RMSE"].values)
    ax.set_title(title)
    ax.set_xlabel("hours_ahead")
    ax.set_ylabel("RMSE")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True, type=str)
    ap.add_argument("--tft-only", required=True, type=str)
    ap.add_argument("--tft-pvlib", required=True, type=str)
    ap.add_argument("--tft-lstm", required=True, type=str)
    ap.add_argument("--full", required=True, type=str)
    ap.add_argument("--out", required=True, type=str)

    ap.add_argument("--max-hours", type=float, default=24.0)
    ap.add_argument("--daylight-threshold", type=float, default=0.01)
    ap.add_argument("--include-night", action="store_true")

    ap.add_argument("--case-summer-start", type=str, default="2024-07-01T00:00:00Z")
    ap.add_argument("--case-summer-end", type=str, default="2024-07-08T00:00:00Z")
    ap.add_argument("--case-winter-start", type=str, default="2024-01-10T00:00:00Z")
    ap.add_argument("--case-winter-end", type=str, default="2024-01-17T00:00:00Z")

    args = ap.parse_args()

    out_dir = Path(args.out)
    tables = out_dir / "tables"
    figs = out_dir / "figures"
    text = out_dir / "text"
    ensure_dir(tables); ensure_dir(figs); ensure_dir(text)

    truth = read_parquet(Path(args.truth))
    truth["timestamp_utc"] = pd.to_datetime(truth["timestamp_utc"], utc=True)
    if "power_norm" not in truth.columns:
        raise ValueError("truth must contain power_norm")
    truth = truth[["timestamp_utc", "power_norm"]].rename(columns={"power_norm": "y_true"})

    models = {
        "TFT-only": Path(args.tft_only),
        "TFT + PVLib": Path(args.tft_pvlib),
        "TFT + LSTM": Path(args.tft_lstm),
        "TFT + LSTM + PVLib (full)": Path(args.full),
    }

    stitched_series: Dict[str, pd.DataFrame] = {}
    lead_curves: Dict[str, pd.DataFrame] = {}
    overall_rows: List[Dict[str, float]] = []
    tail_rows: List[Dict[str, float]] = []
    worst_rows: List[pd.DataFrame] = []

    for name, p in models.items():
        df = read_parquet(p)
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
        df["forecast_start"] = pd.to_datetime(df["forecast_start"], utc=True)

        # join with truth
        j = df.merge(truth, on="timestamp_utc", how="inner")
        if "predicted_power_norm" not in j.columns:
            raise ValueError(f"{name} missing predicted_power_norm")

        # focus on 0-24h short head evaluation
        j = j[j["hours_ahead"] <= float(args.max_hours)].copy()

        if not args.include_night:
            j = j[j["y_true"] >= float(args.daylight_threshold)].copy()

        j = j.rename(columns={"predicted_power_norm": "y_pred"})
        j["abs_err"] = (j["y_pred"] - j["y_true"]).abs()

        st = stitched(j)
        stitched_series[name] = st
        lead_curves[name] = lead_rmse_curve(j.rename(columns={"y_pred": "y_pred"}), float(args.max_hours))

        overall = metrics_row(st["y_true"].values, st["y_pred"].values)
        overall_rows.append({"model": name, **overall})

        tail = tail_row(st["abs_err"].values)
        tail_rows.append({"model": name, **tail})

        # worst days by daily RMSE on stitched
        st2 = st.copy()
        st2["day"] = st2["timestamp_utc"].dt.date.astype(str)
        daily = st2.groupby("day").apply(lambda g: pd.Series({
            "MAE": mae(g["y_true"].values, g["y_pred"].values),
            "RMSE": rmse(g["y_true"].values, g["y_pred"].values),
            "N": int(len(g)),
        })).reset_index()
        daily = daily.sort_values("RMSE", ascending=False).head(10)
        daily["model"] = name
        worst_rows.append(daily)

    overall_tbl = pd.DataFrame(overall_rows).sort_values("RMSE")
    tail_tbl = pd.DataFrame(tail_rows).sort_values("P95")
    worst_tbl = pd.concat(worst_rows, ignore_index=True)

    overall_tbl.to_csv(tables / "shorthead_ablation_overall_stitched.csv", index=False)
    tail_tbl.to_csv(tables / "shorthead_ablation_tail_stitched.csv", index=False)
    worst_tbl.to_csv(tables / "shorthead_ablation_worst10_by_model.csv", index=False)

    # 4x4 plot grid
    fig, axes = plt.subplots(4, 4, figsize=(20, 14))
    col_names = list(models.keys())

    for c, name in enumerate(col_names):
        st = stitched_series[name]
        curve = lead_curves[name]

        plot_case(axes[0, c], st, args.case_summer_start, args.case_summer_end, f"{name}\nSummer week (stitched)")
        plot_case(axes[1, c], st, args.case_winter_start, args.case_winter_end, f"{name}\nWinter week (stitched)")
        plot_hist(axes[2, c], st, f"{name}\nAbs error hist (stitched)")
        plot_lead_curve(axes[3, c], curve, f"{name}\nLead RMSE curve (0–24h)")

    for r in range(4):
        for c in range(4):
            axes[r, c].grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig((figs / "shorthead_ablation_4x4.png").as_posix(), dpi=200)
    plt.close(fig)

    # write a short markdown summary
    md = []
    md.append("# Short-head ablation evaluation (0–24h)\n")
    md.append(f"- max_hours: {args.max_hours}\n")
    md.append(f"- night filtering: {'OFF' if args.include_night else 'ON'} (y_true >= {args.daylight_threshold})\n\n")
    md.append("## Overall metrics (stitched)\n")
    md.append(overall_tbl.to_markdown(index=False))
    md.append("\n\n## Tail abs error (stitched)\n")
    md.append(tail_tbl.to_markdown(index=False))
    (text / "shorthead_ablation_results.md").write_text("\n".join(md))

    print(f"[OK] wrote: {out_dir}")


if __name__ == "__main__":
    main()
