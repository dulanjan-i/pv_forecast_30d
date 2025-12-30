# src/validation/summarize_plant_finetune.py
"""
Summarize plant-level fine-tuning jobs (warm vs cold) into one CSV.

What it does:
- Finds run directories under a given base (default: experiments/tft/runs/germany/plant_03/15min)
- For each run, reads logs/metrics.csv to compute:
  - best_val_loss
  - best_epoch
  - last_epoch, last_val_loss (if available)
- Optionally parses Slurm jobids from your /shared/$USER/miracle/logs/*.out and pulls:
  - elapsed, node, state, exitcode via sacct
- Writes:
  - CSV summary
  - prints a compact table + stats

Notes:
- Uses metrics.csv as source of truth (more reliable than grepping logs).
- If metrics.csv schema differs slightly, it tries common column names.

Example:
  singularity exec --nv "$IMG" python3 -m src.validation.summarize_plant_finetune \
    --run_base experiments/tft/runs/germany/plant_03/15min \
    --log_glob "/shared/$USER/miracle/logs/plant03_pvlib_*.out" \
    --out_csv experiments/tft/runs/germany/plant_03/15min/finetune_summary.csv
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import pandas as pd


# ----------------------------
# helpers
# ----------------------------

def _read_metrics_csv(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    # Normalize epoch column
    if "epoch" not in df.columns:
        if "Epoch" in df.columns:
            df = df.rename(columns={"Epoch": "epoch"})
        elif "step" in df.columns:
            # Some loggers only have step, epoch might be inferred badly, but keep step as epoch fallback
            df["epoch"] = df["step"]
    return df


def _find_val_loss_col(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "val_loss",
        "val/loss",
        "val_loss_epoch",
        "val_loss_step",
        "val_loss_mean",
        "valid_loss",
        "val/QuantileLoss",
        "val_quantile_loss",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    # last resort: any column containing 'val' and 'loss'
    for c in df.columns:
        cl = c.lower()
        if "val" in cl and "loss" in cl:
            return c
    return None


def _best_from_metrics(metrics_csv: Path) -> Tuple[Optional[float], Optional[int], Optional[float], Optional[int]]:
    """
    Returns:
      best_val, best_epoch, last_val, last_epoch
    """
    if not metrics_csv.exists():
        return None, None, None, None

    df = _read_metrics_csv(metrics_csv)
    val_col = _find_val_loss_col(df)
    if val_col is None:
        return None, None, None, None

    # keep rows where val is present
    d = df.dropna(subset=[val_col]).copy()
    if d.empty:
        return None, None, None, None

    # best
    best_idx = d[val_col].astype(float).idxmin()
    best_val = float(d.loc[best_idx, val_col])
    best_epoch = int(d.loc[best_idx, "epoch"]) if "epoch" in d.columns else None

    # last (by epoch then by file order)
    if "epoch" in d.columns:
        d2 = d.sort_values(["epoch"])
        last_row = d2.iloc[-1]
        last_epoch = int(last_row["epoch"])
        last_val = float(last_row[val_col])
    else:
        last_row = d.iloc[-1]
        last_epoch = None
        last_val = float(last_row[val_col])

    return best_val, best_epoch, last_val, last_epoch


def _sacct(jobid: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    Returns: elapsed, nodelist, state, exitcode
    """
    try:
        out = subprocess.check_output(
            ["sacct", "-j", jobid, "--format=JobID,Elapsed,NodeList%40,State,ExitCode", "-X", "-n", "-P"],
            text=True,
        ).strip().splitlines()

        # take first non-empty row
        for row in out:
            if row.strip():
                job, elapsed, nodelist, state, exitcode = row.split("|")
                return elapsed, nodelist, state, exitcode
    except Exception:
        pass
    return None, None, None, None


@dataclass
class LogJobInfo:
    jobid: str
    regime: str
    log_path: Path


def _parse_jobid_and_regime_from_logname(p: Path) -> Optional[LogJobInfo]:
    """
    Accepts names like:
      plant03_pvlib_warm_24493_42.out
      plant03_pvlib_cold_24496_44.out
    """
    m = re.search(r"plant03_pvlib_(warm|cold)_(\d+_\d+)\.out$", p.name)
    if not m:
        return None
    regime = m.group(1)
    jobid = m.group(2)
    return LogJobInfo(jobid=jobid, regime=regime, log_path=p)


def _guess_regime_from_run_dir(run_dir: Path) -> str:
    s = str(run_dir).lower()
    if "warmstart" in s or "warm_start" in s or "warmstart_from" in s:
        return "warm"
    if "coldstart" in s or "cold_start" in s:
        return "cold"
    return "unknown"


def _collect_run_dirs(run_base: Path) -> List[Path]:
    # collect leaf run dirs that contain logs/metrics.csv
    out = []
    for p in run_base.rglob("logs/metrics.csv"):
        out.append(p.parent.parent)  # .../<run>/logs/metrics.csv -> .../<run>
    return sorted(set(out))


# ----------------------------
# main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_base", type=str, default="experiments/tft/runs/germany/plant_03/15min",
                    help="Base folder where plant fine-tune runs live.")
    ap.add_argument("--log_glob", type=str, default="",
                    help="Glob for slurm .out logs to pull jobid/node/elapsed via sacct. "
                         "Example: /shared/$USER/miracle/logs/plant03_pvlib_*.out")
    ap.add_argument("--out_csv", type=str, default="experiments/tft/runs/germany/plant_03/15min/finetune_summary.csv",
                    help="Output CSV path.")
    args = ap.parse_args()

    run_base = Path(args.run_base)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Load job info from logs if provided
    jobinfo: Dict[str, Dict[str, Any]] = {}
    if args.log_glob:
        # expand $USER
        log_glob = os.path.expandvars(args.log_glob)
        for p in sorted(Path("/").glob(log_glob.lstrip("/")) if log_glob.startswith("/") else Path(".").glob(log_glob)):
            info = _parse_jobid_and_regime_from_logname(p)
            if not info:
                continue
            elapsed, node, state, exitcode = _sacct(info.jobid)
            jobinfo[info.jobid] = {
                "jobid": info.jobid,
                "regime_from_log": info.regime,
                "elapsed": elapsed,
                "node": node,
                "state": state,
                "exitcode": exitcode,
                "slurm_log": str(info.log_path),
            }

    rows = []
    for run_dir in _collect_run_dirs(run_base):
        metrics_csv = run_dir / "logs" / "metrics.csv"
        best_val, best_epoch, last_val, last_epoch = _best_from_metrics(metrics_csv)

        # try to infer jobid from run_dir path (some people embed jobXXXX), else keep None
        mjob = re.search(r"(job\d+)", str(run_dir))
        jobtag = mjob.group(1) if mjob else None

        regime = _guess_regime_from_run_dir(run_dir)

        rows.append({
            "regime": regime,
            "run_dir": str(run_dir),
            "metrics_csv": str(metrics_csv) if metrics_csv.exists() else None,
            "best_val_loss": best_val,
            "best_epoch": best_epoch,
            "last_val_loss": last_val,
            "last_epoch": last_epoch,
            "jobtag": jobtag,
        })

    df = pd.DataFrame(rows)

    # If we have slurm jobids from logs, attach them by regime (best-effort)
    # We do not have a perfect mapping run_dir -> jobid, so we attach a "jobid_guess" using regime counts.
    # If you want exact mapping, embed SLURM_JOB_ID into run_dir name in training later.
    if jobinfo:
        # make a small df from jobinfo
        jdf = pd.DataFrame(list(jobinfo.values()))
        # keep only actual jobs
        jdf = jdf.dropna(subset=["jobid"])
        # Attach job rows by matching regime and preserving order
        df = df.sort_values(["regime", "run_dir"]).reset_index(drop=True)
        jdf = jdf.sort_values(["regime_from_log", "jobid"]).reset_index(drop=True)

        # assign sequentially per regime
        df["jobid"] = None
        df["elapsed"] = None
        df["node"] = None
        df["state"] = None
        df["exitcode"] = None
        df["slurm_log"] = None

        for reg in df["regime"].unique():
            if reg not in ["warm", "cold"]:
                continue
            df_idx = df.index[df["regime"] == reg].tolist()
            j_idx = jdf.index[jdf["regime_from_log"] == reg].tolist()
            for k, di in enumerate(df_idx):
                if k >= len(j_idx):
                    break
                ji = j_idx[k]
                df.loc[di, "jobid"] = jdf.loc[ji, "jobid"]
                df.loc[di, "elapsed"] = jdf.loc[ji, "elapsed"]
                df.loc[di, "node"] = jdf.loc[ji, "node"]
                df.loc[di, "state"] = jdf.loc[ji, "state"]
                df.loc[di, "exitcode"] = jdf.loc[ji, "exitcode"]
                df.loc[di, "slurm_log"] = jdf.loc[ji, "slurm_log"]

    df.to_csv(out_csv, index=False)

    # Print compact view
    show_cols = [c for c in [
        "regime", "best_val_loss", "best_epoch", "last_val_loss", "last_epoch",
        "jobid", "elapsed", "node", "state", "exitcode", "run_dir"
    ] if c in df.columns]
    print("\n=== FINETUNE SUMMARY ===")
    if not df.empty:
        print(df[show_cols].sort_values(["regime", "best_val_loss"], na_position="last").to_string(index=False))
        if "best_val_loss" in df.columns:
            print("\n=== STATS (best_val_loss) ===")
            print(df.groupby("regime")["best_val_loss"].agg(["count", "mean", "std", "min", "max"]))
    else:
        print("No runs found under:", run_base)

    print("\nWROTE:", out_csv)


if __name__ == "__main__":
    main()
