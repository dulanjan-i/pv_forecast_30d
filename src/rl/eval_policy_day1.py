# src/rl/eval_policy_day1.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from src.inference.physics_aware_forecaster import PhysicsAwareForecaster
from src.rl.training import DQNTrainer  # your existing trainer (saves q_net/target_net/config)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"RMSE shape mismatch: {a.shape} vs {b.shape}")
    return float(np.sqrt(np.mean((a - b) ** 2)))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate DDQN policy vs baseline for Day1 RMSE")

    # policy + normalized states
    p.add_argument("--ckpt", required=True, help="DDQN checkpoint .pt saved by DQNTrainer.save()")
    p.add_argument("--sarns_norm", required=True, help="SARNS parquet with normalized state_* columns")

    # data sources
    p.add_argument("--hist_weather_gt", required=True, help="Merged parquet with weather+pvlb+power_norm")
    p.add_argument("--weather_15min", required=True, help="weather_with_pvlib_15min.parquet (15-min, >=30d ahead)")
    p.add_argument("--gt", required=True, help="ground_truth_from_sheet_15min_utc_capnorm.parquet (timestamp_utc,power_norm)")

    # model files
    p.add_argument("--plant_meta", required=True, help="Plant config JSON (used by PhysicsAwareForecaster)")
    p.add_argument("--short_ckpt", required=True, help="Short-head TFT checkpoint")
    p.add_argument("--long_ckpt", required=True, help="Long-head TFT checkpoint")
    p.add_argument("--short_train_parquet", required=True, help="Short-head train parquet (normalization)")
    p.add_argument("--long_train_parquet", required=True, help="Long-head train parquet (normalization)")

    # eval controls
    p.add_argument("--n", type=int, default=20, help="How many forecast_starts to evaluate (default 20)")
    p.add_argument("--seed", type=int, default=0, help="Random seed for sampling forecast_starts")
    p.add_argument("--allowed_actions", default="0,2,3", help="Comma list. Policy is masked to these actions only.")
    p.add_argument("--out", default="experiments/rl/plant_03_phase1/eval_policy_vs_baseline_day1.parquet")

    return p.parse_args()


def _read_parquet(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    return df


def main() -> None:
    args = parse_args()

    allowed_actions: List[int] = [int(x) for x in args.allowed_actions.split(",") if x.strip() != ""]
    allowed_set = set(allowed_actions)
    if len(allowed_actions) == 0:
        raise ValueError("allowed_actions is empty")

    # Explicit action -> blend weights
    # Keys MUST be: short, long, physics and must sum to 1.0
    # Expanded action space (0-7) for full exploration
    action_to_weights: Dict[int, Dict[str, float]] = {
        0: {"short": 0.60, "long": 0.20, "physics": 0.20},  # baseline (ML-heavy, balanced)
        1: {"short": 0.20, "long": 0.60, "physics": 0.20},  # long-head dominant
        2: {"short": 0.45, "long": 0.25, "physics": 0.30},  # balanced with physics
        3: {"short": 0.25, "long": 0.15, "physics": 0.60},  # physics-heavy
        4: {"short": 0.00, "long": 0.00, "physics": 1.00},  # pure physics
        5: {"short": 0.80, "long": 0.10, "physics": 0.10},  # short-head dominant
        6: {"short": 0.10, "long": 0.80, "physics": 0.10},  # long-head aggressive
        7: {"short": 0.33, "long": 0.33, "physics": 0.34},  # equal 3-way blend
    }
    # Validate sums
    for a, w in action_to_weights.items():
        s = float(w["short"] + w["long"] + w["physics"])
        if abs(s - 1.0) > 1e-6:
            raise ValueError(f"Bad weights for action {a}: sum={s}")

    if 0 not in action_to_weights:
        raise ValueError("action_to_weights must define baseline action 0")

    # Load SARNS normalized
    sarns = _read_parquet(args.sarns_norm)
    if "forecast_start" not in sarns.columns:
        raise ValueError(f"{args.sarns_norm} missing forecast_start column")
    sarns["forecast_start"] = pd.to_datetime(sarns["forecast_start"], utc=True, errors="coerce")
    sarns = sarns.dropna(subset=["forecast_start"]).copy()

    state_cols = [c for c in sarns.columns if c.startswith("state_")]
    if not state_cols:
        raise ValueError(f"{args.sarns_norm} has no state_* columns")
    state_cols = sorted(state_cols, key=lambda x: int(x.split("_")[1]))
    state_dim = len(state_cols)

    # Load checkpoint and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Use trainer.load because your checkpoint format is trainer.save() dict
    ckpt_obj = torch.load(args.ckpt, map_location=device)
    if "config" not in ckpt_obj:
        raise ValueError(f"Checkpoint missing 'config': {args.ckpt}")

    config = ckpt_obj["config"]
    # Sanity: state_dim must match
    if hasattr(config, "state_dim") and int(config.state_dim) != int(state_dim):
        raise ValueError(f"State dim mismatch: sarns has {state_dim}, ckpt config has {config.state_dim}")

    trainer = DQNTrainer(config)
    trainer.load(args.ckpt)
    qnet = trainer.q_net.to(device).eval()

    # Load data sources
    hist = _read_parquet(args.hist_weather_gt)
    if "timestamp_utc" not in hist.columns:
        raise ValueError("hist_weather_gt missing timestamp_utc")
    if "power_norm" not in hist.columns:
        raise ValueError("hist_weather_gt missing power_norm, it must be merged in")

    weather = _read_parquet(args.weather_15min)
    if "timestamp_utc" not in weather.columns:
        raise ValueError("weather_15min missing timestamp_utc")

    gt = _read_parquet(args.gt)
    if "timestamp_utc" not in gt.columns or "power_norm" not in gt.columns:
        raise ValueError("gt parquet must have timestamp_utc and power_norm")
    gt = gt[["timestamp_utc", "power_norm"]].drop_duplicates("timestamp_utc").sort_values("timestamp_utc")

    # Build forecaster using your repo signature
    forecaster = PhysicsAwareForecaster(
        short_ckpt=Path(args.short_ckpt),
        long_ckpt=Path(args.long_ckpt),
        plant_metadata=args.plant_meta,
        short_train_parquet=Path(args.short_train_parquet),
        long_train_parquet=Path(args.long_train_parquet),
    )

    # Candidate starts
    starts = sarns["forecast_start"].drop_duplicates().sort_values()
    rng = np.random.default_rng(args.seed)

    # Precompute for coverage checks
    max_weather_ts = weather["timestamp_utc"].max()
    min_hist_ts = hist["timestamp_utc"].min()

    ok: List[pd.Timestamp] = []
    for fs in starts:
        fs = pd.Timestamp(fs)
        # need 30d weather ending at fs+30d-15min
        need_weather_end = fs + pd.Timedelta(days=30) - pd.Timedelta(minutes=15)
        need_hist_start = fs - pd.Timedelta(days=7)

        if need_weather_end > max_weather_ts:
            continue
        if need_hist_start < min_hist_ts:
            continue

        # history must have at least 672 observed power points
        hwin = hist[(hist["timestamp_utc"] < fs) & (hist["timestamp_utc"] >= need_hist_start)]
        if int(hwin["power_norm"].notna().sum()) < 672:
            continue

        # day1 GT must exist
        gday = gt[(gt["timestamp_utc"] >= fs) & (gt["timestamp_utc"] < fs + pd.Timedelta(days=1))]
        if len(gday) < 96:
            continue

        ok.append(fs)

    if len(ok) == 0:
        raise RuntimeError("No valid forecast_starts found with full history, weather, and GT coverage")

    picks = list(rng.choice(ok, size=min(args.n, len(ok)), replace=False))
    picks = sorted(picks)

    results = []
    for fs in picks:
        fs = pd.Timestamp(fs)

        # Build history window: last 7 days, require power_norm, then take last 672 rows
        hwin = hist[(hist["timestamp_utc"] < fs) & (hist["timestamp_utc"] >= fs - pd.Timedelta(days=7))].sort_values("timestamp_utc")
        hwin = hwin.dropna(subset=["power_norm"])
        if len(hwin) < 672:
            continue
        hwin = hwin.tail(672).copy()

        # Weather window: exactly 30 days at 15-min = 2880 steps
        wwin = weather[(weather["timestamp_utc"] >= fs) & (weather["timestamp_utc"] < fs + pd.Timedelta(days=30))].sort_values("timestamp_utc")
        if len(wwin) != 2880:
            continue
        wwin = wwin.copy()

        # Day1 GT: exactly 96 steps
        gday = gt[(gt["timestamp_utc"] >= fs) & (gt["timestamp_utc"] < fs + pd.Timedelta(days=1))].sort_values("timestamp_utc")
        if len(gday) != 96:
            continue
        y_true = gday["power_norm"].to_numpy(dtype=np.float32)

        # Baseline action 0
        w0 = action_to_weights[0]
        comp0 = forecaster.predict_30d(
            forecast_start=str(fs),
            weather_df=wwin,
            historical_df=hwin,
            return_components=True,
            blend_weights=w0,
        )
        y0 = np.asarray(comp0["final"][:96], dtype=np.float32)
        r0 = rmse(y0, y_true)

        # Policy action from normalized state row
        srow = sarns[sarns["forecast_start"] == fs]
        if len(srow) == 0:
            continue
        svec = srow.iloc[0][state_cols].to_numpy(dtype=np.float32)

        st = torch.tensor(svec[None, :], device=device)
        with torch.no_grad():
            q = qnet(st).detach().cpu().numpy().reshape(-1)

        # Mask to allowed actions
        for a in range(len(q)):
            if a not in allowed_set:
                q[a] = -1e9

        ap = int(np.argmax(q))
        wp = action_to_weights.get(ap, action_to_weights[0])

        comp1 = forecaster.predict_30d(
            forecast_start=str(fs),
            weather_df=wwin,
            historical_df=hwin,
            return_components=True,
            blend_weights=wp,
        )
        y1 = np.asarray(comp1["final"][:96], dtype=np.float32)
        r1 = rmse(y1, y_true)

        delta = r1 - r0
        print(f"[{fs}] baseline={r0:.4f} policy={r1:.4f} action={ap} delta={delta:+.4f}")

        results.append(
            {
                "forecast_start": fs,
                "rmse_baseline_a0": r0,
                "rmse_policy": r1,
                "delta_policy_minus_baseline": delta,
                "action_policy": ap,
                "weights_policy_short": wp["short"],
                "weights_policy_long": wp["long"],
                "weights_policy_physics": wp["physics"],
            }
        )

    out_df = pd.DataFrame(results).sort_values("forecast_start")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)

    if len(out_df) == 0:
        print("No rows evaluated. Most likely weather window is not exactly 2880 or GT day1 not exactly 96.")
        print("Check a single fs manually and print window lengths.")
        return

    improved_frac = float((out_df["delta_policy_minus_baseline"] < 0).mean())
    print("\nWROTE:", str(out_path))
    print("n:", len(out_df))
    print("mean baseline:", float(out_df["rmse_baseline_a0"].mean()))
    print("mean policy  :", float(out_df["rmse_policy"].mean()))
    print("fraction improved:", improved_frac)


if __name__ == "__main__":
    main()
