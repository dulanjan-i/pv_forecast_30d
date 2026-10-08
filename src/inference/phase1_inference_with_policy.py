# src/inference/phase1_inference_with_policy.py
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import pyarrow as pa
import pyarrow.parquet as pq

from src.inference.physics_aware_forecaster import PhysicsAwareForecaster

LOGGER = logging.getLogger("phase1_inference_with_policy")


# ----------------------------
# Helpers
# ----------------------------
def _ensure_utc_midnight(x: Any) -> pd.Timestamp:
    """
    Return a tz-aware UTC Timestamp floored to midnight.
    Accepts str, datetime, pd.Timestamp.
    """
    ts = pd.Timestamp(x)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.floor("D")


def _read_parquet_must_exist(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return pd.read_parquet(path)


def _normalize_ts_col(df: pd.DataFrame, col: str = "timestamp_utc") -> pd.DataFrame:
    if col not in df.columns:
        raise ValueError(f"Missing '{col}' column")
    out = df.copy()
    out[col] = pd.to_datetime(out[col], utc=True, errors="coerce")
    out = out.dropna(subset=[col]).sort_values(col)
    return out


def _infer_state_cols(df: pd.DataFrame, qnet_in_dim: int) -> List[str]:
    """
    Best-effort inference of state feature columns from sarns_norm.
    We prefer columns that look like state features.
    """
    cols = list(df.columns)

    # Prefer common prefixes
    preferred = []
    for prefix in ("s_", "state_", "feat_", "x_"):
        preferred = [c for c in cols if c.startswith(prefix)]
        if preferred:
            break

    # Another common style: s0,s1,s2...
    if not preferred:
        preferred = [c for c in cols if (c.startswith("s") and c[1:].isdigit())]

    # Fallback: numeric columns excluding known non-state columns
    if not preferred:
        exclude = {
            "action", "a",
            "reward", "r",
            "done", "terminal",
            "forecast_start",
            "blend_short", "blend_long", "blend_physics",
        }
        # also exclude next-state columns
        numeric = []
        for c in cols:
            if c in exclude:
                continue
            if c.startswith("next_") or c.startswith("ns_") or c.startswith("s_next"):
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                numeric.append(c)
        preferred = numeric

    # Enforce exact dimension
    if len(preferred) < qnet_in_dim:
        raise ValueError(
            f"Could not infer enough state columns. Need {qnet_in_dim}, found {len(preferred)}. "
            f"Candidate cols: {preferred}"
        )
    return preferred[:qnet_in_dim]


class QNet(torch.nn.Module):
    """
    Wrapper matching checkpoint key structure: 'net.0.weight', 'net.2.weight', ...
    """
    def __init__(self, layer_sizes: List[Tuple[int, int]]):
        super().__init__()
        layers: List[torch.nn.Module] = []
        for i, (in_f, out_f) in enumerate(layer_sizes):
            layers.append(torch.nn.Linear(in_f, out_f))
            if i < len(layer_sizes) - 1:
                layers.append(torch.nn.ReLU())
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _extract_qnet_state_dict(ckpt: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """
    Support a few common save formats.
    """
    for key in ("q_net", "online_net", "policy_net", "model_state_dict", "state_dict"):
        if key in ckpt and isinstance(ckpt[key], dict):
            return ckpt[key]
    # Sometimes the checkpoint itself is the state_dict
    if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return ckpt  # type: ignore
    raise ValueError(f"Could not find q-network state_dict keys in checkpoint. Keys: {list(ckpt.keys())}")


def _build_qnet_from_state_dict(sd: Dict[str, torch.Tensor]) -> Tuple[QNet, int, int]:
    """
    Reconstruct MLP architecture purely from state_dict tensor shapes.
    Expects keys like: net.0.weight, net.0.bias, net.2.weight, net.2.bias, ...
    """
    weight_keys = [k for k in sd.keys() if k.endswith(".weight") and k.startswith("net.")]
    if not weight_keys:
        raise ValueError(f"State dict does not look like expected MLP with 'net.*.weight'. Keys: {list(sd.keys())[:20]}")

    # Sort by module index inside 'net.{idx}.weight'
    def _idx(k: str) -> int:
        return int(k.split(".")[1])

    weight_keys = sorted(weight_keys, key=_idx)

    layer_sizes: List[Tuple[int, int]] = []
    for k in weight_keys:
        w = sd[k]
        if w.ndim != 2:
            raise ValueError(f"Unexpected weight tensor shape for {k}: {tuple(w.shape)}")
        out_f, in_f = int(w.shape[0]), int(w.shape[1])
        layer_sizes.append((in_f, out_f))

    state_dim = layer_sizes[0][0]
    action_dim = layer_sizes[-1][1]
    qnet = QNet(layer_sizes)
    qnet.load_state_dict(sd, strict=True)
    return qnet, state_dim, action_dim


def _action_to_blend_weights(sarns_norm: pd.DataFrame) -> Dict[int, Dict[str, float]]:
    required = ["action", "blend_short", "blend_long", "blend_physics"]
    missing = [c for c in required if c not in sarns_norm.columns]
    if missing:
        raise ValueError(f"sarns_norm missing columns: {missing}")

    g = sarns_norm.groupby("action")[["blend_short", "blend_long", "blend_physics"]].mean()

    mapping: Dict[int, Dict[str, float]] = {}
    for a, row in g.iterrows():
        w = row.to_dict()
        s = float(w["blend_short"] + w["blend_long"] + w["blend_physics"])
        if not np.isfinite(s) or abs(s - 1.0) > 1e-2:
            # Do not hard fail, but clamp/renorm for safety
            vals = np.array([w["blend_short"], w["blend_long"], w["blend_physics"]], dtype=float)
            vals = np.clip(vals, 0.0, None)
            ss = float(vals.sum())
            if ss <= 0:
                vals = np.array([0.5, 0.25, 0.25], dtype=float)
                ss = float(vals.sum())
            vals = vals / ss
            w = {"blend_short": float(vals[0]), "blend_long": float(vals[1]), "blend_physics": float(vals[2])}
        mapping[int(a)] = w

    if 0 not in mapping:
        raise ValueError("sarns_norm must include baseline action 0 to allow fallback")

    return mapping


@dataclass
class Paths:
    phase_dir: Path
    out_path: Path
    plant_meta: Path
    short_ckpt: Path
    long_ckpt: Path
    short_train: Path
    long_train: Path
    hist_encoder: Optional[Path]
    policy_ckpt: Path
    sarns_norm: Path
    weather_15min: Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 1 inference with DDQN policy controlling Day-1 blend weights")
    ap.add_argument("--start-date", required=True, type=str, help="YYYY-MM-DD")
    ap.add_argument("--end-date", required=True, type=str, help="YYYY-MM-DD")
    ap.add_argument("--stride-days", type=int, default=1)

    ap.add_argument("--phase-dir", type=str, required=True, help="Freeze phase dir, e.g. freeze/final_thesis_v1/phase1_2024daily_final")
    ap.add_argument("--out", type=str, default=None, help="Output parquet path (default: phase-dir/processed/predictions_phase1_policy.parquet)")

    ap.add_argument("--plant-meta", type=str, required=True)
    ap.add_argument("--short-ckpt", type=str, required=True)
    ap.add_argument("--long-ckpt", type=str, required=True)
    ap.add_argument("--short-train", type=str, required=True)
    ap.add_argument("--long-train", type=str, required=True)
    ap.add_argument("--hist-encoder", type=str, default=None)

    ap.add_argument("--policy-ckpt", type=str, required=True)
    ap.add_argument("--sarns-norm", type=str, required=True)

    ap.add_argument("--weather-15min", type=str, default=None, help="Override weather_with_pvlib_15min.parquet")
    ap.add_argument("--history-days", type=int, default=30, help="How much history to pass into forecaster before fs")
    ap.add_argument("--device", type=str, default=None, choices=[None, "cpu", "cuda"])

    ap.add_argument("--log-level", type=str, default="INFO")

    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s:%(name)s:%(message)s")

    phase_dir = Path(args.phase_dir)
    processed_dir = phase_dir / "processed"
    rl_dir = phase_dir / "rl"

    out_path = Path(args.out) if args.out else (processed_dir / "predictions_phase1_policy.parquet")

    weather_15min = Path(args.weather_15min) if args.weather_15min else (processed_dir / "weather_with_pvlib_15min.parquet")

    paths = Paths(
        phase_dir=phase_dir,
        out_path=out_path,
        plant_meta=Path(args.plant_meta),
        short_ckpt=Path(args.short_ckpt),
        long_ckpt=Path(args.long_ckpt),
        short_train=Path(args.short_train),
        long_train=Path(args.long_train),
        hist_encoder=Path(args.hist_encoder) if args.hist_encoder else None,
        policy_ckpt=Path(args.policy_ckpt),
        sarns_norm=Path(args.sarns_norm),
        weather_15min=weather_15min,
    )

    device = torch.device("cuda" if (args.device == "cuda" or (args.device is None and torch.cuda.is_available())) else "cpu")
    LOGGER.info("Device: %s", device)

    # ----------------------------
    # Load weather and (optional) historical encoder source
    # ----------------------------
    wx15 = _normalize_ts_col(_read_parquet_must_exist(paths.weather_15min), "timestamp_utc")

    hist_df: Optional[pd.DataFrame] = None
    if paths.hist_encoder is not None:
        hist_df = _normalize_ts_col(_read_parquet_must_exist(paths.hist_encoder), "timestamp_utc")

    # ----------------------------
    # Load SARNS_NORM and build mapping + state dataframe indexed by forecast_start
    # ----------------------------
    sarns = _read_parquet_must_exist(paths.sarns_norm)

    if "forecast_start" not in sarns.columns:
        raise ValueError("sarns_norm must contain 'forecast_start' column")

    sarns = sarns.copy()
    sarns["forecast_start"] = pd.to_datetime(sarns["forecast_start"], utc=True, errors="coerce")
    sarns = sarns.dropna(subset=["forecast_start"])
    sarns["forecast_start"] = sarns["forecast_start"].dt.floor("D")
    sarns = sarns.sort_values("forecast_start")
    sarns_1 = sarns.drop_duplicates(subset=["forecast_start"], keep="last").set_index("forecast_start")

    action_to_weights = _action_to_blend_weights(sarns_1.reset_index())
    LOGGER.info("Actions seen in sarns_norm: %s", sorted(action_to_weights.keys()))
    LOGGER.info("Baseline (action 0) weights: %s", action_to_weights[0])

    # ----------------------------
    # Load DDQN checkpoint, reconstruct Q-net from its saved weights
    # ----------------------------
    ckpt_obj = torch.load(paths.policy_ckpt, map_location="cpu")
    q_sd = _extract_qnet_state_dict(ckpt_obj)
    qnet, q_state_dim, q_action_dim = _build_qnet_from_state_dict(q_sd)
    qnet = qnet.to(device).eval()
    LOGGER.info("Loaded Q-net from checkpoint: state_dim=%d action_dim=%d", q_state_dim, q_action_dim)

    # Make sure our action mapping can handle the policy outputs
    # If policy picks an action not present, we fall back to action 0.
    # That is fine for robustness.

    # Infer which columns are the state features used by this checkpoint
    state_cols = _infer_state_cols(sarns_1.reset_index(), q_state_dim)
    LOGGER.info("Using %d state columns: %s", len(state_cols), state_cols)

    # ----------------------------
    # Forecaster (must match PhysicsAwareForecaster.__init__)
    # ----------------------------
    forecaster = PhysicsAwareForecaster(
        short_ckpt=str(paths.short_ckpt),
        long_ckpt=str(paths.long_ckpt),
        plant_metadata=str(paths.plant_meta),
        short_train_parquet=str(paths.short_train),
        long_train_parquet=str(paths.long_train),
        device=str(device),
    )

    # ----------------------------
    # Rolling forecast loop
    # ----------------------------
    start = _ensure_utc_midnight(args.start_date)
    end = _ensure_utc_midnight(args.end_date)
    stride = int(args.stride_days)

    # Forecast starts: inclusive start, inclusive end
    fss = pd.date_range(start=start, end=end, freq=f"{stride}D", tz="UTC")

    # Output streaming writer
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    writer: Optional[pq.ParquetWriter] = None

    stats = {
        "attempted": 0,
        "saved": 0,
        "fallback_no_state": 0,
        "skipped_bad_window": 0,
        "skipped_failed": 0,
    }

    history_td = pd.Timedelta(days=int(args.history_days))

    for fs in fss:
        stats["attempted"] += 1
        fs = _ensure_utc_midnight(fs)

        try:
            # 30d decoder window (15-min). Must be exactly 2880 rows.
            wwin = wx15[(wx15["timestamp_utc"] >= fs) & (wx15["timestamp_utc"] < fs + pd.Timedelta(days=30))].copy()
            if len(wwin) != 2880:
                stats["skipped_bad_window"] += 1
                continue

            # History window for encoder anchoring
            if hist_df is not None:
                hstart = fs - history_td
                hwin = hist_df[(hist_df["timestamp_utc"] < fs) & (hist_df["timestamp_utc"] >= hstart)].copy()
                if len(hwin) == 0:
                    hwin = hist_df[hist_df["timestamp_utc"] < fs].tail(96).copy()
            else:
                # Fallback: use weather as history (power_norm will be derived from PVLib inside forecaster)
                hstart = fs - history_td
                hwin = wx15[(wx15["timestamp_utc"] < fs) & (wx15["timestamp_utc"] >= hstart)].copy()
                if len(hwin) == 0:
                    hwin = wx15[wx15["timestamp_utc"] < fs].tail(96).copy()

            # State -> action (offline evaluation uses sarns_norm state)
            if fs in sarns_1.index:
                st = sarns_1.loc[fs, state_cols].to_numpy(dtype=np.float32)
                st_t = torch.from_numpy(st).to(device).view(1, -1)
                with torch.no_grad():
                    qvals = qnet(st_t)
                    a = int(torch.argmax(qvals, dim=1).item())
            else:
                stats["fallback_no_state"] += 1
                a = 0

            w = action_to_weights.get(a, action_to_weights[0])
            blend_weights = {
                "short": float(w["blend_short"]),
                "long": float(w["blend_long"]),
                "physics": float(w["blend_physics"]),
            }

            # Full 30d forecast. RL affects Day 1 blend only (by design in PhysicsAwareForecaster).
            yhat = forecaster.predict_30d(
                forecast_start=fs,
                weather_df=wwin,
                historical_df=hwin,
                blend_weights=blend_weights,
                return_components=False,
            )

            yhat = np.asarray(yhat, dtype=np.float32).reshape(-1)
            if yhat.shape[0] != 2880 or not np.isfinite(yhat).all():
                raise RuntimeError(f"Bad prediction array: shape={yhat.shape}, finite={np.isfinite(yhat).mean()}")

            ts = pd.date_range(start=fs, periods=2880, freq="15min", tz="UTC")

            out_df = pd.DataFrame(
                {
                    "timestamp_utc": ts,
                    "forecast_start": np.repeat(fs, 2880),
                    "step_ahead": np.arange(2880, dtype=np.int32),
                    "hours_ahead": (np.arange(2880, dtype=np.float32) / 4.0),
                    "predicted_power_norm": yhat,
                    "policy_action": np.repeat(a, 2880).astype(np.int16),
                    "blend_short": np.repeat(blend_weights["short"], 2880).astype(np.float32),
                    "blend_long": np.repeat(blend_weights["long"], 2880).astype(np.float32),
                    "blend_physics": np.repeat(blend_weights["physics"], 2880).astype(np.float32),
                }
            )

            table = pa.Table.from_pandas(out_df, preserve_index=False)

            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema, compression="snappy")

            writer.write_table(table)
            stats["saved"] += 1

        except Exception as e:
            stats["skipped_failed"] += 1
            LOGGER.error("Failed fs=%s: %s", fs, repr(e))
            continue

    if writer is not None:
        writer.close()

    LOGGER.info("WROTE: %s", out_path)
    LOGGER.info("Stats: %s", stats)


if __name__ == "__main__":
    main()
