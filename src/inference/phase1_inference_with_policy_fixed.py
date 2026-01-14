from __future__ import annotations

import argparse
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

LOGGER = logging.getLogger("phase1_inference_with_policy_fixed")

# ==================================================================================
# Action -> (short,long,physics) blend mapping
# ==================================================================================
ACTION_TO_WEIGHTS_FIXED = {
    0: {"short": 0.4875, "long": 0.2625, "physics": 0.25},  # Maintain
    1: {"short": 0.5500, "long": 0.2500, "physics": 0.20},  # Fine-tune short
    2: {"short": 0.3000, "long": 0.4500, "physics": 0.25},  # Fine-tune long
    3: {"short": 0.3250, "long": 0.1750, "physics": 0.50},  # Recalibrate physics
    4: {"short": 0.7000, "long": 0.2000, "physics": 0.10},
    5: {"short": 0.2000, "long": 0.7000, "physics": 0.10},
    6: {"short": 0.2000, "long": 0.2000, "physics": 0.60},
    7: {"short": 0.4875, "long": 0.2625, "physics": 0.25},
}


def _ensure_utc_midnight(x: Any) -> pd.Timestamp:
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
    cols = list(df.columns)

    preferred: List[str] = []
    for prefix in ("s_", "state_", "feat_", "x_"):
        preferred = [c for c in cols if c.startswith(prefix)]
        if preferred:
            break

    if not preferred:
        preferred = [c for c in cols if (c.startswith("s") and c[1:].isdigit())]

    if not preferred:
        exclude = {
            "action",
            "a",
            "reward",
            "r",
            "done",
            "terminal",
            "forecast_start",
            "blend_short",
            "blend_long",
            "blend_physics",
        }
        numeric = [
            c
            for c in cols
            if c not in exclude
            and not c.startswith("next_")
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        preferred = numeric

    if len(preferred) < qnet_in_dim:
        raise ValueError(f"Could not infer enough state columns. Need {qnet_in_dim}, found {len(preferred)}.")
    return preferred[:qnet_in_dim]


class QNet(torch.nn.Module):
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
    for key in ("q_net", "online_net", "policy_net", "model_state_dict", "state_dict"):
        if key in ckpt and isinstance(ckpt[key], dict):
            return ckpt[key]
    if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return ckpt
    raise ValueError("Could not find q-network state_dict keys.")


def _build_qnet_from_state_dict(sd: Dict[str, torch.Tensor]) -> Tuple[QNet, int, int]:
    weight_keys = sorted(
        [k for k in sd.keys() if k.endswith(".weight") and k.startswith("net.")],
        key=lambda k: int(k.split(".")[1]),
    )
    layer_sizes: List[Tuple[int, int]] = []
    for k in weight_keys:
        w = sd[k]
        layer_sizes.append((int(w.shape[1]), int(w.shape[0])))
    qnet = QNet(layer_sizes)
    qnet.load_state_dict(sd, strict=True)
    return qnet, layer_sizes[0][0], layer_sizes[-1][1]


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-date", required=True)
    ap.add_argument("--end-date", required=True)
    ap.add_argument("--stride-days", type=int, default=1)
    ap.add_argument("--phase-dir", required=True)
    ap.add_argument("--out", default=None)

    ap.add_argument("--plant-meta", required=True)
    ap.add_argument("--short-ckpt", required=True)
    ap.add_argument("--long-ckpt", required=True)
    ap.add_argument("--short-train", required=True)
    ap.add_argument("--long-train", required=True)
    ap.add_argument("--hist-encoder", default=None)

    ap.add_argument("--policy-ckpt", required=True)
    ap.add_argument("--sarns-norm", required=True)

    ap.add_argument("--weather-15min", default=None)
    ap.add_argument("--history-days", type=int, default=30)
    ap.add_argument("--device", default=None)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(levelname)s:%(name)s:%(message)s",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Device: %s", device)

    phase_dir = Path(args.phase_dir)
    out_path = Path(args.out) if args.out else phase_dir / "processed" / "predictions_phase1_policy.parquet"
    weather_15min = Path(args.weather_15min) if args.weather_15min else phase_dir / "processed" / "weather_with_pvlib_15min.parquet"

    wx15 = _normalize_ts_col(_read_parquet_must_exist(weather_15min), "timestamp_utc")
    hist_df = _normalize_ts_col(_read_parquet_must_exist(Path(args.hist_encoder)), "timestamp_utc") if args.hist_encoder else None

    sarns = _read_parquet_must_exist(Path(args.sarns_norm))
    sarns["forecast_start"] = pd.to_datetime(sarns["forecast_start"], utc=True).dt.floor("D")
    sarns_1 = sarns.drop_duplicates(subset=["forecast_start"], keep="last").set_index("forecast_start")

    ckpt_obj = torch.load(args.policy_ckpt, map_location="cpu")
    qnet, q_state_dim, q_action_dim = _build_qnet_from_state_dict(_extract_qnet_state_dict(ckpt_obj))
    qnet = qnet.to(device).eval()
    LOGGER.info("Loaded Q-net: state_dim=%d action_dim=%d", q_state_dim, q_action_dim)

    state_cols = _infer_state_cols(sarns_1.reset_index(), q_state_dim)

    forecaster = PhysicsAwareForecaster(
        short_ckpt=str(args.short_ckpt),
        long_ckpt=str(args.long_ckpt),
        plant_metadata=str(args.plant_meta),
        short_train_parquet=str(args.short_train),
        long_train_parquet=str(args.long_train),
        device=str(device),
    )

    start = _ensure_utc_midnight(args.start_date)
    end = _ensure_utc_midnight(args.end_date)
    fss = pd.date_range(start=start, end=end, freq=f"{args.stride_days}D", tz="UTC")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    writer = None
    stats = {
        "attempted": 0,
        "saved": 0,
        "fallback_no_state": 0,
        "skipped_bad_window": 0,
        "skipped_failed": 0,
    }

    for fs in fss:
        stats["attempted"] += 1
        fs = _ensure_utc_midnight(fs)

        try:
            wwin = wx15[(wx15["timestamp_utc"] >= fs) & (wx15["timestamp_utc"] < fs + pd.Timedelta(days=30))].copy()
            if len(wwin) != 2880:
                stats["skipped_bad_window"] += 1
                continue

            hstart = fs - pd.Timedelta(days=args.history_days)
            if hist_df is not None:
                hwin = hist_df[(hist_df["timestamp_utc"] < fs) & (hist_df["timestamp_utc"] >= hstart)].copy()
            else:
                hwin = wwin.iloc[:0]

            if len(hwin) == 0:
                hwin = wx15[wx15["timestamp_utc"] < fs].tail(96).copy()

            if fs in sarns_1.index:
                st = torch.from_numpy(sarns_1.loc[fs, state_cols].to_numpy(dtype=np.float32)).to(device).view(1, -1)
                with torch.no_grad():
                    a = int(torch.argmax(qnet(st), dim=1).item())
            else:
                stats["fallback_no_state"] += 1
                a = 0

            w = ACTION_TO_WEIGHTS_FIXED.get(a, ACTION_TO_WEIGHTS_FIXED[0])
            blend_weights = {"short": float(w["short"]), "long": float(w["long"]), "physics": float(w["physics"])}

            yhat = forecaster.predict_30d(
                forecast_start=fs,
                weather_df=wwin,
                historical_df=hwin,
                use_live_weather=False,
                return_components=False,
                blend_weights=blend_weights,
            )
            yhat = np.asarray(yhat, dtype=np.float32).reshape(-1)

            ts = pd.date_range(start=fs, periods=2880, freq="15min", tz="UTC")
            out_df = pd.DataFrame(
                {
                    "timestamp_utc": ts,
                    "forecast_start": fs,
                    "step_ahead": np.arange(2880, dtype=int),
                    "predicted_power_norm": yhat,
                    "policy_action": int(a),
                    "blend_short": float(blend_weights["short"]),
                    "blend_long": float(blend_weights["long"]),
                    "blend_physics": float(blend_weights["physics"]),
                }
            )

            table = pa.Table.from_pandas(out_df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema, compression="snappy")
            writer.write_table(table)
            stats["saved"] += 1

        except Exception as e:
            LOGGER.error("Failed fs=%s: %s", str(fs), str(e))
            stats["skipped_failed"] += 1
            continue

    if writer:
        writer.close()
    LOGGER.info("Done. Stats: %s", stats)


if __name__ == "__main__":
    main()
