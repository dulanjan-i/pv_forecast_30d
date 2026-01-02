"""
Offline TFT inference for MiRACLE.

Compatible with BOTH "column_roles.json" schemas:
1) Standard PyTorch Forecasting schema (static/time-varying known/unknown, etc.)
2) MiRACLE v1 schema written by train_tft_v1.py:
   {
     "target", "time_col", "time_idx_col", "group_ids",
     "known_time_reals", "unknown_time_reals", "lagged_encoding_cols"
   }

Supports BOTH checkpoint styles:
- checkpoints/best_state_dict.pt  (preferred)
- checkpoints/best.ckpt           (may be Lightning weights-only)

Example:
python -m src.inference.offline_predict_tft \
  --train_parquet data/processed/plant_level/plant_03/15min_pca32/train.parquet \
  --test_parquet  data/processed/plant_level/plant_03/15min_pca32/test.parquet \
  --run_dir       experiments/tft/runs/germany/plant_03/15min/pvlib_warmstart_from_global_noleak/20251229_151100 \
  --out_parquet   outputs/plant03_shorthead_test_preds.parquet \
  --batch_size    512 \
  --strict
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.models import TemporalFusionTransformer


def _must(p: Path) -> Path:
    if not p.exists():
        raise FileNotFoundError(str(p))
    return p


def _read_json(p: Path) -> Dict[str, Any]:
    return json.loads(_must(p).read_text())


def _infer_roles(roles: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize roles into a single dict we can use to build TimeSeriesDataSet.
    """
    # MiRACLE v1 schema
    if "known_time_reals" in roles and "time_idx_col" in roles:
        target = roles["target"]
        time_col = roles.get("time_col", "timestamp_utc")
        time_idx_col = roles.get("time_idx_col", "time_idx")
        group_ids = roles.get("group_ids", ["plant_id"])
        known_reals = roles.get("known_time_reals", [])
        unknown_reals = roles.get("unknown_time_reals", [target])
        lagged = roles.get("lagged_encoding_cols", [])
        return {
            "target": target,
            "time_col": time_col,
            "time_idx_col": time_idx_col,
            "group_ids": group_ids,
            "static_categoricals": [],
            "static_reals": [],
            "time_varying_known_categoricals": [],
            "time_varying_known_reals": known_reals,
            "time_varying_unknown_categoricals": [],
            "time_varying_unknown_reals": unknown_reals,
            "lagged_encoding_cols": lagged,
        }

    # Standard PF schema fallback
    target = roles.get("target", "power_norm")
    time_col = roles.get("time_col", "timestamp_utc")
    time_idx_col = roles.get("time_idx", roles.get("time_idx_col", "time_idx"))
    group_ids = roles.get("group_ids", ["plant_id"])
    return {
        "target": target,
        "time_col": time_col,
        "time_idx_col": time_idx_col,
        "group_ids": group_ids,
        "static_categoricals": roles.get("static_categoricals", []),
        "static_reals": roles.get("static_reals", []),
        "time_varying_known_categoricals": roles.get("time_varying_known_categoricals", []),
        "time_varying_known_reals": roles.get("time_varying_known_reals", roles.get("known_time_reals", [])),
        "time_varying_unknown_categoricals": roles.get("time_varying_unknown_categoricals", []),
        "time_varying_unknown_reals": roles.get("time_varying_unknown_reals", roles.get("unknown_time_reals", [target])),
        "lagged_encoding_cols": roles.get("lagged_encoding_cols", []),
    }


def _ensure_time_columns(df: pd.DataFrame, roles: Dict[str, Any]) -> pd.DataFrame:
    """
    Ensure df has time_col (datetime) and time_idx_col (int) consistent with training.
    Recompute time_idx via cumcount per group to avoid missing timestep assertions.
    """
    df = df.copy()

    time_col = roles["time_col"]
    time_idx_col = roles["time_idx_col"]
    group_ids = roles["group_ids"]

    # best-effort fallback if time_col isn't present
    if time_col not in df.columns:
        for cand in ["timestamp_utc", "timestamp", "time", "datetime"]:
            if cand in df.columns:
                time_col = cand
                break
        else:
            raise KeyError(f"time column not found. expected '{roles['time_col']}'")

    df[time_col] = pd.to_datetime(df[time_col], utc=True)

    # Ensure group columns exist
    for g in group_ids:
        if g not in df.columns:
            df[g] = "plant_unk"

    df = df.sort_values(group_ids + [time_col]).reset_index(drop=True)

    # Always recompute to guarantee step=1 per group
    df[time_idx_col] = df.groupby(group_ids).cumcount().astype("int64")

    return df


def _load_weights_into_model(model: torch.nn.Module, ckpt_dir: Path, strict: bool) -> None:
    sd_path = ckpt_dir / "best_state_dict.pt"
    ckpt_path = ckpt_dir / "best.ckpt"

    state: Dict[str, Any] | None = None

    if sd_path.exists():
        obj = torch.load(sd_path, map_location="cpu")
        state = obj if isinstance(obj, dict) else None
    elif ckpt_path.exists():
        obj = torch.load(ckpt_path, map_location="cpu")
        if isinstance(obj, dict) and "state_dict" in obj:
            state = obj["state_dict"]
        elif isinstance(obj, dict):
            state = obj
        else:
            state = None
    else:
        raise FileNotFoundError(f"Missing weights in {ckpt_dir} (need best_state_dict.pt or best.ckpt)")

    if state is None:
        raise RuntimeError("Could not interpret checkpoint format.")

    def _strip_prefix(sd: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
        if not all(k.startswith(prefix) for k in sd.keys()):
            return sd
        return {k[len(prefix):]: v for k, v in sd.items()}

    for prefix in ["model.", "tft.", "net."]:
        state = _strip_prefix(state, prefix)

    try:
        model.load_state_dict(state, strict=strict)
    except RuntimeError as e:
        raise RuntimeError(
            "State dict load failed. Most likely your venv versions differ from the HPC container.\n"
            "Fix: run inference inside the same singularity container, or recreate the venv with matching versions.\n"
            f"Original error: {e}"
        )


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _build_timestamp_lookup(df: pd.DataFrame, roles: Dict[str, Any]) -> Dict[Tuple[Any, int], pd.Timestamp]:
    time_col = roles["time_col"]
    time_idx_col = roles["time_idx_col"]
    group_ids = roles["group_ids"]

    cols = group_ids + [time_idx_col, time_col]
    sub = df[cols].drop_duplicates()

    if len(group_ids) == 1:
        g = group_ids[0]
        return {(row[g], int(row[time_idx_col])): row[time_col] for _, row in sub.iterrows()}

    return {(tuple(row[group_ids]), int(row[time_idx_col])): row[time_col] for _, row in sub.iterrows()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_parquet", required=True)
    ap.add_argument("--test_parquet", required=True)
    ap.add_argument("--run_dir", required=True, help="Run dir containing run_config.json, column_roles.json, checkpoints/")
    ap.add_argument("--out_parquet", required=True)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--strict", action="store_true", help="Strict state_dict loading (recommended).")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    roles_raw = _read_json(run_dir / "column_roles.json")
    roles = _infer_roles(roles_raw)

    run_cfg = _read_json(run_dir / "run_config.json")
    cfg = run_cfg.get("cfg", run_cfg)

    max_encoder_length = int(cfg.get("max_encoder_length", 96))
    max_prediction_length = int(cfg.get("max_prediction_length", 96))

    hidden_size = int(cfg.get("hidden_size", 64))
    lstm_layers = int(cfg.get("lstm_layers", 2))
    attention_head_size = int(cfg.get("attention_head_size", 4))
    dropout = float(cfg.get("dropout", 0.1))
    # Default to pytorch_forecasting.metrics.QuantileLoss default quantiles (7 quantiles)
    quantiles = cfg.get("quantiles", [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98])

    train_df = _ensure_time_columns(pd.read_parquet(args.train_parquet), roles)
    test_df = _ensure_time_columns(pd.read_parquet(args.test_parquet), roles)

    train_ds = TimeSeriesDataSet(
        train_df,
        time_idx=roles["time_idx_col"],
        target=roles["target"],
        group_ids=roles["group_ids"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=roles["static_categoricals"],
        static_reals=roles["static_reals"],
        time_varying_known_categoricals=roles["time_varying_known_categoricals"],
        time_varying_known_reals=roles["time_varying_known_reals"],
        time_varying_unknown_categoricals=roles["time_varying_unknown_categoricals"],
        time_varying_unknown_reals=roles["time_varying_unknown_reals"],
        target_normalizer=GroupNormalizer(groups=roles["group_ids"], transformation="softplus"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )

    test_ds = TimeSeriesDataSet.from_dataset(train_ds, test_df, predict=False, stop_randomization=True)
    test_dl = test_ds.to_dataloader(train=False, batch_size=args.batch_size, num_workers=args.num_workers)

    # Build group ID decoder: encoded integer -> original value
    # PyTorch Forecasting encodes group IDs as integers in sorted order
    group_decoders = {}
    for g in roles["group_ids"]:
        unique_vals = sorted(train_df[g].unique())
        group_decoders[g] = {i: val for i, val in enumerate(unique_vals)}


    loss = QuantileLoss(quantiles=quantiles)

    model = TemporalFusionTransformer.from_dataset(
        train_ds,
        learning_rate=float(cfg.get("learning_rate", 1e-3)),
        hidden_size=hidden_size,
        lstm_layers=lstm_layers,
        attention_head_size=attention_head_size,
        dropout=dropout,
        loss=loss,
        reduce_on_plateau_patience=int(cfg.get("patience", 3)),
    )

    _load_weights_into_model(model, run_dir / "checkpoints", strict=args.strict)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    print(f"[INFO] Running inference on {device} with batch_size={args.batch_size}")

    # Manual inference loop to avoid Lightning distributed issues
    all_preds = []
    all_targets = []
    all_group_ids = {g: [] for g in roles["group_ids"]}
    all_time_idx = []
    
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_dl):
            # Move to device
            x = {k: v.to(device) if torch.is_tensor(v) else v for k, v in x.items()}
            if isinstance(y, (list, tuple)):
                targets = y[0].to(device)
            else:
                targets = y.to(device)
            
            # Forward pass
            output = model(x)
            pred = output.prediction  # (B, T, Q) or (B, T)
            
            all_preds.append(pred.cpu())
            all_targets.append(targets.cpu())
            
            # Extract group IDs from 'groups' key and decode to original values
            if 'groups' in x:
                groups_tensor = x['groups']  # (B, num_group_ids), encoded integers
                if torch.is_tensor(groups_tensor):
                    groups_np = groups_tensor.cpu().numpy()
                    # groups_np contains encoded integers; decode them
                    for idx, g in enumerate(roles["group_ids"]):
                        if groups_np.ndim == 2 and idx < groups_np.shape[1]:
                            encoded_vals = groups_np[:, idx].astype(int).tolist()
                        elif groups_np.ndim == 1:
                            encoded_vals = groups_np.astype(int).tolist()
                        else:
                            encoded_vals = []
                        
                        # Decode using the group_decoders mapping
                        if g in group_decoders:
                            decoded = [group_decoders[g].get(enc, enc) for enc in encoded_vals]
                            all_group_ids[g].extend(decoded)
                        else:
                            # No decoder, use raw values
                            all_group_ids[g].extend(encoded_vals)
            else:
                # Fallback to decoder_{group_id} keys
                for g in roles["group_ids"]:
                    decoder_key = f"decoder_{g}"
                    if decoder_key in x and torch.is_tensor(x[decoder_key]):
                        vals = x[decoder_key][:, 0].cpu().numpy().tolist()
                        all_group_ids[g].extend(vals)
            
            # Time idx: try decoder_time_idx first, then encoder_time_idx
            time_idx_key = 'decoder_time_idx'
            if time_idx_key in x and torch.is_tensor(x[time_idx_key]):
                # decoder_time_idx has shape (B, decoder_length)
                # First timestep in decoder is the start of prediction
                start_idx = x[time_idx_key][:, 0].cpu().numpy()
                all_time_idx.extend(start_idx.tolist())
            else:
                encoder_time_key = f"encoder_{roles['time_idx_col']}"
                if encoder_time_key in x and torch.is_tensor(x[encoder_time_key]):
                    # Last encoder time_idx + 1 = first prediction time_idx
                    start_idx = x[encoder_time_key][:, -1].cpu().numpy() + 1
                    all_time_idx.extend(start_idx.tolist())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{len(test_dl)} batches")
    
    # Concatenate all batches
    y_hat = torch.cat(all_preds, dim=0)
    y_true = torch.cat(all_targets, dim=0)
    
    # Build index dataframe
    index_data = {**all_group_ids, roles["time_idx_col"]: all_time_idx}
    index_df = pd.DataFrame(index_data)

    y_hat = _to_numpy(y_hat)
    y_true = _to_numpy(y_true)

    if y_hat.ndim == 2:
        y_hat = y_hat[:, :, None]
    if y_true.ndim == 2:
        y_true = y_true[:, :, None]

    N, P, Q = y_hat.shape

    all_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    lookup = _build_timestamp_lookup(all_df, roles)

    time_idx_col = roles["time_idx_col"]
    group_ids = roles["group_ids"]

    if time_idx_col not in index_df.columns:
        if "time_idx" in index_df.columns:
            index_df[time_idx_col] = index_df["time_idx"]
        else:
            raise KeyError(f"Prediction index missing time index. got columns: {list(index_df.columns)}")

    rows = []
    for i in range(N):
        if len(group_ids) == 1:
            gval = index_df.iloc[i][group_ids[0]]
            gkey = gval
            group_payload = {group_ids[0]: gval}
        else:
            gkey = tuple(index_df.iloc[i][g] for g in group_ids)
            group_payload = {g: index_df.iloc[i][g] for g in group_ids}

        start_idx = int(index_df.iloc[i][time_idx_col])

        for h in range(P):
            ti = start_idx + h
            ts = lookup.get((gkey, ti), pd.NaT)

            row = {
                **group_payload,
                "pred_start_time_idx": start_idx,
                "horizon": h + 1,
                "time_idx": ti,
                "timestamp_utc": ts,
                "y_true": float(y_true[i, h, 0]) if y_true.size else np.nan,
            }

            if Q == len(quantiles):
                for qi, q in enumerate(quantiles):
                    row[f"y_hat_q{int(round(float(q) * 100)):02d}"] = float(y_hat[i, h, qi])
            else:
                for qi in range(Q):
                    row[f"y_hat_q{qi}"] = float(y_hat[i, h, qi])

            rows.append(row)

    out = pd.DataFrame(rows)
    out_path = Path(args.out_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)

    print(f"[DONE] wrote: {out_path} rows={len(out):,} N={N} P={P} Q={Q} device={device}")


if __name__ == "__main__":
    main()
