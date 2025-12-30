"""
Evaluate short-head TFT candidates (15-min, 96-step) with RMSE/MAE on validation.

Important:
- Your train_tft_v1.py saves checkpoints as plain state_dict via torch.save(model.state_dict()).
  So we must rebuild the TFT from the dataset and load_state_dict. load_from_checkpoint will fail.

Robust to:
- Parquets that have timestamp_utc but not time_idx (we build time_idx per plant_id)
- Quantile outputs (B, T, Q) (we take median quantile)

Outputs:
- <out_dir>/short_head_eval.csv
- <out_dir>/short_head_model_selection.md
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.models import TemporalFusionTransformer


KEY_TIME = "timestamp_utc"
KEY_GROUP = "plant_id"
KEY_TARGET = "power_norm"
KEY_TIME_IDX = "time_idx"


def ensure_time_idx(df: pd.DataFrame) -> pd.DataFrame:
    if KEY_TIME_IDX in df.columns:
        return df

    for c in (KEY_GROUP, KEY_TIME):
        if c not in df.columns:
            raise KeyError(f"Missing '{c}' and '{KEY_TIME_IDX}' not present, cannot build time_idx")

    out = df.copy()
    out[KEY_TIME] = pd.to_datetime(out[KEY_TIME], utc=True, errors="coerce")
    out = out.dropna(subset=[KEY_TIME]).sort_values([KEY_GROUP, KEY_TIME]).reset_index(drop=True)
    out[KEY_TIME_IDX] = out.groupby(KEY_GROUP, sort=False).cumcount().astype("int64")
    return out


def load_json(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return json.loads(path.read_text())


def build_datasets(train_df: pd.DataFrame, val_df: pd.DataFrame, roles: Dict, enc_len: int, pred_len: int):
    # Validate required cols
    required = {KEY_GROUP, KEY_TARGET, KEY_TIME_IDX}
    missing_train = sorted([c for c in required if c not in train_df.columns])
    missing_val = sorted([c for c in required if c not in val_df.columns])
    if missing_train:
        raise KeyError(f"Train parquet missing columns: {missing_train}")
    if missing_val:
        raise KeyError(f"Val parquet missing columns: {missing_val}")

    known_time_reals = roles.get("known_time_reals", [])
    unknown_time_reals = roles.get("unknown_time_reals", [KEY_TARGET])

    # Ensure feature columns exist
    feat_missing = [c for c in known_time_reals if c not in train_df.columns]
    if feat_missing:
        raise KeyError(f"Train parquet missing known_time_reals columns (from column_roles.json): {feat_missing[:20]}")

    train_ds = TimeSeriesDataSet(
        train_df,
        time_idx=KEY_TIME_IDX,
        target=KEY_TARGET,
        group_ids=[KEY_GROUP],
        min_encoder_length=enc_len,
        max_encoder_length=enc_len,
        min_prediction_length=pred_len,
        max_prediction_length=pred_len,
        time_varying_known_reals=known_time_reals,
        time_varying_unknown_reals=unknown_time_reals,
        target_normalizer=GroupNormalizer(groups=[KEY_GROUP], transformation="softplus"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )

    val_ds = TimeSeriesDataSet.from_dataset(train_ds, val_df, predict=False, stop_randomization=True)
    return train_ds, val_ds


def point_from_quantiles(pred: torch.Tensor, model: TemporalFusionTransformer) -> torch.Tensor:
    """
    pred: (B,T) or (B,T,Q)
    return: (B,T)
    """
    if pred.ndim == 2:
        return pred
    if pred.ndim != 3:
        raise ValueError(f"Unexpected prediction shape: {tuple(pred.shape)}")

    qs = getattr(getattr(model, "loss", None), "quantiles", None)
    if qs is None:
        return pred.mean(dim=-1)

    qs_arr = torch.tensor([float(q) for q in qs], device=pred.device)
    med_i = int(torch.argmin(torch.abs(qs_arr - 0.5)).item())
    return pred[:, :, med_i]


def update_streaming_metrics(sum_abs: float, sum_sq: float, n: int, y_true: torch.Tensor, y_pred: torch.Tensor):
    yt = y_true.detach().float().reshape(-1)
    yp = y_pred.detach().float().reshape(-1)
    diff = yp - yt
    sum_abs += float(torch.sum(torch.abs(diff)).item())
    sum_sq += float(torch.sum(diff * diff).item())
    n += int(diff.numel())
    return sum_abs, sum_sq, n


def eval_one(mode: str, run_dir: Path, train_parquet: Path, val_parquet: Path, batch_size: int = 1024) -> Dict:
    run_dir = run_dir.resolve()
    ckpt = run_dir / "checkpoints" / "best.ckpt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt}")

    roles = load_json(run_dir / "column_roles.json")
    run_cfg = load_json(run_dir / "run_config.json")
    cli = run_cfg.get("cli_args", {})
    tft_cfg = run_cfg.get("tft_config", {})

    enc_len = int(cli.get("encoder_len", 96))
    pred_len = int(cli.get("pred_len", 96))

    hidden_size = int(tft_cfg.get("hidden_size", cli.get("hidden_size", 64)))
    lstm_layers = int(tft_cfg.get("lstm_layers", cli.get("lstm_layers", 2)))
    attn_heads = int(tft_cfg.get("attention_head_size", cli.get("attn_heads", 4)))
    dropout = float(tft_cfg.get("dropout", cli.get("dropout", 0.1)))
    lr = float(cli.get("lr", 1e-3))

    train_df = ensure_time_idx(pd.read_parquet(train_parquet))
    val_df = ensure_time_idx(pd.read_parquet(val_parquet))

    train_ds, val_ds = build_datasets(train_df, val_df, roles, enc_len, pred_len)

    val_dl = val_ds.to_dataloader(train=False, batch_size=batch_size, num_workers=0, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Rebuild model exactly like training, then load state_dict
    model = TemporalFusionTransformer.from_dataset(
        train_ds,
        hidden_size=hidden_size,
        lstm_layers=lstm_layers,
        attention_head_size=attn_heads,
        dropout=dropout,
        learning_rate=lr,
        log_interval=-1,
        reduce_on_plateau_patience=0,
    )
    sd = torch.load(str(ckpt), map_location="cpu")
    model.load_state_dict(sd, strict=True)

    model.to(device)
    model.eval()

    sum_abs, sum_sq, n = 0.0, 0.0, 0
    with torch.no_grad():
        for x, y in val_dl:
            # Move batch to device
            x = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in x.items()}
            if isinstance(y, (list, tuple)):
                targets = y[0]
            else:
                targets = y
            targets = targets.to(device)

            out = model(x)
            pred = out.prediction
            pred_point = point_from_quantiles(pred, model)

            sum_abs, sum_sq, n = update_streaming_metrics(sum_abs, sum_sq, n, targets, pred_point)

    mae = sum_abs / max(n, 1)
    rmse = math.sqrt(sum_sq / max(n, 1))

    return {
        "mode": mode,
        "rmse": float(rmse),
        "mae": float(mae),
        "enc_len": enc_len,
        "pred_len": pred_len,
        "hidden_size": hidden_size,
        "lstm_layers": lstm_layers,
        "attn_heads": attn_heads,
        "dropout": dropout,
        "lr": lr,
        "run_dir": str(run_dir),
        "ckpt": str(ckpt),
        "train_parquet": str(train_parquet),
        "val_parquet": str(val_parquet),
        "device": str(device),
    }


def write_markdown(df: pd.DataFrame, out_md: Path) -> None:
    df2 = df.sort_values("rmse").reset_index(drop=True)
    winner = df2.iloc[0]

    lines = []
    lines.append("# Short-head (15-min, 24h) evaluation\n\n")
    lines.append("Metrics are computed on validation as RMSE/MAE over all horizons (flattened).\n\n")
    lines.append("| mode | rmse | mae | enc_len | pred_len | hidden | lstm_layers | attn_heads | dropout | lr |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for _, r in df2.iterrows():
        lines.append(
            f"| {r['mode']} | {r['rmse']:.6f} | {r['mae']:.6f} | {int(r['enc_len'])} | {int(r['pred_len'])} | "
            f"{int(r['hidden_size'])} | {int(r['lstm_layers'])} | {int(r['attn_heads'])} | {float(r['dropout']):.3f} | {float(r['lr']):.2e} |\n"
        )

    lines.append("\n## Selected model\n")
    lines.append(f"Winner by RMSE: **{winner['mode']}**\n\n")
    lines.append(f"- run_dir: {winner['run_dir']}\n")
    lines.append(f"- ckpt: {winner['ckpt']}\n")
    out_md.write_text("".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_tft_only", required=True)
    ap.add_argument("--run_tft_pvlib", required=True)
    ap.add_argument("--train_tft_only", required=True)
    ap.add_argument("--val_tft_only", required=True)
    ap.add_argument("--train_tft_pvlib", required=True)
    ap.add_argument("--val_tft_pvlib", required=True)
    ap.add_argument("--out_dir", default="experiments/tft/notes")
    ap.add_argument("--batch_size", type=int, default=1024)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    rows.append(
        eval_one(
            "tft_only",
            Path(args.run_tft_only),
            Path(args.train_tft_only),
            Path(args.val_tft_only),
            batch_size=args.batch_size,
        )
    )
    rows.append(
        eval_one(
            "tft_pvlib",
            Path(args.run_tft_pvlib),
            Path(args.train_tft_pvlib),
            Path(args.val_tft_pvlib),
            batch_size=args.batch_size,
        )
    )

    df = pd.DataFrame(rows)
    out_csv = out_dir / "short_head_eval.csv"
    df.to_csv(out_csv, index=False)

    out_md = out_dir / "short_head_model_selection.md"
    write_markdown(df, out_md)

    print(df.sort_values("rmse")[["mode", "rmse", "mae", "enc_len", "pred_len", "dropout", "lr"]].to_string(index=False))
    print(f"\n[DONE] wrote {out_csv}")
    print(f"[DONE] wrote {out_md}")


if __name__ == "__main__":
    main()
