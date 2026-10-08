from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch


# -----------------------------
# Helpers
# -----------------------------
def to_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def ensure_exists(p: Path, kind: str) -> None:
    if not p.exists():
        raise FileNotFoundError(f"Missing {kind}: {p}")


def strip_prefix(state_dict: dict) -> dict:
    prefixes = ["model.", "net.", "encoder.", "lstm_encoder.", "module."]
    out = {}
    for k, v in state_dict.items():
        kk = k
        for p in prefixes:
            if kk.startswith(p):
                kk = kk[len(p) :]
        out[kk] = v
    return out


def infer_lstm_shapes(state: dict) -> Tuple[int, int, int]:
    keys = list(state.keys())
    wih_keys = [k for k in keys if "weight_ih_l0" in k]
    if not wih_keys:
        raise RuntimeError("Cannot infer LSTM shapes, no weight_ih_l0 in state_dict")
    wih = state[wih_keys[0]]
    hidden4, input_size = wih.shape
    if hidden4 % 4 != 0:
        raise RuntimeError(f"Unexpected weight_ih_l0 shape: {tuple(wih.shape)}")
    hidden = hidden4 // 4

    layer_ids = []
    for k in keys:
        if "weight_ih_l" in k:
            try:
                layer_ids.append(int(k.split("weight_ih_l")[-1].split(".")[0]))
            except Exception:
                pass
    num_layers = max(layer_ids) + 1 if layer_ids else 1
    return int(input_size), int(hidden), int(num_layers)


def load_lstm_encoder(ckpt: Path, device: torch.device) -> torch.nn.Module:
    obj = torch.load(str(ckpt), map_location="cpu")
    sd = None
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        sd = obj["state_dict"]
    elif isinstance(obj, dict) and all(isinstance(v, torch.Tensor) for v in obj.values()):
        sd = obj
    else:
        raise RuntimeError(f"Unsupported checkpoint format: {ckpt}")

    sd = strip_prefix(sd)
    input_size, hidden_size, num_layers = infer_lstm_shapes(sd)

    model = torch.nn.LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        batch_first=True,
    )
    model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    print(f"[INFO] Loaded encoder {ckpt.name}: input_size={input_size} hidden={hidden_size} layers={num_layers}")
    return model


def make_windows(X: np.ndarray, seq_len: int) -> np.ndarray:
    T, F = X.shape
    if T < seq_len:
        raise RuntimeError(f"Not enough rows: T={T} < seq_len={seq_len}")
    n = T - seq_len + 1
    s0, s1 = X.strides
    win = np.lib.stride_tricks.as_strided(
        X,
        shape=(n, seq_len, F),
        strides=(s0, s0, s1),
        writeable=False,
    )
    return win


def batched_encode(model: torch.nn.Module, windows: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    outs = []
    with torch.no_grad():
        for i in range(0, windows.shape[0], batch_size):
            xb = torch.from_numpy(windows[i : i + batch_size]).to(device)
            out, (h, c) = model(xb)
            emb = h[-1]  # (B, H)
            outs.append(emb.detach().cpu().numpy().astype(np.float32))
    return np.vstack(outs)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1-in", required=True, type=str)
    ap.add_argument("--phase1-out", required=True, type=str)
    ap.add_argument("--encoder-ckpt", required=True, type=str)
    ap.add_argument("--hist-stream", required=True, type=str, help="Must include power_norm + required weather cols")
    ap.add_argument("--pca-pkl", default="", type=str, help="Optional. If provided, applies PCA to 32 dims.")
    ap.add_argument("--plant-id", default="plant_03", type=str)
    ap.add_argument("--time-col", default="timestamp_utc", type=str)
    ap.add_argument("--plant-col", default="plant_id", type=str)
    ap.add_argument("--seq-len", default=96, type=int)
    ap.add_argument("--lag", default=96, type=int)
    ap.add_argument("--batch-size", default=2048, type=int)
    ap.add_argument("--device", default="cuda", type=str)

    # EXACT 15-feature schema from your YAML, order locked
    ap.add_argument(
        "--feature-cols",
        default="power_norm,poa_irradiance,temperature_2m,relative_humidity_2m,precipitation,weather_code,cloud_cover,wind_speed_10m,wind_direction_10m,shortwave_radiation_instant,direct_radiation_instant,diffuse_radiation_instant,direct_normal_irradiance_instant,global_tilted_irradiance_instant,surface_pressure",
        type=str,
    )

    args = ap.parse_args()

    phase_in = Path(args.phase1_in)
    phase_out = Path(args.phase1_out)
    phase_out.mkdir(parents=True, exist_ok=True)

    weather_in = phase_in / "weather_with_pvlib_15min.parquet"
    ensure_exists(weather_in, "phase weather parquet")

    enc_ckpt = Path(args.encoder_ckpt)
    ensure_exists(enc_ckpt, "encoder ckpt")

    hist_p = Path(args.hist_stream)
    ensure_exists(hist_p, "hist stream parquet")

    # Load data
    w = pd.read_parquet(weather_in, engine="pyarrow")
    h = pd.read_parquet(hist_p, engine="pyarrow")

    if args.time_col not in w.columns or args.time_col not in h.columns:
        raise KeyError(f"Missing {args.time_col} in weather or hist stream parquet")

    w[args.time_col] = to_utc(w[args.time_col])
    h[args.time_col] = to_utc(h[args.time_col])

    # Filter plant if possible
    if args.plant_col in h.columns:
        h = h[h[args.plant_col].astype(str) == str(args.plant_id)].copy()
    if args.plant_col in w.columns:
        if str(args.plant_id) in w[args.plant_col].astype(str).unique():
            w = w[w[args.plant_col].astype(str) == str(args.plant_id)].copy()

    h = h.sort_values(args.time_col).reset_index(drop=True)

    # Feature cols (exact order)
    feature_cols: List[str] = [c.strip() for c in args.feature_cols.split(",") if c.strip()]
    missing = [c for c in feature_cols if c not in h.columns]
    if missing:
        raise KeyError(
            "Hist stream parquet is missing required feature cols for encoder:\n"
            + "\n".join(missing)
            + "\n\nFix by providing a hist stream parquet that includes these columns "
            "with the same preprocessing used during LSTM training."
        )

    X = h[feature_cols].to_numpy(dtype=np.float32)

    # Check input_size matches encoder
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    model = load_lstm_encoder(enc_ckpt, device=device)

    # Build windows and align to window end time
    L = int(args.seq_len)
    windows = make_windows(X, seq_len=L)
    end_times = h[args.time_col].iloc[L - 1 :].reset_index(drop=True)

    enc64 = batched_encode(model, windows, batch_size=int(args.batch_size), device=device)

    # Optional PCA32
    use_pca = bool(args.pca_pkl.strip())
    if use_pca:
        pca_p = Path(args.pca_pkl)
        ensure_exists(pca_p, "PCA pkl")
        with open(pca_p, "rb") as f:
            pca = pickle.load(f)
        enc = pca.transform(enc64).astype(np.float32)
        if enc.shape[1] != 32:
            raise RuntimeError(f"PCA output dim={enc.shape[1]}, expected 32")
        base_names = [f"lstm_enc_pca_{j:03d}" for j in range(32)]
        print("[INFO] Applied PCA: enc64 -> enc32")
    else:
        enc = enc64
        base_names = [f"lstm_enc_{j:03d}" for j in range(enc.shape[1])]
        print(f"[INFO] No PCA: using raw enc dim={enc.shape[1]}")

    df_enc = pd.DataFrame({args.time_col: end_times})
    for j, name in enumerate(base_names):
        df_enc[name] = enc[:, j]

    lag = int(args.lag)
    for name in base_names:
        df_enc[f"{name}_lag{lag}"] = df_enc[name].shift(lag)

    # Merge into phase weather
    out = w.merge(df_enc, on=args.time_col, how="left")

    # Coverage checks
    sample_col = base_names[0]
    frac = float(out[sample_col].notna().mean())
    frac_lag = float(out[f"{sample_col}_lag{lag}"].notna().mean())
    print(f"[INFO] coverage {sample_col}: {frac:.4f}")
    print(f"[INFO] coverage {sample_col}_lag{lag}: {frac_lag:.4f}")

    out_path = phase_out / "weather_with_pvlib_15min.parquet"
    out.to_parquet(out_path, index=False)
    print(f"[OK] wrote: {out_path}")


if __name__ == "__main__":
    main()
