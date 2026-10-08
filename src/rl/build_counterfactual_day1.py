# src/rl/build_counterfactual_day1.py
"""
Build counterfactual Day-1 evaluations for each forecast_start by replaying
the same underlying short/long/physics components but swapping blend weights
(action -> weights).

Output rows = (#forecast_starts successfully evaluated) * (#actions evaluated)

This is meant for fast offline policy evaluation for the DDQN stage.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

from src.inference.physics_aware_forecaster import PhysicsAwareForecaster
from src.inference.physics_glue import upsample_with_pvlib_shape, blend_hierarchical

logger = logging.getLogger("build_counterfactual_day1")


# -----------------------------
# Helpers
# -----------------------------
def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def must_cols(df: pd.DataFrame, cols: List[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {name}: {missing}")


def to_utc(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, utc=True, errors="coerce")
    return ts


def ensure_power_norm(
    hist: pd.DataFrame,
    gt: pd.DataFrame,
    plant_id: str,
) -> pd.DataFrame:
    """
    Ensure hist has power_norm.
    If missing, merge it from ground truth by timestamp.
    """
    if "power_norm" in hist.columns:
        return hist

    logger.warning("hist_weather_gt has no power_norm, merging from --gt ...")

    must_cols(hist, ["timestamp_utc"], "hist_weather_gt")
    must_cols(gt, ["timestamp_utc", "power_norm"], "gt")

    h = hist.copy()
    h["timestamp_utc"] = to_utc(h["timestamp_utc"])
    h["plant_id"] = plant_id

    g = gt.copy()
    g["timestamp_utc"] = to_utc(g["timestamp_utc"])
    if "plant_id" in g.columns:
        g = g[g["plant_id"].astype(str) == plant_id].copy()

    g = g[["timestamp_utc", "power_norm"]].drop_duplicates("timestamp_utc")

    out = h.merge(g, on="timestamp_utc", how="left")
    # For safety: missing power_norm should not crash TFT
    out["power_norm"] = pd.to_numeric(out["power_norm"], errors="coerce").fillna(0.0)
    return out


def safe_sort_dedup(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(subset=["timestamp_utc"])
    df = df.sort_values("timestamp_utc")
    df = df.drop_duplicates("timestamp_utc", keep="last")
    return df


def resample_hourly(df15: pd.DataFrame, plant_id: str) -> pd.DataFrame:
    """
    Resample 15-min dataframe to hourly. Keeps numeric columns as mean.
    Forces plant_id after resample. Fills NaNs to avoid TFT crashes.
    """
    df15 = df15.copy()
    df15["timestamp_utc"] = to_utc(df15["timestamp_utc"])
    df15 = safe_sort_dedup(df15)

    # numeric aggregation
    num_cols = df15.select_dtypes(include=[np.number]).columns.tolist()
    if "timestamp_utc" in num_cols:
        num_cols.remove("timestamp_utc")

    dfh = (
        df15.set_index("timestamp_utc")[num_cols]
        .resample("1h")
        .mean()
        .reset_index()
    )
    dfh["plant_id"] = plant_id

    # Fill NaNs defensively (TFT does not allow NaNs in real-valued features)
    for c in num_cols:
        if c == "power_norm":
            dfh[c] = pd.to_numeric(dfh[c], errors="coerce").fillna(0.0)
        else:
            dfh[c] = pd.to_numeric(dfh[c], errors="coerce").ffill().bfill().fillna(0.0)

    return dfh


def slice_window(
    df: pd.DataFrame,
    start: pd.Timestamp,
    steps: int,
    freq: str,
) -> pd.DataFrame:
    """
    Slice a fixed-length window starting at `start`.
    """
    # Interpret `freq` as a per-step timedelta (e.g., '15min' or 'h').
    try:
        per_step = pd.Timedelta(freq)
    except Exception:
        # Fallback: assume `freq` is a unit string usable by Timedelta
        per_step = pd.Timedelta(1, unit=freq)

    end = start + per_step * int(steps)
    out = df[(df["timestamp_utc"] >= start) & (df["timestamp_utc"] < end)].copy()
    return out


def action_weight_map_default() -> Dict[int, Dict[str, float]]:
    """
    Action -> weights for day1 blending.
    We use weights in "short/long/physics" space.
    Then convert to alpha_short/alpha_long/alpha_ml for blend_hierarchical.

    Adjust these later if you want, but this is a sane, stable starting point.
    """
    return {
        0: {"short": 0.60, "long": 0.20, "physics": 0.20},
        1: {"short": 0.20, "long": 0.60, "physics": 0.20},
        2: {"short": 0.45, "long": 0.25, "physics": 0.30},
        3: {"short": 0.25, "long": 0.15, "physics": 0.60},
        4: {"short": 0.00, "long": 0.00, "physics": 1.00},
    }


def weights_to_alphas(w: Dict[str, float]) -> Tuple[float, float, float]:
    """
    Convert weights short/long/physics to:
      alpha_short + alpha_long = 1 (within ML)
      alpha_ml = ML vs physics
    """
    w_short = float(w.get("short", 0.0))
    w_long = float(w.get("long", 0.0))
    w_phys = float(w.get("physics", 0.0))

    w_ml = w_short + w_long
    if w_ml <= 1e-9:
        # pure physics
        return 0.5, 0.5, 0.0

    alpha_short = w_short / w_ml
    alpha_long = w_long / w_ml
    alpha_ml = np.clip(w_ml, 0.0, 1.0)
    return float(alpha_short), float(alpha_long), float(alpha_ml)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    setup_logging()

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--plant_meta", required=True)

    ap.add_argument("--short_ckpt", required=True)
    ap.add_argument("--long_ckpt", required=True)
    ap.add_argument("--short_train_parquet", required=True)
    ap.add_argument("--long_train_parquet", required=True)

    ap.add_argument("--sarns_norm", required=True, help="Transitions with forecast_start list")
    ap.add_argument("--hist_weather_gt", required=True, help="15-min historical with weather (+ ideally power_norm)")
    ap.add_argument("--weather_15min", required=True, help="15-min weather (+ pvlib) for the forecast horizon")
    ap.add_argument("--weather_hourly", default=None, help="Optional hourly weather (+ pvlib) for long head. If omitted, derived from weather_15min.")
    ap.add_argument("--gt", required=True, help="15-min ground truth parquet with power_norm")

    args = ap.parse_args()

    plant_meta = Path(args.plant_meta)
    if not plant_meta.exists():
        raise FileNotFoundError(str(plant_meta))

    meta = pd.read_json(plant_meta, typ="series")
    plant_id = str(meta.get("plant_id", "plant_03"))

    logger.info("Plant: %s", plant_id)

    # Load transitions
    dfT = pd.read_parquet(args.sarns_norm)
    must_cols(dfT, ["forecast_start"], "sarns_norm")
    dfT["forecast_start"] = to_utc(dfT["forecast_start"])
    forecast_starts = sorted(dfT["forecast_start"].dropna().unique().tolist())
    logger.info("Forecast starts: %d", len(forecast_starts))

    # Load historical and gt
    dfH15 = pd.read_parquet(args.hist_weather_gt)
    dfH15["timestamp_utc"] = to_utc(dfH15["timestamp_utc"])
    dfH15["plant_id"] = plant_id
    dfH15 = safe_sort_dedup(dfH15)

    dfGT = pd.read_parquet(args.gt)
    dfGT["timestamp_utc"] = to_utc(dfGT["timestamp_utc"])
    if "plant_id" in dfGT.columns:
        dfGT = dfGT[dfGT["plant_id"].astype(str) == plant_id].copy()
    dfGT = safe_sort_dedup(dfGT)
    must_cols(dfGT, ["timestamp_utc", "power_norm"], "gt")
    dfGT["power_norm"] = pd.to_numeric(dfGT["power_norm"], errors="coerce").fillna(0.0)

    # Ensure power_norm exists in hist
    dfH15 = ensure_power_norm(dfH15, dfGT, plant_id)

    # Load weather
    dfW15 = pd.read_parquet(args.weather_15min)
    dfW15["timestamp_utc"] = to_utc(dfW15["timestamp_utc"])
    dfW15["plant_id"] = plant_id
    dfW15 = safe_sort_dedup(dfW15)

    if args.weather_hourly:
        dfW_h = pd.read_parquet(args.weather_hourly)
        dfW_h["timestamp_utc"] = to_utc(dfW_h["timestamp_utc"])
        dfW_h["plant_id"] = plant_id
        dfW_h = safe_sort_dedup(dfW_h)
    else:
        logger.info("Deriving weather_hourly from weather_15min ...")
        dfW_h = resample_hourly(dfW15.assign(power_norm=0.0), plant_id)  # dummy power_norm for fill logic

    # Build hourly historical for long head
    dfH_h = resample_hourly(dfH15, plant_id)

    # Initialize forecaster (match PhysicsAwareForecaster signature)
    forecaster = PhysicsAwareForecaster(
        short_ckpt=args.short_ckpt,
        long_ckpt=args.long_ckpt,
        plant_metadata=str(plant_meta),
        short_train_parquet=args.short_train_parquet,
        long_train_parquet=args.long_train_parquet,
        device="cuda",
    )

    # Action map
    action_to_w = action_weight_map_default()
    actions = sorted(action_to_w.keys())

    out_rows = []
    stats = {"attempted": 0, "saved": 0, "skip_missing": 0, "skip_failed": 0}

    for fs in forecast_starts:
        stats["attempted"] += 1

        try:
            # Need at least: encoder windows + decoder windows
            # Short head: encoder 96 steps (1 day), decoder 96 steps (day1)
            enc15 = dfH15[dfH15["timestamp_utc"] < fs].tail(96).copy()
            # Ensure encoder has power_norm filled (use forecaster helper after init)
            enc15 = forecaster._ensure_encoder_power_norm(enc15)
            dec15_day1 = slice_window(dfW15, fs, steps=96, freq="15min")

            if len(enc15) < 96 or len(dec15_day1) < 96:
                stats["skip_missing"] += 1
                continue

            # Long head: encoder 168 hours (7 days), decoder 720 hours (30 days)
            encH = dfH_h[dfH_h["timestamp_utc"] < fs].tail(168).copy()
            decH_30d = slice_window(dfW_h, fs, steps=720, freq="h")

            if len(encH) < 168 or len(decH_30d) < 720:
                stats["skip_missing"] += 1
                continue

            # Ground truth day1
            gt_day1 = dfGT[(dfGT["timestamp_utc"] >= fs) & (dfGT["timestamp_utc"] < fs + pd.Timedelta(days=1))].copy()
            if len(gt_day1) < 96:
                stats["skip_missing"] += 1
                continue
            gt_day1 = gt_day1.sort_values("timestamp_utc").head(96)
            gt_y = gt_day1["power_norm"].to_numpy(dtype=np.float32)

            # PVLib baseline day1 norm
            if "pvlib_ac_kw" not in dec15_day1.columns:
                raise ValueError("weather_15min missing pvlib_ac_kw")
            cap_kw = float(meta.get("installed_capacity_kw", 1.0))
            pvlib_day1_norm = (pd.to_numeric(dec15_day1["pvlib_ac_kw"], errors="coerce").fillna(0.0).to_numpy() / max(cap_kw, 1e-6)).astype(np.float32)
            pvlib_day1_norm = np.clip(pvlib_day1_norm, 0.0, 2.0)

            # Run component predictions once
            # NOTE: PhysicsAwareForecaster._predict_short_head_for_day signature is
            # (day_start, day_idx, historical_df, weather_df)
            # so pass the forecast start timestamp `fs`, day index 0, then
            # the encoder/history and decoder/weather windows.
            short_pred = forecaster._predict_short_head_for_day(fs, 0, enc15, dec15_day1)
            long_hourly = forecaster._predict_long_head(fs, encH, decH_30d)  # 720
            long_hourly_day1 = np.asarray(long_hourly[:24], dtype=np.float32)

            long_day1_upsampled = upsample_with_pvlib_shape(
                hourly_predictions=long_hourly_day1,
                pvlib_15min=pvlib_day1_norm,
                method="proportional",
            ).astype(np.float32)

            # Evaluate all actions for this forecast_start
            for a in actions:
                w = action_to_w[a]
                alpha_short, alpha_long, alpha_ml = weights_to_alphas(w)

                pred_day1 = blend_hierarchical(
                    short_pred=np.asarray(short_pred, dtype=np.float32),
                    long_upsampled=np.asarray(long_day1_upsampled, dtype=np.float32),
                    pvlib_baseline=np.asarray(pvlib_day1_norm, dtype=np.float32),
                    alpha_short=alpha_short,
                    alpha_long=alpha_long,
                    alpha_ml=alpha_ml,
                    constraints=True,
                    max_capacity_multiplier=1.2,
                ).astype(np.float32)

                rmse = float(np.sqrt(np.mean((pred_day1 - gt_y) ** 2)))

                out_rows.append(
                    {
                        "forecast_start": fs,
                        "action": int(a),
                        "alpha_short": alpha_short,
                        "alpha_long": alpha_long,
                        "alpha_ml": alpha_ml,
                        "rmse_day1": rmse,
                    }
                )

            stats["saved"] += 1

        except Exception as e:
            stats["skip_failed"] += 1
            logger.warning("Forecast_start %s failed: %s", str(fs), repr(e))
            continue

    out_df = pd.DataFrame(out_rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)

    logger.info("WROTE: %s", str(out_path))
    logger.info("Stats: %s", stats)
    logger.info("Rows: %d (forecast_starts_saved * actions)", len(out_df))


if __name__ == "__main__":
    main()
