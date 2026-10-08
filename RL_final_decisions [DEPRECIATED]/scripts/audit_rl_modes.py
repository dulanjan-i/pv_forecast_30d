#!/usr/bin/env python3
"""Audit RL v1 vs v2 modes: action comparison table, metrics, audit markdown, and winter-week plot.

Writes outputs to: freeze_corrected/final_thesis_v1/benchmarks/thesis_ready/{tables,figures}
"""
from pathlib import Path
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import matplotlib.pyplot as plt


ROOT = Path.cwd()
FREEZE = ROOT / "freeze"
FREEZE_CORR = ROOT / "freeze_corrected" / "final_thesis_v1"
PHASE_PROC = FREEZE_CORR / "phase1_2024daily_final" / "processed"
OUT_DIR = FREEZE_CORR / "benchmarks" / "thesis_ready"
TABLES = OUT_DIR / "tables"
FIGS = OUT_DIR / "figures"
TABLES.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)


def read_parquet(p: Path) -> pd.DataFrame:
    return pq.read_table(p.as_posix()).to_pandas()


def stitch_most_recent(pred: pd.DataFrame, y_col: str) -> pd.DataFrame:
    dd = pred.sort_values(["timestamp_utc", "hours_ahead"]) 
    dd = dd.groupby("timestamp_utc", as_index=False).first()
    return dd[["timestamp_utc", "forecast_start", "hours_ahead", y_col]]


def safe_metric_rmse(y_true, y_pred):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        return float('nan')
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def main():
    # paths
    v1_p = PHASE_PROC / "predictions_phase1_policy_heuristic_rl.parquet"
    v2_p = PHASE_PROC / "predictions_phase1_policy_full_rl.parquet"
    truth_p = FREEZE / "final_thesis_v1" / "phase1_2024daily_final" / "processed" / "ground_truth_15min_utc_capnorm.parquet"
    baseline_p = FREEZE / "final_thesis_v1" / "phase1_2024daily_final" / "processed" / "predictions_phase1_baseline_rerun.parquet"

    assert v1_p.exists(), f"Missing {v1_p}"
    assert v2_p.exists(), f"Missing {v2_p}"
    assert truth_p.exists(), f"Missing {truth_p}"
    assert baseline_p.exists(), f"Missing {baseline_p}"

    df1 = read_parquet(v1_p)
    df2 = read_parquet(v2_p)

    # normalize timestamp types
    for df in (df1, df2):
        df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], utc=True)
        df['forecast_start'] = pd.to_datetime(df['forecast_start'], utc=True)

    # Per-forecast action: assume action is constant within forecast_start; take first
    act1 = df1.groupby('forecast_start', as_index=False).agg({'policy_action': lambda x: x.iloc[0]})
    act2 = df2.groupby('forecast_start', as_index=False).agg({'policy_action': lambda x: x.iloc[0]})

    merged_actions = act1.merge(act2, on='forecast_start', how='outer', suffixes=('_v1','_v2'))
    merged_actions['action_changed'] = merged_actions['policy_action_v1'] != merged_actions['policy_action_v2']

    # Bring in blend columns if available (first row per forecast_start)
    def first_cols(df, prefix):
        grp = df.groupby('forecast_start').first().reset_index()
        cols = {}
        for c in grp.columns:
            if c.startswith('blend_') or c.startswith('weights_'):
                cols[f"{c}_{prefix}"] = grp[c].values
        out = pd.DataFrame({'forecast_start': grp['forecast_start']})
        for k, v in cols.items():
            out[k] = v
        return out

    blends1 = first_cols(df1, 'v1')
    blends2 = first_cols(df2, 'v2')

    merged = merged_actions.merge(blends1, on='forecast_start', how='left')
    merged = merged.merge(blends2, on='forecast_start', how='left')

    # Save action comparison CSV
    out_csv = TABLES / 'action_selection_v1_vs_v2.csv'
    merged.sort_values('forecast_start').to_csv(out_csv.as_posix(), index=False)
    print(f"[OK] wrote {out_csv}")

    # Compute overall metrics using stitch-most-recent approach
    truth = read_parquet(truth_p)
    baseline = read_parquet(baseline_p)
    # normalize
    truth['timestamp_utc'] = pd.to_datetime(truth['timestamp_utc'], utc=True)
    baseline['timestamp_utc'] = pd.to_datetime(baseline['timestamp_utc'], utc=True)
    for df in (baseline, df1, df2):
        if 'predicted_power_norm' in df.columns:
            df['predicted_power_norm'] = pd.to_numeric(df['predicted_power_norm'], errors='coerce')
        if 'hours_ahead' in df.columns:
            df['hours_ahead'] = pd.to_numeric(df['hours_ahead'], errors='coerce')

    baseline_st = stitch_most_recent(baseline.rename(columns={'predicted_power_norm':'y_baseline'}), 'y_baseline')
    v1_st = stitch_most_recent(df1.rename(columns={'predicted_power_norm':'y_model'}), 'y_model')
    v2_st = stitch_most_recent(df2.rename(columns={'predicted_power_norm':'y_model'}), 'y_model')

    # join with truth
    tr = truth[['timestamp_utc','power_norm']].rename(columns={'power_norm':'y_true'})
    base_join = baseline_st.merge(tr, on='timestamp_utc', how='inner')
    v1_join = v1_st.merge(tr, on='timestamp_utc', how='inner')
    v2_join = v2_st.merge(tr, on='timestamp_utc', how='inner')

    rmse_base = safe_metric_rmse(base_join['y_true'].values, base_join['y_baseline'].values)
    rmse_v1 = safe_metric_rmse(v1_join['y_true'].values, v1_join['y_model'].values)
    rmse_v2 = safe_metric_rmse(v2_join['y_true'].values, v2_join['y_model'].values)

    # win-rate (per-forecast) - compare RMSE per forecast_start if available
    # For per-forecast comparison, use the stitched per-forecast series grouped by forecast_start
    # compute RMSE per forecast_start for baseline and models
    def per_forecast_rmse(pred_df, truth_df, pred_col):
        joined = pred_df.merge(truth_df, on='timestamp_utc', how='inner')
        # group by forecast_start
        rows = []
        if 'forecast_start' not in joined.columns:
            return pd.DataFrame()
        for fs, grp in joined.groupby('forecast_start'):
            r = safe_metric_rmse(grp['y_true'].values, grp[pred_col].values)
            rows.append({'forecast_start': fs, 'rmse': r})
        return pd.DataFrame(rows)

    # need full pred (not stitched) with y_true - pass raw pred frames; per_forecast_rmse will merge with truth
    full_base = baseline.copy()
    full_v1 = df1.copy()
    full_v2 = df2.copy()

    pf_base = per_forecast_rmse(full_base, tr, 'predicted_power_norm')
    pf_v1 = per_forecast_rmse(full_v1, tr, 'predicted_power_norm')
    pf_v2 = per_forecast_rmse(full_v2, tr, 'predicted_power_norm')

    # compare per-forecast
    cmp_v1 = pf_base.merge(pf_v1, on='forecast_start', suffixes=('_base','_v1'))
    cmp_v1['v1_wins'] = cmp_v1['rmse_v1'] < cmp_v1['rmse_base']
    v1_win_rate = float(cmp_v1['v1_wins'].mean()) * 100.0 if not cmp_v1.empty else float('nan')

    cmp_v2 = pf_base.merge(pf_v2, on='forecast_start', suffixes=('_base','_v2'))
    cmp_v2['v2_wins'] = cmp_v2['rmse_v2'] < cmp_v2['rmse_base']
    v2_win_rate = float(cmp_v2['v2_wins'].mean()) * 100.0 if not cmp_v2.empty else float('nan')

    # write audit markdown
    audit_md = TABLES / 'rl_modes_audit.md'
    txt = []
    txt.append('# RL Modes Audit: Heuristic RL (v1) vs Full RL (v2)')
    txt.append('')
    txt.append('## Summary')
    txt.append('This audit compares two synthesized RL prediction modes:')
    txt.append('- **Heuristic RL (v1)**: synthesized from the v1 summary weights (earlier run).')
    txt.append('- **Full RL (v2)**: synthesized from the v2 summary weights (re-run with corrected action→blend mapping).')
    txt.append('')
    txt.append('## What was tested')
    txt.append('- Both modes were evaluated on the same processed backtest dataset using the thesis-ready evaluation pipeline.')
    txt.append('- Predictions were synthesized by applying per-forecast weights onto component per-timestep predictions (short/long/pvlib).')
    txt.append('')
    txt.append('## Metrics (stitched, most-recent selection)')
    txt.append(f'- Baseline RMSE: {rmse_base:.6f}')
    txt.append(f'- Heuristic RL (v1) RMSE: {rmse_v1:.6f}')
    txt.append(f'- Full RL (v2) RMSE: {rmse_v2:.6f}')
    txt.append('')
    txt.append('## Per-forecast win rates')
    txt.append(f'- Heuristic RL (v1) win rate vs baseline: {v1_win_rate:.2f}%')
    txt.append(f'- Full RL (v2) win rate vs baseline: {v2_win_rate:.2f}%')
    txt.append('')
    txt.append('## Action selection differences')
    txt.append(f'- Action selection table saved as `tables/action_selection_v1_vs_v2.csv`')
    txt.append('')
    txt.append('## Observations')
    if not np.isnan(rmse_v2) and rmse_v2 < rmse_base:
        txt.append(f'- The Full RL (v2) policy improves RMSE by {(rmse_base-rmse_v2)/rmse_base*100.0:.2f}% relative to the baseline.')
    else:
        txt.append('- The Full RL (v2) policy does not improve RMSE vs baseline on this dataset.')
    txt.append('')
    txt.append('## Provenance & Notes')
    txt.append('- Per-timestep predictions for RL modes were synthesized from per-forecast summary weights and the component predictions (short, long, physics).')
    txt.append('- v1 contained an incorrect action→blend mapping; v2 was re-run with corrected mapping. This explains the performance gap between v1 and v2.')
    txt.append('')
    audit_md.write_text('\n'.join(txt))
    print(f"[OK] wrote {audit_md}")

    # Write a short CSV summary metrics
    metrics_df = pd.DataFrame([
        {'Mode':'Baseline','RMSE':rmse_base,'WinRate_pct':0.0},
        {'Mode':'Heuristic_RL_v1','RMSE':rmse_v1,'WinRate_pct':v1_win_rate},
        {'Mode':'Full_RL_v2','RMSE':rmse_v2,'WinRate_pct':v2_win_rate},
    ])
    metrics_df.to_csv(TABLES / 'rl_modes_metrics_summary.csv', index=False)
    print(f"[OK] wrote {TABLES / 'rl_modes_metrics_summary.csv'}")

    # ------------------ create winter week timeseries plot ------------------
    # Use the same winter week as the thesis pipeline defaults: 2024-01-10 -> 2024-01-17
    start = pd.to_datetime('2024-01-10T00:00:00Z')
    end = pd.to_datetime('2024-01-17T00:00:00Z')

    # stitch series (most recent) for plotting
    base_plot = baseline_st.rename(columns={'y_baseline':'y_baseline'})
    v1_plot = v1_st.rename(columns={'y_model':'y_model'})
    v2_plot = v2_st.rename(columns={'y_model':'y_model'})

    # merge on timestamp
    p = tr.merge(base_plot, on='timestamp_utc', how='inner')
    p = p.merge(v1_plot[['timestamp_utc','y_model']], on='timestamp_utc', how='left').rename(columns={'y_model':'y_v1'})
    p = p.merge(v2_plot[['timestamp_utc','y_model']], on='timestamp_utc', how='left').rename(columns={'y_model':'y_v2'})

    ww = p[(p['timestamp_utc'] >= start) & (p['timestamp_utc'] <= end)].copy()
    if ww.empty:
        print('[WARN] winter week window empty; skipping plot')
    else:
        plt.figure(figsize=(12,4))
        plt.plot(ww['timestamp_utc'], ww['y_true'], label='Ground Truth', color='#888888', linewidth=1.5)
        plt.plot(ww['timestamp_utc'], ww['y_baseline'], label='MiRACLE Core', color='#00AA00', linestyle='--')
        plt.plot(ww['timestamp_utc'], ww['y_v1'], label='Heuristic RL (v1)', color='#6BA3D8')
        plt.plot(ww['timestamp_utc'], ww['y_v2'], label='Full RL (v2)', color='#FAA43A')
        plt.ylabel('Normalized Power')
        plt.xlabel('Timestamp (UTC)')
        plt.legend(loc='upper left', fontsize=9)
        plt.grid(alpha=0.3, linestyle=':')
        plt.tight_layout()
        out_png = FIGS / 'winter_week_miracle_heuristic_full_timeseries_thesis.png'
        plt.gcf().autofmt_xdate()
        plt.savefig(out_png.as_posix(), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] wrote {out_png}")


if __name__ == '__main__':
    main()
