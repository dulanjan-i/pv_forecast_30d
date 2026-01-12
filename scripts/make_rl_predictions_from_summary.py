#!/usr/bin/env python3
"""
Create full per-timestamp prediction parquet for RL policy by applying blend weights
from a summary per-forecast parquet to component model per-timestep parquets.
"""
import argparse
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq


def read_parquet(p):
    t = pq.read_table(p)
    return t.to_pandas()


def detect_weights(df):
    # look for common weight column names
    if {'weights_policy_short','weights_policy_long','weights_policy_physics'}.issubset(df.columns):
        return 'weights_policy_short','weights_policy_long','weights_policy_physics'
    if {'alpha_short','alpha_long','alpha_ml'}.issubset(df.columns):
        # alpha_ml corresponds to physics in some variants
        return 'alpha_short','alpha_long','alpha_ml'
    if {'blend_short','blend_long','blend_physics'}.issubset(df.columns):
        return 'blend_short','blend_long','blend_physics'
    # fallback: try common action mapping later
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--summary', required=True)
    ap.add_argument('--short', required=True)
    ap.add_argument('--long', required=True)
    ap.add_argument('--pvlib', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--action-col', default=None)
    args = ap.parse_args()

    summary = read_parquet(args.summary)
    short = read_parquet(args.short)
    long = read_parquet(args.long)
    pvlib = read_parquet(args.pvlib)

    # unify datetimes
    for df in (summary, short, long, pvlib):
        for c in ['forecast_start','timestamp_utc']:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], utc=True)

    # ensure key columns exist on component dfs
    for name, df in [('short', short), ('long', long), ('pvlib', pvlib)]:
        if 'forecast_start' not in df.columns or 'step_ahead' not in df.columns:
            raise SystemExit(f"Component parquet {name} missing required columns: forecast_start, step_ahead")

    # detect weights columns
    wcols = detect_weights(summary)
    if wcols is None:
        raise SystemExit('Could not detect weight columns in summary parquet; expected weights_policy_* or alpha_* or blend_*')
    ws, wl, wp = wcols

    # If policy action column provided, keep it
    action_col = args.action_col if args.action_col else ('action_policy' if 'action_policy' in summary.columns else ('action' if 'action' in summary.columns else None))

    out_rows = []
    # merge component predictions on forecast_start + step_ahead for each forecast in summary
    # to avoid huge memory, process per-forecast
    summary = summary.sort_values('forecast_start')
    short_idxed = short.set_index(['forecast_start','step_ahead'])
    long_idxed = long.set_index(['forecast_start','step_ahead'])
    pv_idxed = pvlib.set_index(['forecast_start','step_ahead'])

    for _, row in summary.iterrows():
        fs = row['forecast_start']
        # get weights
        s_w = float(row[ws])
        l_w = float(row[wl])
        p_w = float(row[wp])
        # select component rows for this forecast
        try:
            short_block = short_idxed.loc[(fs, slice(None))]
            long_block = long_idxed.loc[(fs, slice(None))]
            pv_block = pv_idxed.loc[(fs, slice(None))]
        except KeyError:
            # no matching rows for this forecast_start, skip
            continue

        # When selecting a single level from a MultiIndex, the selected frame
        # may drop the outer level; ensure forecast_start is present as a column
        short_block = short_block.reset_index()
        long_block = long_block.reset_index()
        pv_block = pv_block.reset_index()
        if 'forecast_start' not in short_block.columns:
            short_block['forecast_start'] = fs
        if 'forecast_start' not in long_block.columns:
            long_block['forecast_start'] = fs
        if 'forecast_start' not in pv_block.columns:
            pv_block['forecast_start'] = fs

        # align on step_ahead
        merged = short_block.merge(long_block, on=['forecast_start','step_ahead','hours_ahead','timestamp_utc'], suffixes=('_short','_long'))
        merged = merged.merge(pv_block, on=['forecast_start','step_ahead','hours_ahead','timestamp_utc'])

        # predicted_power_norm columns
        ps = merged['predicted_power_norm_short']
        pl = merged['predicted_power_norm_long']
        pp = merged['predicted_power_norm']
        merged['predicted_power_norm'] = (s_w * ps + l_w * pl + p_w * pp).astype('float32')

        # keep policy metadata
        if action_col and action_col in row.index:
            merged['policy_action'] = int(row[action_col])
        merged['blend_short'] = float(s_w)
        merged['blend_long'] = float(l_w)
        merged['blend_physics'] = float(p_w)

        keep_cols = ['timestamp_utc','forecast_start','step_ahead','hours_ahead','predicted_power_norm','policy_action','blend_short','blend_long','blend_physics']
        # some columns might be missing (policy_action)
        keep = [c for c in keep_cols if c in merged.columns]
        out_rows.append(merged[keep].copy())

    if not out_rows:
        raise SystemExit('No merged rows produced; check that forecast_start values align between summary and component parquets')

    out_df = pd.concat(out_rows, ignore_index=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path.as_posix(), index=False)
    print(f'WROTE: {out_path.as_posix()}')


if __name__ == '__main__':
    main()
