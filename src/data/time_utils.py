"""
Time helpers for the weather ingestion path.

Why this exists
---------------
Open-Meteo returns hourly data as a start epoch (UTC) plus a fixed interval, i.e. a
*uniform UTC* series. Earlier versions of the pipeline turned that into naive Europe/Berlin
wall-clock labels with a uniform 1 h step and later re-localised the labels as Berlin time.
Because Berlin's UTC offset changes with DST but the labels do not, the weather ended up
shifted by 1 hour from late March to late October. Keep everything in UTC instead.
"""
from __future__ import annotations

import warnings

import pandas as pd

LOCAL_TZ = "Europe/Berlin"


def utc_index_from_epoch(start_epoch_s: int, interval_s: int, n_points: int) -> pd.DatetimeIndex:
    """Uniform, timezone-aware UTC index from an Open-Meteo start epoch and interval."""
    start = pd.to_datetime(int(start_epoch_s), unit="s", utc=True)
    return pd.date_range(start=start, periods=int(n_points), freq=pd.Timedelta(seconds=int(interval_s)))


def repair_legacy_naive_local(dates: pd.Series, tz: str = LOCAL_TZ) -> pd.DatetimeIndex:
    """
    Recover true UTC timestamps from legacy `date` columns.

    Legacy CSVs hold naive labels generated as `first_local_label + k * 1h`. They are uniform in
    real time, so the UTC time of row k is `utc(first_label) + k * 1h`. This is exact as long as
    the labels are uniform, which is checked here.
    """
    d = pd.to_datetime(dates).reset_index(drop=True)
    if d.empty:
        return pd.DatetimeIndex([], tz="UTC")
    step = d.diff().dropna()
    if not (step == step.iloc[0]).all():
        raise ValueError("legacy `date` labels are not uniformly spaced; cannot repair safely")
    first_utc = d.iloc[0].tz_localize(tz, ambiguous=True, nonexistent="shift_forward").tz_convert("UTC")
    warnings.warn(
        "Hourly weather has legacy naive local `date` labels; reconstructing UTC from the first label. "
        "Re-fetch with the current call_openmeteo_hist_germany.py to get a `timestamp_utc` column.",
        stacklevel=2,
    )
    return pd.DatetimeIndex(first_utc + step.iloc[0] * pd.RangeIndex(len(d)))
