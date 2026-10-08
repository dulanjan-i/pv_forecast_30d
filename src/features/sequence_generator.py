# src/features/sequence_generator.py

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class SimpleWindowDataset(Dataset):
    """
    Sliding-window dataset for sequence-to-one or sequence-to-few forecasting.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe sorted by time (and group, if provided).
    time_col : str
        Name of time index column. Currently assumed to be sortable (int or datetime).
    group_col : Optional[str]
        Optional column that identifies different time series (e.g. site_id).
        If None, the whole df is treated as one long sequence.
    feature_cols : List[str]
        Names of input feature columns.
    target_col : str
        Name of target column.
    input_window : int
        Number of past steps to use as input (encoder length).
    forecast_horizon : int
        Number of future steps to predict. In pretraining we usually set this to 1.
    require_contiguous : bool
        If True, drop every window whose timestamps are not exactly `step` apart, i.e. windows that
        span a gap left by dropped/missing rows. Default False keeps the v1.0 behaviour (positional
        windows), which is what the published checkpoints were trained with.
    step : pandas.Timedelta | int | float | None
        Expected spacing between consecutive rows for `require_contiguous`. Defaults to 15 minutes
        for datetime columns and 1 for numeric ones.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        time_col: str,
        group_col: Optional[str],
        feature_cols: List[str],
        target_col: str,
        input_window: int,
        forecast_horizon: int = 1,
        require_contiguous: bool = False,
        step=None,
    ):
        self.time_col = time_col
        self.group_col = group_col
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.input_window = int(input_window)
        self.forecast_horizon = int(forecast_horizon)
        self.require_contiguous = bool(require_contiguous)
        self.step = step
        self.n_dropped_gap_windows = 0

        # Sort by group + time to ensure correct ordering
        if group_col is not None:
            df = df.sort_values([group_col, time_col]).reset_index(drop=True)
        else:
            df = df.sort_values(time_col).reset_index(drop=True)

        self.df = df

        # Precompute valid index ranges for sliding windows
        self.indices = self._build_indices()

    def _contiguous_mask(self, times: pd.Series, n_windows: int) -> np.ndarray:
        """True for window starts whose full span (input + horizon) has no gap."""
        span = self.input_window + self.forecast_horizon - 1
        if pd.api.types.is_datetime64_any_dtype(times):
            step = self.step if self.step is not None else pd.Timedelta(minutes=15)
            t = times.to_numpy(dtype="datetime64[ns]").astype("int64")
            step_ns = int(pd.Timedelta(step).value)
        else:
            step_ns = int(self.step if self.step is not None else 1)
            t = times.to_numpy().astype("int64")
        return (t[span:span + n_windows] - t[:n_windows]) == span * step_ns

    def _build_indices(self):
        """
        Build a list of (start_idx, end_idx_input, start_idx_target, end_idx_target)
        for each valid window.
        """
        indices = []
        groups = (
            [g for _, g in self.df.groupby(self.group_col, sort=False)]
            if self.group_col is not None
            else [self.df]
        )
        for g in groups:
            n = len(g)
            n_windows = n - (self.input_window + self.forecast_horizon) + 1
            if n_windows <= 0:
                continue
            keep = np.ones(n_windows, dtype=bool)
            if self.require_contiguous:
                keep = self._contiguous_mask(g[self.time_col], n_windows)
                self.n_dropped_gap_windows += int((~keep).sum())
            idx = g.index
            for start in np.flatnonzero(keep):
                indices.append((
                    idx[start],
                    idx[start + self.input_window - 1],
                    idx[start + self.input_window],
                    idx[start + self.input_window + self.forecast_horizon - 1],
                ))
        return indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        in_start, in_end, out_start, out_end = self.indices[idx]

        df = self.df

        x = df.loc[in_start:in_end, self.feature_cols].to_numpy(dtype=np.float32)
        y = df.loc[out_start:out_end, self.target_col].to_numpy(dtype=np.float32)

        # For horizon == 1, return scalar instead of length-1 array for convenience
        if self.forecast_horizon == 1:
            y = y[0]

        return x, y