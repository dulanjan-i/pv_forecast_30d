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
    ):
        self.time_col = time_col
        self.group_col = group_col
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.input_window = int(input_window)
        self.forecast_horizon = int(forecast_horizon)

        # Sort by group + time to ensure correct ordering
        if group_col is not None:
            df = df.sort_values([group_col, time_col]).reset_index(drop=True)
        else:
            df = df.sort_values(time_col).reset_index(drop=True)

        self.df = df

        # Precompute valid index ranges for sliding windows
        self.indices = self._build_indices()

    def _build_indices(self):
        """
        Build a list of (start_idx, end_idx_input, start_idx_target, end_idx_target)
        for each valid window.
        """
        indices = []

        if self.group_col is not None:
            grouped = self.df.groupby(self.group_col, sort=False)
            for _, g in grouped:
                n = len(g)
                max_start = n - (self.input_window + self.forecast_horizon) + 1
                for start in range(max_start):
                    in_start = g.index[start]
                    in_end = g.index[start + self.input_window - 1]
                    out_start = g.index[start + self.input_window]
                    out_end = g.index[start + self.input_window + self.forecast_horizon - 1]
                    indices.append((in_start, in_end, out_start, out_end))
        else:
            n = len(self.df)
            max_start = n - (self.input_window + self.forecast_horizon) + 1
            for start in range(max_start):
                in_start = start
                in_end = start + self.input_window - 1
                out_start = start + self.input_window
                out_end = start + self.input_window + self.forecast_horizon - 1
                indices.append((in_start, in_end, out_start, out_end))

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