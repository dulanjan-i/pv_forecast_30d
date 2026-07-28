# src/utils/metrics.py
"""
Basic regression metrics for PV forecasting.
"""

from __future__ import annotations

import torch


def rmse(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Root Mean Squared Error."""
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2))


def mae(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Mean Absolute Error."""
    return torch.mean(torch.abs(y_true - y_pred))


def mape(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Mean Absolute Percentage Error.

    eps prevents division by zero when y_true is near zero (night-time values).
    """
    denom = torch.clamp(torch.abs(y_true), min=eps)
    return torch.mean(torch.abs((y_true - y_pred) / denom))
