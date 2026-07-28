"""
tests/test_utils.py — Unit tests for src/utils/metrics.py

All tests are data-free (synthetic tensors only) and run without GPU or real data.
"""
from __future__ import annotations

import math
import pytest
import torch


from src.utils.metrics import rmse, mae, mape


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def perfect():
    """Identical prediction — all metrics should be 0."""
    t = torch.tensor([1.0, 2.0, 3.0, 4.0])
    return t, t.clone()


@pytest.fixture
def simple():
    """y_true=[0,1,2,3], y_pred=[1,2,3,4] — constant offset of 1."""
    y_true = torch.tensor([0.0, 1.0, 2.0, 3.0])
    y_pred = torch.tensor([1.0, 2.0, 3.0, 4.0])
    return y_true, y_pred


# ---------------------------------------------------------------------------
# RMSE
# ---------------------------------------------------------------------------

class TestRMSE:
    def test_perfect_prediction_is_zero(self, perfect):
        y_true, y_pred = perfect
        assert rmse(y_true, y_pred).item() == pytest.approx(0.0, abs=1e-6)

    def test_constant_offset(self, simple):
        """RMSE of a constant offset of 1.0 should be 1.0."""
        y_true, y_pred = simple
        assert rmse(y_true, y_pred).item() == pytest.approx(1.0, abs=1e-6)

    def test_known_value(self):
        """RMSE([0,0], [3,4]) = sqrt((9+16)/2) = sqrt(12.5)."""
        y_true = torch.tensor([0.0, 0.0])
        y_pred = torch.tensor([3.0, 4.0])
        expected = math.sqrt(12.5)
        assert rmse(y_true, y_pred).item() == pytest.approx(expected, rel=1e-5)

    def test_non_negative(self):
        y_true = torch.randn(100)
        y_pred = torch.randn(100)
        assert rmse(y_true, y_pred).item() >= 0.0

    def test_symmetric(self):
        """RMSE(a, b) == RMSE(b, a)."""
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([4.0, 5.0, 6.0])
        assert rmse(a, b).item() == pytest.approx(rmse(b, a).item(), rel=1e-6)

    def test_scalar_tensors(self):
        assert rmse(torch.tensor(3.0), torch.tensor(0.0)).item() == pytest.approx(3.0)

    def test_single_element(self):
        assert rmse(torch.tensor([5.0]), torch.tensor([2.0])).item() == pytest.approx(3.0)

    def test_large_batch(self):
        """Smoke test — no crash on large input."""
        y = torch.zeros(10_000)
        y_hat = torch.ones(10_000)
        assert rmse(y, y_hat).item() == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# MAE
# ---------------------------------------------------------------------------

class TestMAE:
    def test_perfect_prediction_is_zero(self, perfect):
        y_true, y_pred = perfect
        assert mae(y_true, y_pred).item() == pytest.approx(0.0, abs=1e-6)

    def test_constant_offset(self, simple):
        """MAE of constant offset 1.0 should be 1.0."""
        y_true, y_pred = simple
        assert mae(y_true, y_pred).item() == pytest.approx(1.0, abs=1e-6)

    def test_known_value(self):
        """MAE([0,2,4], [1,1,1]) = mean(1,1,3) = 5/3."""
        y_true = torch.tensor([0.0, 2.0, 4.0])
        y_pred = torch.tensor([1.0, 1.0, 1.0])
        assert mae(y_true, y_pred).item() == pytest.approx(5.0 / 3.0, rel=1e-5)

    def test_non_negative(self):
        y_true = torch.randn(100)
        y_pred = torch.randn(100)
        assert mae(y_true, y_pred).item() >= 0.0

    def test_symmetric(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([4.0, 5.0, 6.0])
        assert mae(a, b).item() == pytest.approx(mae(b, a).item(), rel=1e-6)

    def test_mae_leq_rmse_not_always(self):
        """MAE <= RMSE always holds (Cauchy-Schwarz)."""
        y_true = torch.randn(50)
        y_pred = torch.randn(50)
        assert mae(y_true, y_pred).item() <= rmse(y_true, y_pred).item() + 1e-5


# ---------------------------------------------------------------------------
# MAPE
# ---------------------------------------------------------------------------

class TestMAPE:
    def test_perfect_prediction_is_zero(self, perfect):
        y_true, y_pred = perfect
        assert mape(y_true, y_pred).item() == pytest.approx(0.0, abs=1e-5)

    def test_known_value(self):
        """MAPE([1,2], [2,4]) = mean(1/1, 2/2) = 1.0 (100%)."""
        y_true = torch.tensor([1.0, 2.0])
        y_pred = torch.tensor([2.0, 4.0])
        assert mape(y_true, y_pred).item() == pytest.approx(1.0, rel=1e-5)

    def test_zero_true_values_no_crash(self):
        """Near-zero y_true should not produce inf/nan — eps guards division."""
        y_true = torch.tensor([0.0, 0.0, 0.0])
        y_pred = torch.tensor([1.0, 2.0, 3.0])
        result = mape(y_true, y_pred)
        assert not torch.isnan(result)
        assert not torch.isinf(result)

    def test_non_negative(self):
        y_true = torch.rand(50) + 0.1   # avoid near-zero
        y_pred = torch.rand(50) + 0.1
        assert mape(y_true, y_pred).item() >= 0.0

    def test_eps_parameter(self):
        """Custom eps should behave consistently — larger eps = more regularisation."""
        y_true = torch.tensor([0.0])
        y_pred = torch.tensor([1.0])
        r1 = mape(y_true, y_pred, eps=1e-6).item()
        r2 = mape(y_true, y_pred, eps=1.0).item()
        # With larger eps denominator is larger → smaller MAPE
        assert r2 <= r1
