"""
tests/test_reward.py — Unit tests for src/rl/reward.py

All tests are data-free (synthetic numpy arrays). No GPU, no real data, no file I/O.
"""
from __future__ import annotations

import pytest
import numpy as np

from src.rl.reward import (
    ACTION_COSTS,
    RewardWeightsV1,
    RewardWeightsV2,
    compute_reward_v1,
    compute_reward_v2,
    _rmse,
    _tv,
)


# ---------------------------------------------------------------------------
# ACTION_COSTS
# ---------------------------------------------------------------------------

class TestActionCosts:
    def test_all_eight_actions_present(self):
        assert set(ACTION_COSTS.keys()) == set(range(8))

    def test_maintain_has_zero_cost(self):
        assert ACTION_COSTS[0] == 0.0

    def test_suggest_retrain_is_most_expensive(self):
        assert ACTION_COSTS[7] == max(ACTION_COSTS.values())

    def test_all_costs_non_negative(self):
        assert all(v >= 0.0 for v in ACTION_COSTS.values())


# ---------------------------------------------------------------------------
# RewardWeights dataclasses
# ---------------------------------------------------------------------------

class TestRewardWeights:
    def test_v1_defaults_are_frozen(self):
        w = RewardWeightsV1()
        with pytest.raises((AttributeError, TypeError)):
            w.w_short = 99.0  # type: ignore[misc]

    def test_v2_defaults_are_frozen(self):
        w = RewardWeightsV2()
        with pytest.raises((AttributeError, TypeError)):
            w.w_acc = 99.0  # type: ignore[misc]

    def test_v1_custom_weights(self):
        w = RewardWeightsV1(w_short=2.0, w_long=1.0)
        assert w.w_short == 2.0
        assert w.w_long == 1.0

    def test_v2_scale_denom_positive(self):
        w = RewardWeightsV2()
        assert w.scale_denom > 0.0


# ---------------------------------------------------------------------------
# Helper functions _rmse and _tv
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_rmse_perfect(self):
        a = np.array([1.0, 2.0, 3.0])
        assert _rmse(a, a) == pytest.approx(0.0, abs=1e-9)

    def test_rmse_known(self):
        a = np.array([0.0, 0.0])
        b = np.array([3.0, 4.0])
        assert _rmse(a, b) == pytest.approx(np.sqrt(12.5), rel=1e-6)

    def test_rmse_empty_returns_zero(self):
        assert _rmse(np.array([]), np.array([])) == 0.0

    def test_tv_constant_is_zero(self):
        assert _tv(np.ones(10)) == pytest.approx(0.0, abs=1e-9)

    def test_tv_unit_step(self):
        """TV of [0,1,1,1] = mean([1,0,0]) = 1/3."""
        a = np.array([0.0, 1.0, 1.0, 1.0])
        assert _tv(a) == pytest.approx(1.0 / 3.0, rel=1e-6)

    def test_tv_single_element_returns_zero(self):
        assert _tv(np.array([5.0])) == 0.0

    def test_tv_empty_returns_zero(self):
        assert _tv(np.array([])) == 0.0


# ---------------------------------------------------------------------------
# compute_reward_v1
# ---------------------------------------------------------------------------

class TestComputeRewardV1:
    def _state(self, short, long, phys=0.0):
        return np.array([short, long, phys])

    def test_improvement_gives_positive_reward(self):
        """RMSE goes down → positive reward."""
        s = self._state(0.10, 0.15)
        ns = self._state(0.08, 0.12)
        r = compute_reward_v1(s, ns, action=0)
        assert r > 0.0

    def test_worsening_gives_negative_reward(self):
        s = self._state(0.05, 0.05)
        ns = self._state(0.10, 0.10)
        r = compute_reward_v1(s, ns, action=0)
        assert r < 0.0

    def test_no_change_maintain_is_zero(self):
        s = self._state(0.10, 0.10, phys=0.0)
        r = compute_reward_v1(s, s, action=0)
        assert r == pytest.approx(0.0, abs=1e-9)

    def test_expensive_action_reduces_reward(self):
        """Same transition, action=7 (SUGGEST_RETRAIN, cost=1.0) < action=0 (cost=0.0)."""
        s = self._state(0.10, 0.10)
        ns = self._state(0.08, 0.08)
        r_cheap = compute_reward_v1(s, ns, action=0)
        r_expensive = compute_reward_v1(s, ns, action=7)
        assert r_expensive < r_cheap

    def test_physics_residual_penalises_reward(self):
        """Higher physics residual → lower reward."""
        s = self._state(0.10, 0.10)
        ns_low = self._state(0.08, 0.08, phys=0.01)
        ns_high = self._state(0.08, 0.08, phys=0.50)
        assert compute_reward_v1(s, ns_low, action=0) > compute_reward_v1(s, ns_high, action=0)

    def test_state_missing_phys_defaults_zero(self):
        """2-element state should not crash — phys defaults to 0."""
        s = np.array([0.10, 0.10])
        ns = np.array([0.08, 0.08])
        r = compute_reward_v1(s, ns, action=0)
        assert isinstance(r, float)

    def test_custom_weights_respected(self):
        w = RewardWeightsV1(w_short=10.0, w_long=0.0, w_phys=0.0, w_cost=0.0, scale_denom=1.0)
        s = self._state(0.10, 0.10)
        ns = self._state(0.09, 0.10)  # only short improves by 0.01
        r = compute_reward_v1(s, ns, action=0, weights=w)
        assert r == pytest.approx(10.0 * 0.01, rel=1e-5)

    def test_returns_float(self):
        s = self._state(0.10, 0.10)
        ns = self._state(0.09, 0.09)
        assert isinstance(compute_reward_v1(s, ns, action=0), float)


# ---------------------------------------------------------------------------
# compute_reward_v2
# ---------------------------------------------------------------------------

class TestComputeRewardV2:
    def _make_inputs(self, n=48, improvement=True):
        rng = np.random.default_rng(42)
        y = np.abs(rng.normal(0.5, 0.1, n))
        y_hat = y + rng.normal(0.0, 0.02, n)
        state = np.array([0.10, 0.10, 0.0])
        next_state = np.array([0.08 if improvement else 0.12, 0.10, 0.0])
        return state, next_state, y, y_hat

    def test_improvement_gives_higher_reward_than_worsening(self):
        s, ns_good, y, yh = self._make_inputs(improvement=True)
        _, ns_bad, _, _ = self._make_inputs(improvement=False)
        r_good = compute_reward_v2(s, ns_good, action=0, y=y, y_hat=yh)
        r_bad = compute_reward_v2(s, ns_bad, action=0, y=y, y_hat=yh)
        assert r_good > r_bad

    def test_over_smoothed_prediction_penalised(self):
        """A flat (over-smooth) y_hat should score lower than a noisy one."""
        rng = np.random.default_rng(0)
        y = np.abs(rng.normal(0.5, 0.2, 48))
        y_flat = np.full(48, y.mean())         # over-smooth
        y_noisy = y + rng.normal(0, 0.05, 48)  # preserves variability

        s = np.array([0.10, 0.10, 0.0])
        ns = np.array([0.08, 0.10, 0.0])
        r_flat = compute_reward_v2(s, ns, action=0, y=y, y_hat=y_flat)
        r_noisy = compute_reward_v2(s, ns, action=0, y=y, y_hat=y_noisy)
        assert r_flat < r_noisy

    def test_daylight_mask_shape_mismatch_raises(self):
        s = np.array([0.10, 0.10, 0.0])
        ns = np.array([0.08, 0.10, 0.0])
        y = np.ones(48)
        yh = np.ones(48)
        bad_mask = np.ones(10, dtype=bool)  # wrong size
        with pytest.raises(ValueError, match="is_daylight shape"):
            compute_reward_v2(s, ns, action=0, y=y, y_hat=yh, is_daylight=bad_mask)

    def test_all_night_mask_no_crash(self):
        """All-False daylight mask — daylight arrays are empty, should not crash."""
        s = np.array([0.10, 0.10, 0.0])
        ns = np.array([0.08, 0.10, 0.0])
        y = np.ones(48)
        yh = np.ones(48)
        mask = np.zeros(48, dtype=bool)
        r = compute_reward_v2(s, ns, action=0, y=y, y_hat=yh, is_daylight=mask)
        assert isinstance(r, float)

    def test_expensive_action_penalised_more_than_cheap(self):
        s, ns, y, yh = self._make_inputs()
        r0 = compute_reward_v2(s, ns, action=0, y=y, y_hat=yh)
        r7 = compute_reward_v2(s, ns, action=7, y=y, y_hat=yh)
        assert r7 < r0

    def test_returns_float(self):
        s, ns, y, yh = self._make_inputs()
        assert isinstance(compute_reward_v2(s, ns, action=0, y=y, y_hat=yh), float)
