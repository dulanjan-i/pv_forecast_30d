"""
tests/test_rl_checkpoint.py

Unit tests for RL checkpoint loading and Q-network reconstruction.

These tests are data-free — they build synthetic state dicts and verify
the loading/inference logic without requiring any trained checkpoint files.

For integration tests against real checkpoints, see the `--run-integration`
marker (skipped by default in CI).
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers mirrored from src/rl/test_checkpoint.py (now tested properly)
# ---------------------------------------------------------------------------

def _extract_qnet_state_dict(ckpt: dict) -> dict:
    """Extract Q-network state dict from a checkpoint dict."""
    for k in ["q_net", "policy_net", "state_dict", "model_state_dict", "net"]:
        if k in ckpt and isinstance(ckpt[k], dict):
            return ckpt[k]
    # If checkpoint IS the state dict (all tensors)
    if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return ckpt
    raise KeyError(f"No q-network state dict found. Keys: {list(ckpt.keys())}")


def _infer_dims_from_state_dict(sd: dict) -> Tuple[int, int]:
    """Infer (state_dim, action_dim) from a Linear MLP state dict."""
    weight_keys = sorted([k for k in sd.keys() if k.endswith(".weight")])
    if not weight_keys:
        raise ValueError("No .weight tensors in state_dict, cannot infer dims.")
    first_w = sd[weight_keys[0]]
    last_w = sd[weight_keys[-1]]
    return int(first_w.shape[1]), int(last_w.shape[0])


def _build_mlp(state_dim: int, hidden_sizes: list[int], action_dim: int) -> nn.Sequential:
    """Build a simple MLP matching the MiRACLE DQN structure."""
    layers = []
    in_dim = state_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(in_dim, h))
        layers.append(nn.ReLU())
        in_dim = h
    layers.append(nn.Linear(in_dim, action_dim))
    return nn.Sequential(*layers)


def _make_synthetic_checkpoint(
    state_dim: int = 35,
    hidden_sizes: list[int] = [64, 64],
    action_dim: int = 8,
    wrap_key: str | None = "q_net",
) -> Tuple[dict, nn.Sequential]:
    """Create a synthetic checkpoint (as saved by MiRACLE training scripts)."""
    net = _build_mlp(state_dim, hidden_sizes, action_dim)
    sd = net.state_dict()
    if wrap_key:
        ckpt = {
            wrap_key: sd,
            "total_steps": 10000,
            "eps": 0.05,
        }
    else:
        ckpt = dict(sd)  # raw state dict as checkpoint
    return ckpt, net


# ---------------------------------------------------------------------------
# Tests: _extract_qnet_state_dict
# ---------------------------------------------------------------------------

class TestExtractQnetStateDict:

    def test_extracts_q_net_key(self):
        net = _build_mlp(35, [64], 8)
        ckpt = {"q_net": net.state_dict(), "eps": 0.1}
        sd = _extract_qnet_state_dict(ckpt)
        assert "0.weight" in sd or any(".weight" in k for k in sd)

    def test_extracts_policy_net_key(self):
        net = _build_mlp(35, [64], 8)
        ckpt = {"policy_net": net.state_dict()}
        sd = _extract_qnet_state_dict(ckpt)
        assert isinstance(sd, dict)

    def test_falls_back_to_raw_state_dict(self):
        net = _build_mlp(35, [64], 8)
        # Checkpoint IS the state dict (no wrapper key)
        raw_sd = {k: v for k, v in net.state_dict().items()}
        sd = _extract_qnet_state_dict(raw_sd)
        assert sd is raw_sd

    def test_raises_on_unknown_format(self):
        bad_ckpt = {"optimizer": {"some": "dict"}, "config": {"lr": 1e-4}}
        with pytest.raises(KeyError, match="No q-network state dict found"):
            _extract_qnet_state_dict(bad_ckpt)

    @pytest.mark.parametrize("key", ["q_net", "policy_net", "state_dict", "model_state_dict"])
    def test_all_supported_keys(self, key):
        net = _build_mlp(10, [32], 4)
        ckpt = {key: net.state_dict()}
        sd = _extract_qnet_state_dict(ckpt)
        assert isinstance(sd, dict)
        assert len(sd) > 0


# ---------------------------------------------------------------------------
# Tests: _infer_dims_from_state_dict
# ---------------------------------------------------------------------------

class TestInferDimsFromStateDict:

    @pytest.mark.parametrize("state_dim,hidden,action_dim", [
        (35, [64, 64], 8),    # MiRACLE v1 canonical config
        (10, [32], 4),        # Small test network
        (128, [256, 128, 64], 16),  # Larger network
    ])
    def test_infers_correct_dims(self, state_dim, hidden, action_dim):
        net = _build_mlp(state_dim, hidden, action_dim)
        sd = net.state_dict()
        inferred_state, inferred_action = _infer_dims_from_state_dict(sd)
        assert inferred_state == state_dim, f"Expected state_dim={state_dim}, got {inferred_state}"
        assert inferred_action == action_dim, f"Expected action_dim={action_dim}, got {inferred_action}"

    def test_raises_on_empty_state_dict(self):
        with pytest.raises(ValueError, match="No .weight tensors"):
            _infer_dims_from_state_dict({})

    def test_raises_on_no_weight_keys(self):
        sd = {"some_param": torch.tensor([1.0, 2.0])}
        with pytest.raises(ValueError, match="No .weight tensors"):
            _infer_dims_from_state_dict(sd)


# ---------------------------------------------------------------------------
# Tests: checkpoint round-trip (save → load → forward)
# ---------------------------------------------------------------------------

class TestCheckpointRoundTrip:

    @pytest.mark.parametrize("wrap_key", ["q_net", "policy_net", None])
    def test_roundtrip_via_tempfile(self, wrap_key, tmp_path):
        """Save a checkpoint to disk, reload it, extract state dict, run forward."""
        state_dim, action_dim = 35, 8
        ckpt, original_net = _make_synthetic_checkpoint(
            state_dim=state_dim,
            hidden_sizes=[64, 64],
            action_dim=action_dim,
            wrap_key=wrap_key,
        )
        ckpt_path = tmp_path / "test_ckpt.pt"
        torch.save(ckpt, ckpt_path)

        # Reload
        loaded_ckpt = torch.load(ckpt_path, map_location="cpu")
        sd = _extract_qnet_state_dict(loaded_ckpt)

        # Reconstruct network from state dict alone
        inferred_state_dim, inferred_action_dim = _infer_dims_from_state_dict(sd)
        assert inferred_state_dim == state_dim
        assert inferred_action_dim == action_dim

        # Verify all original weights are recovered exactly
        original_sd = original_net.state_dict()
        for k in original_sd:
            assert k in sd, f"Key {k} missing after reload"
            assert torch.allclose(original_sd[k], sd[k]), f"Weight mismatch for {k}"

    def test_forward_pass_after_load(self, tmp_path):
        """Loaded checkpoint must produce finite Q-values on a random state."""
        state_dim, action_dim = 35, 8
        ckpt, _ = _make_synthetic_checkpoint(state_dim=state_dim, action_dim=action_dim)
        ckpt_path = tmp_path / "ckpt.pt"
        torch.save(ckpt, ckpt_path)

        loaded = torch.load(ckpt_path, map_location="cpu")
        sd = _extract_qnet_state_dict(loaded)
        inferred_state, inferred_action = _infer_dims_from_state_dict(sd)

        # Rebuild MLP from inferred dims
        weight_keys = sorted([k for k in sd if k.endswith(".weight")])
        shapes = [sd[k].shape for k in weight_keys]
        hidden = [int(s[0]) for s in shapes[:-1]]
        net = _build_mlp(inferred_state, hidden, inferred_action)
        net.load_state_dict(sd, strict=False)
        net.eval()

        x = torch.randn(1, state_dim)
        with torch.no_grad():
            q_values = net(x)

        assert q_values.shape == (1, action_dim)
        assert torch.isfinite(q_values).all(), "Q-values contain NaN/Inf"

    def test_argmax_action_is_valid(self, tmp_path):
        """The greedy action from a loaded Q-net must be in [0, action_dim)."""
        state_dim, action_dim = 35, 8
        ckpt, _ = _make_synthetic_checkpoint(state_dim=state_dim, action_dim=action_dim)
        ckpt_path = tmp_path / "ckpt.pt"
        torch.save(ckpt, ckpt_path)

        loaded = torch.load(ckpt_path, map_location="cpu")
        sd = _extract_qnet_state_dict(loaded)
        weight_keys = sorted([k for k in sd if k.endswith(".weight")])
        shapes = [sd[k].shape for k in weight_keys]
        hidden = [int(s[0]) for s in shapes[:-1]]
        net = _build_mlp(state_dim, hidden, action_dim)
        net.load_state_dict(sd, strict=False)
        net.eval()

        x = torch.randn(4, state_dim)  # batch of 4 states
        with torch.no_grad():
            q_values = net(x)
        actions = torch.argmax(q_values, dim=1).numpy()

        assert all(0 <= a < action_dim for a in actions), f"Out-of-range actions: {actions}"


# ---------------------------------------------------------------------------
# Tests: blend weight mechanics (from test_direct_blend.py logic)
# ---------------------------------------------------------------------------

class TestBlendWeights:
    """
    Verify that different blend weight presets produce different ensemble outputs.
    These are pure numpy tests — no model loading required.
    """

    @pytest.fixture
    def mock_component_forecasts(self) -> Dict[str, np.ndarray]:
        """Synthetic Day-1 forecasts from three component models."""
        rng = np.random.default_rng(42)
        n = 96  # Day 1: 96 × 15-min steps
        return {
            "short":   rng.uniform(0.1, 0.9, size=n).astype(np.float32),
            "long":    rng.uniform(0.0, 0.8, size=n).astype(np.float32),
            "physics": rng.uniform(0.0, 0.6, size=n).astype(np.float32),
        }

    def _blend(self, components: Dict[str, np.ndarray], weights: Dict[str, float]) -> np.ndarray:
        return (
            weights["short"] * components["short"]
            + weights["long"] * components["long"]
            + weights["physics"] * components["physics"]
        )

    BLEND_PRESETS = {
        "MAINTAIN":         {"short": 0.4875, "long": 0.2625, "physics": 0.25},
        "BLEND_HIGH_SHORT": {"short": 0.70,   "long": 0.20,   "physics": 0.10},
        "BLEND_HIGH_LONG":  {"short": 0.20,   "long": 0.70,   "physics": 0.10},
        "BLEND_HIGH_PHYSICS":{"short": 0.20,  "long": 0.20,   "physics": 0.60},
    }

    def test_weights_sum_to_one(self):
        for name, w in self.BLEND_PRESETS.items():
            total = w["short"] + w["long"] + w["physics"]
            assert abs(total - 1.0) < 1e-6, f"{name} weights sum to {total}, expected 1.0"

    def test_different_blends_give_different_predictions(self, mock_component_forecasts):
        results = {}
        for name, weights in self.BLEND_PRESETS.items():
            results[name] = self._blend(mock_component_forecasts, weights)

        # All blend outputs must be mutually different (not identical)
        names = list(results.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = results[names[i]], results[names[j]]
                assert not np.allclose(a, b), (
                    f"Blend '{names[i]}' and '{names[j]}' gave identical outputs — "
                    "blend weights are not affecting predictions"
                )

    def test_rmse_varies_across_blends(self, mock_component_forecasts):
        """RMSE must vary across blend strategies against a synthetic ground truth."""
        rng = np.random.default_rng(99)
        ground_truth = rng.uniform(0.1, 0.8, size=96).astype(np.float32)

        rmse_values = {}
        for name, weights in self.BLEND_PRESETS.items():
            blended = self._blend(mock_component_forecasts, weights)
            rmse_values[name] = float(np.sqrt(np.mean((blended - ground_truth) ** 2)))

        rmse_range = max(rmse_values.values()) - min(rmse_values.values())
        assert rmse_range > 1e-6, (
            f"RMSE is identical across all blends (range={rmse_range:.2e}). "
            "Blend weights are having no effect."
        )

    def test_extreme_blend_extreme_output(self, mock_component_forecasts):
        """A 100% physics blend must equal the physics forecast exactly."""
        physics_only = {"short": 0.0, "long": 0.0, "physics": 1.0}
        blended = self._blend(mock_component_forecasts, physics_only)
        np.testing.assert_array_almost_equal(
            blended, mock_component_forecasts["physics"],
            decimal=6,
            err_msg="100% physics blend did not equal physics forecast",
        )

    def test_blend_output_is_weighted_average(self, mock_component_forecasts):
        """Verify the blend formula is a true convex combination."""
        w = {"short": 0.5, "long": 0.3, "physics": 0.2}
        blended = self._blend(mock_component_forecasts, w)
        expected = (
            0.5 * mock_component_forecasts["short"]
            + 0.3 * mock_component_forecasts["long"]
            + 0.2 * mock_component_forecasts["physics"]
        )
        np.testing.assert_array_almost_equal(blended, expected, decimal=6)


# ---------------------------------------------------------------------------
# Tests: RL agent API (data-free, pure logic)
# ---------------------------------------------------------------------------

class TestRLMetaControllerSystem:
    """
    Smoke tests for the RL meta-controller system.
    No checkpoints or data files required.
    """

    @pytest.fixture
    def rl_system(self):
        from src.rl.rl_meta_controller import RLMetaControllerSystem, RLConfig
        config = RLConfig(mode="heuristic")
        return RLMetaControllerSystem(config=config)

    def test_initialization(self, rl_system):
        assert rl_system is not None
        assert rl_system.total_state_dim == 35
        assert rl_system.meta_controller.action_dim == 8

    def test_build_meta_state_shape(self, rl_system):
        metrics = {}  # All defaults → zeros
        state = rl_system.build_meta_state(metrics)
        assert state.shape == (35,), f"Expected (35,), got {state.shape}"
        assert np.isfinite(state).all(), "State contains NaN/Inf"

    def test_heuristic_action_is_valid(self, rl_system):
        metrics = {}
        state = rl_system.build_meta_state(metrics)
        action = rl_system.meta_controller.select_action(state, mode="heuristic")
        assert 0 <= action < 8, f"Action {action} out of range [0, 8)"

    def test_step_returns_expected_keys(self, rl_system):
        result = rl_system.step({})
        assert "action" in result
        assert "action_name" in result
        assert "blend_weights" in result
        assert "advisor_alerts" in result

    def test_all_8_actions_named(self, rl_system):
        expected_actions = [
            "MAINTAIN", "FINE_TUNE_SHORT_TFT", "FINE_TUNE_LONG_TFT",
            "RECALIBRATE_PVLIB", "BLEND_HIGH_SHORT", "BLEND_HIGH_LONG",
            "BLEND_HIGH_PHYSICS", "SUGGEST_RETRAIN",
        ]
        for i, name in enumerate(expected_actions):
            assert rl_system.meta_controller.get_action_name(i) == name

    def test_blend_weights_change_on_blend_actions(self, rl_system):
        """Actions A4-A6 must change the current blend weights."""
        initial_weights = rl_system.meta_controller.current_weights.copy()
        rl_system.meta_controller.execute_action(4)  # BLEND_HIGH_SHORT
        assert rl_system.meta_controller.current_weights != initial_weights

    def test_status_dict_has_expected_structure(self, rl_system):
        status = rl_system.get_status()
        assert "mode" in status
        assert "meta_controller" in status
        assert "advisors" in status
        assert set(status["advisors"].keys()) == {"short_tft", "long_tft", "pvlib"}

    def test_reward_is_finite(self, rl_system):
        metrics_a = {"ensemble_rmse": 0.05, "data_drift_score": 0.1}
        metrics_b = {"ensemble_rmse": 0.04, "data_drift_score": 0.08}
        # Trigger step so current_action is set
        rl_system.step(metrics_a)
        reward = rl_system.compute_reward(metrics_a, metrics_b)
        assert np.isfinite(reward), f"Reward is not finite: {reward}"
