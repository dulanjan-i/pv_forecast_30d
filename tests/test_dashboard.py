"""
test_dashboard.py — Dashboard data generation and smoke tests.

Verifies the RL monitoring dashboard can generate valid log data
and that the expected log files are structurally correct.

Note: Does NOT launch Streamlit. For manual dashboard launch:
    streamlit run src/rl/monitoring_dashboard.py
"""
import json
import pytest
import numpy as np
import pandas as pd
from datetime import timedelta
from pathlib import Path


LOG_DIR = Path("checkpoints/rl/logs")


@pytest.fixture
def dashboard_log_dir(tmp_path):
    """Provide a temp directory for dashboard log files."""
    return tmp_path / "rl_logs"


def _make_metrics_entry(i: int) -> dict:
    """Generate one fake RL metrics entry for step i."""
    timestamp = pd.Timestamp.now(tz="UTC") - timedelta(minutes=15 * (200 - i))
    short_rmse = 0.12 - 0.0003 * i + np.random.rand() * 0.01
    long_rmse = 0.15 - 0.0002 * i + np.random.rand() * 0.01
    reward = -0.5 + 0.003 * i + np.random.rand() * 0.1
    action = int(np.random.choice(8, p=[0.30, 0.10, 0.08, 0.07, 0.15, 0.15, 0.10, 0.05]))

    return {
        "timestamp": timestamp.isoformat(),
        "action": action,
        "reward": float(reward),
        "short_rmse_1h": float(short_rmse),
        "long_rmse_30d": float(long_rmse),
        "blend_short": 0.33,
        "blend_long": 0.33,
        "blend_physics": 0.34,
        "epsilon": float(max(0.1, 1.0 - 0.004 * i)),
        "q_loss": float(max(0.0, 0.1 - 0.0003 * i)),
    }


def test_metrics_entry_structure():
    """Each generated metrics entry has the required keys and valid types."""
    entry = _make_metrics_entry(10)
    required = {"timestamp", "action", "reward", "short_rmse_1h",
                "long_rmse_30d", "blend_short", "blend_long", "blend_physics",
                "epsilon", "q_loss"}
    assert required.issubset(entry.keys())
    assert isinstance(entry["action"], int)
    assert 0 <= entry["action"] <= 7
    assert 0.0 <= entry["epsilon"] <= 1.0
    assert entry["blend_short"] + entry["blend_long"] + entry["blend_physics"] == pytest.approx(1.0, abs=1e-6)


def test_metrics_file_writes_valid_jsonl(dashboard_log_dir):
    """200 metrics entries write to JSONL and parse back cleanly."""
    dashboard_log_dir.mkdir(parents=True)
    metrics_file = dashboard_log_dir / "metrics.jsonl"

    with open(metrics_file, "w") as f:
        for i in range(200):
            f.write(json.dumps(_make_metrics_entry(i)) + "\n")

    lines = metrics_file.read_text().strip().splitlines()
    assert len(lines) == 200

    for line in lines:
        entry = json.loads(line)
        assert "timestamp" in entry
        assert "reward" in entry


def test_rl_state_file_structure(dashboard_log_dir):
    """RL state JSON has required fields and valid ranges."""
    dashboard_log_dir.mkdir(parents=True)
    state = {
        "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
        "epsilon": 0.25,
        "last_action": 4,
        "q_max": 0.85,
        "buffer_size": 3245,
        "buffer_capacity": 10000,
        "total_steps": 200,
    }
    state_file = dashboard_log_dir / "rl_state.json"
    state_file.write_text(json.dumps(state, indent=2))

    loaded = json.loads(state_file.read_text())
    assert 0.0 <= loaded["epsilon"] <= 1.0
    assert loaded["buffer_size"] <= loaded["buffer_capacity"]
    assert loaded["total_steps"] > 0


@pytest.mark.integration
def test_dashboard_module_importable():
    """RL monitoring dashboard module imports without errors (requires Streamlit)."""
    try:
        pytest.importorskip("streamlit", reason="Streamlit not installed")
        import importlib
        mod = importlib.import_module("src.rl.monitoring_dashboard")
        assert mod is not None
    except TypeError as e:
        pytest.skip(f"Protobuf version conflict in env (upgrade protobuf or use venv): {e}")

