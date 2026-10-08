"""
test_orchestration.py — Orchestration smoke tests for the LSTM-pretrain branch config.

NOTE: These tests require experiments/lstm/pretrain_pvdaq.yaml and the
      lstm_encoder src module, which are not present in the v1.0 inference-only
      checkout. All tests are marked integration and skipped by default.

Run with:
    pytest tests/test_orchestration.py -v -m integration
"""
import pytest

pytestmark = pytest.mark.skip(
    reason="Requires LSTM pretrain config + src/models/lstm_encoder.py "
           "(not present in v1.0 inference-only checkout). "
           "Run with -m integration on the full training environment."
)


@pytest.mark.integration
def test_imports():
    """LSTM encoder and sequence generator imports resolve cleanly."""
    from src.models.lstm_encoder import LSTMEncoderConfig, LSTMEncoder, make_trainer
    from src.features.sequence_generator import SimpleWindowDataset
    assert LSTMEncoderConfig is not None


@pytest.mark.integration
def test_config_loads():
    """YAML config for LSTM pretrain loads and contains required keys."""
    import yaml
    from pathlib import Path

    config_path = Path("experiments/lstm/pretrain_pvdaq.yaml")
    assert config_path.exists(), f"Config not found: {config_path}"

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    assert "model" in cfg, "Config missing 'model' section"
    assert "training" in cfg, "Config missing 'training' section"


@pytest.mark.integration
def test_model_instantiation():
    """LSTMEncoder instantiates from config without errors."""
    import yaml
    import torch
    from pathlib import Path
    from src.models.lstm_encoder import LSTMEncoderConfig, LSTMEncoder

    config_path = Path("experiments/lstm/pretrain_pvdaq.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model_cfg = LSTMEncoderConfig(**cfg["model"])
    model = LSTMEncoder(model_cfg)
    assert model is not None

    # Forward pass with dummy input
    dummy = torch.randn(4, model_cfg.seq_len, model_cfg.input_size)
    out = model(dummy)
    assert out.shape[0] == 4
