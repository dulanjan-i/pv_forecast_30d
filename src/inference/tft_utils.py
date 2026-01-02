"""
TFT Utility Functions for Hierarchical 30-Day Forecasting.

Shared utilities for loading TFT models, creating datasets, and extracting predictions.
Used by PhysicsAwareForecaster for real-time inference.

Based on offline_predict_tft.py patterns but adapted for production use.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data.encoders import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss


def load_tft_config(run_dir: Path) -> Dict[str, Any]:
    """
    Load TFT configuration from run directory.
    
    Args:
        run_dir: Path to run directory containing run_config.json and column_roles.json
    
    Returns:
        config: Dictionary with keys:
            - 'run_config': Full run configuration
            - 'cli_args': CLI arguments (hyperparameters)
            - 'roles': Column roles (normalized)
            - 'encoder_len': Encoder length
            - 'pred_len': Prediction length
            - 'hidden_size': Hidden layer size
            - 'lstm_layers': Number of LSTM layers
            - 'attention_head_size': Attention heads
            - 'dropout': Dropout rate
            - 'quantiles': Prediction quantiles
    
    Raises:
        FileNotFoundError: If config files don't exist
    """
    run_dir = Path(run_dir)
    
    # Load run_config.json
    run_config_path = run_dir / "run_config.json"
    if not run_config_path.exists():
        raise FileNotFoundError(f"run_config.json not found in {run_dir}")
    
    with open(run_config_path, 'r') as f:
        run_cfg = json.load(f)
    
    # Extract CLI args (support both nested and flat structures)
    cfg = run_cfg.get("cfg", run_cfg)
    if "cli_args" in cfg:
        cli_args = cfg["cli_args"]
    else:
        cli_args = cfg
    
    # Load column_roles.json
    roles_path = run_dir / "column_roles.json"
    if not roles_path.exists():
        raise FileNotFoundError(f"column_roles.json not found in {run_dir}")
    
    with open(roles_path, 'r') as f:
        roles_raw = json.load(f)
    
    # Normalize roles (handle both schema styles)
    roles = _infer_roles(roles_raw)
    
    # Extract key hyperparameters with fallbacks
    encoder_len = int(cli_args.get("max_encoder_length", cli_args.get("encoder_len", 96)))
    pred_len = int(cli_args.get("max_prediction_length", cli_args.get("pred_len", 96)))
    hidden_size = int(cli_args.get("hidden_size", 64))
    lstm_layers = int(cli_args.get("lstm_layers", 2))
    attention_head_size = int(cli_args.get("attention_head_size", cli_args.get("attn_heads", 4)))
    dropout = float(cli_args.get("dropout", 0.1))
    quantiles = cli_args.get("quantiles", [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98])
    
    return {
        'run_config': run_cfg,
        'cli_args': cli_args,
        'roles': roles,
        'encoder_len': encoder_len,
        'pred_len': pred_len,
        'hidden_size': hidden_size,
        'lstm_layers': lstm_layers,
        'attention_head_size': attention_head_size,
        'dropout': dropout,
        'quantiles': quantiles,
    }


def _infer_roles(roles: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize column roles into standard PyTorch Forecasting format.
    Handles both MiRACLE v1 schema and standard PF schema.
    
    Args:
        roles: Raw roles dictionary from column_roles.json
    
    Returns:
        Normalized roles dictionary
    """
    # MiRACLE v1 schema
    if "known_time_reals" in roles and "time_idx_col" in roles:
        target = roles["target"]
        time_col = roles.get("time_col", "timestamp_utc")
        time_idx_col = roles.get("time_idx_col", "time_idx")
        group_ids = roles.get("group_ids", ["plant_id"])
        known_reals = roles.get("known_time_reals", [])
        unknown_reals = roles.get("unknown_time_reals", [target])
        lagged = roles.get("lagged_encoding_cols", [])
        return {
            "target": target,
            "time_col": time_col,
            "time_idx_col": time_idx_col,
            "group_ids": group_ids,
            "static_categoricals": [],
            "static_reals": [],
            "time_varying_known_categoricals": [],
            "time_varying_known_reals": known_reals,
            "time_varying_unknown_categoricals": [],
            "time_varying_unknown_reals": unknown_reals,
            "lagged_encoding_cols": lagged,
        }
    
    # Standard PyTorch Forecasting schema fallback
    target = roles.get("target", "power_norm")
    time_col = roles.get("time_col", "timestamp_utc")
    time_idx_col = roles.get("time_idx", roles.get("time_idx_col", "time_idx"))
    group_ids = roles.get("group_ids", ["plant_id"])
    return {
        "target": target,
        "time_col": time_col,
        "time_idx_col": time_idx_col,
        "group_ids": group_ids,
        "static_categoricals": roles.get("static_categoricals", []),
        "static_reals": roles.get("static_reals", []),
        "time_varying_known_categoricals": roles.get("time_varying_known_categoricals", []),
        "time_varying_known_reals": roles.get("time_varying_known_reals", roles.get("known_time_reals", [])),
        "time_varying_unknown_categoricals": roles.get("time_varying_unknown_categoricals", []),
        "time_varying_unknown_reals": roles.get("time_varying_unknown_reals", roles.get("unknown_time_reals", [target])),
        "lagged_encoding_cols": roles.get("lagged_encoding_cols", []),
    }


def ensure_time_columns(df: pd.DataFrame, roles: Dict[str, Any]) -> pd.DataFrame:
    """
    Ensure DataFrame has proper time columns with correct types and no gaps.
    
    Critical: Recomputes time_idx via cumcount per group to avoid missing timestep assertions.
    
    Args:
        df: Input DataFrame
        roles: Column roles dictionary
    
    Returns:
        DataFrame with validated time columns
    """
    df = df.copy()
    
    time_col = roles["time_col"]
    time_idx_col = roles["time_idx_col"]
    group_ids = roles["group_ids"]
    
    # Best-effort fallback if time_col isn't present
    if time_col not in df.columns:
        for cand in ["timestamp_utc", "timestamp", "time", "datetime"]:
            if cand in df.columns:
                time_col = cand
                break
        else:
            raise KeyError(f"Time column not found. Expected '{roles['time_col']}'")
    
    # Ensure datetime type with UTC timezone
    df[time_col] = pd.to_datetime(df[time_col], utc=True)
    
    # Ensure group columns exist
    for g in group_ids:
        if g not in df.columns:
            df[g] = "plant_unk"
    
    # Sort by group and time
    df = df.sort_values(group_ids + [time_col]).reset_index(drop=True)
    
    # CRITICAL: Always recompute time_idx to guarantee step=1 per group (no gaps)
    df[time_idx_col] = df.groupby(group_ids, observed=True).cumcount().astype("int64")
    
    return df


def create_training_dataset(
    train_df: pd.DataFrame,
    roles: Dict[str, Any],
    encoder_len: int,
    pred_len: int,
    target_normalizer: Optional[Any] = None
) -> TimeSeriesDataSet:
    """
    Create TimeSeriesDataSet from training data.
    
    This dataset is used for:
    1. Training TFT models
    2. As reference for test datasets (via .from_dataset() to inherit normalization)
    
    Args:
        train_df: Training DataFrame
        roles: Column roles from config
        encoder_len: Encoder length (e.g., 96 for short-head, 168 for long-head)
        pred_len: Prediction length (e.g., 96 for short-head, 720 for long-head)
        target_normalizer: Optional normalizer (default: GroupNormalizer with softplus)
    
    Returns:
        TimeSeriesDataSet instance
    """
    # Ensure time columns are valid
    train_df = ensure_time_columns(train_df, roles)
    
    # Default normalizer if not provided
    if target_normalizer is None:
        target_normalizer = GroupNormalizer(
            groups=roles["group_ids"],
            transformation="softplus"
        )
    
    dataset = TimeSeriesDataSet(
        train_df,
        time_idx=roles["time_idx_col"],
        target=roles["target"],
        group_ids=roles["group_ids"],
        max_encoder_length=encoder_len,
        max_prediction_length=pred_len,
        static_categoricals=roles["static_categoricals"],
        static_reals=roles["static_reals"],
        time_varying_known_categoricals=roles["time_varying_known_categoricals"],
        time_varying_known_reals=roles["time_varying_known_reals"],
        time_varying_unknown_categoricals=roles["time_varying_unknown_categoricals"],
        time_varying_unknown_reals=roles["time_varying_unknown_reals"],
        target_normalizer=target_normalizer,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )
    
    return dataset


def load_tft_model(
    config: Dict[str, Any],
    train_dataset: TimeSeriesDataSet,
    checkpoint_dir: Path,
    strict: bool = True,
    device: Optional[str] = None
) -> TemporalFusionTransformer:
    """
    Load TFT model with trained weights.
    
    Args:
        config: Config dictionary from load_tft_config()
        train_dataset: Training dataset for model architecture
        checkpoint_dir: Directory containing checkpoints/ folder
        strict: Strict state dict loading (recommended True)
        device: Device to load model on ('cuda', 'cpu', or None for auto)
    
    Returns:
        Loaded TFT model in eval mode
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create loss function
    loss = QuantileLoss(quantiles=config['quantiles'])
    
    # Create model from dataset
    model = TemporalFusionTransformer.from_dataset(
        train_dataset,
        learning_rate=float(config['cli_args'].get('learning_rate', config['cli_args'].get('lr', 1e-3))),
        hidden_size=config['hidden_size'],
        lstm_layers=config['lstm_layers'],
        attention_head_size=config['attention_head_size'],
        dropout=config['dropout'],
        loss=loss,
        reduce_on_plateau_patience=int(config['cli_args'].get('patience', 3)),
    )
    
    # Load weights
    _load_weights_into_model(model, checkpoint_dir, strict)
    
    # Move to device and set to eval mode
    model.to(device)
    model.eval()
    
    return model


def _load_weights_into_model(
    model: torch.nn.Module,
    ckpt_dir: Path,
    strict: bool
) -> None:
    """
    Load checkpoint weights into model.
    Handles both best_state_dict.pt and best.ckpt formats.
    
    Args:
        model: Model to load weights into
        ckpt_dir: Checkpoints directory
        strict: Strict loading flag
    
    Raises:
        FileNotFoundError: If no checkpoint found
        RuntimeError: If loading fails
    """
    sd_path = ckpt_dir / "best_state_dict.pt"
    ckpt_path = ckpt_dir / "best.ckpt"
    
    state: Dict[str, Any] | None = None
    
    # Try best_state_dict.pt first (preferred format)
    if sd_path.exists():
        obj = torch.load(sd_path, map_location="cpu")
        state = obj if isinstance(obj, dict) else None
    # Fallback to best.ckpt
    elif ckpt_path.exists():
        obj = torch.load(ckpt_path, map_location="cpu")
        if isinstance(obj, dict) and "state_dict" in obj:
            state = obj["state_dict"]
        elif isinstance(obj, dict):
            state = obj
        else:
            state = None
    else:
        raise FileNotFoundError(
            f"Missing weights in {ckpt_dir} "
            f"(need best_state_dict.pt or best.ckpt)"
        )
    
    if state is None:
        raise RuntimeError("Could not interpret checkpoint format.")
    
    # Strip common prefixes that may exist from Lightning wrappers
    def _strip_prefix(sd: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
        if not all(k.startswith(prefix) for k in sd.keys()):
            return sd
        return {k[len(prefix):]: v for k, v in sd.items()}
    
    for prefix in ["model.", "tft.", "net."]:
        state = _strip_prefix(state, prefix)
    
    # Load state dict
    try:
        model.load_state_dict(state, strict=strict)
    except RuntimeError as e:
        raise RuntimeError(
            "State dict load failed. Most likely your venv versions differ from training.\n"
            "Fix: run inference inside the same container, or recreate venv with matching versions.\n"
            f"Original error: {e}"
        )


def extract_q50_prediction(
    output: Any,
    quantiles: List[float]
) -> np.ndarray:
    """
    Extract median (q50) quantile from TFT model output.
    
    Args:
        output: Model output object (has .prediction attribute)
        quantiles: List of quantiles used by model (e.g., [0.02, 0.1, ..., 0.98])
    
    Returns:
        predictions: Array shape (B, P) with q50 predictions
            B = batch size
            P = prediction length
    
    Raises:
        ValueError: If q50 not in quantiles
    """
    # Get prediction tensor: (B, P, Q)
    pred = output.prediction
    
    # Find q50 index
    if 0.5 not in quantiles:
        # Fallback: use middle quantile
        q50_idx = len(quantiles) // 2
    else:
        q50_idx = quantiles.index(0.5)
    
    # Extract q50: (B, P, Q) → (B, P)
    if pred.ndim == 3:
        pred_q50 = pred[:, :, q50_idx]
    elif pred.ndim == 2:
        # Already 2D (single quantile or mean)
        pred_q50 = pred
    else:
        raise ValueError(f"Unexpected prediction shape: {pred.shape}")
    
    # Convert to numpy
    if torch.is_tensor(pred_q50):
        pred_q50 = pred_q50.detach().cpu().numpy()
    else:
        pred_q50 = np.asarray(pred_q50)
    
    return pred_q50


def create_inference_dataframe(
    encoder_df: pd.DataFrame,
    decoder_df: pd.DataFrame,
    roles: Dict[str, Any],
    plant_id: str = "plant_03"
) -> pd.DataFrame:
    """
    Create properly formatted inference DataFrame from encoder + decoder windows.
    
    Args:
        encoder_df: Historical data (encoder window)
        decoder_df: Future data (decoder window)
        roles: Column roles from config
        plant_id: Plant identifier
    
    Returns:
        Combined DataFrame ready for TimeSeriesDataSet
    """
    # Concatenate encoder + decoder
    inference_df = pd.concat([encoder_df, decoder_df], ignore_index=True)
    
    # Ensure plant_id column exists
    if 'plant_id' not in inference_df.columns:
        inference_df['plant_id'] = plant_id
    
    # Ensure proper time columns
    inference_df = ensure_time_columns(inference_df, roles)
    
    return inference_df


def validate_inference_window(
    df: pd.DataFrame,
    expected_length: int,
    window_name: str = "window"
) -> None:
    """
    Validate inference window has correct length.
    
    Args:
        df: DataFrame to validate
        expected_length: Expected number of rows
        window_name: Name for error message
    
    Raises:
        ValueError: If length doesn't match
    """
    actual = len(df)
    if actual != expected_length:
        raise ValueError(
            f"{window_name} must be {expected_length} steps, got {actual}"
        )


# ==================== Testing & Debugging ====================

if __name__ == "__main__":
    """Quick test of utility functions."""
    import sys
    from pathlib import Path
    
    print("\n" + "="*70)
    print("TFT UTILS - QUICK TEST")
    print("="*70)
    
    # Test 1: Load short-head config
    print("\n[TEST 1] Load short-head config")
    try:
        short_run = Path("experiments/tft/runs/germany/plant_03/15min/pvlib_coldstart/20251229_134850")
        if short_run.exists():
            config = load_tft_config(short_run)
            print(f"  ✓ Encoder length: {config['encoder_len']}")
            print(f"  ✓ Prediction length: {config['pred_len']}")
            print(f"  ✓ Hidden size: {config['hidden_size']}")
            print(f"  ✓ Quantiles: {len(config['quantiles'])} quantiles")
            print(f"  ✓ Target: {config['roles']['target']}")
        else:
            print(f"  ⚠️  Short-head run not found: {short_run}")
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
    
    # Test 2: Load long-head config
    print("\n[TEST 2] Load long-head config")
    try:
        import subprocess
        result = subprocess.run(
            ["find", "experiments/tft/runs/germany/plant_03", "-path", "*longhead*", "-name", "run_config.json"],
            capture_output=True, text=True
        )
        long_configs = result.stdout.strip().split('\n')
        if long_configs and long_configs[0]:
            long_run = Path(long_configs[0]).parent
            config = load_tft_config(long_run)
            print(f"  ✓ Encoder length: {config['encoder_len']}")
            print(f"  ✓ Prediction length: {config['pred_len']}")
            print(f"  ✓ LSTM layers: {config['lstm_layers']}")
        else:
            print(f"  ⚠️  Long-head run not found")
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
    
    # Test 3: Test time column handling
    print("\n[TEST 3] Test time column handling")
    try:
        test_df = pd.DataFrame({
            'timestamp_utc': pd.date_range('2023-01-01', periods=100, freq='15min'),
            'plant_id': 'plant_03',
            'power_norm': np.random.rand(100)
        })
        roles = {
            'time_col': 'timestamp_utc',
            'time_idx_col': 'time_idx',
            'group_ids': ['plant_id']
        }
        test_df = ensure_time_columns(test_df, roles)
        print(f"  ✓ Shape: {test_df.shape}")
        print(f"  ✓ time_idx range: [{test_df['time_idx'].min()}, {test_df['time_idx'].max()}]")
        print(f"  ✓ No gaps: {(test_df['time_idx'].diff()[1:] == 1).all()}")
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
    
    # Test 4: Test quantile extraction
    print("\n[TEST 4] Test quantile extraction")
    try:
        # Simulate model output
        class MockOutput:
            def __init__(self):
                # (B=2, P=96, Q=7)
                self.prediction = torch.randn(2, 96, 7)
        
        quantiles = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
        output = MockOutput()
        q50 = extract_q50_prediction(output, quantiles)
        
        print(f"  ✓ Input shape: {output.prediction.shape}")
        print(f"  ✓ Output shape: {q50.shape}")
        print(f"  ✓ Q50 index: {quantiles.index(0.5)} (0.5 in quantiles)")
        assert q50.shape == (2, 96), f"Expected (2, 96), got {q50.shape}"
        print(f"  ✓ Shape correct!")
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
    
    print("\n" + "="*70)
    print("✅ TFT UTILS BASIC TESTS PASSED")
    print("="*70)
    print("\nReady for integration into PhysicsAwareForecaster!")
    print("="*70)
