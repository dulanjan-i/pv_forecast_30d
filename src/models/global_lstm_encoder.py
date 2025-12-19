"""
src/models/global_lstm_encoder.py

Version 3 - Global Forecasting Model: LSTM Encoder with Transfer Learning

PURPOSE
-------
This module defines the GlobalLSTMEncoder - an LSTM that processes the Super Matrix
(all 5 plants) with zero-padded transfer learning from Farm2107 pretrained weights.

KEY FEATURES
------------
1. **Input Expansion**: 15 features (Farm2107) → 20 features (15 original + 5 plant IDs)
2. **Zero-Padding Transfer**: Farm2107 weights preserved for first 15 features,
   new plant_id columns initialized to zero (model learns them from scratch)
3. **PyTorch Lightning**: Compatible with existing training infrastructure
4. **Multi-Task Learning**: One model handles all 5 plants simultaneously

ARCHITECTURE
------------
Input: (batch_size, seq_len, 20)
    ↓
LSTM(input_size=20, hidden_size=64, num_layers=2, dropout=0.1)
    ↓
Final hidden state: (batch_size, 64)
    ↓
Linear(64 → 1) [Next-step prediction head]
    ↓
Output: (batch_size, 1) - predicted power_norm at t+1

TRANSFER LEARNING MECHANICS
----------------------------
Farm2107 LSTM weights shape:
    weight_ih_l0: (hidden_size*4, 15)  # Input-to-hidden for 4 gates (i,f,g,o)
    weight_hh_l0: (hidden_size*4, hidden_size)  # Hidden-to-hidden (unchanged)

Global LSTM weights shape:
    weight_ih_l0: (hidden_size*4, 20)  # Expanded to 20 input features
    weight_hh_l0: (hidden_size*4, hidden_size)  # Same (no change needed)

Zero-Padding Strategy:
    new_weight[:, 0:15] = farm2107_weight  # Copy first 15 columns
    new_weight[:, 15:20] = 0.0              # Initialize plant_id weights to zero

Why zeros? 
- Conservative approach: model starts with no plant-specific bias
- Gradients will learn optimal plant_id weights during training
- Alternative: small random init (e.g., Normal(0, 0.01))

USAGE EXAMPLE
-------------
```python
from src.models.global_lstm_encoder import GlobalLSTMEncoder, transfer_from_farm2107

# Option 1: Transfer learning (recommended)
model = GlobalLSTMEncoder(input_size=20, hidden_size=64, num_layers=2)
model = transfer_from_farm2107(
    model, 
    farm2107_ckpt_path="experiments/lstm/encoders/lstm_encoder_farm2107_CANONICAL.pt"
)

# Option 2: Train from scratch
model = GlobalLSTMEncoder(input_size=20, hidden_size=64, num_layers=2)

# Training with PyTorch Lightning
trainer = pl.Trainer(max_epochs=20, gpus=2)
trainer.fit(model, train_dataloader, val_dataloader)
```

DESIGN DECISIONS
----------------
1. **Inherit from LSTMEncoder**: Reuses existing training logic (loss, optimizer, logging)
2. **Override input_size**: Only change is input dimension (15→20)
3. **Zero-padding in separate function**: Keeps transfer logic modular and testable
4. **Compatible with existing training scripts**: Minimal code changes needed

NOTES
-----
- This model expects GLOBAL_LSTM_INPUT_FEATURES (20 features in correct order)
- One-hot plant_id columns should be binary (0.0 or 1.0)
- Use SimpleWindowDataset with window_size matching seq_len
- Monitor per-plant performance during validation (may vary)

Author: PV Forecast Team
Date: December 2024
Version: 3.0 (Global Forecasting Model)
"""

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

# Add repo root to path for imports
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Import base LSTM encoder (Version 02 implementation)
from src.models.lstm_encoder import LSTMEncoder, LSTMEncoderConfig


class GlobalLSTMEncoder(LSTMEncoder):
    """
    LSTM Encoder for Global Forecasting Model (all 5 plants).
    
    Inherits from LSTMEncoder but expects 20 input features instead of 15:
        - 15 original LSTM features (weather + power_norm)
        - 5 one-hot plant_id features (plant_01, plant_02, plant_03, plant_05, plant_06)
    
    Parameters
    ----------
    input_size : int
        Input feature dimension (should be 20 for Global Model)
    hidden_size : int
        LSTM hidden state dimension (default: 64, matching Farm2107)
    num_layers : int
        Number of LSTM layers (default: 2, matching Farm2107)
    dropout : float
        Dropout probability between LSTM layers (default: 0.1)
    lr : float
        Learning rate for Adam optimizer (default: 1e-4)
        
    Attributes
    ----------
    lstm : nn.LSTM
        LSTM module with expanded input_size=20
    fc : nn.Linear
        Final prediction head (hidden_size → 1)
        
    Methods
    -------
    forward(x)
        Forward pass: returns dict with 'next_pred' and 'embedding'
        
    Examples
    --------
    >>> config = LSTMEncoderConfig(input_size=20, hidden_size=64, num_layers=2)
    >>> model = GlobalLSTMEncoder(config)
    >>> x = torch.randn(32, 24, 20)  # (batch, seq_len, features)
    >>> output = model(x)
    >>> output['next_pred'].shape  # (32,) - predicted power_norm
    >>> output['embedding'].shape  # (32, 64) - learned representation
    """
    
    def __init__(self, config: LSTMEncoderConfig):
        """
        Initialize Global LSTM Encoder with 20 input features.
        
        Validates that input_size=20 (15 original + 5 plant IDs).
        """
        # Validate input size for global model
        if config.input_size != 20:
            raise ValueError(
                f"GlobalLSTMEncoder expects input_size=20 (15 features + 5 plant IDs), "
                f"got {config.input_size}. Check GLOBAL_LSTM_INPUT_FEATURES in schema.py"
            )
        
        # Initialize parent class (LSTMEncoder)
        super().__init__(config)
        
        print(f"[INFO] GlobalLSTMEncoder initialized:")
        print(f"  Input size: {config.input_size} (15 original + 5 plant IDs)")
        print(f"  Hidden size: {config.hidden_size}")
        print(f"  Num layers: {config.num_layers}")
        print(f"  Dropout: {config.dropout}")


def transfer_from_farm2107(
    global_model: GlobalLSTMEncoder,
    farm2107_ckpt_path: str,
    device: str = 'cpu',
) -> GlobalLSTMEncoder:
    """
    Transfer Farm2107 pretrained weights to Global Model with zero-padding.
    
    This function loads Farm2107's LSTM encoder weights (trained on 15 features)
    and transfers them to the Global Model (which expects 20 features) by:
        1. Copying first 15 columns of input-to-hidden weights
        2. Zero-initializing last 5 columns (plant_id features)
        3. Copying hidden-to-hidden weights unchanged
        4. Copying final prediction head unchanged
    
    Parameters
    ----------
    global_model : GlobalLSTMEncoder
        Target model with input_size=20 (randomly initialized)
    farm2107_ckpt_path : str
        Path to Farm2107 checkpoint file (.pt)
        Expected: 'experiments/lstm/encoders/lstm_encoder_farm2107_CANONICAL.pt'
    device : str
        Device for loading checkpoint ('cpu' or 'cuda')
        
    Returns
    -------
    GlobalLSTMEncoder
        Model with transferred weights (ready for fine-tuning)
        
    Raises
    ------
    FileNotFoundError
        If farm2107_ckpt_path doesn't exist
    ValueError
        If checkpoint structure doesn't match expected format
        
    Notes
    -----
    - Farm2107 had 15 input features → weight_ih shape: (hidden*4, 15)
    - Global Model has 20 input features → weight_ih shape: (hidden*4, 20)
    - Zero-padding initializes plant_id weights neutrally (model learns them)
    - Alternative: Random init for columns 15-20 (use if zeros underperform)
    
    Weight Shapes After Transfer:
        lstm.weight_ih_l0: (256, 20)  # 64*4 gates, 20 inputs
        lstm.weight_hh_l0: (256, 64)  # Unchanged (hidden→hidden)
        lstm.bias_ih_l0: (256,)       # Unchanged (input biases)
        lstm.bias_hh_l0: (256,)       # Unchanged (hidden biases)
        fc.weight: (1, 64)            # Unchanged (prediction head)
        fc.bias: (1,)                 # Unchanged
    
    Examples
    --------
    >>> model = GlobalLSTMEncoder(config)
    >>> model = transfer_from_farm2107(
    ...     model,
    ...     farm2107_ckpt_path="experiments/lstm/encoders/lstm_encoder_farm2107_CANONICAL.pt"
    ... )
    >>> # Now model has Farm2107 knowledge + trainable plant_id weights
    """
    ckpt_path = Path(farm2107_ckpt_path)
    
    # Validate checkpoint exists
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Farm2107 checkpoint not found: {ckpt_path}\n"
            f"Expected: experiments/lstm/encoders/lstm_encoder_farm2107_CANONICAL.pt"
        )
    
    print(f"\n{'='*80}")
    print("TRANSFER LEARNING: Farm2107 → Global Model")
    print(f"{'='*80}")
    print(f"Loading Farm2107 checkpoint: {ckpt_path}")
    
    # Load Farm2107 checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # Extract Farm2107 LSTM weights
    # Expected keys: 'lstm.weight_ih_l0', 'lstm.weight_hh_l0', 'lstm.bias_ih_l0', 'lstm.bias_hh_l0'
    farm2107_state_dict = checkpoint  # Assumes checkpoint is state_dict
    
    # Get input-to-hidden weight (this needs expansion)
    if 'lstm.weight_ih_l0' not in farm2107_state_dict:
        raise ValueError(
            f"Checkpoint missing 'lstm.weight_ih_l0'. "
            f"Available keys: {list(farm2107_state_dict.keys())}"
        )
    
    farm_weight_ih = farm2107_state_dict['lstm.weight_ih_l0']  # Shape: (hidden*4, 15)
    
    print(f"\nFarm2107 LSTM weights:")
    print(f"  Input-to-hidden shape: {farm_weight_ih.shape} (hidden*4, 15 features)")
    
    # Validate shape
    hidden_size = global_model.cfg.hidden_size
    expected_shape = (hidden_size * 4, 15)
    if farm_weight_ih.shape != expected_shape:
        raise ValueError(
            f"Farm2107 weight_ih shape mismatch. "
            f"Expected {expected_shape}, got {farm_weight_ih.shape}"
        )
    
    # Zero-pad: Expand from 15 → 20 columns
    print(f"\nZero-padding input weights: 15 → 20 features")
    expanded_weight_ih = torch.zeros(hidden_size * 4, 20, device=device)
    
    # Copy first 15 columns (Farm2107 knowledge)
    expanded_weight_ih[:, :15] = farm_weight_ih
    
    # Last 5 columns (plant_id) remain zero (model learns them)
    # Alternative: Small random init
    # expanded_weight_ih[:, 15:] = torch.randn(hidden_size * 4, 5) * 0.01
    
    print(f"  Columns 0-14: Copied from Farm2107 (weather + power features)")
    print(f"  Columns 15-19: Initialized to ZERO (plant_id one-hot)")
    
    # Build new state dict for Global Model
    global_state_dict = global_model.state_dict()
    
    # Transfer LSTM weights (layer 0)
    global_state_dict['lstm.weight_ih_l0'] = expanded_weight_ih
    global_state_dict['lstm.weight_hh_l0'] = farm2107_state_dict['lstm.weight_hh_l0']  # Hidden-to-hidden (unchanged)
    global_state_dict['lstm.bias_ih_l0'] = farm2107_state_dict['lstm.bias_ih_l0']      # Input biases
    global_state_dict['lstm.bias_hh_l0'] = farm2107_state_dict['lstm.bias_hh_l0']      # Hidden biases
    
    # Transfer LSTM layer 1 (if exists)
    if 'lstm.weight_ih_l1' in farm2107_state_dict:
        print(f"  Transferring layer 1 weights (hidden-to-hidden, no expansion needed)")
        global_state_dict['lstm.weight_ih_l1'] = farm2107_state_dict['lstm.weight_ih_l1']
        global_state_dict['lstm.weight_hh_l1'] = farm2107_state_dict['lstm.weight_hh_l1']
        global_state_dict['lstm.bias_ih_l1'] = farm2107_state_dict['lstm.bias_ih_l1']
        global_state_dict['lstm.bias_hh_l1'] = farm2107_state_dict['lstm.bias_hh_l1']
    
    # Transfer prediction head (final FC layer)
    if 'fc.weight' in farm2107_state_dict:
        print(f"  Transferring prediction head (FC layer)")
        global_state_dict['fc.weight'] = farm2107_state_dict['fc.weight']
        global_state_dict['fc.bias'] = farm2107_state_dict['fc.bias']
    
    # Load transferred weights into Global Model
    global_model.load_state_dict(global_state_dict)
    
    print(f"\n✅ Transfer learning complete!")
    print(f"{'='*80}\n")
    
    # Verify: Check that plant_id weights are zero
    current_weight_ih = global_model.lstm.weight_ih_l0
    plant_id_weights = current_weight_ih[:, 15:20]
    
    if torch.allclose(plant_id_weights, torch.zeros_like(plant_id_weights)):
        print("[INFO] Verification PASSED: Plant_id weights are zero (as expected)")
    else:
        print(f"[WARNING] Plant_id weights are NOT zero! Max abs value: {plant_id_weights.abs().max():.6f}")
    
    return global_model


def load_global_encoder_for_inference(
    checkpoint_path: str,
    device: str = 'cpu',
) -> GlobalLSTMEncoder:
    """
    Load a trained Global LSTM Encoder for inference.
    
    Parameters
    ----------
    checkpoint_path : str
        Path to saved .pt checkpoint
    device : str
        Device to load model onto
        
    Returns
    -------
    GlobalLSTMEncoder
        Loaded model in eval mode
        
    Examples
    --------
    >>> model = load_global_encoder_for_inference(
    ...     "experiments/lstm/encoders/global_lstm_encoder_fold1.pt"
    ... )
    >>> model.eval()
    >>> with torch.no_grad():
    ...     output = model(x_test)
    """
    config = LSTMEncoderConfig(input_size=20, hidden_size=64, num_layers=2)
    model = GlobalLSTMEncoder(config)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    
    print(f"[INFO] Loaded Global Encoder from {checkpoint_path}")
    
    return model


# Example usage (for testing)
if __name__ == "__main__":
    print("Testing GlobalLSTMEncoder...")
    
    # Test 1: Create model
    config = LSTMEncoderConfig(input_size=20, hidden_size=64, num_layers=2)
    model = GlobalLSTMEncoder(config)
    print(f"\n✅ Model created: {model}")
    
    # Test 2: Forward pass
    batch_size, seq_len, n_features = 32, 24, 20
    x = torch.randn(batch_size, seq_len, n_features)
    output = model(x)
    
    print(f"\n✅ Forward pass successful:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output 'next_pred' shape: {output['next_pred'].shape}")
    print(f"  Output 'embedding' shape: {output['embedding'].shape}")
    
    # Test 3: Zero-padding (simulation, no real checkpoint)
    print(f"\n✅ Zero-padding test (simulated):")
    print("  Would expand Farm2107 weights from (256, 15) → (256, 20)")
    print("  Columns 0-14: Farm2107 knowledge")
    print("  Columns 15-19: Zero-initialized (plant IDs)")
