#!/usr/bin/env python
"""
Test script to validate end-to-end orchestration of lstm-pretrain branch.
Tests config -> data pipeline -> model instantiation WITHOUT real data.
"""

import yaml
import torch
import pandas as pd
import numpy as np
from pathlib import Path

# Test imports
print("=" * 80)
print("TESTING LSTM-PRETRAIN BRANCH ORCHESTRATION")
print("=" * 80)

print("\n[1/6] Testing imports...")
try:
    from src.models.lstm_encoder import LSTMEncoderConfig, LSTMEncoder, make_trainer
    from src.features.sequence_generator import SimpleWindowDataset
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    exit(1)

print("\n[2/6] Loading config file...")
try:
    config_path = Path("experiments/lstm/pretrain_pvdaq.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"✓ Config loaded: {config_path}")
    print(f"  - Data config keys: {list(config['data'].keys())}")
    print(f"  - Model config keys: {list(config['model'].keys())}")
    print(f"  - Training config keys: {list(config['training'].keys())}")
except Exception as e:
    print(f"✗ Config loading failed: {e}")
    exit(1)

print("\n[3/6] Mapping config to LSTMEncoderConfig...")
try:
    # Extract values
    data_cfg = config['data']
    model_cfg = config['model']
    train_cfg = config['training']
    
    # Count features
    num_features = len(data_cfg['feature_cols'])
    
    # Map to LSTMEncoderConfig
    encoder_config = LSTMEncoderConfig(
        input_size=num_features,                    # Derived from feature_cols
        hidden_size=model_cfg['hidden_size'],       # From model.hidden_size
        num_layers=model_cfg['num_layers'],         # From model.num_layers
        dropout=model_cfg['dropout'],               # From model.dropout
        lr=train_cfg['learning_rate'],              # From training.learning_rate
        weight_decay=train_cfg['weight_decay'],     # From training.weight_decay
        aux_predict=True,                           # Default (for pretraining)
        embedding_dim=None,                         # Default (use hidden_size)
    )
    print("✓ Config mapping successful")
    print(f"  - input_size: {encoder_config.input_size} (from {len(data_cfg['feature_cols'])} feature_cols)")
    print(f"  - hidden_size: {encoder_config.hidden_size}")
    print(f"  - num_layers: {encoder_config.num_layers}")
    print(f"  - dropout: {encoder_config.dropout}")
    print(f"  - lr: {encoder_config.lr}")
    print(f"  - weight_decay: {encoder_config.weight_decay}")
except Exception as e:
    print(f"✗ Config mapping failed: {e}")
    exit(1)

print("\n[4/6] Creating synthetic data matching config expectations...")
try:
    # Generate synthetic dataframe matching YAML structure
    num_sites = 3
    timesteps_per_site = 200  # Must be > window_size + horizon
    
    data_list = []
    for site_id in range(num_sites):
        for t in range(timesteps_per_site):
            row = {
                data_cfg['time_col']: t,                    # time_idx
                data_cfg['id_col']: f"site_{site_id}",      # site_id
                data_cfg['target_col']: np.random.rand(),   # pv_power_norm
            }
            # Add all feature columns
            for feat in data_cfg['feature_cols']:
                row[feat] = np.random.rand()
            data_list.append(row)
    
    df = pd.DataFrame(data_list)
    print(f"✓ Synthetic data created: {len(df)} rows, {len(df.columns)} columns")
    print(f"  - Columns: {list(df.columns)}")
    print(f"  - Sites: {df[data_cfg['id_col']].nunique()}")
    print(f"  - Timesteps per site: ~{len(df) // num_sites}")
except Exception as e:
    print(f"✗ Synthetic data creation failed: {e}")
    exit(1)

print("\n[5/6] Testing SimpleWindowDataset with config parameters...")
try:
    dataset = SimpleWindowDataset(
        df=df,
        time_col=data_cfg['time_col'],
        group_col=data_cfg['id_col'],
        feature_cols=data_cfg['feature_cols'],
        target_col=data_cfg['target_col'],
        input_window=data_cfg['window_size'],
        forecast_horizon=data_cfg['horizon'],
    )
    
    print(f"✓ Dataset created: {len(dataset)} windows")
    
    # Test sample
    x_sample, y_sample = dataset[0]
    print(f"  - Sample X shape: {x_sample.shape} (expected: ({data_cfg['window_size']}, {num_features}))")
    print(f"  - Sample y shape: {y_sample.shape if hasattr(y_sample, 'shape') else 'scalar'} (expected: scalar for horizon=1)")
    
    # Verify shapes match expectations
    assert x_sample.shape == (data_cfg['window_size'], num_features), \
        f"X shape mismatch: got {x_sample.shape}, expected ({data_cfg['window_size']}, {num_features})"
    assert isinstance(y_sample, (float, np.float32, np.float64)), \
        f"y should be scalar for horizon=1, got {type(y_sample)}"
    
    print("  ✓ Shape validation passed")
except Exception as e:
    print(f"✗ Dataset creation failed: {e}")
    exit(1)

print("\n[6/6] Testing model instantiation and forward pass...")
try:
    # Create model
    model = LSTMEncoder(encoder_config)
    print(f"✓ Model instantiated: {model.__class__.__name__}")
    
    # Create a batch
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=train_cfg['batch_size'], shuffle=False)
    batch_x, batch_y = next(iter(loader))
    
    print(f"  - Batch X shape: {batch_x.shape} (B={batch_x.shape[0]}, T={batch_x.shape[1]}, F={batch_x.shape[2]})")
    print(f"  - Batch y shape: {batch_y.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(batch_x)
    
    print(f"  - Output keys: {list(output.keys())}")
    print(f"  - Embedding shape: {output['embedding'].shape}")
    if output['next_pred'] is not None:
        print(f"  - Next_pred shape: {output['next_pred'].shape}")
    
    # Verify shapes
    expected_emb_dim = encoder_config.embedding_dim or encoder_config.hidden_size
    assert output['embedding'].shape == (batch_x.shape[0], expected_emb_dim), \
        f"Embedding shape mismatch: got {output['embedding'].shape}, expected ({batch_x.shape[0]}, {expected_emb_dim})"
    
    if encoder_config.aux_predict:
        assert output['next_pred'].shape == (batch_x.shape[0],), \
            f"Next_pred shape mismatch: got {output['next_pred'].shape}, expected ({batch_x.shape[0]},)"
    
    print("  ✓ Forward pass successful")
    print("  ✓ Shape validation passed")
    
except Exception as e:
    print(f"✗ Model instantiation or forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 80)
print("✅ ALL ORCHESTRATION TESTS PASSED")
print("=" * 80)
print("\nSummary:")
print("  ✓ Config → Code mapping is correct")
print("  ✓ Data pipeline (SimpleWindowDataset) works with config params")
print("  ✓ Model (LSTMEncoder) instantiates correctly")
print("  ✓ Forward pass produces expected shapes")
print("\nMISSING for end-to-end execution:")
print("  ✗ Real PVDAQ data files (data/raw/pvdaq_nsrd_train.csv, val.csv)")
print("  ✗ Training script (src/training/train_lstm.py is a placeholder)")
print("  ✗ Trainer integration (need to wire dataset + model + Lightning Trainer)")
print("\nNext steps:")
print("  1. Implement src/training/train_lstm_pretrain.py")
print("  2. Add data loading logic")
print("  3. Wire Lightning Trainer with callbacks, logging")
