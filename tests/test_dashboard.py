#!/usr/bin/env python3
"""
Test dashboard with simulated RL data.

Generates fake metrics and RL state to verify dashboard displays correctly.
"""

import sys
from pathlib import Path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta

# Create log directory
log_dir = Path("checkpoints/rl/logs")
log_dir.mkdir(parents=True, exist_ok=True)

print(f"📊 Generating test data for dashboard in: {log_dir}")

# Generate 200 fake metrics
metrics_file = log_dir / "metrics.jsonl"
with open(metrics_file, 'w') as f:
    for i in range(200):
        timestamp = datetime.now() - timedelta(minutes=15*(200-i))
        
        # Simulate RL learning: RMSE decreases, rewards increase
        short_rmse_1h = 0.12 - 0.0003 * i + np.random.rand() * 0.01
        long_rmse_30d = 0.15 - 0.0002 * i + np.random.rand() * 0.01
        reward = -0.5 + 0.003 * i + np.random.rand() * 0.1
        
        # Random actions with bias toward MAINTAIN and BLEND
        action = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7], 
                                   p=[0.30, 0.10, 0.08, 0.07, 0.15, 0.15, 0.10, 0.05])
        
        # Blend weights drift based on actions
        if action == 4:  # BLEND_SHORT
            blend_short, blend_long, blend_physics = 0.7, 0.2, 0.1
        elif action == 5:  # BLEND_LONG
            blend_short, blend_long, blend_physics = 0.2, 0.7, 0.1
        elif action == 6:  # BLEND_PHYSICS
            blend_short, blend_long, blend_physics = 0.2, 0.2, 0.6
        else:
            # Small random drift
            blend_short = 0.33 + np.random.rand() * 0.1 - 0.05
            blend_long = 0.33 + np.random.rand() * 0.1 - 0.05
            blend_physics = 1.0 - blend_short - blend_long
        
        # Forecasts
        pred_15min = 0.5 + np.random.rand() * 0.3
        pred_1h = 0.5 + np.random.rand() * 0.3
        pred_24h = 0.4 + np.random.rand() * 0.3
        
        entry = {
            'timestamp': timestamp.isoformat(),
            'action': int(action),
            'reward': float(reward),
            'short_rmse_1h': float(short_rmse_1h),
            'short_rmse_6h': float(short_rmse_1h * 1.2),
            'short_rmse_24h': float(short_rmse_1h * 1.5),
            'long_rmse_7d': float(long_rmse_30d * 0.9),
            'long_rmse_30d': float(long_rmse_30d),
            'blend_short': float(blend_short),
            'blend_long': float(blend_long),
            'blend_physics': float(blend_physics),
            'pred_power_15min': float(pred_15min),
            'pred_power_1h': float(pred_1h),
            'pred_power_24h': float(pred_24h),
            'q_loss': float(0.1 - 0.0003 * i + np.random.rand() * 0.02) if i > 10 else 0.0,
            'epsilon': float(max(0.1, 1.0 - 0.004 * i))
        }
        
        f.write(json.dumps(entry) + '\n')

print(f"✅ Generated {metrics_file}")

# Generate current RL state
rl_state = {
    'timestamp': datetime.now().isoformat(),
    'epsilon': 0.25,
    'epsilon_delta': 0.15,
    'last_action': 4,  # BLEND_SHORT
    'q_max': 0.85,
    'buffer_size': 3245,
    'buffer_capacity': 10000,
    'total_steps': 200
}

state_file = log_dir / "rl_state.json"
with open(state_file, 'w') as f:
    json.dump(rl_state, f, indent=2)

print(f"✅ Generated {state_file}")

print("\n" + "="*60)
print("🚀 DASHBOARD READY!")
print("="*60)
print(f"\nRun dashboard with:")
print(f"  streamlit run src/rl/monitoring_dashboard.py")
print(f"\nOr specify custom log directory:")
print(f"  streamlit run src/rl/monitoring_dashboard.py -- --log-dir {log_dir}")
print("\n" + "="*60)
