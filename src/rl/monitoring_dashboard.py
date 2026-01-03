#!/usr/bin/env python3
"""
MiRACLE RL Monitoring Dashboard

Real-time visualization of:
- RL meta-controller diagnostics (ε, Q-values, loss, actions)
- TFT model performance (short/long RMSE)
- Forecasts (15min, 1h, 24h predictions)
- Blend weights evolution
- Reward trends
- Replay buffer status

Usage:
    streamlit run src/rl/monitoring_dashboard.py -- --log-dir checkpoints/rl/logs

Author: MiRACLE Team
Date: 2026-01-02
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import json
import time
from datetime import datetime, timedelta
import argparse

# Page config
st.set_page_config(
    page_title="MiRACLE RL Monitor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .status-good {
        color: #28a745;
        font-weight: bold;
    }
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
    .status-bad {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Action names mapping
ACTION_NAMES = {
    0: "MAINTAIN",
    1: "FINE_TUNE_SHORT",
    2: "FINE_TUNE_LONG",
    3: "RECALIBRATE_PVLIB",
    4: "BLEND_SHORT",
    5: "BLEND_LONG",
    6: "BLEND_PHYSICS",
    7: "SUGGEST_RETRAIN"
}

ACTION_COLORS = {
    0: "#2ecc71",  # Green (maintain)
    1: "#3498db",  # Blue (short)
    2: "#9b59b6",  # Purple (long)
    3: "#f39c12",  # Orange (pvlib)
    4: "#1abc9c",  # Teal (blend short)
    5: "#e74c3c",  # Red (blend long)
    6: "#95a5a6",  # Gray (blend physics)
    7: "#e67e22"   # Dark orange (retrain)
}


def load_metrics_log(log_dir: Path):
    """Load latest metrics from log directory."""
    log_file = log_dir / "metrics.jsonl"
    if not log_file.exists():
        return None
    
    # Read last 1000 lines (for performance)
    with open(log_file, 'r') as f:
        lines = f.readlines()[-1000:]
    
    metrics = [json.loads(line) for line in lines if line.strip()]
    return pd.DataFrame(metrics)


def load_rl_state(log_dir: Path):
    """Load current RL state."""
    state_file = log_dir / "rl_state.json"
    if not state_file.exists():
        return None
    
    with open(state_file, 'r') as f:
        return json.load(f)


def render_header():
    """Render dashboard header."""
    st.markdown('<div class="main-header">🤖 MiRACLE RL Meta-Controller Monitor</div>', 
                unsafe_allow_html=True)
    st.markdown("---")


def render_current_status(rl_state):
    """Render current RL status cards."""
    if rl_state is None:
        st.warning("⚠️ No RL state data available")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        epsilon = rl_state.get('epsilon', 0.0)
        st.metric("Exploration (ε)", f"{epsilon:.3f}", 
                  delta=f"{rl_state.get('epsilon_delta', 0.0):.4f}")
    
    with col2:
        action_idx = rl_state.get('last_action', 0)
        action_name = ACTION_NAMES.get(action_idx, "UNKNOWN")
        st.metric("Current Action", action_name)
    
    with col3:
        q_max = rl_state.get('q_max', 0.0)
        st.metric("Max Q-Value", f"{q_max:.3f}")
    
    with col4:
        buffer_size = rl_state.get('buffer_size', 0)
        buffer_capacity = rl_state.get('buffer_capacity', 10000)
        st.metric("Replay Buffer", f"{buffer_size}/{buffer_capacity}",
                  delta=f"{100*buffer_size/buffer_capacity:.1f}%")


def render_rmse_comparison(df):
    """Render RMSE comparison for short and long TFT."""
    if df is None or len(df) == 0:
        st.warning("No metrics data")
        return
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Short-TFT RMSE (1-24h)", "Long-TFT RMSE (1-30d)"),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Short-TFT RMSE
    if 'short_rmse_1h' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['short_rmse_1h'], 
                      name="1h", line=dict(color='#3498db', width=2)),
            row=1, col=1
        )
    if 'short_rmse_6h' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['short_rmse_6h'], 
                      name="6h", line=dict(color='#9b59b6', width=2)),
            row=1, col=1
        )
    if 'short_rmse_24h' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['short_rmse_24h'], 
                      name="24h", line=dict(color='#e74c3c', width=2)),
            row=1, col=1
        )
    
    # Long-TFT RMSE
    if 'long_rmse_7d' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['long_rmse_7d'], 
                      name="7d", line=dict(color='#2ecc71', width=2)),
            row=1, col=2
        )
    if 'long_rmse_30d' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['long_rmse_30d'], 
                      name="30d", line=dict(color='#f39c12', width=2)),
            row=1, col=2
        )
    
    fig.update_xaxes(title_text="Time Step", row=1, col=1)
    fig.update_xaxes(title_text="Time Step", row=1, col=2)
    fig.update_yaxes(title_text="RMSE", row=1, col=1)
    fig.update_yaxes(title_text="RMSE", row=1, col=2)
    
    fig.update_layout(height=400, showlegend=True, hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)


def render_forecasts(df):
    """Render predicted power for 15min, 1h, 24h."""
    if df is None or len(df) == 0:
        st.warning("No forecast data")
        return
    
    # Get latest predictions
    latest = df.iloc[-1]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pred_15min = latest.get('pred_power_15min', 0.0)
        st.metric("15min Forecast", f"{pred_15min:.3f} kW",
                  help="Next 15-minute prediction")
    
    with col2:
        pred_1h = latest.get('pred_power_1h', 0.0)
        st.metric("1h Forecast", f"{pred_1h:.3f} kW",
                  help="1-hour ahead prediction")
    
    with col3:
        pred_24h = latest.get('pred_power_24h', 0.0)
        st.metric("24h Forecast", f"{pred_24h:.3f} kW",
                  help="24-hour ahead prediction")
    
    # Plot forecast horizon
    if all(k in df.columns for k in ['pred_power_15min', 'pred_power_1h', 'pred_power_24h']):
        fig = go.Figure()
        
        # Show last 100 steps
        plot_df = df.tail(100)
        
        fig.add_trace(go.Scatter(
            x=plot_df.index, y=plot_df['pred_power_15min'],
            name="15min", line=dict(color='#2ecc71', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=plot_df.index, y=plot_df['pred_power_1h'],
            name="1h", line=dict(color='#3498db', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=plot_df.index, y=plot_df['pred_power_24h'],
            name="24h", line=dict(color='#e74c3c', width=2)
        ))
        
        fig.update_layout(
            title="Predicted Power Over Time",
            xaxis_title="Time Step",
            yaxis_title="Power (kW)",
            height=300,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)


def render_blend_weights(df):
    """Render blend weight evolution."""
    if df is None or len(df) == 0:
        st.warning("No blend weight data")
        return
    
    if not all(k in df.columns for k in ['blend_short', 'blend_long', 'blend_physics']):
        st.info("Blend weights not yet available")
        return
    
    fig = go.Figure()
    
    plot_df = df.tail(100)
    
    fig.add_trace(go.Scatter(
        x=plot_df.index, y=plot_df['blend_short'],
        name="Short-TFT", fill='tonexty',
        line=dict(color='#3498db', width=0),
        stackgroup='one'
    ))
    fig.add_trace(go.Scatter(
        x=plot_df.index, y=plot_df['blend_long'],
        name="Long-TFT", fill='tonexty',
        line=dict(color='#9b59b6', width=0),
        stackgroup='one'
    ))
    fig.add_trace(go.Scatter(
        x=plot_df.index, y=plot_df['blend_physics'],
        name="PVLib", fill='tonexty',
        line=dict(color='#f39c12', width=0),
        stackgroup='one'
    ))
    
    fig.update_layout(
        title="Blend Weight Evolution (Stacked Area)",
        xaxis_title="Time Step",
        yaxis_title="Weight",
        height=300,
        hovermode='x unified',
        yaxis=dict(range=[0, 1])
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_reward_trends(df):
    """Render reward and Q-loss over time."""
    if df is None or len(df) == 0:
        st.warning("No reward data")
        return
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Reward Trends", "Q-Loss (Training)"),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    plot_df = df.tail(100)
    
    # Rewards
    if 'reward' in df.columns:
        fig.add_trace(
            go.Scatter(x=plot_df.index, y=plot_df['reward'],
                      name="Reward", line=dict(color='#2ecc71', width=2)),
            row=1, col=1
        )
        
        # Add moving average
        if len(plot_df) > 10:
            ma = plot_df['reward'].rolling(10).mean()
            fig.add_trace(
                go.Scatter(x=plot_df.index, y=ma,
                          name="MA(10)", line=dict(color='#e74c3c', width=2, dash='dash')),
                row=1, col=1
            )
    
    # Q-Loss
    if 'q_loss' in df.columns:
        fig.add_trace(
            go.Scatter(x=plot_df.index, y=plot_df['q_loss'],
                      name="Q-Loss", line=dict(color='#9b59b6', width=2)),
            row=1, col=2
        )
    
    fig.update_xaxes(title_text="Time Step", row=1, col=1)
    fig.update_xaxes(title_text="Time Step", row=1, col=2)
    fig.update_yaxes(title_text="Reward", row=1, col=1)
    fig.update_yaxes(title_text="Loss", row=1, col=2)
    
    fig.update_layout(height=300, showlegend=True, hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)


def render_action_distribution(df):
    """Render action distribution histogram."""
    if df is None or len(df) == 0 or 'action' not in df.columns:
        st.info("No action data yet")
        return
    
    # Count actions
    action_counts = df['action'].value_counts().sort_index()
    
    fig = go.Figure()
    
    colors = [ACTION_COLORS.get(i, '#95a5a6') for i in action_counts.index]
    labels = [ACTION_NAMES.get(i, f"Action {i}") for i in action_counts.index]
    
    fig.add_trace(go.Bar(
        x=labels,
        y=action_counts.values,
        marker_color=colors,
        text=action_counts.values,
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Action Distribution (Total Counts)",
        xaxis_title="Action",
        yaxis_title="Count",
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)


def main():
    """Main dashboard loop."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir', type=str, default='checkpoints/rl/logs',
                       help='Directory containing RL logs')
    parser.add_argument('--refresh-rate', type=int, default=5,
                       help='Refresh rate in seconds')
    
    # Streamlit args handling
    try:
        args = parser.parse_args()
    except SystemExit:
        args = argparse.Namespace(log_dir='checkpoints/rl/logs', refresh_rate=5)
    
    log_dir = Path(args.log_dir)
    
    # Render header
    render_header()
    
    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Controls")
        
        auto_refresh = st.checkbox("Auto-refresh", value=True)
        refresh_rate = st.slider("Refresh Rate (s)", 1, 60, args.refresh_rate)
        
        if st.button("🔄 Refresh Now"):
            st.rerun()
        
        st.markdown("---")
        st.markdown(f"**Log Directory:**  \n`{log_dir}`")
        st.markdown(f"**Last Updated:**  \n{pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M:%S UTC')}")
    
    # Load data
    df = load_metrics_log(log_dir)
    rl_state = load_rl_state(log_dir)
    
    # Current Status
    st.header("📊 Current Status")
    render_current_status(rl_state)
    
    st.markdown("---")
    
    # RMSE Performance
    st.header("📈 Model Performance (RMSE)")
    render_rmse_comparison(df)
    
    st.markdown("---")
    
    # Forecasts
    st.header("🔮 Power Forecasts")
    render_forecasts(df)
    
    st.markdown("---")
    
    # Blend Weights & Rewards in two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚖️ Blend Weights")
        render_blend_weights(df)
    
    with col2:
        st.subheader("💰 Rewards & Loss")
        render_reward_trends(df)
    
    st.markdown("---")
    
    # Action Distribution
    st.header("🎯 Action History")
    render_action_distribution(df)
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(refresh_rate)
        st.rerun()


if __name__ == "__main__":
    main()
