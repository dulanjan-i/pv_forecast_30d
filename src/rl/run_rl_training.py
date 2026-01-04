#!/usr/bin/env python3
"""
Offline DDQN Training Script (Thin Wrapper)

This script is a thin wrapper around src.rl.training module.
All core logic lives in src/rl/ for reusability.

Usage:
    PYTHONPATH=/home/dwijenayake/pv_forecast_30d python scripts/train_rl_offline.py \
        --data data/rl_transitions/training_batch_001.parquet \
        --epochs 50 \
        --checkpoint-dir checkpoints/rl_meta_controller \
        --device cuda
"""

import argparse
import logging
from pathlib import Path
import json
import matplotlib.pyplot as plt

from src.rl.rl_meta_controller import MetaController, RLConfig
from src.rl.training import (
    load_transitions,
    prepare_training_data,
    train_offline,
    save_checkpoint
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def plot_training_metrics(metrics: dict, output_dir: Path):
    """Plot training curves."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    axes[0].plot(metrics['losses'])
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].grid(True)
    
    # Q-values
    axes[1].plot(metrics['avg_q_values'])
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Avg Q-value')
    axes[1].set_title('Average Q-values')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_metrics.png', dpi=150)
    logger.info(f"Training plots saved to {output_dir / 'training_metrics.png'}")


def main():
    parser = argparse.ArgumentParser(description='Train DDQN meta-controller offline')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to transition parquet file')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs (reduced from 100 to prevent overfitting)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/rl_meta_controller',
                        help='Directory to save checkpoints')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device for training (cpu or cuda)')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.95,
                        help='Discount factor')
    
    args = parser.parse_args()
    
    # Setup paths
    data_path = Path(args.data)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("DDQN OFFLINE TRAINING")
    logger.info("=" * 80)
    logger.info(f"Data: {data_path}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Checkpoint dir: {checkpoint_dir}")
    logger.info(f"Device: {args.device}")
    logger.info("=" * 80)
    
    # Load and prepare data
    df = load_transitions(data_path)
    states, actions, rewards, next_states, dones = prepare_training_data(df)
    
    # Initialize MetaController
    config = RLConfig(
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        batch_size=args.batch_size,
        mode="rl"
    )
    
    state_dim = states.shape[1]
    n_actions = 8
    
    logger.info(f"Initializing MetaController (state_dim={state_dim}, n_actions={n_actions})")
    controller = MetaController(state_dim=state_dim, config=config)
    
    # Train
    metrics = train_offline(
        controller=controller,
        states=states,
        actions=actions,
        rewards=rewards,
        next_states=next_states,
        dones=dones,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device
    )
    
    # Save checkpoint
    metadata = {
        'data_path': str(data_path),
        'n_transitions': len(df),
        'state_dim': state_dim,
        'n_actions': n_actions,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'final_loss': metrics['losses'][-1] if metrics['losses'] else None,
        'final_avg_q': metrics['avg_q_values'][-1] if metrics['avg_q_values'] else None,
        'total_steps': controller.steps
    }
    
    checkpoint_path = checkpoint_dir / 'ddqn_meta_controller.pt'
    save_checkpoint(controller, checkpoint_path, metadata)
    
    # Save metadata as JSON
    with open(checkpoint_dir / 'training_summary.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Plot metrics
    plot_training_metrics(metrics, checkpoint_dir)
    
    logger.info("=" * 80)
    logger.info("✅ TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Final Loss: {metadata['final_loss']:.4f}")
    logger.info(f"Final Avg Q: {metadata['final_avg_q']:.4f}")
    logger.info(f"Total Steps: {metadata['total_steps']}")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
