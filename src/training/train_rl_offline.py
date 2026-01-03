"""
Offline DDQN training from collected RL transitions.

Trains the DDQN meta-controller using experience replay on historical data.
This allows us to bootstrap the RL system before deploying it online.

Usage:
    python scripts/train_rl_offline.py --data data/rl_transitions/historical_batch.parquet --epochs 5000
"""
import sys
import argparse
from pathlib import Path
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Fix imports for running from src/
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.rl.rl_meta_controller import RLMetaControllerSystem, RLConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OfflineTrainer:
    """Train DDQN from offline dataset."""
    
    def __init__(
        self,
        rl_system: RLMetaControllerSystem,
        batch_size: int = 32,
        device: str = 'cuda'
    ):
        self.rl_system = rl_system
        self.batch_size = batch_size
        self.device = device
        
        # Training metrics
        self.losses = []
        self.q_values = []
        self.rewards = []
        
    def load_transitions(self, data_path: str) -> int:
        """
        Load transitions from parquet into replay buffer.
        
        Returns:
            Number of transitions loaded
        """
        logger.info(f"Loading transitions from {data_path}")
        df = pd.read_parquet(data_path)
        
        logger.info(f"Loaded {len(df)} transitions")
        logger.info(f"Columns: {list(df.columns)}")
        
        # Extract state, action, reward, next_state
        state_cols = [c for c in df.columns if c.startswith('state_') and not c.startswith('next_')]
        next_state_cols = [c for c in df.columns if c.startswith('next_state_')]
        
        logger.info(f"State dimensions: {len(state_cols)}")
        
        # Load into replay buffer
        for idx, row in df.iterrows():
            state = np.array([row[c] for c in state_cols], dtype=np.float32)
            action = int(row['action'])
            reward = float(row['reward'])
            next_state = np.array([row[c] for c in next_state_cols], dtype=np.float32)
            done = False  # Not terminal in offline data
            
            self.rl_system.meta_controller.replay_buffer.push(
                state, action, reward, next_state, done
            )
        
        logger.info(f"✅ Loaded {len(self.rl_system.meta_controller.replay_buffer)} transitions into replay buffer")
        return len(df)
    
    def train_step(self) -> dict:
        """
        Perform one DDQN training step.
        
        Returns:
            dict with loss, q_value, etc.
        """
        if len(self.rl_system.meta_controller.replay_buffer) < self.batch_size:
            return {'loss': 0.0, 'q_value': 0.0}
        
        # Sample batch
        transitions = self.rl_system.meta_controller.replay_buffer.sample(self.batch_size)
        batch = list(zip(*transitions))
        
        state_batch = torch.FloatTensor(np.array(batch[0])).to(self.device)
        action_batch = torch.LongTensor(batch[1]).to(self.device)
        reward_batch = torch.FloatTensor(batch[2]).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch[3])).to(self.device)
        done_batch = torch.FloatTensor(batch[4]).to(self.device)
        
        # Current Q values
        current_q_values = self.rl_system.meta_controller.policy_net(state_batch)
        current_q = current_q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)
        
        # DDQN: use policy_net to select action, target_net to evaluate
        with torch.no_grad():
            next_q_policy = self.rl_system.meta_controller.policy_net(next_state_batch)
            next_actions = next_q_policy.max(1)[1]
            
            next_q_target = self.rl_system.meta_controller.target_net(next_state_batch)
            next_q = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            
            # TD target
            target_q = reward_batch + self.rl_system.meta_controller.gamma * next_q * (1 - done_batch)
        
        # Compute loss
        loss = F.smooth_l1_loss(current_q, target_q)
        
        # Optimize
        self.rl_system.meta_controller.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.rl_system.meta_controller.policy_net.parameters(), 1.0)
        self.rl_system.meta_controller.optimizer.step()
        
        return {
            'loss': loss.item(),
            'q_value': current_q.mean().item(),
            'target_q': target_q.mean().item(),
            'reward': reward_batch.mean().item()
        }
    
    def train(self, epochs: int, log_interval: int = 100, save_interval: int = 1000):
        """
        Train DDQN for specified epochs.
        
        Args:
            epochs: Number of training iterations
            log_interval: Log every N epochs
            save_interval: Save checkpoint every N epochs
        """
        logger.info(f"Starting offline DDQN training for {epochs} epochs")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Replay buffer size: {len(self.rl_system.meta_controller.replay_buffer)}")
        
        pbar = tqdm(range(epochs), desc="Training DDQN")
        
        for epoch in pbar:
            metrics = self.train_step()
            
            self.losses.append(metrics['loss'])
            self.q_values.append(metrics['q_value'])
            self.rewards.append(metrics['reward'])
            
            # Update target network
            if (epoch + 1) % self.rl_system.meta_controller.target_update == 0:
                self.rl_system.meta_controller.target_net.load_state_dict(
                    self.rl_system.meta_controller.policy_net.state_dict()
                )
            
            # Logging
            if (epoch + 1) % log_interval == 0:
                recent_loss = np.mean(self.losses[-log_interval:])
                recent_q = np.mean(self.q_values[-log_interval:])
                recent_reward = np.mean(self.rewards[-log_interval:])
                
                pbar.set_postfix({
                    'loss': f'{recent_loss:.4f}',
                    'Q': f'{recent_q:.3f}',
                    'R': f'{recent_reward:.3f}'
                })
                
                logger.info(f"Epoch {epoch+1}/{epochs} | Loss: {recent_loss:.4f} | Q-value: {recent_q:.3f} | Reward: {recent_reward:.3f}")
            
            # Save checkpoint
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(epoch + 1)
        
        pbar.close()
        
        # Final save
        self.save_checkpoint(epochs, final=True)
        
        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info(f"Training Complete!")
        logger.info(f"{'='*60}")
        logger.info(f"Total epochs: {epochs}")
        logger.info(f"Final loss: {np.mean(self.losses[-100:]):.4f}")
        logger.info(f"Final Q-value: {np.mean(self.q_values[-100:]):.3f}")
        logger.info(f"Average reward: {np.mean(self.rewards):.3f}")
        logger.info(f"Loss trend: {self.losses[0]:.4f} → {self.losses[-1]:.4f}")
    
    def save_checkpoint(self, epoch: int, final: bool = False):
        """Save model checkpoint."""
        checkpoint_dir = Path("checkpoints/rl")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        if final:
            filename = checkpoint_dir / "ddqn_final.pt"
        else:
            filename = checkpoint_dir / f"ddqn_epoch_{epoch}.pt"
        
        checkpoint = {
            'epoch': epoch,
            'policy_net_state_dict': self.rl_system.meta_controller.policy_net.state_dict(),
            'target_net_state_dict': self.rl_system.meta_controller.target_net.state_dict(),
            'optimizer_state_dict': self.rl_system.meta_controller.optimizer.state_dict(),
            'losses': self.losses,
            'q_values': self.q_values,
            'rewards': self.rewards,
            'epsilon': self.rl_system.meta_controller.epsilon,
            'training_steps': epoch
        }
        
        torch.save(checkpoint, filename)
        logger.info(f"💾 Saved checkpoint: {filename}")
    
    def save_training_curves(self):
        """Save training metrics to CSV for plotting."""
        metrics_df = pd.DataFrame({
            'epoch': range(len(self.losses)),
            'loss': self.losses,
            'q_value': self.q_values,
            'reward': self.rewards
        })
        
        output_path = Path("checkpoints/rl/training_metrics.csv")
        metrics_df.to_csv(output_path, index=False)
        logger.info(f"📊 Saved training curves: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train DDQN offline from collected transitions")
    parser.add_argument('--data', type=str, required=True,
                       help='Path to parquet file with RL transitions')
    parser.add_argument('--epochs', type=int, default=5000,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--buffer-size', type=int, default=10000,
                       help='Replay buffer capacity')
    parser.add_argument('--target-update', type=int, default=100,
                       help='Target network update frequency')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device for training')
    parser.add_argument('--log-interval', type=int, default=100,
                       help='Log every N epochs')
    parser.add_argument('--save-interval', type=int, default=1000,
                       help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    # Check data exists
    if not Path(args.data).exists():
        logger.error(f"Data file not found: {args.data}")
        return
    
    # Initialize RL system with config
    logger.info("Initializing RL Meta-Controller System...")
    config = RLConfig(
        learning_rate=args.lr,
        gamma=args.gamma,
        buffer_capacity=args.buffer_size,
        target_update_freq=args.target_update,
        batch_size=args.batch_size
    )
    
    rl_system = RLMetaControllerSystem(
        config=config,
        checkpoint_dir=Path("/home/dwijenayake/pv_forecast_30d/checkpoints/rl")
    )
    
    # Create trainer
    trainer = OfflineTrainer(
        rl_system=rl_system,
        batch_size=args.batch_size,
        device=args.device
    )
    
    # Load data
    num_transitions = trainer.load_transitions(args.data)
    
    if num_transitions < args.batch_size:
        logger.error(f"Not enough transitions ({num_transitions}) for batch size ({args.batch_size})")
        logger.error(f"Need at least {args.batch_size} transitions to train")
        return
    
    # Train
    trainer.train(
        epochs=args.epochs,
        log_interval=args.log_interval,
        save_interval=args.save_interval
    )
    
    # Save training curves
    trainer.save_training_curves()
    
    logger.info("\n🎉 Training complete!")
    logger.info(f"Model saved to: checkpoints/rl/ddqn_final.pt")
    logger.info(f"Training curves: checkpoints/rl/training_metrics.csv")
    logger.info(f"\nNext steps:")
    logger.info(f"  1. Plot training curves: python scripts/plot_rl_training.py")
    logger.info(f"  2. Test trained model: python scripts/test_rl_agent.py --checkpoint checkpoints/rl/ddqn_final.pt")
    logger.info(f"  3. Deploy online: Set mode='rl' in RLIntegratedForecaster")


if __name__ == "__main__":
    main()
