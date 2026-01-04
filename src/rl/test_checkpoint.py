"""
RL checkpoint testing utilities.

This module provides functions to test and validate trained RL checkpoints.
"""

import logging
from pathlib import Path
import numpy as np
import torch

from src.rl.rl_meta_controller import MetaController, RLConfig

logger = logging.getLogger(__name__)


def test_checkpoint_loading(checkpoint_path: Path, state_dim: int = 35, device: str = 'cpu') -> bool:
    """
    Test if a checkpoint can be loaded and used for inference.
    
    Args:
        checkpoint_path: Path to checkpoint file
        state_dim: State dimension (default 35 from 3 advisors)
        device: Device to run on
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Testing checkpoint: {checkpoint_path}")
        
        # Initialize controller
        config = RLConfig(mode="rl")
        controller = MetaController(state_dim=state_dim, config=config)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        controller.policy_net.load_state_dict(checkpoint['policy_net'])
        controller.target_net.load_state_dict(checkpoint['target_net'])
        controller.epsilon = checkpoint['epsilon']
        controller.steps = checkpoint['steps']
        
        logger.info(f"✅ Checkpoint loaded successfully!")
        logger.info(f"   Training steps: {checkpoint['steps']}")
        logger.info(f"   Final epsilon: {checkpoint['epsilon']:.4f}")
        
        # Move to device
        controller.policy_net.to(device)
        controller.target_net.to(device)
        
        # Test inference on sample states
        logger.info("\nTesting inference on 5 random sample states...")
        for i in range(5):
            state = np.random.randn(state_dim).astype(np.float32)
            action = controller.select_action(state)
            
            # Get Q-values for debugging
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = controller.policy_net(state_tensor).squeeze().cpu().numpy()
            
            logger.info(f"State {i+1}: Selected action: {action}, Q-values: {q_values}, Max Q: {q_values.max():.4f}")
        
        logger.info("\n✅ Checkpoint is functional and ready for deployment!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Checkpoint test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    # Can be run as standalone script for quick testing
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description='Test RL checkpoint loading')
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/rl_meta_controller/ddqn_meta_controller.pt',
                        help='Path to checkpoint file')
    parser.add_argument('--state-dim', type=int, default=35,
                        help='State dimension (default 35 from 3 advisors)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to run on (cpu or cuda)')
    
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    
    if not checkpoint_path.exists():
        logger.error(f"❌ Checkpoint not found: {checkpoint_path}")
        exit(1)
    
    logger.info("=" * 80)
    logger.info("RL CHECKPOINT TEST")
    logger.info("=" * 80)
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"State dim: {args.state_dim}")
    logger.info(f"Device: {args.device}")
    logger.info("=" * 80)
    
    success = test_checkpoint_loading(
        checkpoint_path=checkpoint_path,
        state_dim=args.state_dim,
        device=args.device
    )
    
    if success:
        logger.info("\n✅ All tests passed! Checkpoint is ready for deployment.")
    else:
        logger.error("\n❌ Tests failed. Check errors above.")
        exit(1)
