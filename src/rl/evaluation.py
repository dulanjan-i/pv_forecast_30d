"""
Evaluation utilities for trained RL agents.
"""

import logging
from pathlib import Path
from typing import Dict, List
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


def compare_action_distributions(
    heuristic_actions: List[int],
    learned_actions: List[int],
    action_names: Dict[int, str]
) -> Dict:
    """
    Compare action distributions between heuristic and learned policies.
    
    Args:
        heuristic_actions: Actions from heuristic policy
        learned_actions: Actions from learned policy
        action_names: Mapping from action ID to action name
        
    Returns:
        Dict with comparison statistics
    """
    heuristic_counts = np.bincount(heuristic_actions, minlength=8)
    learned_counts = np.bincount(learned_actions, minlength=8)
    
    heuristic_dist = heuristic_counts / len(heuristic_actions)
    learned_dist = learned_counts / len(learned_actions)
    
    logger.info("\nAction Distribution Comparison:")
    logger.info("=" * 70)
    logger.info(f"{'Action':<30} {'Heuristic':>15} {'Learned':>15} {'Diff':>10}")
    logger.info("=" * 70)
    
    for action_id in range(8):
        action_name = action_names.get(action_id, f"Action {action_id}")
        h_pct = heuristic_dist[action_id] * 100
        l_pct = learned_dist[action_id] * 100
        diff = l_pct - h_pct
        
        logger.info(f"{action_name:<30} {h_pct:>14.1f}% {l_pct:>14.1f}% {diff:>9.1f}%")
    
    logger.info("=" * 70)
    
    return {
        'heuristic_distribution': heuristic_dist,
        'learned_distribution': learned_dist,
        'heuristic_counts': heuristic_counts,
        'learned_counts': learned_counts
    }


def evaluate_policy(
    controller: MetaController,
    eval_states: np.ndarray,
    eval_labels: np.ndarray = None,
    device: str = 'cpu'
) -> Dict:
    """
    Evaluate a trained policy on a set of states.
    
    Args:
        controller: Trained MetaController
        eval_states: States to evaluate on (N, state_dim)
        eval_labels: Optional ground truth actions/rewards
        device: Device to run on
        
    Returns:
        Evaluation metrics dict
    """
    controller.policy_net.to(device)
    controller.policy_net.eval()
    
    actions = []
    q_values = []
    
    with torch.no_grad():
        for state in eval_states:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_vals = controller.policy_net(state_tensor).squeeze().cpu().numpy()
            action = int(np.argmax(q_vals))
            
            actions.append(action)
            q_values.append(q_vals)
    
    actions = np.array(actions)
    q_values = np.array(q_values)
    
    results = {
        'actions': actions,
        'q_values': q_values,
        'action_distribution': np.bincount(actions, minlength=8) / len(actions),
        'mean_q_value': q_values.mean(),
        'std_q_value': q_values.std()
    }
    
    logger.info("\nEvaluation Results:")
    logger.info(f"  Mean Q-value: {results['mean_q_value']:.4f}")
    logger.info(f"  Std Q-value: {results['std_q_value']:.4f}")
    logger.info(f"  Action distribution: {results['action_distribution']}")
    
    return results
