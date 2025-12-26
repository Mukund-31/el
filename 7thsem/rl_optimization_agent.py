"""
Enhanced Dynamic Optimization Agent with Reinforcement Learning
================================================================

Research-grade implementation for academic publication.

Key Enhancements:
1. Deep Q-Network (DQN) for dynamic decision-making
2. Adaptive cost weight learning
3. Experience replay for stable learning
4. Comparison with traditional rule-based baseline
5. Performance metrics tracking

Author: Research Team
Date: 2025
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Experience tuple for replay buffer
Experience = namedtuple('Experience', 
                       ['state', 'action', 'reward', 'next_state', 'done'])


@dataclass
class SystemState:
    """Enhanced state representation for RL."""
    num_tellers: int
    current_queue: int
    avg_fatigue: float
    max_fatigue: float
    burnt_out_count: int
    lobby_anger: float
    predicted_arrivals_mean: float
    predicted_arrivals_ucb: float
    prediction_uncertainty: float
    current_wait: float
    hour_of_day: int  # NEW: Time context
    recent_renege_rate: float  # NEW: Recent performance
    
    def to_tensor(self) -> torch.Tensor:
        """Convert state to normalized tensor for neural network."""
        return torch.tensor([
            self.num_tellers / 10.0,  # Normalize to [0, 1]
            self.current_queue / 50.0,
            self.avg_fatigue,
            self.max_fatigue,
            self.burnt_out_count / 5.0,
            self.lobby_anger / 10.0,
            self.predicted_arrivals_mean / 30.0,
            self.predicted_arrivals_ucb / 50.0,
            self.prediction_uncertainty / 10.0,
            self.current_wait / 20.0,
            self.hour_of_day / 24.0,
            self.recent_renege_rate
        ], dtype=torch.float32)


class DQNetwork(nn.Module):
    """Deep Q-Network for action-value estimation."""
    
    def __init__(self, state_dim: int = 12, action_dim: int = 4, hidden_dim: int = 128):
        super(DQNetwork, self).__init__()
        
        # Deeper network for complex decision-making
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass: state -> Q-values for each action."""
        return self.network(state)


class AdaptiveCostLearner:
    """Learns optimal cost weights from system performance."""
    
    def __init__(self, learning_rate: float = 0.01):
        # Initialize cost weights (will be learned)
        self.weights = {
            'staffing': 50.0,
            'wait_time': 5.0,
            'renege': 100.0,
            'fatigue': 30.0
        }
        self.lr = learning_rate
        self.performance_history = deque(maxlen=100)
        
    def update_weights(self, performance_metrics: Dict[str, float]):
        """Adapt cost weights based on system performance."""
        self.performance_history.append(performance_metrics)
        
        if len(self.performance_history) < 10:
            return
        
        # Simple gradient-based adaptation
        # Increase weight for metrics that are getting worse
        recent_avg = np.mean([m['renege_rate'] for m in list(self.performance_history)[-10:]])
        older_avg = np.mean([m['renege_rate'] for m in list(self.performance_history)[-20:-10]])
        
        if recent_avg > older_avg:
            # Renege rate increasing - increase its cost weight
            self.weights['renege'] *= (1 + self.lr)
            logger.info(f"Adapted renege cost weight to {self.weights['renege']:.1f}")
        
        # Similar logic for other metrics
        recent_wait = np.mean([m['avg_wait'] for m in list(self.performance_history)[-10:]])
        older_wait = np.mean([m['avg_wait'] for m in list(self.performance_history)[-20:-10]])
        
        if recent_wait > older_wait:
            self.weights['wait_time'] *= (1 + self.lr)
            logger.info(f"Adapted wait_time cost weight to {self.weights['wait_time']:.1f}")


class RLOptimizationAgent:
    """
    Reinforcement Learning-based optimization agent.
    
    Uses Deep Q-Learning with:
    - Experience replay
    - Target network
    - Epsilon-greedy exploration
    - Adaptive cost learning
    """
    
    def __init__(
        self,
        state_dim: int = 12,
        action_dim: int = 4,
        learning_rate: float = 0.001,  # Reverted to stable 0.001
        gamma: float = 0.99,           # Kept high for long-term planning
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.98,   # Reverted to 0.98 for better exploration
        buffer_size: int = 10000,
        batch_size: int = 64           # Reverted to 64 for stability
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        # Q-networks
        self.q_network = DQNetwork(state_dim, action_dim)
        self.target_network = DQNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Experience replay buffer
        self.replay_buffer = deque(maxlen=buffer_size)
        
        # Adaptive cost learner
        self.cost_learner = AdaptiveCostLearner()
        
        # Performance tracking
        self.episode_rewards = []
        self.training_losses = []
        self.decisions_made = 0
        
        # Action mapping
        self.actions = ['DO_NOTHING', 'ADD_TELLER', 'REMOVE_TELLER', 'GIVE_BREAK']
        
    def select_action(self, state: SystemState, training: bool = True) -> Tuple[int, str]:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current system state
            training: If True, use exploration; if False, use greedy policy
            
        Returns:
            (action_index, action_name)
        """
        if training and random.random() < self.epsilon:
            # Explore: random action
            action_idx = random.randint(0, self.action_dim - 1)
        else:
            # Exploit: best action according to Q-network
            with torch.no_grad():
                state_tensor = state.to_tensor().unsqueeze(0)
                q_values = self.q_network(state_tensor)
                action_idx = q_values.argmax().item()
        
        return action_idx, self.actions[action_idx]
    
    def compute_reward(self, state: SystemState, action: str, next_state: SystemState) -> float:
        """
        Compute reward for state transition.
        
        Reward design:
        - Negative reward for high wait times, reneges, fatigue
        - Positive reward for efficient service
        - Penalty for unnecessary actions
        """
        # Get adaptive cost weights
        weights = self.cost_learner.weights
        
        # Compute costs
        # Compute costs with STABLE penalties
        # Use high linear weights instead of exponents for stable learning
        wait_cost = weights['wait_time'] * next_state.current_wait * 2.0  # Double importance
        
        # Heavy penalty for reneging but linear
        renege_cost = weights['renege'] * next_state.recent_renege_rate * 150 
        
        fatigue_cost = weights['fatigue'] * next_state.avg_fatigue
        staffing_cost = weights['staffing'] * next_state.num_tellers / 10.0
        
        # Total cost (negative reward)
        total_cost = wait_cost + renege_cost + fatigue_cost + staffing_cost
        
        # Bonus for low anger
        anger_bonus = max(0, 8 - next_state.lobby_anger)
        
        # Penalty for unnecessary actions
        action_penalty = 0
        if action == 'ADD_TELLER' and next_state.current_queue < 5:
            action_penalty = 12
        elif action == 'REMOVE_TELLER' and next_state.current_queue > 8:
            action_penalty = 20
        
        reward = -total_cost + anger_bonus - action_penalty
        
        return reward
    
    def store_experience(self, state: SystemState, action: int, reward: float,
                        next_state: SystemState, done: bool):
        """Store experience in replay buffer."""
        self.replay_buffer.append(
            Experience(state.to_tensor(), action, reward, 
                      next_state.to_tensor(), done)
        )
    
    def train_step(self) -> Optional[float]:
        """Perform one training step using experience replay."""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        batch = random.sample(self.replay_buffer, self.batch_size)
        
        # Unpack batch
        states = torch.stack([e.state for e in batch])
        actions = torch.tensor([e.action for e in batch], dtype=torch.long)
        rewards = torch.tensor([e.reward for e in batch], dtype=torch.float32)
        next_states = torch.stack([e.next_state for e in batch])
        dones = torch.tensor([e.done for e in batch], dtype=torch.float32)
        
        # Compute current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        self.training_losses.append(loss.item())
        return loss.item()
    
    def update_target_network(self):
        """Update target network with current Q-network weights."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save_model(self, path: str):
        """Save policy network weights."""
        torch.save(self.q_network.state_dict(), path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load policy network weights."""
        try:
            self.q_network.load_state_dict(torch.load(path))
            self.target_network.load_state_dict(self.q_network.state_dict())
            self.epsilon = self.epsilon_end 
            logger.info(f"Model loaded from {path}")
        except FileNotFoundError:
            logger.warning(f"No model found at {path}")
    
    def save_model(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'cost_weights': self.cost_learner.weights
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.cost_learner.weights = checkpoint['cost_weights']
        logger.info(f"Model loaded from {path}")


class TraditionalBaselineAgent:
    """
    Traditional rule-based agent for comparison.
    
    Uses fixed thresholds and heuristics (no learning).
    """
    
    def __init__(self):
        self.decisions_made = 0
        
    def decide(self, state: SystemState) -> Tuple[str, Dict]:
        """Make decision using fixed rules."""
        self.decisions_made += 1
        
        # Rule 1: Give break if fatigue > 70%
        if state.max_fatigue > 0.7:
            return 'GIVE_BREAK', {'reason': 'High fatigue'}
        
        # Rule 2: Add teller if queue > 10 OR predicted arrivals high
        if state.current_queue > 10 or state.predicted_arrivals_ucb > 15:
            if state.num_tellers < 10:
                return 'ADD_TELLER', {'reason': 'High demand'}
        
        # Rule 3: Remove teller if queue < 3 AND low predictions
        if state.current_queue < 3 and state.predicted_arrivals_ucb < 5:
            if state.num_tellers > 1:
                return 'REMOVE_TELLER', {'reason': 'Low demand'}
        
        # Default: do nothing
        return 'DO_NOTHING', {'reason': 'System balanced'}


class PerformanceComparator:
    """
    Compares RL agent with traditional baseline.
    
    Tracks metrics for statistical validation.
    """
    
    def __init__(self):
        self.rl_metrics = {
            'avg_wait': [],
            'renege_rate': [],
            'staffing_cost': [],
            'total_cost': []
        }
        self.baseline_metrics = {
            'avg_wait': [],
            'renege_rate': [],
            'staffing_cost': [],
            'total_cost': []
        }
        
    def record_rl(self, metrics: Dict[str, float]):
        """Record RL agent performance."""
        for key in self.rl_metrics:
            if key in metrics:
                self.rl_metrics[key].append(metrics[key])
    
    def record_baseline(self, metrics: Dict[str, float]):
        """Record baseline agent performance."""
        for key in self.baseline_metrics:
            if key in metrics:
                self.baseline_metrics[key].append(metrics[key])
    
    def get_comparison_stats(self) -> Dict:
        """Compute statistical comparison."""
        stats = {}
        
        for metric in self.rl_metrics:
            if len(self.rl_metrics[metric]) > 0 and len(self.baseline_metrics[metric]) > 0:
                rl_mean = np.mean(self.rl_metrics[metric])
                baseline_mean = np.mean(self.baseline_metrics[metric])
                improvement = ((baseline_mean - rl_mean) / baseline_mean) * 100
                
                stats[metric] = {
                    'rl_mean': rl_mean,
                    'baseline_mean': baseline_mean,
                    'improvement_pct': improvement,
                    'rl_std': np.std(self.rl_metrics[metric]),
                    'baseline_std': np.std(self.baseline_metrics[metric])
                }
        
        return stats
    
    def export_for_paper(self, filename: str):
        """Export results in publication-ready format."""
        stats = self.get_comparison_stats()
        
        with open(filename, 'w') as f:
            f.write("# Performance Comparison: RL vs Traditional Baseline\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            
            f.write("## Summary Statistics\n\n")
            f.write("| Metric | RL Agent | Baseline | Improvement |\n")
            f.write("|--------|----------|----------|-------------|\n")
            
            for metric, data in stats.items():
                f.write(f"| {metric} | {data['rl_mean']:.2f} ± {data['rl_std']:.2f} | "
                       f"{data['baseline_mean']:.2f} ± {data['baseline_std']:.2f} | "
                       f"{data['improvement_pct']:.1f}% |\n")
            
            f.write("\n## Raw Data\n\n")
            f.write(json.dumps({
                'rl_metrics': {k: [float(v) for v in vals] 
                              for k, vals in self.rl_metrics.items()},
                'baseline_metrics': {k: [float(v) for v in vals] 
                                    for k, vals in self.baseline_metrics.items()}
            }, indent=2))
        
        logger.info(f"Results exported to {filename}")


if __name__ == "__main__":
    # Demo: Train RL agent
    logger.info("Initializing RL Optimization Agent...")
    agent = RLOptimizationAgent()
    
    # Simulate some training
    for episode in range(10):
        # Dummy state
        state = SystemState(
            num_tellers=3, current_queue=5, avg_fatigue=0.3,
            max_fatigue=0.5, burnt_out_count=0, lobby_anger=2.0,
            predicted_arrivals_mean=10, predicted_arrivals_ucb=15,
            prediction_uncertainty=3, current_wait=2.5,
            hour_of_day=12, recent_renege_rate=0.05
        )
        
        action_idx, action_name = agent.select_action(state)
        logger.info(f"Episode {episode}: Selected action {action_name}")
    
    logger.info("RL agent demo complete!")
