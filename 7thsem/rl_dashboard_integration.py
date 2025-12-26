# Add this at the top of dashboard.py after imports
from pathlib import Path
import torch

# Load trained RL model
def load_trained_model():
    """Load the trained RL model for real-time decisions."""
    from rl_optimization_agent import RLOptimizationAgent
    
    model_path = Path("trained_model.pth")
    if not model_path.exists():
        return None
    
    agent = RLOptimizationAgent()
    agent.load_model(str(model_path))
    return agent

# Add this function to make RL decisions
def get_rl_decision(agent, state_data):
    """Get decision from trained RL model."""
    from rl_optimization_agent import SystemState
    
    if agent is None:
        return "MAINTAIN", 0.0
    
    # Create state from dashboard data
    state = SystemState(
        num_tellers=state_data.get('num_tellers', 3),
        current_queue=state_data.get('queue_length', 0),
        avg_fatigue=state_data.get('avg_fatigue', 0.3),
        max_fatigue=state_data.get('max_fatigue', 0.5),
        burnt_out_count=state_data.get('burnt_out', 0),
        lobby_anger=state_data.get('lobby_anger', 1.0),
        predicted_arrivals_mean=state_data.get('pred_mean', 10.0),
        predicted_arrivals_ucb=state_data.get('pred_ucb', 15.0),
        prediction_uncertainty=state_data.get('uncertainty', 0.5),
        current_wait=state_data.get('current_wait', 0.0),
        hour_of_day=state_data.get('hour', 9),
        recent_renege_rate=state_data.get('renege_rate', 0.0)
    )
    
    # Get action from trained model (inference mode)
    q_values, action_name = agent.select_action(state, training=False)
    
    return action_name, q_values.max().item()
