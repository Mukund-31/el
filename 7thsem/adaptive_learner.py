"""
Adaptive Arrival Rate Learner
==============================

Learns time-varying arrival patterns from observed data instead of using
hardcoded schedules. Uses Gaussian Process Regression for smooth interpolation.

For research paper: Demonstrates fully dynamic system that adapts to
real-world patterns without manual tuning.
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from typing import List, Tuple, Dict
from collections import deque
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdaptiveArrivalRateLearner:
    """
    Learns arrival rate λ(t) from observed data using Gaussian Process.
    
    Key Features:
    - No hardcoded schedules
    - Learns from actual arrival patterns
    - Provides uncertainty estimates
    - Adapts to changing patterns over time
    """
    
    def __init__(
        self,
        window_size: int = 200,
        update_frequency: int = 20,
        initial_rate: float = 10.0
    ):
        """
        Initialize adaptive learner.
        
        Args:
            window_size: Number of observations to keep
            update_frequency: Retrain GP every N observations
            initial_rate: Starting estimate before learning
        """
        self.window_size = window_size
        self.update_frequency = update_frequency
        self.initial_rate = initial_rate
        
        # Observation buffer: (hour_of_day, day_of_week, observed_rate)
        self.observations = deque(maxlen=window_size)
        
        # Gaussian Process model
        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3)) *
            RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) +
            WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-3, 1e1))
        )
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            alpha=1e-6,
            normalize_y=True
        )
        
        # Training state
        self.is_trained = False
        self.observations_since_update = 0
        self.training_history = []
        
        # Fallback to simple moving average initially
        self.hourly_averages = {}  # hour -> list of observed rates
        
    def observe_arrivals(
        self,
        hour: float,
        day_of_week: int,
        arrivals_in_interval: int,
        interval_minutes: float = 5.0
    ):
        """
        Record observed arrivals.
        
        Args:
            hour: Hour of day (0-23, can be fractional)
            day_of_week: Day of week (0=Monday, 6=Sunday)
            arrivals_in_interval: Number of arrivals observed
            interval_minutes: Length of observation interval
        """
        # Convert to arrivals per hour
        rate = (arrivals_in_interval / interval_minutes) * 60.0
        
        # Store observation
        self.observations.append((hour, day_of_week, rate))
        self.observations_since_update += 1
        
        # Update hourly averages for fallback
        hour_int = int(hour)
        if hour_int not in self.hourly_averages:
            self.hourly_averages[hour_int] = []
        self.hourly_averages[hour_int].append(rate)
        if len(self.hourly_averages[hour_int]) > 50:
            self.hourly_averages[hour_int].pop(0)
        
        # Retrain if enough new data
        if self.observations_since_update >= self.update_frequency:
            self._train_gp()
            self.observations_since_update = 0
    
    def _train_gp(self):
        """Train Gaussian Process on observed data."""
        if len(self.observations) < 20:
            logger.info("Not enough data to train GP (need 20+ observations)")
            return
        
        # Prepare training data
        X = np.array([[hour, day] for hour, day, _ in self.observations])
        y = np.array([rate for _, _, rate in self.observations])
        
        try:
            # Fit GP
            self.gp.fit(X, y)
            self.is_trained = True
            
            # Log training info
            score = self.gp.score(X, y)
            self.training_history.append({
                'timestamp': datetime.now().isoformat(),
                'n_observations': len(self.observations),
                'r2_score': score
            })
            
            logger.info(f"GP trained on {len(self.observations)} observations, R² = {score:.3f}")
            
        except Exception as e:
            logger.warning(f"GP training failed: {e}")
            self.is_trained = False
    
    def predict_rate(
        self,
        hour: float,
        day_of_week: int,
        return_std: bool = True
    ) -> Tuple[float, float]:
        """
        Predict arrival rate for given time.
        
        Args:
            hour: Hour of day (0-23)
            day_of_week: Day of week (0-6)
            return_std: If True, also return uncertainty
            
        Returns:
            (predicted_rate, std) if return_std else predicted_rate
        """
        if not self.is_trained:
            # Fallback to hourly average or initial rate
            hour_int = int(hour)
            if hour_int in self.hourly_averages and len(self.hourly_averages[hour_int]) > 0:
                mean_rate = np.mean(self.hourly_averages[hour_int])
                std_rate = np.std(self.hourly_averages[hour_int]) if len(self.hourly_averages[hour_int]) > 1 else 2.0
            else:
                mean_rate = self.initial_rate
                std_rate = 5.0  # High uncertainty
            
            return (mean_rate, std_rate) if return_std else mean_rate
        
        # Use GP prediction
        X_pred = np.array([[hour, day_of_week]])
        
        if return_std:
            mean, std = self.gp.predict(X_pred, return_std=True)
            return float(mean[0]), float(std[0])
        else:
            mean = self.gp.predict(X_pred)
            return float(mean[0])
    
    def get_daily_pattern(self, day_of_week: int = 0) -> Dict[int, Tuple[float, float]]:
        """
        Get predicted pattern for entire day.
        
        Args:
            day_of_week: Which day to predict (0=Monday)
            
        Returns:
            Dict mapping hour -> (mean_rate, std_rate)
        """
        pattern = {}
        for hour in range(24):
            mean, std = self.predict_rate(hour, day_of_week, return_std=True)
            pattern[hour] = (mean, std)
        return pattern
    
    def export_learned_pattern(self, filename: str):
        """Export learned pattern for visualization/paper."""
        import json
        
        patterns = {}
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        for day in range(7):
            day_pattern = self.get_daily_pattern(day)
            patterns[day_names[day]] = {
                str(hour): {'mean': mean, 'std': std}
                for hour, (mean, std) in day_pattern.items()
            }
        
        with open(filename, 'w') as f:
            json.dump({
                'learned_patterns': patterns,
                'training_history': self.training_history,
                'n_observations': len(self.observations)
            }, f, indent=2)
        
        logger.info(f"Learned patterns exported to {filename}")


class AdaptiveCustomerAttributeLearner:
    """
    Learns customer attribute distributions from observed data.
    
    Instead of hardcoded patience/complexity distributions,
    learns from actual customer behavior.
    """
    
    def __init__(self, window_size: int = 500):
        self.window_size = window_size
        
        # Observed attributes
        self.observed_patience = deque(maxlen=window_size)
        self.observed_complexity = deque(maxlen=window_size)
        self.observed_contagion = deque(maxlen=window_size)
        
        # Learned parameters
        self.patience_params = {'mean': 15.0, 'std': 5.0}
        self.complexity_params = {'mean': 1.0, 'std': 0.2}
        self.contagion_params = {'mean': 0.5, 'std': 0.2}
        
    def observe_customer(
        self,
        patience: float,
        complexity: float,
        contagion: float
    ):
        """Record observed customer attributes."""
        self.observed_patience.append(patience)
        self.observed_complexity.append(complexity)
        self.observed_contagion.append(contagion)
        
        # Update parameters periodically
        if len(self.observed_patience) >= 50 and len(self.observed_patience) % 50 == 0:
            self._update_parameters()
    
    def _update_parameters(self):
        """Update distribution parameters from observations."""
        if len(self.observed_patience) > 10:
            self.patience_params = {
                'mean': np.mean(self.observed_patience),
                'std': np.std(self.observed_patience)
            }
        
        if len(self.observed_complexity) > 10:
            self.complexity_params = {
                'mean': np.mean(self.observed_complexity),
                'std': np.std(self.observed_complexity)
            }
        
        if len(self.observed_contagion) > 10:
            self.contagion_params = {
                'mean': np.mean(self.observed_contagion),
                'std': np.std(self.observed_contagion)
            }
        
        logger.info(f"Updated attribute distributions: "
                   f"patience={self.patience_params['mean']:.1f}±{self.patience_params['std']:.1f}, "
                   f"complexity={self.complexity_params['mean']:.2f}±{self.complexity_params['std']:.2f}")
    
    def generate_patience(self, is_lunch: bool = False) -> float:
        """Generate patience using learned distribution."""
        # Use learned parameters instead of hardcoded
        mean = self.patience_params['mean']
        std = self.patience_params['std']
        
        # Adjust for lunch rush (people are less patient)
        if is_lunch:
            mean *= 0.7
        
        patience = np.random.normal(mean, std)
        return max(1.0, patience)  # Ensure positive
    
    def generate_complexity(self) -> float:
        """Generate complexity using learned distribution."""
        complexity = np.random.normal(
            self.complexity_params['mean'],
            self.complexity_params['std']
        )
        return np.clip(complexity, 0.3, 2.5)
    
    def generate_contagion_factor(self) -> float:
        """Generate contagion factor using learned distribution."""
        contagion = np.random.normal(
            self.contagion_params['mean'],
            self.contagion_params['std']
        )
        return np.clip(contagion, 0.0, 1.0)


if __name__ == "__main__":
    # Demo: Adaptive arrival rate learning
    logger.info("Testing Adaptive Arrival Rate Learner...")
    
    learner = AdaptiveArrivalRateLearner()
    
    # Simulate observations (morning rush pattern)
    for i in range(100):
        hour = 9 + (i % 8)  # Hours 9-16
        day = i % 5  # Weekdays
        
        # Simulate higher arrivals during lunch (12-13)
        if 12 <= hour < 13:
            arrivals = np.random.poisson(12)
        else:
            arrivals = np.random.poisson(5)
        
        learner.observe_arrivals(hour, day, arrivals, interval_minutes=5.0)
    
    # Test prediction
    predicted_rate, uncertainty = learner.predict_rate(12.5, 2, return_std=True)
    logger.info(f"Predicted rate at 12:30 on Wednesday: {predicted_rate:.1f} ± {uncertainty:.1f} customers/hour")
    
    # Export pattern
    learner.export_learned_pattern("learned_arrival_pattern.json")
    logger.info("Demo complete!")
