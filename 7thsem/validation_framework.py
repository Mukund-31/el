"""
Validation Framework for Research Paper
========================================

Runs controlled experiments comparing RL agent vs baseline,
performs statistical tests, and generates publication-ready results.

Usage:
    python validation_framework.py --episodes 100 --output results/
"""

import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import json
from datetime import datetime, timedelta
import argparse
from pathlib import Path

# Import our agents
from rl_optimization_agent import (
    RLOptimizationAgent, TraditionalBaselineAgent,
    PerformanceComparator, SystemState
)


class ValidationEnvironment:
    """
    Interactive environment where actions actually change the state.
    Ensures identical arrival patterns for fairness.
    """
    def __init__(self, seed: int):
        self.rng = np.random.RandomState(seed)
        self.step_count = 0
        self.max_steps = 100
        
        # State variables
        self.num_tellers = 3
        self.queue = 5
        self.fatigue = 0.2
        self.anger = 1.0
        
        # Scenario parameters for this episode
        self.has_rush_hour = self.rng.random() < 0.7
        self.has_emergency = self.rng.random() < 0.3
        self.variability = self.rng.uniform(0.5, 1.5)

    def reset(self) -> SystemState:
        self.step_count = 0
        self.num_tellers = 3
        self.queue = 5
        self.fatigue = 0.2
        self.anger = 1.0
        self.total_served = 0
        return self._get_state()

    def step(self, action_name: str) -> Tuple[SystemState, float, bool]:
        self.step_count += 1
        
        # 1. Apply Action
        if action_name == 'ADD_TELLER' and self.num_tellers < 10:
            self.num_tellers += 1
        elif action_name == 'REMOVE_TELLER' and self.num_tellers > 1:
            self.num_tellers -= 1
        elif action_name == 'GIVE_BREAK':
            self.fatigue = max(0.0, self.fatigue - 0.3)
            # Giving break reduces capacity temporarily (modeled as effective tellers -1)
            effective_tellers = max(1, self.num_tellers - 1)
        else:
            effective_tellers = self.num_tellers

        # 2. Simulate Dynamics (Arrivals vs Service)
        hour = 9 + (self.step_count / self.max_steps) * 8
        
        # Arrivals (Base + Rush Hour + Noise)
        base_arrival = 5 + 10 * np.sin(self.step_count / self.max_steps * np.pi)
        if self.has_rush_hour and 11 <= hour < 13:
            base_arrival += 15 * np.sin((hour - 11) * np.pi)
        if self.has_emergency and 40 < self.step_count < 60:
            base_arrival += 10
            
        arrivals = int(max(0, base_arrival + self.rng.normal(0, 2 * self.variability)))
        
        # Service Rate (approx 4 customers per teller per step)
        service_capacity = int(self.num_tellers * 4 * (1.0 - self.fatigue * 0.5))
        served = min(self.queue + arrivals, service_capacity)
        self.total_served += served
        
        # Update Queue
        self.queue = max(0, self.queue + arrivals - served)
        
        # Update Fatigue (increases with work)
        step_fatigue = 0.02 + (self.queue / 100.0) 
        self.fatigue = min(1.0, self.fatigue + step_fatigue)

        # 3. Calculate Derived Metrics
        wait_time = max(0, (self.queue / max(1, self.num_tellers * 4)) * 60) # minutes approx
        
        return self._get_state(wait_time), wait_time, self.step_count >= self.max_steps

    def _get_state(self, wait_time=0.0) -> SystemState:
        # Generate predictions (noisy lookahead)
        hour = 9 + (self.step_count / self.max_steps) * 8
        pred_mean = 10
        if self.has_rush_hour and 10.5 <= hour < 12.5:
             pred_mean = 25
        
        renege = min(0.5, wait_time / 15.0) # Simple renege model
        
        return SystemState(
            num_tellers=self.num_tellers,
            current_queue=self.queue,
            avg_fatigue=self.fatigue,
            max_fatigue=self.fatigue,
            burnt_out_count=0,
            lobby_anger=self.queue / 2.0,
            predicted_arrivals_mean=pred_mean,
            predicted_arrivals_ucb=pred_mean + 5,
            prediction_uncertainty=2.0,
            current_wait=wait_time,
            hour_of_day=int(hour),
            recent_renege_rate=renege
        )

# --- Helper for Dates ---
def parse_weird_date(date_str):
    try:
        date_part, time_part = date_str.split(' ')
        hours = int(float(time_part))
        minutes = int(round((float(time_part) - hours) * 100))
        return datetime.strptime(f"{date_part} {hours:02d}:{minutes:02d}", "%d-%m-%Y %H:%M")
    except:
        return None

class TraceValidationEnvironment:
    """
    Validation environment driven by real-world trace data (CSV).
    Adds stochastic jitter to arrivals/services to test robustness over multiple episodes.
    """
    def __init__(self, seed: int, dataframe: pd.DataFrame):
        self.rng = np.random.RandomState(seed)
        self.original_df = dataframe
        self.events = []
        self.current_time = None
        self.num_tellers = 3
        self.queue = [] # List of (arrival_time, service_duration)
        self.active_services = [] # List of completion_times
        self.stats = {'total_wait': 0, 'total_renege': 0, 'served': 0, 'steps': 0}
        
        # State tracking
        self.fatigue = 0.2
        
    def reset(self) -> SystemState:
        # 1. Jitter the dataframe to create a variation of the day
        jittered_df = self.original_df.copy()
        
        # Add random noise to arrivals (gaussian, sigma=5 mins)
        arrival_noise = self.rng.normal(0, 5*60, size=len(jittered_df))
        jittered_df['sim_arrival'] = jittered_df['parsed_arrival'] + pd.to_timedelta(arrival_noise, unit='s')
        
        # Add random noise to service (gaussian, sigma=1 min, min 0.5 min)
        service_noise = self.rng.normal(0, 60, size=len(jittered_df))
        jittered_df['sim_service'] = jittered_df['inferred_service_min'] * 60 + service_noise
        jittered_df['sim_service'] = jittered_df['sim_service'].clip(lower=30) # Min 30 sec service
        
        # Sort by new arrival times
        jittered_df = jittered_df.sort_values('sim_arrival').reset_index(drop=True)
        
        # Create event queue: list of (time, type, data)
        # We need to process this step-by-step
        self.events = []
        for _, row in jittered_df.iterrows():
            self.events.append({
                'time': row['sim_arrival'],
                'type': 'ARRIVAL',
                'service_sec': row['sim_service']
            })
        
        self.event_idx = 0
        self.current_time = self.events[0]['time'] if self.events else datetime.now()
        self.num_tellers = 3
        self.queue = []
        self.active_services = []
        self.stats = {'total_wait': 0, 'total_renege': 0, 'served': 0, 'steps': 0}
        self.fatigue = 0.2
        
        return self._get_state()

    def step(self, action_name: str) -> Tuple[SystemState, float, bool]:
        # Advance time by 10 minutes (decision interval)
        step_duration = timedelta(minutes=10)
        next_time = self.current_time + step_duration
        
        # Apply Action
        if action_name == 'ADD_TELLER' and self.num_tellers < 10:
            self.num_tellers += 1
        elif action_name == 'REMOVE_TELLER' and self.num_tellers > 1:
            self.num_tellers -= 1
            
        # Process events within this window
        step_wait_accum = 0
        step_served_count = 0
        
        # 1. Process Arrivals & Services in chronological order
        while True:
            # Find next event (Arrival vs Service Completion)
            next_arrival = self.events[self.event_idx] if self.event_idx < len(self.events) else None
            
            # Filter valid active services
            self.active_services.sort()
            next_completion = self.active_services[0] if self.active_services else None
            
            # Determine nearest event
            candidates = []
            if next_arrival and next_arrival['time'] <= next_time:
                candidates.append((next_arrival['time'], 'ARRIVAL', next_arrival))
            if next_completion and next_completion <= next_time:
                candidates.append((next_completion, 'COMPLETED', None))
                
            if not candidates:
                break # No more events in this 10-min window
                
            # Pop earliest
            candidates.sort(key=lambda x: x[0])
            evt_time, evt_type, evt_data = candidates[0]
            
            # Update Simulated Time
            self.current_time = evt_time
            
            if evt_type == 'ARRIVAL':
                # Add to queue
                self.queue.append({'arrival': evt_time, 'service_sec': evt_data['service_sec']})
                self.event_idx += 1
                
            elif evt_type == 'COMPLETED':
                self.active_services.pop(0)
                step_served_count += 1
                self.stats['served'] += 1
                
            # Try to start service
            while len(self.active_services) < self.num_tellers and self.queue:
                cust = self.queue.pop(0)
                wait_seconds = (self.current_time - cust['arrival']).total_seconds()
                
                # Check renege (simplified probability)
                if wait_seconds > 15 * 60: # >15 mins
                    if self.rng.random() < 0.5: # 50% chance to leave
                        self.stats['total_renege'] += 1
                        continue # Reneged
                
                self.stats['total_wait'] += wait_seconds / 60.0
                step_wait_accum += wait_seconds / 60.0
                
                completion_time = self.current_time + timedelta(seconds=cust['service_sec'])
                self.active_services.append(completion_time)
        
        # Jump to end of step if we finished early
        self.current_time = next_time
        self.stats['steps'] += 1
        
        # Metrics for this step interval (averaged)
        avg_step_wait = step_wait_accum / max(1, step_served_count)
        
        done = (self.event_idx >= len(self.events) and not self.queue and not self.active_services)
        
        return self._get_state(avg_step_wait), avg_step_wait, done

    def _get_state(self, current_wait=0.0):
        # Calculate state based on snapshot
        q_len = len(self.queue)
        
        # Fatigue increases with queue
        self.fatigue = min(1.0, self.fatigue + (q_len * 0.001))
        if q_len == 0: self.fatigue = max(0.2, self.fatigue - 0.01)
        
        return SystemState(
            num_tellers=self.num_tellers,
            current_queue=q_len,
            avg_fatigue=self.fatigue,
            max_fatigue=self.fatigue,
            burnt_out_count=0,
            lobby_anger=q_len * 0.5,
            predicted_arrivals_mean=10, # Mock
            predicted_arrivals_ucb=15,
            prediction_uncertainty=2.0,
            current_wait=current_wait,
            hour_of_day=self.current_time.hour,
            recent_renege_rate=0.0
        )

class StatisticalValidator:
    """Performs statistical validation of results."""
    
    def __init__(self, rl_results: List[Dict], baseline_results: List[Dict]):
        self.rl_results = pd.DataFrame(rl_results)
        self.baseline_results = pd.DataFrame(baseline_results)
    
    def compute_statistics(self) -> Dict:
        """Compute comprehensive statistics."""
        results = {}
        
        for metric in ['avg_wait', 'renege_rate', 'staffing_cost', 'total_cost']:
            rl_data = self.rl_results[metric].values
            baseline_data = self.baseline_results[metric].values
            
            # Descriptive statistics
            rl_mean = np.mean(rl_data)
            rl_std = np.std(rl_data)
            baseline_mean = np.mean(baseline_data)
            baseline_std = np.std(baseline_data)
            
            # Improvement
            improvement = ((baseline_mean - rl_mean) / baseline_mean) * 100
            
            # Paired t-test
            t_stat, p_value = stats.ttest_rel(rl_data, baseline_data)
            
            # Mann-Whitney U test (non-parametric)
            u_stat, u_p_value = stats.mannwhitneyu(rl_data, baseline_data, alternative='two-sided')
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt((rl_std**2 + baseline_std**2) / 2)
            cohens_d = (baseline_mean - rl_mean) / pooled_std
            
            # Confidence intervals (95%)
            rl_ci = stats.t.interval(0.95, len(rl_data)-1, loc=rl_mean, scale=stats.sem(rl_data))
            baseline_ci = stats.t.interval(0.95, len(baseline_data)-1, loc=baseline_mean, scale=stats.sem(baseline_data))
            
            results[metric] = {
                'rl_mean': rl_mean,
                'rl_std': rl_std,
                'rl_ci': rl_ci,
                'baseline_mean': baseline_mean,
                'baseline_std': baseline_std,
                'baseline_ci': baseline_ci,
                'improvement_pct': improvement,
                't_statistic': t_stat,
                'p_value': p_value,
                'u_statistic': u_stat,
                'u_p_value': u_p_value,
                'cohens_d': cohens_d,
                'effect_size_interpretation': self._interpret_effect_size(cohens_d)
            }
        
        return results
    
    def _interpret_effect_size(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def generate_report(self, output_path: Path):
        """Generate statistical report."""
        stats_results = self.compute_statistics()
        
        # Create markdown report
        report = "# Statistical Validation Report\n\n"
        report += f"Generated: {datetime.now().isoformat()}\n\n"
        report += f"Sample size: {len(self.rl_results)} episodes per agent\n\n"
        
        report += "## Results Summary\n\n"
        report += "| Metric | RL Agent | Baseline | Improvement | p-value | Cohen's d | Effect Size |\n"
        report += "|--------|----------|----------|-------------|---------|-----------|-------------|\n"
        
        for metric, data in stats_results.items():
            report += f"| {metric} | "
            report += f"{data['rl_mean']:.2f} ± {data['rl_std']:.2f} | "
            report += f"{data['baseline_mean']:.2f} ± {data['baseline_std']:.2f} | "
            report += f"{data['improvement_pct']:.1f}% | "
            report += f"{data['p_value']:.4f} | "
            report += f"{data['cohens_d']:.2f} | "
            report += f"{data['effect_size_interpretation']} |\n"
        
        report += "\n## Interpretation\n\n"
        
        for metric, data in stats_results.items():
            if data['p_value'] < 0.05:
                report += f"- **{metric}**: Statistically significant difference (p < 0.05) "
                report += f"with {data['effect_size_interpretation']} effect size. "
                report += f"RL agent shows {data['improvement_pct']:.1f}% improvement.\n"
        
        report += "\n## Detailed Statistics\n\n"
        report += "```json\n"
        report += json.dumps(stats_results, indent=2, default=str)
        report += "\n```\n"
        
        # Save report
        with open(output_path / "statistical_report.md", 'w') as f:
            f.write(report)
        
        print(f"Statistical report saved to {output_path / 'statistical_report.md'}")
    
    def plot_results(self, output_path: Path):
        """Generate visualization plots."""
        sns.set_style("whitegrid")
        
        # Create comparison plots
        metrics = ['avg_wait', 'renege_rate', 'total_cost']
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            # Box plots
            data_to_plot = [
                self.rl_results[metric].values,
                self.baseline_results[metric].values
            ]
            bp = ax.boxplot(data_to_plot, labels=['RL Agent', 'Baseline'],
                           patch_artist=True)
            
            # Color boxes
            bp['boxes'][0].set_facecolor('lightblue')
            bp['boxes'][1].set_facecolor('lightcoral')
            
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'comparison_boxplots.png', dpi=300)
        print(f"Comparison plots saved to {output_path / 'comparison_boxplots.png'}")
        
        # Learning curve
        fig, ax = plt.subplots(figsize=(10, 6))
        
        window = 10
        rl_smoothed = pd.Series(self.rl_results['total_cost']).rolling(window).mean()
        baseline_smoothed = pd.Series(self.baseline_results['total_cost']).rolling(window).mean()
        
        ax.plot(rl_smoothed, label='RL Agent', color='blue', linewidth=2)
        ax.plot(baseline_smoothed, label='Baseline', color='red', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Cost')
        ax.set_title('Learning Curve: Total Cost Over Episodes')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'learning_curve.png', dpi=300)
        print(f"Learning curve saved to {output_path / 'learning_curve.png'}")

class ExperimentRunner:
    def __init__(self, n_episodes: int = 100, seed: int = 42, trace_df=None):
        self.n_episodes = n_episodes
        self.seed = seed
        self.trace_df = trace_df
        self.rl_agent = RLOptimizationAgent()
        self.baseline_agent = TraditionalBaselineAgent()
        self.comparator = PerformanceComparator()
        self.rl_episode_results = []
        self.baseline_episode_results = []
        
    def run_experiments(self):
        print(f"Running {self.n_episodes} episodes...")
        
        for episode in range(self.n_episodes):
            episode_seed = self.seed + episode
            
            # --- Select Environment (Synthetic or Trace) ---
            if self.trace_df is not None:
                env_rl = TraceValidationEnvironment(episode_seed, self.trace_df)
            else:
                env_rl = ValidationEnvironment(episode_seed)
                
            state = env_rl.reset()
            rl_metrics = {'avg_wait': 0, 'renege_rate': 0, 'staffing_cost': 0, 'steps': 0}
            
            done = False
            while not done:
                action_idx, action_name = self.rl_agent.select_action(state, training=True)
                next_state, wait, done = env_rl.step(action_name)
                
                reward = self.rl_agent.compute_reward(state, action_name, next_state)
                self.rl_agent.store_experience(state, action_idx, reward, next_state, done)
                self.rl_agent.train_step()
                
                rl_metrics['avg_wait'] += wait
                rl_metrics['renege_rate'] += next_state.recent_renege_rate
                rl_metrics['staffing_cost'] += next_state.num_tellers * 50
                rl_metrics['steps'] += 1
                state = next_state
            
            if episode % 10 == 0:
                self.rl_agent.update_target_network()
            
            # Final RL Metrics
            steps = max(1, rl_metrics['steps'])
            # Get served count
            served = getattr(env_rl, 'total_served', env_rl.stats.get('served', 0) if hasattr(env_rl, 'stats') else 0)
            
            final_rl = {
                'avg_wait': rl_metrics['avg_wait'] / steps,
                'renege_rate': (rl_metrics['renege_rate'] / steps) * 100,
                'staffing_cost': rl_metrics['staffing_cost'] / steps,
                'avg_tellers': (rl_metrics['staffing_cost'] / steps) / 50.0,
                'served': served,
                'total_cost': (rl_metrics['staffing_cost'] / steps) + (rl_metrics['avg_wait'] / steps)*5
            }
            self.rl_episode_results.append(final_rl)
            self.comparator.record_rl(final_rl)
            
            # --- Run Baseline ---
            if self.trace_df is not None:
                env_bl = TraceValidationEnvironment(episode_seed, self.trace_df)
            else:
                env_bl = ValidationEnvironment(episode_seed)
            
            state = env_bl.reset()
            bl_metrics = {'avg_wait': 0, 'renege_rate': 0, 'staffing_cost': 0, 'steps': 0}
            
            done = False
            while not done:
                action_name, _ = self.baseline_agent.decide(state)
                next_state, wait, done = env_bl.step(action_name)
                
                bl_metrics['avg_wait'] += wait
                bl_metrics['renege_rate'] += next_state.recent_renege_rate
                bl_metrics['staffing_cost'] += next_state.num_tellers * 50
                bl_metrics['steps'] += 1
                state = next_state

            steps = max(1, bl_metrics['steps'])
            served_bl = getattr(env_bl, 'total_served', env_bl.stats.get('served', 0) if hasattr(env_bl, 'stats') else 0)

            final_bl = {
                'avg_wait': bl_metrics['avg_wait'] / steps,
                'renege_rate': (bl_metrics['renege_rate'] / steps) * 100,
                'staffing_cost': bl_metrics['staffing_cost'] / steps,
                'avg_tellers': (bl_metrics['staffing_cost'] / steps) / 50.0,
                'served': served_bl,
                'total_cost': (bl_metrics['staffing_cost'] / steps) + (bl_metrics['avg_wait'] / steps)*5
            }
            self.baseline_episode_results.append(final_bl)
            self.comparator.record_baseline(final_bl)

            if (episode + 1) % 20 == 0:
                print(f"Episode {episode+1}: RL Cost={final_rl['total_cost']:.0f} vs Baseline={final_bl['total_cost']:.0f}")

        print("\nExperiments complete!")

def main():
    parser = argparse.ArgumentParser(description="Run validation experiments")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    parser.add_argument("--trace_file", type=str, default=None, help="Path to real-world trace data (CSV)")
    
    args = parser.parse_args()
    
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    trace_df = None
    if args.trace_file:
        print(f"[LOAD] Loading Trace Data from {args.trace_file}...")
        df = pd.read_csv(args.trace_file)
        # Pre-process once
        df['parsed_arrival'] = df['arrival_time'].apply(parse_weird_date)
        df['parsed_finish'] = pd.to_datetime(df['finish_time'])
        df['inferred_service_min'] = (df['parsed_finish'] - df['parsed_arrival']).dt.total_seconds()/60.0 - df['wait_time']
        df['inferred_service_min'] = df['inferred_service_min'].clip(lower=0.5)
        trace_df = df
        print(f"   Loaded {len(df)} events. Starting Trace-Driven Validation.")

    runner = ExperimentRunner(n_episodes=args.episodes, seed=args.seed, trace_df=trace_df)
    runner.run_experiments()
    
    # Statistical Validation & Plots
    validator = StatisticalValidator(runner.rl_episode_results, runner.baseline_episode_results)
    
    # Generate report
    validator.generate_report(output_path)
    
    # Generate plots
    validator.plot_results(output_path)
    
    # Export comparison data
    runner.comparator.export_for_paper(str(output_path / 'comparison_data.md'))
    
    # Save trained RL model
    model_path = Path("trained_model.pth")
    runner.rl_agent.save_model(str(model_path))
    print(f"\n[SUCCESS] Trained RL model saved to: {model_path}")
    
    # Save raw data
    pd.DataFrame(runner.rl_episode_results).to_csv(output_path / 'rl_results.csv', index=False)
    pd.DataFrame(runner.baseline_episode_results).to_csv(output_path / 'baseline_results.csv', index=False)
    
    print("\n" + "="*60)
    print("VALIDATION COMPLETE!")
    print(f"Results saved to: {output_path}")
    print("Dashboard will automatically visualize this data.")
    print("="*60)

if __name__ == "__main__":
    main()
