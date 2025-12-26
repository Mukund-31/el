"""
Direct Trace Comparison - Simple & Clear
==========================================

Takes the EXACT historical data and runs it through:
1. Real World (what actually happened)
2. Baseline Agent (simple rules)
3. RL Agent (your AI)

No jitter, no variations, just direct comparison.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Import your agents
from rl_optimization_agent import RLOptimizationAgent, SystemState
from baseline_agent import BaselineAgent


def parse_weird_date(date_str):
    """Parse the DD-MM-YYYY H.MM format."""
    try:
        return pd.to_datetime(date_str, format='%d-%m-%Y %H.%M')
    except:
        return pd.to_datetime(date_str)


class DirectTraceSimulator:
    """Simulates the exact historical trace without any jitter."""
    
    def __init__(self, trace_df, agent_name="Agent"):
        self.trace_df = trace_df.copy()
        self.agent_name = agent_name
        
        # Metrics
        self.total_wait = 0
        self.total_served = 0
        self.total_reneged = 0
        self.teller_history = []
        
        # Simulation state
        self.num_tellers = 3
        self.queue = []
        self.active_services = []
        self.current_time = None
        
    def run(self, agent):
        """Run the simulation with the given agent."""
        print(f"\n{'='*60}")
        print(f"Running {self.agent_name}...")
        print(f"{'='*60}")
        
        # Sort events by arrival time
        events = []
        for _, row in self.trace_df.iterrows():
            events.append({
                'time': row['parsed_arrival'],
                'service_duration': row['inferred_service_min']
            })
        
        events.sort(key=lambda x: x['time'])
        
        # Start simulation
        self.current_time = events[0]['time']
        event_idx = 0
        step_count = 0
        
        while event_idx < len(events) or self.queue or self.active_services:
            step_count += 1
            
            # Decision point every 10 minutes
            next_decision_time = self.current_time + timedelta(minutes=10)
            
            # Process all events until next decision point
            while event_idx < len(events) and events[event_idx]['time'] < next_decision_time:
                evt = events[event_idx]
                self.current_time = evt['time']
                
                # Customer arrives
                self.queue.append({
                    'arrival': evt['time'],
                    'service_min': evt['service_duration']
                })
                event_idx += 1
                
                # Try to start service
                self._process_queue()
            
            # Complete any services that finished
            self._complete_services()
            
            # Agent makes decision
            if agent is not None:
                state = self._get_state()
                if hasattr(agent, 'select_action'):
                    # RL Agent
                    _, action_name = agent.select_action(state, training=False)
                else:
                    # Baseline Agent
                    action_name, _ = agent.decide(state)
                
                # Apply action
                if action_name == 'ADD_TELLER' and self.num_tellers < 10:
                    self.num_tellers += 1
                elif action_name == 'REMOVE_TELLER' and self.num_tellers > 1:
                    self.num_tellers -= 1
            
            self.teller_history.append(self.num_tellers)
            
            # Move time forward
            self.current_time = next_decision_time
            
            if step_count > 1000:  # Safety limit
                break
        
        # Calculate final metrics
        avg_tellers = np.mean(self.teller_history) if self.teller_history else self.num_tellers
        avg_wait = self.total_wait / max(1, self.total_served)
        renege_rate = (self.total_reneged / max(1, self.total_served + self.total_reneged)) * 100
        
        staffing_cost = avg_tellers * 50  # $50 per teller per hour
        wait_cost = avg_wait * 5  # $5 per minute
        renege_cost = renege_rate * 10  # $10 per percent
        total_cost = staffing_cost + wait_cost + renege_cost
        
        return {
            'avg_wait': avg_wait,
            'renege_rate': renege_rate,
            'avg_tellers': avg_tellers,
            'served': self.total_served,
            'total_cost': total_cost,
            'staffing_cost': staffing_cost
        }
    
    def _process_queue(self):
        """Start service for waiting customers."""
        while len(self.active_services) < self.num_tellers and self.queue:
            customer = self.queue.pop(0)
            wait_time = (self.current_time - customer['arrival']).total_seconds() / 60.0
            
            # Check if customer reneges (waits too long)
            if wait_time > 20:  # 20 minute patience threshold
                self.total_reneged += 1
                continue
            
            self.total_wait += wait_time
            self.total_served += 1
            
            # Schedule service completion
            completion_time = self.current_time + timedelta(minutes=customer['service_min'])
            self.active_services.append(completion_time)
    
    def _complete_services(self):
        """Complete any services that have finished."""
        self.active_services = [t for t in self.active_services if t > self.current_time]
    
    def _get_state(self):
        """Get current state for agent decision."""
        return SystemState(
            num_tellers=self.num_tellers,
            current_queue=len(self.queue),
            avg_fatigue=0.3,
            max_fatigue=0.5,
            avg_anger=1.0,
            max_anger=2.0,
            predicted_arrivals=10.0,
            uncertainty=0.5,
            hour_of_day=self.current_time.hour,
            recent_renege_rate=0.0,
            staffing_cost=self.num_tellers * 50,
            wait_penalty=5.0
        )


def main():
    print("\n" + "="*60)
    print("DIRECT TRACE COMPARISON - No Jitter, Just Facts")
    print("="*60)
    
    # Load trace data
    trace_file = Path("../queue_data.csv")
    if not trace_file.exists():
        print(f"Error: {trace_file} not found!")
        return
    
    print(f"\nLoading trace data from {trace_file}...")
    df = pd.read_csv(trace_file)
    
    # Preprocess
    df['parsed_arrival'] = df['arrival_time'].apply(parse_weird_date)
    df['parsed_finish'] = pd.to_datetime(df['finish_time'])
    df['inferred_service_min'] = (df['parsed_finish'] - df['parsed_arrival']).dt.total_seconds()/60.0 - df['wait_time']
    df['inferred_service_min'] = df['inferred_service_min'].clip(lower=0.5)
    
    print(f"Loaded {len(df)} customer events from March 30th")
    
    # Get real-world metrics
    real_world = {
        'avg_wait': df['wait_time'].mean(),
        'renege_rate': 0.0,  # Not in dataset
        'avg_tellers': float('nan'),  # Unknown
        'served': len(df),
        'total_cost': float('nan'),  # Can't calculate without teller count
        'staffing_cost': float('nan')
    }
    
    print(f"\nReal World Performance (from CSV):")
    print(f"  Avg Wait Time: {real_world['avg_wait']:.2f} minutes")
    print(f"  Customers Served: {real_world['served']}")
    
    # Run Baseline Agent
    baseline_sim = DirectTraceSimulator(df, "Baseline Agent")
    baseline_agent = BaselineAgent()
    baseline_results = baseline_sim.run(baseline_agent)
    
    print(f"\nBaseline Agent Performance:")
    print(f"  Avg Wait Time: {baseline_results['avg_wait']:.2f} minutes")
    print(f"  Avg Tellers: {baseline_results['avg_tellers']:.2f}")
    print(f"  Customers Served: {baseline_results['served']}")
    print(f"  Renege Rate: {baseline_results['renege_rate']:.2f}%")
    print(f"  Total Cost: ${baseline_results['total_cost']:.2f}")
    
    # Run RL Agent
    rl_sim = DirectTraceSimulator(df, "RL Agent")
    rl_agent = RLOptimizationAgent()
    
    # Load trained model if exists
    model_path = Path("trained_model.pth")
    if model_path.exists():
        print("\nLoading trained RL model...")
        rl_agent.load_model(str(model_path))
    else:
        print("\nWarning: No trained model found. Using untrained agent.")
    
    rl_results = rl_sim.run(rl_agent)
    
    print(f"\nRL Agent Performance:")
    print(f"  Avg Wait Time: {rl_results['avg_wait']:.2f} minutes")
    print(f"  Avg Tellers: {rl_results['avg_tellers']:.2f}")
    print(f"  Customers Served: {rl_results['served']}")
    print(f"  Renege Rate: {rl_results['renege_rate']:.2f}%")
    print(f"  Total Cost: ${rl_results['total_cost']:.2f}")
    
    # Comparison
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    comparison_df = pd.DataFrame({
        'Metric': ['Avg Wait (min)', 'Avg Tellers', 'Customers Served', 'Renege Rate (%)', 'Total Cost ($)'],
        'Real World': [
            f"{real_world['avg_wait']:.2f}",
            "Unknown",
            f"{real_world['served']}",
            "Unknown",
            "Unknown"
        ],
        'Baseline': [
            f"{baseline_results['avg_wait']:.2f}",
            f"{baseline_results['avg_tellers']:.2f}",
            f"{baseline_results['served']}",
            f"{baseline_results['renege_rate']:.2f}",
            f"{baseline_results['total_cost']:.2f}"
        ],
        'RL Agent': [
            f"{rl_results['avg_wait']:.2f}",
            f"{rl_results['avg_tellers']:.2f}",
            f"{rl_results['served']}",
            f"{rl_results['renege_rate']:.2f}",
            f"{rl_results['total_cost']:.2f}"
        ],
        'RL vs Baseline': [
            f"{((baseline_results['avg_wait'] - rl_results['avg_wait'])/baseline_results['avg_wait']*100):+.1f}%",
            f"{baseline_results['avg_tellers'] - rl_results['avg_tellers']:+.2f}",
            f"{rl_results['served'] - baseline_results['served']:+d}",
            f"{baseline_results['renege_rate'] - rl_results['renege_rate']:+.2f}pp",
            f"{((baseline_results['total_cost'] - rl_results['total_cost'])/baseline_results['total_cost']*100):+.1f}%"
        ]
    })
    
    print("\n" + comparison_df.to_string(index=False))
    
    # Save results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    results_summary = pd.DataFrame([
        {'Agent': 'Real World', **real_world},
        {'Agent': 'Baseline', **baseline_results},
        {'Agent': 'RL', **rl_results}
    ])
    
    results_summary.to_csv(output_dir / 'direct_comparison.csv', index=False)
    print(f"\n✅ Results saved to {output_dir / 'direct_comparison.csv'}")
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)


if __name__ == "__main__":
    main()
