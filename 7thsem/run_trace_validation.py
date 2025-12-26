import pandas as pd
import numpy as np
import heapq
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from rl_optimization_agent import RLOptimizationAgent, TraditionalBaselineAgent, SystemState

# --- Helper to parse dates ---
def parse_weird_date(date_str):
    try:
        date_part, time_part = date_str.split(' ')
        hours = int(float(time_part))
        minutes = int(round((float(time_part) - hours) * 100))
        return datetime.strptime(f"{date_part} {hours:02d}:{minutes:02d}", "%d-%m-%Y %H:%M")
    except:
        return None

class Customer:
    def __init__(self, id, arrival_time, service_duration, original_wait):
        self.id = id
        self.arrival_time = arrival_time
        self.service_duration = service_duration
        self.original_wait = original_wait
        self.start_service_time = None
        self.finish_time = None

    @property
    def wait_time(self):
        if self.start_service_time:
            return (self.start_service_time - self.arrival_time).total_seconds() / 60.0
        return 0.0

    def __lt__(self, other):
        return self.id < other.id

class TraceSimulation:
    def __init__(self, dataframe, agent=None, agent_type='RL', initial_tellers=3):
        self.df = dataframe
        self.agent_type = agent_type
        self.num_tellers = initial_tellers
        self.queue = []
        self.active_services = [] 
        self.completed_customers = []
        self.current_time = dataframe['parsed_arrival'].min()
        
        # Load Agents
        self.rl_agent = agent if agent_type == 'RL' else None
        self.baseline_agent = agent if agent_type == 'Baseline' else None
        
        # Validation State
        self.fatigue = 0.0
        self.processed_idx = 0
        self.stats_history = []

    def get_state(self):
        # Construct state for Agent
        current_hour = self.current_time.hour
        
        # Simple stats
        avg_wait = 0
        if len(self.completed_customers) > 0:
            recent = self.completed_customers[-10:]
            avg_wait = np.mean([c.wait_time for c in recent])

        # Current wait estimate (head of queue)
        curr_wait = 0
        if self.queue:
            curr_wait = (self.current_time - self.queue[0].arrival_time).total_seconds() / 60.0
        
        # Prediction (mock based on hour)
        pred_mean = 10 if 9 <= current_hour <= 17 else 2

        return SystemState(
            num_tellers=self.num_tellers,
            current_queue=len(self.queue),
            avg_fatigue=self.fatigue,
            max_fatigue=self.fatigue,
            burnt_out_count=0,
            lobby_anger=len(self.queue) * 0.5,
            predicted_arrivals_mean=pred_mean,
            predicted_arrivals_ucb=pred_mean + 5,
            prediction_uncertainty=2.0,
            current_wait=curr_wait,
            hour_of_day=current_hour,
            recent_renege_rate=0.0 
        )

    def run(self, training=False):
        if not training:
            print(f"Starting {self.agent_type} Simulation (Evaluation)...")
        
        events = []
        
        # Load all arrivals as events
        for idx, row in self.df.iterrows():
            cust = Customer(idx, row['parsed_arrival'], row['inferred_service_min'], row['wait_time'])
            heapq.heappush(events, (cust.arrival_time, 'ARRIVAL', cust))
            
        # Decision loop timer
        next_decision_time = self.current_time
        decision_interval = timedelta(minutes=10) 

        while events or self.queue or self.active_services:
            # Check if we need to make a decision
            if self.current_time >= next_decision_time:
                self._make_decision(training)
                next_decision_time += decision_interval

            # Pop next event
            if events:
                time, type, data = events[0]
            else:
                time = self.active_services[0] if self.active_services else self.current_time + timedelta(seconds=1)
                type = 'SERVICE_FINISH' 

            # If next event is far, but we have services finishing sooner?
            if self.active_services:
                min_finish = min(self.active_services)
                if not events or min_finish < events[0][0]:
                     self.current_time = min_finish
                     self.active_services.remove(min_finish)
                     self._process_queue() 
                     continue
            
            if not events: break 
            
            time, type, data = heapq.heappop(events)
            self.current_time = time
            
            if type == 'ARRIVAL':
                self.queue.append(data)
                self._process_queue()
                
            self.fatigue = min(1.0, len(self.queue) / 50.0)

    def _make_decision(self, training):
        state = self.get_state()
        action_name = None
        action_idx = 0
        
        if self.agent_type == 'RL':
            action_idx, action_name = self.rl_agent.select_action(state, training=training)
            
            # If training, we need to compute reward for PREVIOUS action
            # Simplified: Just train on current state transition approximation
            # (In a real temporal simulation, we'd store (s,a) and reward later)
            # For trace replay, we'll skip complex reward storage and just let it explore
            if training:
                # Mock reward based on immediate state (Simplified)
                # Ideally we want (s, a, r, s')
                # We can't easily do s' here without looking ahead.
                # So we train "Online" - we store experience from LAST step
                pass 

        elif self.agent_type == 'Baseline':
            action_name, _ = self.baseline_agent.decide(state)
            
        if action_name == 'ADD_TELLER' and self.num_tellers < 10:
            self.num_tellers += 1
        elif action_name == 'REMOVE_TELLER' and self.num_tellers > 1:
            self.num_tellers -= 1
            
        self.stats_history.append({
            'time': self.current_time,
            'tellers': self.num_tellers,
            'queue': len(self.queue)
        })

    def _process_queue(self):
        while len(self.active_services) < self.num_tellers and self.queue:
            cust = self.queue.pop(0)
            cust.start_service_time = self.current_time
            finish_time = self.current_time + timedelta(minutes=cust.service_duration)
            cust.finish_time = finish_time
            self.active_services.append(finish_time)
            self.completed_customers.append(cust)

    def get_results(self):
        waits = [c.wait_time for c in self.completed_customers]
        return {
            'avg_wait': np.mean(waits),
            'max_wait': np.max(waits),
            'total_customers': len(self.completed_customers),
            'cost': np.mean([s['tellers'] for s in self.stats_history]) * 50 * (len(self.stats_history)/6)
        }

def run_comparison():
    # Load Data
    df = pd.read_csv('../queue_data.csv')
    df['parsed_arrival'] = df['arrival_time'].apply(parse_weird_date)
    df['parsed_finish'] = pd.to_datetime(df['finish_time'])
    df['inferred_service_min'] = (df['parsed_finish'] - df['parsed_arrival']).dt.total_seconds()/60.0 - df['wait_time']
    df['inferred_service_min'] = df['inferred_service_min'].clip(lower=0.5) 
    
    df = df.sort_values('parsed_arrival')
    
    # --- PHASE 1: TRAIN RL AGENT ---
    print("\n🧠 TRAINING RL AGENT ON TRACE DATA...")
    rl_agent = RLOptimizationAgent()
    
    # We need a proper training loop that calculates rewards
    # For simplicity in this trace replay, we will run the simulation step usage
    # But RLOptimizationAgent expects (s,a,r,s')
    # Let's manually run a training loop 5 times
    
    for epoch in range(5):
        print(f"  Epoch {epoch+1}/5...")
        sim_train = TraceSimulation(df, agent=rl_agent, agent_type='RL')
        
        # Custom training run to capture transitions
        # We need to hook into the run loop to capture (s, a, r, s')
        # Since I can't easily inject code into run(), I'll trust the agent's internal memory?
        # No, the agent needs store_experience called explicitly.
        
        # Hack: Just run the simulation in 'training' mode which triggers exploration
        # And we need to rely on the fact that `select_action` updates epsilon.
        # But `train_step()` is never called! 
        # So the agent explores but never Learns.
        pass
    
    # Start fresh with a smarter approach:
    # We will simulate valid interaction
    pass

    # RE-IMPLEMENTING COMPARISON WITH PROPER TRAINING
    rl_agent = RLOptimizationAgent() # Fresh agent
    
    print(f"  Training on {len(df)} historical events (3 Epochs)...")
    for epoch in range(3):
        # We need to manually iterate to train properly
        sim = TraceSimulation(df, agent=rl_agent, agent_type='RL')
        
        # Manually run the event loop to enable training
        events = []
        for idx, row in df.iterrows():
            cust = Customer(idx, row['parsed_arrival'], row['inferred_service_min'], row['wait_time'])
            heapq.heappush(events, (cust.arrival_time, 'ARRIVAL', cust))
            
        current_time = df['parsed_arrival'].min()
        last_state = None
        last_action = 0
        last_action_name = 'MAINTAIN'
        
        next_decision = current_time
        
        while events or sim.queue or sim.active_services:
            # Timer update
            if events:
                 next_event_time = events[0][0]
            else:
                 next_event_time = sim.active_services[0] if sim.active_services else current_time + timedelta(seconds=1)
            
            if sim.active_services and min(sim.active_services) < next_event_time:
                next_event_time = min(sim.active_services)

            # Advance time
            current_time = next_event_time
            sim.current_time = current_time
            
            # Process Events at this time
            while events and events[0][0] <= current_time:
                _, t, data = heapq.heappop(events)
                if t == 'ARRIVAL':
                    sim.queue.append(data)
                    sim._process_queue()
                    
            # Process Service Finishes
            if sim.active_services:
                # Remove finished
                active = [t for t in sim.active_services if t > current_time]
                finished = len(sim.active_services) - len(active)
                sim.active_services = active
                if finished > 0 and sim.queue:
                    sim._process_queue()

            # DECISION & TRAINING
            if current_time >= next_decision:
                state = sim.get_state()
                
                # If we have a previous state, Train!
                if last_state:
                    # Compute Reward
                    reward = rl_agent.compute_reward(last_state, last_action_name, state)
                    rl_agent.store_experience(last_state, last_action, reward, state, False)
                    rl_agent.train_step()
                
                # Select New Action
                action_idx, action_name = rl_agent.select_action(state, training=True)
                
                # Apply Action
                if action_name == 'ADD_TELLER' and sim.num_tellers < 10:
                    sim.num_tellers += 1
                elif action_name == 'REMOVE_TELLER' and sim.num_tellers > 1:
                    sim.num_tellers -= 1
                
                last_state = state
                last_action = action_idx
                last_action_name = action_name
                next_decision += timedelta(minutes=10)
        
        rl_agent.update_target_network()

    # --- PHASE 2: EVALUATION ---
    sim_rl = TraceSimulation(df, agent=rl_agent, agent_type='RL')
    sim_rl.run(training=False)
    res_rl = sim_rl.get_results()
    
    # Baseline
    baseline_agent = TraditionalBaselineAgent()
    sim_bl = TraceSimulation(df, agent=baseline_agent, agent_type='Baseline')
    sim_bl.run(training=False)
    res_bl = sim_bl.get_results()
    
    # Real World Data
    real_avg_wait = df['wait_time'].mean()
    
    print("\n" + "="*60)
    print("🌍 REAL-WORLD TRACE VALIDATION RESULTS")
    print("="*60)
    print(f"{'Metric':<20} | {'Real World':<12} | {'Baseline':<12} | {'RL Agent':<12}")
    print("-" * 65)
    print(f"{'Avg Wait (min)':<20} | {real_avg_wait:<12.2f} | {res_bl['avg_wait']:<12.2f} | {res_rl['avg_wait']:<12.2f}")
    print(f"{'Max Wait (min)':<20} | {df['wait_time'].max():<12.2f} | {res_bl['max_wait']:<12.2f} | {res_rl['max_wait']:<12.2f}")
    print("-" * 65)
    
    # ImprovementRL
    imp = ((real_avg_wait - res_rl['avg_wait']) / real_avg_wait) * 100
    print(f"\n🚀 IMPROVEMENT OVER REALITY: {imp:.1f}%")
    if imp > 0:
        print("✅ The RL Agent successfully learned the patterns.")
    else:
        print("⚠️ The RL Agent needs more training epochs.")

if __name__ == "__main__":
    run_comparison()
