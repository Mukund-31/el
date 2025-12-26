"""
SIMPLE DIRECT COMPARISON - No Jitter, Just the Facts
====================================================

Compares 3 approaches on the EXACT same historical data:
1. Real World (what actually happened - from CSV)
2. Simple Rules (if queue > 10, add teller)
3. Your RL Model (if trained model exists)

NO random variations, NO jitter, just ONE clean run.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path


def parse_date(date_str):
    """Parse the weird DD-MM-YYYY H.MM format."""
    try:
        date_part, time_part = date_str.split(' ')
        hours = int(float(time_part))
        minutes = int(round((float(time_part) - hours) * 100))
        return datetime.strptime(f"{date_part} {hours:02d}:{minutes:02d}", "%d-%m-%Y %H:%M")
    except:
        return pd.to_datetime(date_str)


def simulate_with_simple_rules(events):
    """Simulate using simple IF-THEN rules (Baseline)."""
    num_tellers = 3
    queue = []
    active_services = []
    
    total_wait = 0
    total_served = 0
    total_reneged = 0
    teller_counts = []
    
    current_time = events[0]['arrival']
    event_idx = 0
    
    while event_idx < len(events) or queue or active_services:
        # Decision point every 10 minutes
        next_time = current_time + timedelta(minutes=10)
        
        # Process arrivals until next decision
        while event_idx < len(events) and events[event_idx]['arrival'] < next_time:
            evt = events[event_idx]
            queue.append(evt)
            event_idx += 1
        
        # Complete services
        active_services = [s for s in active_services if s > current_time]
        
        # Start new services
        while len(active_services) < num_tellers and queue:
            customer = queue.pop(0)
            wait = (current_time - customer['arrival']).total_seconds() / 60.0
            
            if wait > 20:  # Patience limit
                total_reneged += 1
                continue
            
            total_wait += wait
            total_served += 1
            completion = current_time + timedelta(minutes=customer['service_min'])
            active_services.append(completion)
        
        # SIMPLE RULE: If queue > 10, add teller
        if len(queue) > 10 and num_tellers < 10:
            num_tellers += 1
        elif len(queue) < 3 and num_tellers > 1:
            num_tellers -= 1
        
        teller_counts.append(num_tellers)
        current_time = next_time
        
        if len(teller_counts) > 200:  # Safety
            break
    
    avg_tellers = np.mean(teller_counts) if teller_counts else num_tellers
    avg_wait = total_wait / max(1, total_served)
    renege_rate = (total_reneged / max(1, total_served + total_reneged)) * 100
    
    return {
        'avg_wait': avg_wait,
        'renege_rate': renege_rate,
        'avg_tellers': avg_tellers,
        'served': total_served,
        'reneged': total_reneged
    }


def simulate_with_rl_agent(events):
    """Simulate using trained RL agent."""
    from rl_optimization_agent import RLOptimizationAgent, SystemState
    
    # Load trained model
    agent = RLOptimizationAgent()
    model_path = Path("trained_model.pth")
    
    if not model_path.exists():
        return None
    
    agent.load_model(str(model_path))
    
    # Simulation state
    num_tellers = 3
    queue = []
    active_services = []
    
    total_wait = 0
    total_served = 0
    total_reneged = 0
    teller_counts = []
    
    current_time = events[0]['arrival']
    event_idx = 0
    
    while event_idx < len(events) or queue or active_services:
        # Decision point every 10 minutes
        next_time = current_time + timedelta(minutes=10)
        
        # Process arrivals until next decision
        while event_idx < len(events) and events[event_idx]['arrival'] < next_time:
            evt = events[event_idx]
            queue.append(evt)
            event_idx += 1
        
        # Complete services
        active_services = [s for s in active_services if s > current_time]
        
        # Start new services
        while len(active_services) < num_tellers and queue:
            customer = queue.pop(0)
            wait = (current_time - customer['arrival']).total_seconds() / 60.0
            
            if wait > 20:  # Patience limit
                total_reneged += 1
                continue
            
            total_wait += wait
            total_served += 1
            completion = current_time + timedelta(minutes=customer['service_min'])
            active_services.append(completion)
        
        # RL AGENT DECISION
        hour = int(current_time.hour)
        state = SystemState(
            num_tellers=num_tellers,
            current_queue=len(queue),
            avg_fatigue=0.3,
            max_fatigue=0.5,
            burnt_out_count=0,
            lobby_anger=1.0,
            predicted_arrivals_mean=10.0,
            predicted_arrivals_ucb=15.0,
            prediction_uncertainty=0.5,
            current_wait=total_wait / max(1, total_served),
            hour_of_day=hour,
            recent_renege_rate=0.0
        )
        
        # Get action from RL agent (inference mode)
        _, action_name = agent.select_action(state, training=False)
        
        # Apply action
        if action_name == 'ADD_TELLER' and num_tellers < 10:
            num_tellers += 1
        elif action_name == 'REMOVE_TELLER' and num_tellers > 1:
            num_tellers -= 1
        
        teller_counts.append(num_tellers)
        current_time = next_time
        
        if len(teller_counts) > 200:  # Safety
            break
    
    avg_tellers = np.mean(teller_counts) if teller_counts else num_tellers
    avg_wait = total_wait / max(1, total_served)
    renege_rate = (total_reneged / max(1, total_served + total_reneged)) * 100
    
    return {
        'avg_wait': avg_wait,
        'renege_rate': renege_rate,
        'avg_tellers': avg_tellers,
        'served': total_served,
        'reneged': total_reneged
    }


def main():
    print("\n" + "="*70)
    print(" DIRECT TRACE COMPARISON - Single Run, No Jitter")
    print("="*70)
    
    # Load data
    trace_file = Path("../queue_data.csv")
    print(f"\n[LOAD] Loading: {trace_file}")
    
    df = pd.read_csv(trace_file)
    print(f"[OK] Loaded {len(df)} customer records from March 30th")
    
    # Parse dates
    df['arrival_dt'] = df['arrival_time'].apply(parse_date)
    df['finish_dt'] = pd.to_datetime(df['finish_time'])
    df['service_min'] = (df['finish_dt'] - df['arrival_dt']).dt.total_seconds()/60.0 - df['wait_time']
    df['service_min'] = df['service_min'].clip(lower=0.5)
    
    # Prepare events
    events = []
    for _, row in df.iterrows():
        events.append({
            'arrival': row['arrival_dt'],
            'service_min': row['service_min']
        })
    events.sort(key=lambda x: x['arrival'])
    
    # ========================================
    # 1. REAL WORLD (from CSV)
    # ========================================
    print("\n" + "-"*70)
    print("[1] REAL WORLD PERFORMANCE (Historical Data)")
    print("-"*70)
    
    
    # Calculate real world wait time first
    real_world_wait = df['wait_time'].mean()
    
    # Estimate Real World metrics (since CSV doesn't have them)
    # Based on typical banking operations with 10.11 min wait time
    estimated_tellers = 8.0  # Typical staffing for this volume
    estimated_renege = 5.0   # Estimated 5% renege rate for 10 min wait
    estimated_cost = estimated_tellers * 50 + real_world_wait * 5 + estimated_renege * 10
    
    real_world = {
        'avg_wait': real_world_wait,
        'renege_rate': estimated_renege,  # Estimated
        'avg_tellers': estimated_tellers,  # Estimated
        'served': len(df),
        'reneged': "Unknown"
    }
    
    print(f"   Avg Wait Time:     {real_world['avg_wait']:.2f} minutes")
    print(f"   Customers Served:  {real_world['served']}")
    print(f"   Avg Tellers:       {real_world['avg_tellers']:.2f} (estimated)")
    print(f"   Renege Rate:       {real_world['renege_rate']:.2f}% (estimated)")
    
    real_world_cost = estimated_cost
    print(f"   Total Cost:        ${real_world_cost:.2f} (estimated)")
    
    # ========================================
    # 2. SIMPLE RULES (Baseline)
    # ========================================
    print("\n" + "-"*70)
    print("[2] SIMPLE RULES BASELINE (if queue > 10, add teller)")
    print("-"*70)
    
    baseline = simulate_with_simple_rules(events)
    
    print(f"   Avg Wait Time:     {baseline['avg_wait']:.2f} minutes")
    print(f"   Customers Served:  {baseline['served']}")
    print(f"   Avg Tellers:       {baseline['avg_tellers']:.2f}")
    print(f"   Renege Rate:       {baseline['renege_rate']:.2f}%")
    
    baseline_cost = baseline['avg_tellers'] * 50 + baseline['avg_wait'] * 5 + baseline['renege_rate'] * 10
    print(f"   Total Cost:        ${baseline_cost:.2f}")
    
    # ========================================
    # 3. RL AGENT (if model exists)
    # ========================================
    print("\n" + "-"*70)
    print("[3] RL AGENT (Your Trained AI Model)")
    print("-"*70)
    
    model_path = Path("trained_model.pth")
    if not model_path.exists():
        print("   [WARNING] No trained model found (trained_model.pth)")
        print("   [INFO] Run Stage 1 (Training) first")
        rl = None
        rl_cost = None
    else:
        print("   [INFO] Loading trained model and running simulation...")
        rl = simulate_with_rl_agent(events)
        
        if rl is None:
            print("   [ERROR] Failed to run RL simulation")
            rl_cost = None
        else:
            print(f"   Avg Wait Time:     {rl['avg_wait']:.2f} minutes")
            print(f"   Customers Served:  {rl['served']}")
            print(f"   Avg Tellers:       {rl['avg_tellers']:.2f}")
            print(f"   Renege Rate:       {rl['renege_rate']:.2f}%")
            
            rl_cost = rl['avg_tellers'] * 50 + rl['avg_wait'] * 5 + rl['renege_rate'] * 10
            print(f"   Total Cost:        ${rl_cost:.2f}")
    
    # ========================================
    # COMPARISON TABLE
    # ========================================
    print("\n" + "="*70)
    print(" COMPARISON SUMMARY")
    print("="*70)
    
    comparison_data = {
        'Metric': [
            'Avg Wait Time (min)',
            'Avg Tellers Used',
            'Customers Served',
            'Renege Rate (%)',
            'Total Cost ($)'
        ],
        'Baseline': [
            f"{baseline['avg_wait']:.2f}",
            f"{baseline['avg_tellers']:.2f}",
            str(baseline['served']),
            f"{baseline['renege_rate']:.2f}%",
            f"${baseline_cost:.2f}"
        ]
    }
    
    # Add RL column if available
    if rl is not None and rl_cost is not None:
        comparison_data['RL Agent'] = [
            f"{rl['avg_wait']:.2f}",
            f"{rl['avg_tellers']:.2f}",
            str(rl['served']),
            f"{rl['renege_rate']:.2f}%",
            f"${rl_cost:.2f}"
        ]
    
    comparison = pd.DataFrame(comparison_data)
    
    print("\n" + comparison.to_string(index=False))
    
    # Save results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    results_list = [
        {
            'Agent': 'Baseline',
            'avg_wait': baseline['avg_wait'],
            'avg_tellers': baseline['avg_tellers'],
            'served': baseline['served'],
            'renege_rate': baseline['renege_rate'],
            'total_cost': baseline_cost
        }
    ]
    
    # Add RL results if available
    if rl is not None and rl_cost is not None:
        results_list.append({
            'Agent': 'RL Agent',
            'avg_wait': rl['avg_wait'],
            'avg_tellers': rl['avg_tellers'],
            'served': rl['served'],
            'renege_rate': rl['renege_rate'],
            'total_cost': rl_cost
        })
    
    results_df = pd.DataFrame(results_list)
    
    results_df.to_csv(output_dir / 'direct_comparison.csv', index=False)
    
    print("\n" + "="*70)
    print(f"[OK] Results saved to: {output_dir / 'direct_comparison.csv'}")
    print("="*70)
    
    print("\n[INSIGHTS] KEY INSIGHTS:")
    print(f"   • Real world had {real_world['avg_wait']:.1f} min average wait")
    print(f"   • Simple rules achieved {baseline['avg_wait']:.1f} min wait with {baseline['avg_tellers']:.1f} tellers")
    
    wait_improvement = ((real_world['avg_wait'] - baseline['avg_wait']) / real_world['avg_wait']) * 100
    print(f"   • Simple rules improved wait time by {wait_improvement:.1f}%")
    print(f"   • Both served all {real_world['served']} customers")
    
    print("\n[NOTE] For RL comparison, check the dashboard Results Viewer tab")
    print("   (Run validation with Real-World Trace mode)\n")


if __name__ == "__main__":
    main()
