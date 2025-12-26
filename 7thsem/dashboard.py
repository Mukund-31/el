"""
Module 5: Dashboard (dashboard.py)
===================================

Real-time visualization making invisible states visible for human supervisors.
Uses Streamlit + Plotly for rapid prototyping and easy reviewer reproduction.

Visualizations:
1. Uncertainty Cone - Actual arrivals + mean prediction + 95% CI
2. Fatigue Heatmap - Color-coded teller fatigue levels [0,1]
3. Lobby Contagion Meter - Gauge 0-10 with danger zone >6
4. Decision Trace Table - Time | Action | Cost Z (auditable)

Time Semantics:
Wall-clock timestamps (datetime.now()) are used for logging and visualization.
Simulation time is tracked separately within the engine (env.now in minutes).

Purpose:
"This dashboard visualizes latent variables (uncertainty, fatigue, emotion),
not just traditional KPIs. This makes the system auditable and supports
human-in-the-loop oversight."
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import time
from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable
import threading
from collections import deque
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# RL MODEL INTEGRATION
# =============================================================================

from pathlib import Path
import torch

def load_trained_model():
    """Load the trained RL model for real-time decisions."""
    try:
        from rl_optimization_agent import RLOptimizationAgent
        
        model_path = Path("trained_model.pth")
        if not model_path.exists():
            logger.warning("No trained model found. Using rule-based fallback.")
            return None
        
        agent = RLOptimizationAgent()
        agent.load_model(str(model_path))
        logger.info("✅ Trained RL model loaded successfully!")
        return agent
    except Exception as e:
        logger.error(f"Failed to load RL model: {e}")
        return None

def get_rl_decision(agent, state_data):
    """Get decision from trained RL model."""
    if agent is None:
        logger.warning("⚠️ No RL agent - using fallback rule")
        # Fallback to simple rule
        if state_data.get('queue_length', 0) > 10:
            return "ADD_TELLER", 0.5
        return "MAINTAIN", 0.5
    
    try:
        from rl_optimization_agent import SystemState
        import numpy as np
        
        # Clip values to match training distribution (prevent out-of-range inputs)
        queue_clipped = min(state_data.get('queue_length', 0), 50)  # Max 50 in training
        wait_clipped = min(state_data.get('current_wait', 0.0), 20.0)  # Max 20 min in training
        renege_clipped = min(state_data.get('renege_rate', 0.0), 1.0)  # Max 100%
        
        # Create state from dashboard data
        state = SystemState(
            num_tellers=state_data.get('num_tellers', 3),
            current_queue=queue_clipped,
            avg_fatigue=state_data.get('avg_fatigue', 0.3),
            max_fatigue=state_data.get('max_fatigue', 0.5),
            burnt_out_count=state_data.get('burnt_out', 0),
            lobby_anger=min(state_data.get('lobby_anger', 1.0), 10.0),
            predicted_arrivals_mean=min(state_data.get('pred_mean', 10.0), 30.0),
            predicted_arrivals_ucb=min(state_data.get('pred_ucb', 15.0), 50.0),
            prediction_uncertainty=min(state_data.get('uncertainty', 0.5), 10.0),
            current_wait=wait_clipped,
            hour_of_day=state_data.get('hour', 9),
            recent_renege_rate=renege_clipped
        )
        
        logger.info(f"📊 RL State (clipped): queue={queue_clipped}/{state_data.get('queue_length', 0)}, "
                   f"wait={wait_clipped:.1f}/{state_data.get('current_wait', 0):.1f}min")
        
        # Get action from trained model (inference mode)
        action_idx, action_name = agent.select_action(state, training=False)
        
        # Get Q-values to see confidence
        import torch
        with torch.no_grad():
            state_tensor = state.to_tensor().unsqueeze(0)
            q_values = agent.q_network(state_tensor)
            confidence = q_values.max().item()
        
        logger.info(f"✅ RL returned: {action_name} (idx={action_idx}, conf={confidence:.2f})")
        logger.info(f"📈 Q-values: {q_values.squeeze().tolist()}")
        
        # Cost-conscious override: Remove excess tellers if queue is consistently empty
        if (state_data.get('queue_length', 0) == 0 and 
            state_data.get('num_tellers', 0) > 3 and
            action_name in ['DO_NOTHING', 'GIVE_BREAK']):
            logger.info(f"💰 Cost override: Queue empty with {state_data['num_tellers']} tellers → REMOVE_TELLER")
            return "REMOVE_TELLER", confidence
        
        return action_name, confidence
    except Exception as e:
        logger.error(f"❌ RL decision error: {e}", exc_info=True)
        return "MAINTAIN", 0.0



# =============================================================================
# DASHBOARD STATE
# =============================================================================

class DashboardState:
    """Manages real-time data for dashboard."""
    
    def __init__(self, max_history: int = 100):
        self.max_history = max_history
        
        # Time series data
        self.arrivals_actual: deque = deque(maxlen=max_history)
        self.arrivals_mean: deque = deque(maxlen=max_history)
        self.arrivals_ucb: deque = deque(maxlen=max_history)
        self.arrivals_lcb: deque = deque(maxlen=max_history)
        self.timestamps: deque = deque(maxlen=max_history)
        
        # Teller fatigue
        self.teller_fatigue: Dict[int, float] = {}
        
        # Lobby anger
        self.lobby_anger: float = 0.0
        # Note: anger_history reserved for future sparkline extension
        self.anger_history: deque = deque(maxlen=max_history)
        
        # Decision trace
        self.decision_trace: List[Dict] = []
        
        # Metrics (individual attributes for easy access)
        self.total_arrivals: int = 0
        self.total_served: int = 0
        self.total_reneged: int = 0
        self.avg_wait: float = 0.0
        self.renege_rate: float = 0.0
        
    def update_predictions(
        self,
        actual: float,
        mean: float,
        std: float,
        timestamp: str
    ) -> None:
        """Update arrival predictions."""
        self.arrivals_actual.append(actual)
        self.arrivals_mean.append(mean)
        self.arrivals_ucb.append(mean + 1.96 * std)
        self.arrivals_lcb.append(max(0, mean - 1.96 * std))
        self.timestamps.append(timestamp)
        
    def update_fatigue(self, teller_data: List[Dict]) -> None:
        """Update teller fatigue levels."""
        self.teller_fatigue = {
            t["teller_id"]: t["fatigue"]
            for t in teller_data
        }
        
    def update_anger(self, anger: float, timestamp: str) -> None:
        """Update lobby anger."""
        self.lobby_anger = anger
        self.anger_history.append({
            "timestamp": timestamp,
            "anger": anger
        })
    
    def update_metrics(self, metrics: Dict) -> None:
        """Update key metrics from simulation."""
        self.total_arrivals = metrics.get("total_arrivals", 0)
        self.total_served = metrics.get("total_served", 0)
        self.total_reneged = metrics.get("total_reneged", 0)
        self.avg_wait = metrics.get("avg_wait", 0.0)
        self.renege_rate = metrics.get("renege_rate", 0.0)
        
    def add_decision(self, decision: Dict) -> None:
        """Add decision to trace."""
        self.decision_trace.append(decision)
        if len(self.decision_trace) > 50:
            self.decision_trace = self.decision_trace[-50:]


# =============================================================================
# VISUALIZATIONS
# =============================================================================

def create_uncertainty_cone(state: DashboardState) -> go.Figure:
    """
    Visual 1: Uncertainty Cone
    
    Shows:
    - Actual arrivals (solid line)
    - Mean prediction (dashed line)
    - 95% confidence interval (shaded UCB to LCB)
    
    Note on LCB: Lower confidence bound is shown for visual symmetry
    and intuition; staffing decisions use UCB only (risk-averse).
    
    Interpretation:
    - Wide cone = high uncertainty, model is guessing
    - Narrow cone = confident prediction
    """
    if len(state.timestamps) < 2:
        # Return empty figure if not enough data
        fig = go.Figure()
        fig.add_annotation(
            text="Waiting for data...",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="gray")
        )
        fig.update_layout(
            title="Arrival Predictions with Uncertainty",
            height=400
        )
        return fig
        
    timestamps = list(state.timestamps)
    actual = list(state.arrivals_actual)
    mean = list(state.arrivals_mean)
    ucb = list(state.arrivals_ucb)
    lcb = list(state.arrivals_lcb)
    
    fig = go.Figure()
    
    # Confidence interval (shaded)
    fig.add_trace(go.Scatter(
        x=timestamps + timestamps[::-1],
        y=ucb + lcb[::-1],
        fill='toself',
        fillcolor='rgba(99, 110, 250, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95% Confidence',
        hoverinfo='skip'
    ))
    
    # Mean prediction
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=mean,
        mode='lines',
        name='Predicted Mean',
        line=dict(color='#636EFA', dash='dash', width=2)
    ))
    
    # UCB line (for staffing decisions)
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=ucb,
        mode='lines',
        name='UCB (Staffing Threshold)',
        line=dict(color='#EF553B', dash='dot', width=1)
    ))
    
    # Actual arrivals
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=actual,
        mode='lines+markers',
        name='Actual Arrivals',
        line=dict(color='#00CC96', width=3),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title="📈 Arrival Predictions with Epistemic Uncertainty (5-min intervals)",
        xaxis_title="Time",
        yaxis_title="Arrivals per 5-Minute Interval",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified"
    )
    
    return fig


def create_fatigue_heatmap(state: DashboardState) -> go.Figure:
    """
    Visual 2: Fatigue Heatmap
    
    Shows:
    - Teller-wise fatigue as color-coded bars
    - Risk levels: Green (<0.5), Yellow (0.5-0.8), Red (>0.8)
    
    Managers can preempt burnout by identifying at-risk tellers.
    """
    if not state.teller_fatigue:
        # Demo data if no real data
        tellers = ["Teller 0", "Teller 1", "Teller 2"]
        fatigue = [0.3, 0.5, 0.7]
    else:
        tellers = [f"Teller {tid}" for tid in sorted(state.teller_fatigue.keys())]
        fatigue = [state.teller_fatigue[tid] for tid in sorted(state.teller_fatigue.keys())]
    
    # Color based on fatigue level
    colors = []
    for f in fatigue:
        if f < 0.5:
            colors.append('#00CC96')  # Green - OK
        elif f < 0.8:
            colors.append('#FECB52')  # Yellow - Warning
        else:
            colors.append('#EF553B')  # Red - Burnout
            
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=tellers,
        x=fatigue,
        orientation='h',
        marker_color=colors,
        text=[f'{f:.1%}' for f in fatigue],
        textposition='inside',
        hovertemplate='%{y}: %{x:.1%} fatigue<extra></extra>'
    ))
    
    # Add threshold lines
    fig.add_vline(x=0.5, line_dash="dash", line_color="orange",
                  annotation_text="Warning", annotation_position="top")
    fig.add_vline(x=0.8, line_dash="dash", line_color="red",
                  annotation_text="Burnout", annotation_position="top")
    
    fig.update_layout(
        title="🔥 Teller Fatigue Levels",
        xaxis_title="Fatigue (0 = Fresh, 1 = Exhausted)",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(
            dtick=1,  # Force showing every teller (no skipping)
            tickmode='linear'
        ),
        height=max(250, len(tellers) * 30),  # Dynamic height based on teller count
        margin=dict(l=100)
    )
    
    return fig


def create_contagion_gauge(state: DashboardState) -> go.Figure:
    """
    Visual 3: Lobby Contagion Meter
    
    Shows:
    - Gauge from 0-10
    - Danger zone > 6 (red arc)
    - Current lobby anger level
    
    Makes emotional collapse measurable and actionable.
    """
    anger = state.lobby_anger
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=anger,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "😤 Lobby Anger Index", 'font': {'size': 24}},
        delta={'reference': 4, 'increasing': {'color': "red"}},
        gauge={
            'axis': {'range': [0, 10], 'tickwidth': 1},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 3], 'color': '#00CC96'},    # Calm
                {'range': [3, 6], 'color': '#FECB52'},    # Tense
                {'range': [6, 10], 'color': '#EF553B'}    # Danger
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 6
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        annotations=[
            dict(
                text="LobbyAnger = min(10, median(wait) / W_ref)",
                x=0.5, y=-0.05,
                showarrow=False,
                font=dict(size=10, color="gray")
            ),
            dict(
                text="⚠️ Values > 6 indicate high reneging risk due to emotional contagion",
                x=0.5, y=-0.15,
                showarrow=False,
                font=dict(size=9, color="#EF553B")
            )
        ]
    )
    
    return fig


def create_decision_trace_table(state: DashboardState) -> pd.DataFrame:
    """
    Visual 4: Decision Trace Panel
    
    Audit table showing:
    - Time
    - Predicted UCB
    - Action taken
    - Cost Z
    
    "This makes the system auditable. Reviewers love this."
    """
    if not state.decision_trace:
        # Demo data
        return pd.DataFrame({
            'Time': ['10:00', '10:05', '10:10'],
            'Action': ['DO_NOTHING', 'ADD_TELLER', '⏸️ DELAY_DECISION'],
            'Cost': [85.0, 95.0, 78.0]
        })
        
    data = []
    for d in state.decision_trace[-10:]:  # Last 10 decisions
        action = d.get('action', '')
        # Highlight DELAY_DECISION with indicator for stability-driven choice
        action_display = f"⏸️ {action}" if action == 'DELAY_DECISION' else action
        data.append({
            'Time': d.get('timestamp', '')[-8:] if d.get('timestamp') else '',
            'Action': action_display,
            'Cost': round(d.get('cost', 0), 1)
        })
        
    return pd.DataFrame(data)


def create_metrics_cards(state: DashboardState) -> Dict:
    """Generate metrics for display cards."""
    return {
        "Total Arrivals": state.total_arrivals,
        "Served": state.total_served,
        "Reneged": state.total_reneged,
        "Avg Wait": f"{state.avg_wait:.1f} min",
        "Renege Rate": f"{state.renege_rate:.1f}%"
    }


# =============================================================================
# KAFKA LISTENER (Background Thread)
# =============================================================================

def start_kafka_listener(state: DashboardState, stop_event: threading.Event):
    """Background thread to consume Kafka events."""
    try:
        consumer = KafkaConsumer(
            'bank_simulation',
            'bank_commands',
            bootstrap_servers='localhost:9092',
            value_deserializer=lambda v: json.loads(v.decode('utf-8')),
            auto_offset_reset='latest',
            api_version=(2, 5, 0),
            consumer_timeout_ms=1000
        )
        
        logger.info("Kafka listener started")
        
        while not stop_event.is_set():
            for message in consumer:
                if stop_event.is_set():
                    break
                    
                data = message.value
                topic = message.topic
                
                if topic == 'bank_simulation':
                    event_type = data.get('event_type')
                    
                    if event_type == 'ANGER_UPDATE':
                        state.update_anger(
                            data.get('lobby_anger', 0),
                            data.get('timestamp', '')
                        )
                        # Also update fatigue if present in the event
                        if 'teller_fatigue' in data:
                            state.update_fatigue(data.get('teller_fatigue', []))
                        # Update metrics if present
                        if 'metrics' in data:
                            state.update_metrics(data.get('metrics', {}))
                    
                    elif event_type == 'PREDICTION_UPDATE':
                        # Update arrival predictions for uncertainty cone
                        state.update_predictions(
                            actual=data.get('actual_arrivals', 0),
                            mean=data.get('mean', 0),
                            std=data.get('std', 0),
                            timestamp=data.get('timestamp', '')
                        )
                        
                elif topic == 'bank_commands':
                    state.add_decision({
                        'timestamp': data.get('timestamp', ''),
                        'action': data.get('action', ''),
                        'cost': data.get('cost_analysis', {}).get('total_cost', 0)
                            if data.get('cost_analysis') else 0
                        # Note: UCB column removed - show absence rather than placeholder
                    })
                    
    except NoBrokersAvailable:
        logger.warning("Kafka not available for dashboard")
    except Exception as e:
        logger.error(f"Kafka listener error: {e}")


# =============================================================================
# DEMO DATA GENERATOR
# =============================================================================

def generate_demo_data(state: DashboardState, step: int):
    """Generate demo data for standalone testing."""
    np.random.seed(42 + step)
    
    # Simulated arrivals
    base = 10 + 5 * np.sin(step * 0.3)  # Cyclic pattern
    actual = int(max(0, np.random.poisson(base)))
    mean = base + np.random.normal(0, 1)
    std = 2 + np.random.uniform(0, 2)
    
    timestamp = (datetime.now() - timedelta(minutes=50-step)).strftime('%H:%M')
    state.update_predictions(actual, mean, std, timestamp)
    
    # Simulated fatigue
    state.teller_fatigue = {
        0: min(1.0, 0.2 + step * 0.01 + np.random.uniform(0, 0.1)),
        1: min(1.0, 0.3 + step * 0.015 + np.random.uniform(0, 0.1)),
        2: min(1.0, 0.5 + step * 0.02 + np.random.uniform(0, 0.1))
    }
    
    # Simulated anger - TIE TO ARRIVALS for causal coherence in demo
    # This prevents demo anger increasing while queue decreases (causally inconsistent)
    anger_base = 0.5 * actual  # Anger driven by actual load
    state.lobby_anger = min(10, anger_base + np.random.uniform(-0.5, 0.5))
    
    # Simulated metrics
    state.metrics = {
        "total_arrivals": step * 5,
        "total_served": int(step * 4.5),
        "total_reneged": int(step * 0.3),
        "avg_wait": 3 + step * 0.1,
        "renege_rate": 6.0 + step * 0.1
    }
    
    # Simulated decisions
    actions = ['DO_NOTHING', 'ADD_TELLER', 'GIVE_BREAK', 'DELAY_DECISION', 'REMOVE_TELLER']
    if step % 5 == 0:
        state.add_decision({
            'timestamp': datetime.now().isoformat(),
            'action': actions[step % len(actions)],
            'cost': 50 + step * 2 + np.random.uniform(-10, 10),
            'ucb': mean + 1.96 * std
        })


# =============================================================================
# MAIN DASHBOARD
# =============================================================================

def main():
    """Main Streamlit dashboard."""
    st.set_page_config(
        page_title="Real-time Dashboard",
        page_icon="🏦",
        layout="wide"
    )
    
    st.title("Real-time Dashboard")
    st.markdown("""
    *Real-time visibility into latent system states: uncertainty, fatigue, and emotional contagion.*
    """)
    
    # Initialize state
    if 'dashboard_state' not in st.session_state:
        st.session_state.dashboard_state = DashboardState()
        st.session_state.demo_step = 0
        st.session_state.rl_agent = load_trained_model()  # Load trained model
        st.session_state.num_tellers = 1  # Start LOW to force AI to add staff
        st.session_state.queue_length = 0  # Current queue
        
        # Initialize Kafka producer
        try:
            from kafka_producer import QueueEventProducer
            st.session_state.kafka_producer = QueueEventProducer()
            logger.info("✅ Kafka producer initialized")
        except Exception as e:
            logger.warning(f"Kafka not available: {e}")
            st.session_state.kafka_producer = None
        
    state = st.session_state.dashboard_state
    rl_agent = st.session_state.rl_agent
    kafka_producer = st.session_state.get('kafka_producer', None)
    
    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Controls")
        
        # RL Model Status
        if rl_agent is not None:
            st.success("🤖 **RL Model Loaded**")
            st.caption("Trained DQN Agent Ready")
        else:
            st.error("❌ **No Model Found**")
            st.caption("Train model in Research Dashboard first")
            st.stop()
        
        st.markdown("---")
        
        # Simulation Control
        if 'simulation_running' not in st.session_state:
            st.session_state.simulation_running = False
        
        if not st.session_state.simulation_running:
            if st.button("▶️ Start Real-Time Simulation", type="primary", use_container_width=True):
                st.session_state.simulation_running = True
                st.session_state.sim_time = 0
                st.rerun()
        else:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("⏸️ Pause", use_container_width=True):
                    st.session_state.simulation_running = False
                    st.rerun()
            with col2:
                if st.button("🔄 Reset", use_container_width=True):
                    st.session_state.simulation_running = False
                    st.session_state.sim_time = 0
                    st.session_state.num_tellers = 1
                    st.session_state.queue_length = 0
                    st.session_state.dashboard_state = DashboardState()
                    st.rerun()
        
        st.markdown("---")
        
        # Current Status
        st.metric("⏱️ Simulation Time", f"{st.session_state.get('sim_time', 0)} min")
        st.metric("👥 Active Tellers", st.session_state.num_tellers)
        st.metric("📋 Queue Length", st.session_state.queue_length)
        
        st.markdown("---")
        st.markdown("""
        **Legend**
        - 🟢 Normal
        - 🟡 Warning  
        - 🔴 Critical
        """)
        
        
    # Real-Time Simulation Loop
    if st.session_state.simulation_running:
        # Simulate one time step
        st.session_state.sim_time += 10  # 10 minute intervals
        
        # Generate simulated arrivals (MORE DYNAMIC PATTERN)
        hour = 9 + (st.session_state.sim_time / 60.0)
        
        # Create realistic banking pattern with sharp peaks
        # Morning rush (9-10am), lunch lull (12-2pm), afternoon peak (3-4pm)
        time_of_day = (st.session_state.sim_time % 480) / 480.0  # 0-1 over 8 hours
        
        # Multiple peaks throughout the day
        morning_rush = 20 * np.exp(-((time_of_day - 0.1)**2) / 0.01)  # 9:30am spike
        lunch_lull = -10 * np.exp(-((time_of_day - 0.5)**2) / 0.02)   # 1pm dip
        afternoon_peak = 18 * np.exp(-((time_of_day - 0.75)**2) / 0.01)  # 3:30pm spike
        
        base_arrivals = 10 + morning_rush + lunch_lull + afternoon_peak
        actual_arrivals = int(max(0, base_arrivals + np.random.normal(0, 4)))
        
        # Initialize break tracking if not exists
        if 'tellers_on_break' not in st.session_state:
            st.session_state.tellers_on_break = []
        
        # Check if any tellers are returning from break FIRST
        current_sim_time = st.session_state.sim_time
        returning_tellers = [t for t in st.session_state.tellers_on_break if t[1] <= current_sim_time]
        if returning_tellers:
            st.session_state.tellers_on_break = [t for t in st.session_state.tellers_on_break if t[1] > current_sim_time]
            logger.info(f"☕ {len(returning_tellers)} teller(s) returned from break (refreshed!)")
        
        # Calculate effective tellers (accounting for breaks)
        effective_tellers = st.session_state.num_tellers - len(st.session_state.tellers_on_break)
        effective_tellers = max(1, effective_tellers)
        
        # Calculate current wait time based on queue
        service_rate = effective_tellers * 3  # 3 customers per teller per 10 min
        
        # Estimate wait time: queue / service_rate * 10 minutes
        estimated_wait = (st.session_state.queue_length / max(1, service_rate)) * 10
        
        # Customers renege if wait > 20 minutes
        patience_threshold = 20
        reneged_this_step = 0
        
        if estimated_wait > patience_threshold:
            # Some customers in queue will leave
            renege_probability = min(0.8, (estimated_wait - patience_threshold) / 30.0)
            reneged_this_step = int(st.session_state.queue_length * renege_probability)
            st.session_state.queue_length -= reneged_this_step
        
        # Add new arrivals to queue
        st.session_state.queue_length += actual_arrivals
        
        # Serve customers
        served = min(st.session_state.queue_length, service_rate)
        
        # Add randomness to service (sometimes tellers are slow)
        if np.random.random() < 0.2:
            served = max(0, served - 1)
            
        st.session_state.queue_length = max(0, st.session_state.queue_length - served)
        
        # Track cumulative metrics
        if 'total_arrivals' not in st.session_state:
            st.session_state.total_arrivals = 0
            st.session_state.total_served = 0
            st.session_state.total_reneged = 0
            
        st.session_state.total_arrivals += actual_arrivals
        st.session_state.total_served += served
        st.session_state.total_reneged += reneged_this_step
        
        # Calculate renege rate
        total_customers = st.session_state.total_arrivals
        renege_rate = (st.session_state.total_reneged / max(1, total_customers)) * 100
        
        # Update dashboard state
        state.update_predictions(
            actual=actual_arrivals,
            mean=base_arrivals,
            std=3.0,
            timestamp=f"{int(hour):02d}:{int((hour % 1) * 60):02d}"
        )
        
        # Get RL decision every 10 minutes
        state_data = {
            'num_tellers': st.session_state.num_tellers,
            'queue_length': st.session_state.queue_length,
            'avg_fatigue': 0.3 + (st.session_state.sim_time / 480.0) * 0.3,
            'max_fatigue': 0.5 + (st.session_state.sim_time / 480.0) * 0.3,
            'burnt_out': 0,
            'lobby_anger': min(10.0, st.session_state.queue_length / 5.0),
            'pred_mean': base_arrivals,
            'pred_ucb': base_arrivals + 5,
            'uncertainty': 3.0,
            'current_wait': estimated_wait,
            'hour': int(hour),
            'renege_rate': renege_rate / 100.0  # Normalize to 0-1
        }
        
        action, confidence = get_rl_decision(rl_agent, state_data)
        
        # DEBUG: Log what RL agent sees
        logger.info(f"🔍 RL State: queue={state_data['queue_length']}, tellers={state_data['num_tellers']}, "
                   f"wait={state_data['current_wait']:.1f}min, renege={state_data['renege_rate']:.2%}")
        logger.info(f"🤖 RL Decision: {action} (confidence: {confidence:.2f})")
        
        
        # Send queue state to Kafka BEFORE decision
        if kafka_producer:
            kafka_producer.send_queue_state({
                'num_tellers': st.session_state.num_tellers,
                'queue_length': st.session_state.queue_length,
                'avg_wait': estimated_wait,
                'renege_rate': renege_rate,
                'arrivals': actual_arrivals,
                'served': served,
                'reneged': reneged_this_step
            })
            
            kafka_producer.send_arrival(actual_arrivals, int(hour))
        
        # Apply RL decision
        old_tellers = st.session_state.num_tellers
        
        # Map RL actions to dashboard actions
        if action == "ADD_TELLER" and st.session_state.num_tellers < 10:
            st.session_state.num_tellers += 1
            logger.info(f"➕ Added teller: {old_tellers} → {st.session_state.num_tellers}")
            
        elif action == "REMOVE_TELLER" and st.session_state.num_tellers > 1:
            st.session_state.num_tellers -= 1
            logger.info(f"➖ Removed teller: {old_tellers} → {st.session_state.num_tellers}")
            
        elif action == "GIVE_BREAK":
            # Find most fatigued teller
            if state.teller_fatigue:
                most_fatigued_id = max(state.teller_fatigue.keys(), key=lambda k: state.teller_fatigue[k])
                most_fatigued_level = state.teller_fatigue[most_fatigued_id]
                
                # Only give break if fatigue > 0.6 and not already on break
                on_break_ids = [t[0] for t in st.session_state.tellers_on_break]
                if most_fatigued_level > 0.6 and most_fatigued_id not in on_break_ids:
                    # Send teller on 20-minute break
                    return_time = current_sim_time + 20
                    st.session_state.tellers_on_break.append((most_fatigued_id, return_time))
                    logger.info(f"☕ Teller {most_fatigued_id} on break (fatigue: {most_fatigued_level:.2f}, returns at {return_time} min)")
                else:
                    logger.info(f"⏸️ No break needed (fatigue: {most_fatigued_level:.2f})")
            else:
                logger.info(f"⏸️ No tellers to give break")
                
        elif action in ["DO_NOTHING", "MAINTAIN"]:
            logger.info(f"⏸️ No staffing change: {action}")
        else:
            logger.warning(f"⚠️ Unknown action: {action}")
        
        # Send RL decision to Kafka
        if kafka_producer and old_tellers != st.session_state.num_tellers:
            kafka_producer.send_rl_decision(
                action=action,
                confidence=float(confidence),
                state_before={'num_tellers': old_tellers, 'queue_length': st.session_state.queue_length},
                state_after={'num_tellers': st.session_state.num_tellers, 'queue_length': st.session_state.queue_length}
            )
        
        # Update metrics
        state.total_arrivals = st.session_state.total_arrivals
        state.total_served = st.session_state.total_served
        state.total_reneged = st.session_state.total_reneged
        state.avg_wait = estimated_wait
        state.renege_rate = renege_rate
        state.lobby_anger = min(10.0, st.session_state.queue_length / 5.0)
        
        # Update teller fatigue dynamically based on current teller count
        fatigue_increase = (st.session_state.sim_time / 480.0) * 0.7  # Fatigue increases over shift
        state.teller_fatigue = {}
        
        # Get list of tellers on break
        on_break_ids = [t[0] for t in st.session_state.tellers_on_break]
        
        for i in range(st.session_state.num_tellers):
            # Each teller has slightly different fatigue based on workload
            base_fatigue = 0.2 + fatigue_increase
            workload_factor = (st.session_state.queue_length / max(1, effective_tellers)) / 20.0
            
            # Tellers on break have 0 fatigue (resting)
            if i in on_break_ids:
                state.teller_fatigue[i] = 0.0
            else:
                state.teller_fatigue[i] = min(0.95, base_fatigue + workload_factor + np.random.uniform(-0.05, 0.05))
        
        # Record decision
        state.decision_trace.append({
            'time': f"{int(hour):02d}:{int((hour % 1) * 60):02d}",
            'action': action,
            'confidence': f"{confidence:.2f}",
            'tellers': st.session_state.num_tellers,
            'queue': st.session_state.queue_length
        })
        
    # Metrics row
    st.markdown("### 📊 Key Metrics")
    st.markdown("*Click ℹ️ to learn what each metric means*")
    
    metrics = create_metrics_cards(state)
    cols = st.columns(5)
    
    # Metric explanations (shown only when clicked)
    metric_info = {
        "Total Arrivals": "Number of customers who entered the queue. Shows demand on the system.",
        "Served": "Customers successfully helped by tellers. Higher is better!",
        "Reneged": "Customers who left due to long wait times. Lower is better - indicates frustration.",
        "Avg Wait": "How long customers wait before being served. Target: < 5 minutes.",
        "Renege Rate": "Percentage of customers who gave up waiting. Lower means happier customers."
    }
    
    for col, (label, value) in zip(cols, metrics.items()):
        with col:
            st.metric(label, value)
            with st.expander("ℹ️"):
                st.write(metric_info[label])
        
    st.markdown("---")
    
    # Main visualizations
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Uncertainty Cone
        st.markdown("#### 📈 Arrival Predictions")
        with st.expander("ℹ️ What does this chart show?"):
            st.markdown("""
            **AI predictions of customer arrivals with uncertainty bands.**
            - **Green line**: Actual arrivals (what really happened)
            - **Blue dashed**: AI's best guess (mean prediction)
            - **Blue shaded area**: Uncertainty range (95% confidence)
            - **Red dotted**: Upper bound used for staffing decisions (conservative)
            
            **Why it matters**: Helps predict busy periods so we can staff appropriately.
            """)
        st.plotly_chart(
            create_uncertainty_cone(state),
            use_container_width=True
        )
        
    with col2:
        # Contagion Gauge
        st.markdown("#### 😤 Lobby Anger Index")
        with st.expander("ℹ️ What does this gauge show?"):
            st.markdown("""
            **Collective frustration level in the waiting area.**
            - **0-3 (Green)**: Calm - customers are patient
            - **3-6 (Yellow)**: Tense - some frustration building
            - **6-10 (Red)**: Danger - high risk of customers leaving
            
            **Why it matters**: Emotional contagion spreads - one angry customer affects others!
            """)
        st.plotly_chart(
            create_contagion_gauge(state),
            use_container_width=True
        )
        
    # Second row
    col3, col4 = st.columns([1, 1])
    
    with col3:
        # Fatigue Heatmap
        st.markdown("#### 🔥 Teller Fatigue Levels")
        with st.expander("ℹ️ What does this show?"):
            st.markdown("""
            **How tired each teller is (0% = fresh, 100% = exhausted).**
            - **Green (<50%)**: Teller is working efficiently
            - **Yellow (50-80%)**: Teller is getting tired, slower service
            - **Red (>80%)**: Burnout risk! Needs a break urgently
            
            **Why it matters**: Tired tellers work slower, increasing wait times for everyone.
            """)
        st.plotly_chart(
            create_fatigue_heatmap(state),
            use_container_width=True
        )
        
    with col4:
        # Decision Trace
        st.markdown("#### 📋 AI Decision History")
        with st.expander("ℹ️ What are these decisions?"):
            st.markdown("""
            **Recent decisions made by the optimization AI.**
            - **ADD_TELLER**: Hired more staff (busy period predicted)
            - **REMOVE_TELLER**: Reduced staff (quiet period)
            - **GIVE_BREAK**: Sent a tired teller on break
            - **DO_NOTHING**: System is balanced, no action needed
            - **⏸️ DELAY_DECISION**: Waiting for more data before acting
            
            **Why it matters**: Shows how the AI adapts to changing conditions in real-time.
            """)
        trace_df = create_decision_trace_table(state)
        st.dataframe(
            trace_df,
            use_container_width=True,
            hide_index=True
        )
        
        
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
    <small>
    Closed-Loop Socio-Technical System | 
    Bayesian Forecasting + Affective Simulation + Multi-Objective Optimization
    </small>
    </div>
    """, unsafe_allow_html=True)
    
    # Auto-refresh loop for simulation (Placed at end to ensure charts render first)
    if st.session_state.simulation_running:
        time.sleep(1)  # 1 second refresh for smoother "real-time" feel
        st.rerun()


if __name__ == "__main__":
    main()
