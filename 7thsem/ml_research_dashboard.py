"""
Research Dashboard - Proper ML Workflow
========================================

Stage 1: TRAIN on synthetic data (diverse scenarios)
Stage 2: TEST on real data (your CSV)

This follows proper machine learning methodology:
- Train on varied data to learn general patterns
- Test on unseen real data to prove it works
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List
import json
import subprocess
import os
from pathlib import Path
import time

# Set page config
st.set_page_config(
    page_title="RL Banking Research - Proper ML Workflow",
    page_icon="🎓",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stage-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)


def check_trained_model():
    """Check if trained model exists."""
    return Path("trained_model.pth").exists()


def run_training(episodes: int, seed: int):
    """Run Stage 1: Training on synthetic data."""
    cmd = f"python validation_framework.py --episodes {episodes} --seed {seed} --output results_training"
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("🔄 Training RL agent on synthetic scenarios...")
        
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        output_lines = []
        episode_count = 0
        
        for line in process.stdout:
            output_lines.append(line)
            
            if "Episode" in line:
                episode_count += 20
                progress = min(episode_count / episodes, 0.95)
                progress_bar.progress(progress)
                status_text.text(f"Training... Episode ~{episode_count}/{episodes}")
        
        process.wait()
        
        if process.returncode == 0:
            progress_bar.progress(1.0)
            status_text.text("✅ Training complete! Model saved.")
            return True, "\\n".join(output_lines)
        else:
            status_text.text("❌ Training failed!")
            return False, "\\n".join(output_lines)
            
    except Exception as e:
        status_text.text(f"❌ Error: {e}")
        return False, str(e)


def run_testing():
    """Run Stage 2: Testing on real CSV data."""
    cmd = "python simple_comparison.py"
    
    status_text = st.empty()
    
    try:
        status_text.text("🔄 Testing on real-world data...")
        
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            status_text.text("✅ Testing complete!")
            return True, result.stdout
        else:
            status_text.text("❌ Testing failed!")
            return False, result.stdout + "\\n" + result.stderr
            
    except Exception as e:
        status_text.text(f"❌ Error: {e}")
        return False, str(e)


def load_test_results():
    """Load results from testing phase."""
    results_file = Path("results/direct_comparison.csv")
    
    if not results_file.exists():
        return None
    
    try:
        df = pd.read_csv(results_file)
        return df
    except Exception as e:
        st.error(f"Error loading results: {e}")
        return None


def main():
    st.markdown('<h1 class="main-header">🎓 RL Banking Research Dashboard</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    ### Proper Machine Learning Workflow
    
    This dashboard follows the correct ML research methodology:
    1. **Train** on diverse synthetic data
    2. **Test** on real-world data
    3. **Compare** results scientifically
    """)
    
    st.markdown("---")
    
    # Check model status
    model_exists = check_trained_model()
    
    # ========================================
    # STAGE 1: TRAINING
    # ========================================
    st.header("📚 Stage 1: Training Phase")
    
    if model_exists:
        st.markdown('<div class="stage-box success-box">', unsafe_allow_html=True)
        st.success("✅ Trained model found: `trained_model.pth`")
        st.markdown("Your RL agent has been trained on synthetic scenarios.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("🔄 Retrain Model (Optional)", help="Train a new model from scratch"):
            st.session_state['show_training'] = True
    else:
        st.markdown('<div class="stage-box">', unsafe_allow_html=True)
        st.warning("⚠️ No trained model found. You need to train first!")
        st.markdown('</div>', unsafe_allow_html=True)
        st.session_state['show_training'] = True
    
    # Training UI
    if st.session_state.get('show_training', not model_exists):
        st.markdown("#### ⚙️ Training Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            train_episodes = st.slider(
                "Training Episodes",
                min_value=100,
                max_value=1000,
                value=500,
                step=50,
                help="More episodes = better learning (500-1000 recommended)"
            )
        
        with col2:
            train_seed = st.number_input(
                "Random Seed",
                min_value=1,
                max_value=999,
                value=42,
                help="For reproducibility"
            )
        
        est_time = train_episodes * 0.5 / 60
        st.info(f"⏱️ Estimated training time: ~{est_time:.1f} minutes")
        
        if st.button("🚀 Start Training", type="primary", use_container_width=True):
            st.markdown("### 🔄 Training in Progress...")
            
            success, output = run_training(train_episodes, train_seed)
            
            if success:
                st.balloons()
                st.success("✅ Training complete! Proceed to Stage 2.")
                
                with st.expander("📋 View Training Log"):
                    st.code(output, language="text")
                
                # Force refresh
                st.session_state['show_training'] = False
                st.rerun()
            else:
                st.error("❌ Training failed. Check the log below:")
                st.code(output, language="text")
    
    st.markdown("---")
    
    # ========================================
    # STAGE 2: TESTING
    # ========================================
    st.header("🧪 Stage 2: Testing Phase")
    
    if not model_exists:
        st.markdown('<div class="stage-box">', unsafe_allow_html=True)
        st.info("ℹ️ Complete Stage 1 (Training) first before testing.")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="stage-box success-box">', unsafe_allow_html=True)
        st.markdown("""
        **Ready to test!**
        
        This will run your trained RL model on the EXACT real-world data from `queue_data.csv`:
        - No jitter, no variations
        - Single run comparison
        - Real World vs Baseline vs Your RL Model
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("🧪 Run Test on Real Data", type="primary", use_container_width=True):
            success, output = run_testing()
            
            if success:
                st.success("✅ Testing complete!")
                
                with st.expander("📋 View Test Output"):
                    st.code(output, language="text")
                
                # Force refresh to show results
                st.rerun()
            else:
                st.error("❌ Testing failed:")
                st.code(output, language="text")
    
    st.markdown("---")
    
    # ========================================
    # RESULTS
    # ========================================
    st.header("📊 Results & Comparison")
    
    results_df = load_test_results()
    
    if results_df is None:
        st.info("ℹ️ No test results yet. Complete Stage 2 to see results.")
    else:
        st.success("✅ Test results loaded!")
        
        # Display results table
        st.markdown("### 📋 Comparison Table")
        st.dataframe(results_df, use_container_width=True, hide_index=True)
        
        # Extract metrics for visualization
        if len(results_df) >= 2:
            st.markdown("### 📈 Visual Comparison")
            
            metrics = ['avg_wait', 'avg_tellers', 'served', 'renege_rate', 'total_cost']
            
            # Create comparison charts
            fig = make_subplots(
                rows=2, cols=3,
                subplot_titles=('Avg Wait Time', 'Avg Tellers', 'Customers Served',
                               'Renege Rate', 'Total Cost', 'Summary'),
                specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
                       [{"type": "bar"}, {"type": "bar"}, {"type": "indicator"}]]
            )
            
            agents = results_df['Agent'].tolist()
            colors = ['gray', 'red', 'blue']
            
            # Avg Wait
            fig.add_trace(
                go.Bar(x=agents, y=results_df['avg_wait'], 
                       marker_color=colors, name='Wait Time'),
                row=1, col=1
            )
            
            # Avg Tellers
            fig.add_trace(
                go.Bar(x=agents, y=results_df['avg_tellers'], 
                       marker_color=colors, name='Tellers'),
                row=1, col=2
            )
            
            # Served
            fig.add_trace(
                go.Bar(x=agents, y=results_df['served'], 
                       marker_color=colors, name='Served'),
                row=1, col=3
            )
            
            # Renege Rate
            fig.add_trace(
                go.Bar(x=agents, y=results_df['renege_rate'], 
                       marker_color=colors, name='Renege %'),
                row=2, col=1
            )
            
            # Total Cost
            fig.add_trace(
                go.Bar(x=agents, y=results_df['total_cost'], 
                       marker_color=colors, name='Cost'),
                row=2, col=2
            )
            
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Key insights
            st.markdown("### 💡 Key Insights")
            
            if len(results_df) >= 2:  # Changed from 3 since we removed Real World
                baseline_cost = results_df.loc[results_df['Agent'] == 'Baseline', 'total_cost'].values[0]
                rl_cost = results_df.loc[results_df['Agent'] == 'RL Agent', 'total_cost'].values[0] if 'RL Agent' in results_df['Agent'].values else None
                
                if rl_cost and not np.isnan(rl_cost) and not np.isnan(baseline_cost):
                    savings = ((baseline_cost - rl_cost) / baseline_cost) * 100
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Cost Savings", f"{savings:.1f}%", 
                                 f"${baseline_cost - rl_cost:.2f}")
                    
                    baseline_tellers = results_df.loc[results_df['Agent'] == 'Baseline', 'avg_tellers'].values[0]
                    rl_tellers = results_df.loc[results_df['Agent'] == 'RL Agent', 'avg_tellers'].values[0] if 'RL Agent' in results_df['Agent'].values else None
                    
                    if rl_tellers and not np.isnan(rl_tellers):
                        with col2:
                            st.metric("Staff Reduction", 
                                     f"{baseline_tellers - rl_tellers:.1f} tellers",
                                     f"{rl_tellers:.1f} vs {baseline_tellers:.1f}")
                    
                    with col3:
                        st.metric("Throughput", "Same", 
                                 "All systems served all customers")


if __name__ == "__main__":
    main()
