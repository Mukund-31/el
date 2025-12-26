# RL-Based Queue Management System 🏦

**Reinforcement Learning for Dynamic Staffing Optimization in Service Operations**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-green.svg)](https://streamlit.io/)
[![Kafka](https://img.shields.io/badge/Kafka-3.0+-black.svg)](https://kafka.apache.org/)

## 📋 Overview

This project implements a **Deep Q-Network (DQN)** based reinforcement learning system for real-time queue management and dynamic staffing optimization. The system learns optimal staffing policies by balancing customer wait times, abandonment rates, and operational costs.

### Key Features

- ✅ **Deep Reinforcement Learning**: DQN agent with experience replay and target networks
- ✅ **Real-Time Simulation**: Kafka-based event streaming for production deployment
- ✅ **Multi-Objective Optimization**: Balances wait time, renege rate, and staffing cost
- ✅ **Human Factors**: Models teller fatigue and break management
- ✅ **Statistical Validation**: 300-episode Monte Carlo validation on real-world data
- ✅ **Production-Ready**: Complete ML pipeline from training to deployment

### Results

| Metric | Baseline | RL Agent | Improvement |
|--------|----------|----------|-------------|
| **Avg Wait Time** | 34.55 min | 3.92 min | **88% ↓** |
| **Renege Rate** | 10.30% | 1.30% | **87% ↓** |
| **Avg Tellers** | 7.52 | 7.25 | **3.5% ↓** |
| **Total Cost** | High | Low | **Better** |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Docker (for Kafka)
- Git

### Installation

```bash
# Clone repository
git clone https://github.com/Mukund-31/el.git
cd el/7thsem

# Install dependencies
pip install -r requirements.txt

# Start Kafka (optional - for real-time streaming)
docker-compose up -d
```

### Run the System

**Option 1: ML Research Dashboard (Training & Validation)**
```bash
streamlit run ml_research_dashboard.py --server.port 8504
```
- Stage 1: Train RL agent on synthetic data
- Stage 2: Validate on real-world trace data
- View statistical comparison and results

**Option 2: Real-Time Operational Dashboard**
```bash
streamlit run dashboard.py --server.port 8503
```
- Real-time simulation with trained RL model
- Dynamic staffing decisions every 10 minutes
- Kafka event streaming (if enabled)

**Option 3: Kafka Consumer (Monitor Events)**
```bash
python kafka_consumer.py
```
- View real-time queue events
- Monitor RL decisions
- Track system performance

---

## 📊 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ML Research Pipeline                      │
├─────────────────────────────────────────────────────────────┤
│  1. Training (Synthetic Data)                               │
│     └─> 300 episodes, DQN learning                          │
│  2. Validation (Real-World Trace)                           │
│     └─> queue_data.csv (560 customers, March 30)            │
│  3. Statistical Analysis                                     │
│     └─> Paired t-tests, confidence intervals                │
└─────────────────────────────────────────────────────────────┘
                            ↓
                   trained_model.pth
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                 Real-Time Deployment                         │
├─────────────────────────────────────────────────────────────┤
│  Dashboard → Kafka Producer → queue-events topic            │
│                                      ↓                       │
│                            Kafka Consumer (Monitor)          │
│                                      ↓                       │
│                         RL Agent (Inference)                 │
│                                      ↓                       │
│                    Staffing Decisions (ADD/REMOVE)           │
└─────────────────────────────────────────────────────────────┘
```

---

## 🧠 RL Agent Details

### State Space (12 dimensions)
- `num_tellers`: Current staffing level (1-10)
- `current_queue`: Queue length (0-50)
- `avg_fatigue`: Average teller fatigue (0-1)
- `max_fatigue`: Maximum teller fatigue (0-1)
- `burnt_out_count`: Number of exhausted tellers
- `lobby_anger`: Customer frustration level (0-10)
- `predicted_arrivals_mean`: Expected arrivals
- `predicted_arrivals_ucb`: Upper confidence bound
- `prediction_uncertainty`: Forecast uncertainty
- `current_wait`: Average wait time (0-20 min)
- `hour_of_day`: Time context (9-17)
- `recent_renege_rate`: Abandonment rate (0-1)

### Action Space (4 actions)
- `ADD_TELLER`: Hire additional staff
- `REMOVE_TELLER`: Reduce staffing
- `GIVE_BREAK`: Send fatigued teller on 20-min break
- `DO_NOTHING`: Maintain current state

### Reward Function
```python
reward = -(wait_cost + renege_cost + fatigue_cost + staffing_cost) + anger_bonus - action_penalty
```

### Neural Network Architecture
- **Input**: 12-dimensional state vector
- **Hidden Layers**: [128, 64] neurons with ReLU activation
- **Output**: 4 Q-values (one per action)
- **Optimizer**: Adam (lr=0.001)
- **Loss**: Huber Loss (smooth L1)

---

## 📁 Project Structure

```
el/
├── 7thsem/                          # Main project directory
│   ├── dashboard.py                 # Real-time operational dashboard
│   ├── ml_research_dashboard.py    # Training & validation dashboard
│   ├── rl_optimization_agent.py    # DQN implementation
│   ├── validation_framework.py     # Statistical validation
│   ├── kafka_producer.py           # Event streaming producer
│   ├── kafka_consumer.py           # Event streaming consumer
│   ├── simple_comparison.py        # Direct trace comparison
│   ├── trained_model.pth           # Trained RL model weights
│   ├── docker-compose.yml          # Kafka setup
│   ├── requirements.txt            # Python dependencies
│   │
│   ├── Documentation/
│   │   ├── STATE_CALCULATION.md    # How states are computed
│   │   ├── BREAK_SYSTEM.md         # Break management guide
│   │   ├── KAFKA_GUIDE.md          # Kafka integration guide
│   │   ├── SIMULATION_BEHAVIOR.md  # Expected behavior
│   │   ├── OBJECTIVES_VERIFICATION.md  # Project objectives
│   │   └── FINAL_RESULTS_SUMMARY.md    # Results analysis
│   │
│   └── Data/
│       └── queue_data.csv          # Real-world trace data
│
└── README.md                        # This file
```

---

## 🔬 Methodology

### 1. Training Phase
- **Environment**: Synthetic queue simulation
- **Episodes**: 300 training episodes
- **Exploration**: ε-greedy (ε: 1.0 → 0.01)
- **Experience Replay**: 10,000 transitions
- **Target Network**: Updated every 10 episodes

### 2. Validation Phase
- **Data**: Real-world trace (queue_data.csv)
- **Method**: Replay historical arrivals
- **Comparison**: RL Agent vs. Rule-Based Baseline
- **Metrics**: Wait time, renege rate, cost, served customers

### 3. Deployment Phase
- **Mode**: Real-time simulation
- **Decision Frequency**: Every 10 minutes
- **Event Streaming**: Kafka topics
- **Monitoring**: Live dashboard + consumer logs

---

## 📈 Key Insights

### 1. Proactive Staffing
The RL agent learns to **add tellers before** queue buildup by using predicted arrivals:
```
09:20 - Predicted rush at 09:30 → ADD_TELLER
09:30 - Rush arrives → Queue stays low ✅
```

### 2. Cost Optimization
When queue is empty, the agent **reduces excess staffing**:
```
Queue = 0, Tellers = 10 → REMOVE_TELLER
Queue = 0, Tellers = 9 → REMOVE_TELLER
...stabilizes at 5-6 tellers
```

### 3. Human Factors
The agent learns to **give breaks** to prevent burnout:
```
Teller 3 fatigue = 0.78 → GIVE_BREAK
Teller 3 on break (20 min) → Fatigue resets to 0.0
Teller 3 returns refreshed ✅
```

---

## 🎓 Academic Use

This project demonstrates:
- **Queuing Theory**: M/M/c queue modeling with time-varying arrivals
- **Operations Research**: Multi-objective cost optimization
- **Machine Learning**: Deep Q-Learning with function approximation
- **Statistical Validation**: Hypothesis testing, confidence intervals
- **Production Deployment**: Event-driven architecture with Kafka

### Citation
If you use this work, please cite:
```
@misc{queue_rl_2024,
  author = {Mukund},
  title = {RL-Based Queue Management System},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/Mukund-31/el}
}
```

---

## 📄 License

This project is available for academic and research purposes.

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

## 📧 Contact

For questions or collaboration:
- GitHub: [@Mukund-31](https://github.com/Mukund-31)
- Repository: [https://github.com/Mukund-31/el](https://github.com/Mukund-31/el)

---

## 🙏 Acknowledgments

- PyTorch for deep learning framework
- Streamlit for dashboard framework
- Apache Kafka for event streaming
- Real-world queue data from banking operations

---

**Built with ❤️ for advancing AI in operations management**
