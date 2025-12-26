# 🏦 Socio-Technical Service Operations System - User Guide

## 📖 What Is This System?

This is an **intelligent bank queue simulation** that combines:
- 🤖 **AI Forecasting** - Predicts customer arrivals
- 😤 **Emotional Modeling** - Tracks customer frustration and anger
- 💪 **Fatigue Simulation** - Models teller tiredness
- 🎯 **Smart Optimization** - AI makes staffing decisions automatically

---

## 🎮 How to Run the System

### Step 1: Start Kafka (Message Broker)
```bash
docker-compose up -d
```
This starts the messaging system that connects all components.

### Step 2: Run the Main Simulation
```bash
python main.py --dashboard --speed 0.5
```
- `--dashboard`: Opens the visualization dashboard
- `--speed 0.5`: Controls simulation speed (0.1 = fast, 1.0 = real-time)

### Step 3: View the Dashboards
- **Main Dashboard**: http://localhost:8501
- **Scenario Dashboard**: http://localhost:8502

---

## 📊 Understanding the Dashboard

### 🔢 Key Metrics (Top Row)

| Metric | What It Means | Good/Bad |
|--------|---------------|----------|
| **Total Arrivals** ℹ️ | Number of customers who entered the queue | Shows demand |
| **Served** ℹ️ | Customers successfully helped | Higher = Better ✅ |
| **Reneged** ℹ️ | Customers who left due to frustration | Lower = Better ✅ |
| **Avg Wait** ℹ️ | Average waiting time before service | Target: < 5 min ✅ |
| **Renege Rate** ℹ️ | % of customers who gave up | Lower = Better ✅ |

---

### 📈 Arrival Predictions Chart

**What you see:**
- **Green solid line** = Actual customer arrivals (reality)
- **Blue dashed line** = AI's prediction (best guess)
- **Blue shaded area** = Uncertainty range (95% confidence)
- **Red dotted line** = Upper bound (used for conservative staffing)

**Why it matters:**
The AI uses these predictions to decide when to hire more tellers or give breaks.

**How to read it:**
- Wide blue band = AI is uncertain (needs more data)
- Narrow blue band = AI is confident
- Green line inside blue band = Good predictions! ✅
- Green line outside blue band = AI was surprised 😮

---

### 😤 Lobby Anger Index (Gauge)

**What you see:**
A speedometer showing collective frustration (0-10 scale)

**Color zones:**
- **🟢 Green (0-3)**: Calm - customers are patient
- **🟡 Yellow (3-6)**: Tense - frustration building
- **🔴 Red (6-10)**: DANGER - customers will start leaving!

**Why it matters:**
This models **emotional contagion** - when one person gets angry, it spreads to others like a virus! The AI watches this to prevent mass walkouts.

**Formula:**
```
LobbyAnger = min(10, median(wait_times) / reference_wait)
```

---

### 🔥 Teller Fatigue Levels

**What you see:**
Horizontal bars showing how tired each teller is (0% to 100%)

**Color coding:**
- **🟢 Green (<50%)**: Fresh and efficient
- **🟡 Yellow (50-80%)**: Getting tired, slower service
- **🔴 Red (>80%)**: BURNOUT RISK - needs break immediately!

**Why it matters:**
Tired tellers work slower:
- Fresh teller (0% fatigue) = 100% efficiency
- Exhausted teller (100% fatigue) = 40% efficiency

**What the AI does:**
When fatigue hits 70%+, the AI sends tellers on 10-minute breaks to recover.

---

### 📋 AI Decision History (Audit Trail)

**What you see:**
A table of recent AI decisions with timestamps

**Decision types:**

| Action | What It Means | When It Happens |
|--------|---------------|-----------------|
| **ADD_TELLER** | Hired more staff | Busy period predicted (high UCB) |
| **REMOVE_TELLER** | Reduced staff | Quiet period, too many idle tellers |
| **GIVE_BREAK** | Sent teller on break | Fatigue > 70% |
| **DO_NOTHING** | No action needed | System is balanced ✅ |
| **⏸️ DELAY_DECISION** | Waiting for more data | AI is uncertain, being cautious |

**Cost column:**
Shows the "cost" of the decision (combines staffing costs, wait time costs, and renege costs). Lower is better!

---

## 🧠 How the AI Works

### 1. **Bayesian Forecaster** (Predicts Arrivals)
- Uses LSTM neural network
- Learns patterns: "Mondays are busy", "Lunch rush at 12pm"
- Outputs: Mean prediction + Uncertainty
- Uses **Upper Confidence Bound (UCB)** for risk-averse staffing

### 2. **Affective Simulation** (Models Emotions)
- Tracks each customer's patience
- Models emotional contagion (anger spreads!)
- Simulates teller fatigue (efficiency drops over time)
- Customers "renege" (leave) when patience runs out

### 3. **Optimization Agent** (Makes Decisions)
- Balances multiple objectives:
  - ✅ Minimize customer wait time
  - ✅ Minimize reneging (customer satisfaction)
  - ✅ Minimize staffing costs
  - ✅ Prevent teller burnout
- Makes decisions every 2 minutes (simulation time)

---

## 🎛️ Adjusting the Simulation

### Change Arrival Rate
Edit `producer.py`, line 64-68:
```python
self.schedule = [
    (9, 11, self._linear_ramp(2, 6)),     # Morning: 2-6 customers/hour
    (11, 13, lambda h: 12),                # Lunch: 12 customers/hour
    (13, 15, self._linear_ramp(12, 4)),    # Afternoon: 12-4 customers/hour
    (15, 17, lambda h: 3),                 # Late: 3 customers/hour
]
```

### Change Simulation Speed
```bash
python main.py --dashboard --speed 0.1  # 10x faster
python main.py --dashboard --speed 1.0  # Real-time
```

### Change Initial Tellers
Edit `main.py`, line 53:
```python
INITIAL_TELLERS = 3  # Start with 3 tellers
```

---

## 🔬 Key Insights from the System

1. **Uncertainty Matters**: The AI doesn't just predict - it knows when it's uncertain!
2. **Emotions Are Contagious**: One angry customer affects everyone in the lobby
3. **Fatigue Is Real**: Tired workers are slower, creating a vicious cycle
4. **Proactive > Reactive**: AI predicts busy periods and staffs up BEFORE the rush

---

## 🛠️ Troubleshooting

### Dashboard shows "Waiting for data..."
- Wait 30 seconds for the system to collect initial data
- Check that `main.py` is running

### Kafka connection errors
```bash
# Restart Kafka
docker-compose down
docker-compose up -d
```

### Arrivals too fast/slow
- Adjust arrival rates in `producer.py` (see above)
- Change `--speed` parameter

### Dashboard not updating
- Refresh the browser page
- Check "Auto-refresh (2s)" checkbox in sidebar

---

## 📚 Technical Architecture

```
┌─────────────────┐
│  Producer       │ → Generates customers (NHPP)
│  (producer.py)  │
└────────┬────────┘
         │
         ↓ Kafka: bank_arrivals
         │
┌────────┴────────┐
│  Simulation     │ → Models queue, emotions, fatigue
│  (simulation_   │
│   engine.py)    │
└────────┬────────┘
         │
         ↓ Kafka: bank_simulation
         │
┌────────┴────────┐
│  Forecaster     │ → Predicts arrivals (Bayesian LSTM)
│  (forecaster.py)│
└────────┬────────┘
         │
         ↓ Predictions
         │
┌────────┴────────┐
│  Optimizer      │ → Makes staffing decisions
│  (optimization_ │
│   agent.py)     │
└────────┬────────┘
         │
         ↓ Kafka: bank_commands
         │
┌────────┴────────┐
│  Dashboard      │ → Visualizes everything
│  (dashboard.py) │
└─────────────────┘
```

---

## 🎓 Learning Objectives

This system demonstrates:
- ✅ **Closed-loop AI systems** (sense → predict → decide → act)
- ✅ **Multi-objective optimization** (balancing competing goals)
- ✅ **Uncertainty quantification** (knowing what you don't know)
- ✅ **Human-centered AI** (modeling emotions, fatigue, psychology)
- ✅ **Real-time streaming** (Kafka event processing)

---

## 📞 Need Help?

Check the logs in the terminal where you ran `main.py` - they show:
- Customer arrivals
- AI decisions
- System status every 5 seconds

**Example log:**
```
INFO:orchestrator:📊 Status | Queue: 5 | Anger: 2.3 | Served: 45 | Reneged: 2
INFO:orchestrator:🎯 Decision: GIVE_BREAK | Queue: 5 | UCB: 8.7
```

---

**Enjoy exploring the system! 🚀**
