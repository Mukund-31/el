# 🤖 Dynamic vs Hardcoded: What the AI Actually Controls

## Quick Answer

**The forecaster model is REAL AI** - it learns and predicts dynamically!  
**But many system parameters are hardcoded** for realistic simulation.

Let me break down exactly what's dynamic vs hardcoded:

---

## 🧠 DYNAMIC (AI-Controlled)

### 1. **Bayesian Forecaster** (forecaster.py)
**What it does:** Predicts future customer arrivals

**Dynamic components:**
- ✅ **LSTM Neural Network** - Learns patterns from historical data
- ✅ **Predictions** - Mean, standard deviation, UCB/LCB calculated in real-time
- ✅ **Uncertainty Quantification** - Bayesian dropout estimates confidence
- ✅ **Learning** - Model improves as it sees more data

**How it works:**
```python
# The model architecture (lines 96-143 in forecaster.py)
class BayesianLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=32, num_layers=2):
        # LSTM layers with dropout for uncertainty
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           dropout=0.2, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # Predicts arrivals
```

**What it learns:**
- Time-of-day patterns (mornings are busier than afternoons)
- Day-of-week patterns (Mondays vs Fridays)
- Correlation between anger and arrivals
- Seasonal trends (if you run it long enough)

**Training:**
- Pre-trained on synthetic data (forecaster_weights.pth)
- Continues learning during simulation via `update_history()`
- Uses last 10 time steps to predict next interval

---

### 2. **Optimization Agent** (optimization_agent.py)
**What it does:** Makes staffing decisions (add/remove tellers, give breaks)

**Dynamic components:**
- ✅ **Cost Calculation** - Computes real-time costs based on current state
- ✅ **Decision Logic** - Evaluates multiple actions and picks best one
- ✅ **Threshold Adaptation** - Uses forecaster's UCB (dynamic prediction)

**How it works:**
```python
# Decision logic (lines 400-453 in optimization_agent.py)
def decide(self, state: SystemState):
    # Calculate cost for each possible action
    costs = {
        Action.DO_NOTHING: self._cost_do_nothing(state),
        Action.ADD_TELLER: self._cost_add_teller(state),
        Action.REMOVE_TELLER: self._cost_remove_teller(state),
        Action.GIVE_BREAK: self._cost_give_break(state),
    }
    
    # Pick action with LOWEST cost
    best_action = min(costs, key=costs.get)
    return best_action
```

**What it considers:**
- Current queue length (dynamic)
- Predicted arrivals from forecaster (dynamic)
- Teller fatigue levels (dynamic)
- Lobby anger (dynamic)
- Wait times (dynamic)

**Cost formula (dynamic):**
```
Total Cost = 
    (Staffing Cost × num_tellers) +
    (Wait Cost × avg_wait_time) +
    (Renege Cost × renege_rate) +
    (Fatigue Penalty × avg_fatigue)
```

---

### 3. **Simulation State** (simulation_engine.py)
**What it does:** Tracks real-time system state

**Dynamic components:**
- ✅ **Queue length** - Changes as customers arrive/leave
- ✅ **Teller fatigue** - Increases with work, decreases with breaks
- ✅ **Lobby anger** - Calculated from current wait times
- ✅ **Customer patience** - Each customer has random patience
- ✅ **Service times** - Depends on task complexity AND teller fatigue

**Fatigue dynamics (dynamic):**
```python
# Fatigue increases with each customer served
fatigue_increase = 0.02 * complexity  # More complex = more tiring

# Efficiency drops as fatigue increases
efficiency = max(0.3, 1.0 - 0.6 * fatigue)  # 100% → 40%

# Service time increases when tired
actual_service_time = base_time * complexity / efficiency
```

**Lobby anger (dynamic):**
```python
# Calculated from CURRENT wait times
median_wait = median(all_customer_wait_times)
lobby_anger = min(10, median_wait / W_REF)  # W_REF = 10 min
```

---

## 🔧 HARDCODED (Fixed Parameters)

### 1. **Customer Arrival Rates** (producer.py)
**Hardcoded schedule:**
```python
# Lines 64-68 in producer.py
self.schedule = [
    (9, 11, self._linear_ramp(2, 6)),     # Morning: 2-6 customers/hour
    (11, 13, lambda h: 12),                # Lunch: 12 customers/hour
    (13, 15, self._linear_ramp(12, 4)),    # Afternoon: 12-4 customers/hour
    (15, 17, lambda h: 3),                 # Late: 3 customers/hour
]
```

**Why hardcoded?**
- Simulates realistic bank demand patterns
- You can change these values, but they don't adapt automatically

---

### 2. **Customer Psychological Attributes** (producer.py)
**Hardcoded distributions:**
```python
# Patience (lines 119-131)
beta = 10.0 if is_lunch else 15.0  # FIXED values
patience = np.random.exponential(beta)  # Random, but from FIXED distribution

# Complexity (lines 133-145)
complexity = np.random.normal(1.0, 0.2)  # FIXED mean=1.0, std=0.2

# Contagion (lines 147-158)
contagion = np.random.uniform(0, 1)  # FIXED uniform distribution
```

**Why hardcoded?**
- Models realistic human psychology
- Values chosen based on research/assumptions
- Each customer gets random values, but from fixed distributions

---

### 3. **Fatigue Parameters** (simulation_engine.py)
**Hardcoded constants:**
```python
# Lines 56-60
FATIGUE_INCREMENT = 0.02  # How much fatigue per customer
FATIGUE_RECOVERY_RATE = 0.1  # How fast breaks help
MIN_EFFICIENCY = 0.3  # Tired tellers still work at 30% efficiency
EFFICIENCY_SLOPE = 0.6  # How much fatigue affects efficiency
```

**Why hardcoded?**
- Based on human factors research
- Could be learned from real data, but we don't have it

---

### 4. **Cost Weights** (optimization_agent.py)
**Hardcoded cost parameters:**
```python
# Lines 85-89
STAFFING_COST_PER_TELLER = 50.0  # $/hour per teller
WAIT_TIME_COST = 5.0  # $/minute of customer wait
RENEGE_COST = 100.0  # $ per lost customer
FATIGUE_PENALTY = 30.0  # $ per unit of fatigue
```

**Why hardcoded?**
- Business policy decisions (what's a customer worth?)
- Could be tuned, but requires domain expertise

---

### 5. **Decision Thresholds** (optimization_agent.py)
**Hardcoded rules:**
```python
# Lines 250-280
FATIGUE_BREAK_THRESHOLD = 0.7  # Give break when fatigue > 70%
BURNOUT_THRESHOLD = 0.85  # Emergency threshold
MIN_TELLERS = 1  # Never go below 1 teller
MAX_TELLERS = 10  # Cap at 10 tellers
```

**Why hardcoded?**
- Safety constraints
- Business rules (can't have 0 tellers!)

---

## 📊 Summary Table

| Component | Dynamic? | What Decides It |
|-----------|----------|-----------------|
| **Arrival predictions** | ✅ YES | Bayesian LSTM learns patterns |
| **Arrival rates (schedule)** | ❌ NO | Hardcoded in producer.py |
| **Customer patience** | 🟡 RANDOM | Random from fixed distribution |
| **Customer complexity** | 🟡 RANDOM | Random from fixed distribution |
| **Teller fatigue** | ✅ YES | Calculated from work done |
| **Lobby anger** | ✅ YES | Calculated from wait times |
| **Service time** | ✅ YES | Depends on complexity + fatigue |
| **Staffing decisions** | ✅ YES | Optimizer picks best action |
| **Cost weights** | ❌ NO | Hardcoded business values |
| **Break threshold** | ❌ NO | Hardcoded at 70% fatigue |
| **Queue length** | ✅ YES | Real-time arrivals - departures |

---

## 🎯 What This Means

### The AI is REAL but CONSTRAINED:

1. **Forecaster learns** from data and makes predictions
2. **Optimizer makes decisions** based on those predictions
3. **But** it operates within hardcoded constraints (costs, thresholds, distributions)

### Think of it like a self-driving car:
- ✅ **AI decides**: When to brake, accelerate, turn (like our optimizer)
- ✅ **AI predicts**: Where other cars will be (like our forecaster)
- ❌ **Hardcoded**: Speed limits, safety rules, physics (like our parameters)

---

## 🔬 How to Make It MORE Dynamic

If you wanted to make more things adaptive:

### 1. **Learn Arrival Patterns**
Instead of hardcoded schedule, learn from historical data:
```python
# Could train a model to predict λ(t) from features
arrival_rate = learned_model.predict(hour, day, weather, etc.)
```

### 2. **Adaptive Cost Weights**
Learn what customers actually value:
```python
# Multi-armed bandit to learn optimal cost weights
cost_weights = bandit.select_weights(based_on_customer_feedback)
```

### 3. **Reinforcement Learning for Decisions**
Replace rule-based optimizer with RL:
```python
# Deep Q-Network learns optimal policy
action = dqn.select_action(state)  # Learns from rewards
```

### 4. **Learn Fatigue Models**
Fit fatigue dynamics to real worker data:
```python
# Personalized fatigue models per teller
fatigue_model = train_on_real_teller_data(teller_id)
```

---

## 🎓 Bottom Line

**What's AI:**
- Forecasting (LSTM predicts arrivals)
- Decision-making (optimizer picks actions)
- State tracking (queue, fatigue, anger)

**What's Hardcoded:**
- Arrival rate schedule
- Psychological distributions
- Cost parameters
- Safety thresholds

**The system is a HYBRID:**
- AI makes intelligent decisions within realistic constraints
- Like a chess AI: it learns strategy, but the rules are fixed!

---

**Want to see the AI in action?**
Watch the dashboard - every time you see:
- 🎯 Decision: ADD_TELLER → AI predicted busy period
- 🎯 Decision: GIVE_BREAK → AI detected high fatigue
- 📈 Prediction updates → LSTM learning from new data

That's the AI working! 🤖
