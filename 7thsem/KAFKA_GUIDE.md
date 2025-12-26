# Kafka Integration - Quick Start Guide

## 🎯 What's New

Your dashboard now sends **real-time simulation events** to Kafka! You can:
1. See events in Kafka UI
2. Monitor RL decisions in real-time
3. Build downstream analytics

---

## 🚀 Step-by-Step Setup

### Step 1: Start Kafka (if not running)

```bash
cd c:\Users\mukun\Desktop\el\7thsem
docker-compose up -d
```

**Wait 30 seconds** for Kafka to fully start.

### Step 2: Verify Kafka is Running

Open Kafka UI in browser:
```
http://localhost:8080
```

You should see the Kafka UI dashboard.

### Step 3: Start the Consumer (Optional - to see logs)

Open a **new terminal**:
```bash
cd c:\Users\mukun\Desktop\el\7thsem
python kafka_consumer.py
```

You'll see:
```
✅ Kafka Consumer connected to localhost:9092
📥 Listening to topic: queue-events
🎧 Starting to consume events...
```

### Step 4: Start the Dashboard

```bash
streamlit run dashboard.py --server.port 8503
```

### Step 5: Run Simulation

1. Go to http://localhost:8503
2. Click **"🔄 Reset"** (important!)
3. Click **"▶️ Start Real-Time Simulation"**

---

## 📊 What You'll See

### In the Dashboard:
- Queue length building up
- RL agent adding tellers
- Renege rate increasing when wait is high
- Decisions changing from MAINTAIN to ADD_TELLER

### In Kafka Consumer Terminal:
```
📊 Queue State: 15 customers, 1 tellers
👥 Arrivals: 12 customers at hour 9
🤖 RL Decision: ADD_TELLER (confidence: 0.85)
   Before: 1 tellers, 15 queue
   After:  2 tellers, 15 queue
```

### In Kafka UI (http://localhost:8080):
1. Click **"Topics"** in left menu
2. Click **"queue-events"** topic
3. Click **"Messages"** tab
4. You'll see real-time events streaming in!

---

## 📋 Event Types

### 1. QUEUE_STATE
Sent every 10 seconds with current system state:
```json
{
  "timestamp": "2024-03-30T10:15:00",
  "event_type": "QUEUE_STATE",
  "data": {
    "num_tellers": 3,
    "queue_length": 12,
    "avg_wait": 8.5,
    "renege_rate": 2.3,
    "arrivals": 8,
    "served": 9,
    "reneged": 0
  }
}
```

### 2. ARRIVAL
Sent when customers arrive:
```json
{
  "timestamp": "2024-03-30T10:15:00",
  "event_type": "ARRIVAL",
  "data": {
    "num_arrivals": 8,
    "hour": 10
  }
}
```

### 3. RL_DECISION
Sent when RL agent changes staffing:
```json
{
  "timestamp": "2024-03-30T10:15:00",
  "event_type": "RL_DECISION",
  "data": {
    "action": "ADD_TELLER",
    "confidence": 0.87,
    "state_before": {"num_tellers": 2, "queue_length": 15},
    "state_after": {"num_tellers": 3, "queue_length": 15}
  }
}
```

---

## 🔧 Troubleshooting

### "Kafka not available" warning
**Solution:** Start Kafka first:
```bash
docker-compose up -d
```

### No events in Kafka UI
**Solution:** 
1. Make sure simulation is running
2. Refresh Kafka UI page
3. Check topic name is "queue-events"

### Consumer not receiving events
**Solution:**
1. Stop consumer (Ctrl+C)
2. Restart: `python kafka_consumer.py`
3. Start simulation again

---

## 🎓 For Your Thesis

**What to Screenshot:**
1. Kafka UI showing "queue-events" topic with messages
2. Consumer terminal showing RL decisions
3. Dashboard + Kafka UI side-by-side

**What to Explain:**
> "The system implements a producer-consumer architecture using Apache Kafka for real-time event streaming. The simulation dashboard produces queue state events, customer arrivals, and RL agent decisions to a Kafka topic. This enables real-time monitoring, downstream analytics, and demonstrates production-ready deployment architecture."

---

## ✅ Success Criteria

You know it's working when:
- ✅ Dashboard shows changing teller counts
- ✅ Renege rate increases when queue is high
- ✅ Kafka UI shows messages in "queue-events" topic
- ✅ Consumer terminal logs RL decisions

**You now have a complete event-driven ML system!** 🎉
