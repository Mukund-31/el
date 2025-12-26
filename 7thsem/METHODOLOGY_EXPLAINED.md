# Research Methodology: Digital Twin & Trace-Driven Simulation

This document explains the core methodology used to validate the Reinforcement Learning (RL) agent using the `queue_data.csv` dataset.

## 1. Concept: Digital Twin Simulation
We treat the real-world dataset (`queue_data.csv`) as a **Trace**. We do not "predict" this data; we **replay** it.

1.  **Input:** We extract the exact timestamp every customer arrived from the CSV.
2.  **Simulation:** We feed these customers into our Banking Simulation Engine.
3.  **Agent's Role:** 
    *   In the **Real World**, the number of tellers was fixed (mostly unknown, likely static).
    *   In the **Simulation**, our RL Agent has control. It observes the replay and dynamically changes the number of tellers (3, 4, ... 10).
4.  **Outcome:** We measure the performance (Wait Time, Cost) of the Agent's decisions against the original historical performance.

## 2. Comparison Metrics

| Metric | Source | Definition |
| :--- | :--- | :--- |
| **Real World Wait Time** | `queue_data.csv` | The actual time recorded in history (10.11 min). |
| **RL Wait Time** | Simulation | The calculated wait time if the RL agent had been in charge (5.35 min). |
| **Baseline Wait Time** | Simulation | The calculated wait time if a simple rule ("if queue > 5...") was used. |

## 3. Training Process (Online Learning)
We use an **Online Learning** approach because we are optimizing operations for a specific branch.

*   **Training Phase (Episodes 1-150):** 
    The Agent runs through the "Trace Day" repeatedly. 
    *   *Early Episodes:* It explores random staffing levels. Costs are high.
    *   *Learning:* It realizes that adding staff at 10:00 AM prevents a bottleneck at 10:15 AM.
*   **Convergence (Episodes 150-300):** 
    The Agent has learned the specific demand patterns of this dataset. It now executes the optimal policy.

## 4. Key Results (From Latest Run)

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Customers Served** | **560** | The RL agent successfully handled 100% of the historical volume. |
| **Avg Wait Time** | **5.35 min** | **47% Improvement** over the real-world average (10.11 min). |
| **Avg Tellers** | **6.23** | The Baseline needed **8.36** tellers to achieve similar stability. The RL agent saved **2 full salaries** per hour. |
| **Renege Rate** | **0.0%** | Zero customers left the queue (Simulation Estimate). |

## 5. Conclusion
The model demonstrates that by using **Adaptive Reinforcement Learning**, a bank branch with this specific arrival pattern could have:
1.  Reduced customer waiting times by nearly half.
2.  Done so while employing fewer staff on average (cost savings).
3.  Eliminated reliance on manual scheduling or rigid rules.
