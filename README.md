# DQN-Simulation-in-Edge-Computing
# Edge DQN Simulation

This project implements a **Deep Q-Learning (DQN)** based simulation for optimizing resource allocation in Edge Computing environments.

## 🔹 Components
- **edge_env.py** – Simulation environment (CPU, memory, latency, energy)
- **dqn_agent.py** – DQN agent with experience replay and epsilon decay
- **train.py** – Training loop to collect and optimize performance
- **analyze_metrics.py** – Result analysis and visualization
- **requirements.txt** – Dependencies list

## 📊 Metrics
Each simulation generates 30 results for:
- Execution Time
- Energy Consumption
- Memory Usage
- Task Success Rate

## 🧠 Run the project
```bash
pip install -r requirements.txt
python train.py
python analyze_metrics.py
