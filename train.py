# train.py
from edge_env import EdgeEnv
from dqn_agent import Agent
import pandas as pd
import numpy as np
import os

os.makedirs("results", exist_ok=True)

env = EdgeEnv()
agent = Agent(state_dim=4, action_dim=3)

episodes = 30
metrics = {"Episode": [], "ExecutionTime": [], "Energy": [], "Memory": [], "TaskSuccessRate": []}

for episode in range(1, episodes + 1):
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.act(state)
        next_state, reward, done, _, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        agent.replay()
        state = next_state
        total_reward += reward

    # Collect metrics
    exec_time = env.exec_time
    energy_used = env.energy_used
    memory_used = env.memory_used / env.steps
    success_rate = env.tasks_completed / env.total_tasks if env.total_tasks > 0 else 0

    metrics["Episode"].append(episode)
    metrics["ExecutionTime"].append(exec_time)
    metrics["Energy"].append(energy_used)
    metrics["Memory"].append(memory_used)
    metrics["TaskSuccessRate"].append(success_rate)

    print(f"Episode {episode:02d}: Time={exec_time:.3f}s | Energy={energy_used:.3f} | "
          f"Memory={memory_used:.2f} | Success={success_rate*100:.1f}%")

# Save to CSV
df = pd.DataFrame(metrics)
df.to_csv("results/episode_metrics.csv", index=False)
print("\n All metrics saved to results/episode_metrics.csv")
