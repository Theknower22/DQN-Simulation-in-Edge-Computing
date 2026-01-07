# analyze.py
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Read training results
try:
    df = pd.read_csv("results/episode_rewards.csv")
except FileNotFoundError:
    print(" The results file 'episode_rewards.csv' was not found.")
    print(" Make sure you have run train.py first to generate the results.")
    exit()

# Basic statistics
avg_reward = df["Reward"].mean()
max_reward = df["Reward"].max()
min_reward = df["Reward"].min()
std_reward = df["Reward"].std()

print("DQN Simulation Results")
print("────────────────────────────")
print(f"Total Episodes: {len(df)}")
print(f"Average Reward: {avg_reward:.2f}")
print(f"Max Reward: {max_reward:.2f}")
print(f"Min Reward: {min_reward:.2f}")
print(f"Standard Deviation: {std_reward:.2f}")
print("────────────────────────────")

# Plot reward curve
plt.figure(figsize=(10, 5))
plt.plot(df["Episode"], df["Reward"], color="royalblue", linewidth=2)
plt.title("DQN Training Performance Over Episodes")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("results/analyzed_performance.png")
plt.show()

# Save summary file
summary = f"""
DQN Simulation Summary
───────────────────────────────
Total Episodes     : {len(df)}
Average Reward     : {avg_reward:.2f}
Max Reward         : {max_reward:.2f}
Min Reward         : {min_reward:.2f}
Standard Deviation : {std_reward:.2f}
───────────────────────────────
"""
with open("results/summary.txt", "w", encoding="utf-8") as f:
    f.write(summary)

print("\n Summary report saved to results/summary.txt")
print("Performance chart saved to results/analyzed_performance.png")
