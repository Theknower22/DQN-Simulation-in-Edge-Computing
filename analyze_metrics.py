# analyze_metrics.py
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/episode_metrics.csv")

print("\n 30 Results per Concept (Metric Summary):")
print(df.describe())

metrics = ["ExecutionTime", "Energy", "Memory", "TaskSuccessRate"]

for m in metrics:
    plt.figure(figsize=(8,4))
    plt.plot(df["Episode"], df[m], marker='o')
    plt.title(f"{m} Across 30 Episodes")
    plt.xlabel("Episode")
    plt.ylabel(m)
    plt.grid(True)
    plt.savefig(f"results/{m}_trend.png")
    plt.show()
