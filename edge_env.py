# edge_env.py
import gymnasium as gym
import numpy as np

class EdgeEnv(gym.Env):
    def __init__(self):
        super(EdgeEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
        self.reset()

    def reset(self, *, seed=None, options=None):
        self.state = np.random.rand(4)
        self.energy = 1.0
        self.steps = 0
        self.exec_time = 0.0
        self.energy_used = 0.0
        self.memory_used = 0.0
        self.tasks_completed = 0
        self.total_tasks = 0
        return self.state, {}

    def step(self, action):
        cpu, mem, latency, energy = self.state
        reward = 0

        # Simulate dynamic environment
        exec_time = np.random.uniform(0.1, 0.4)  # simulated execution time per step
        energy_cost = np.random.uniform(0.01, 0.05)
        mem_usage = np.random.uniform(0.3, 0.8)

        if action == 1:  # allocate CPU
            reward += 1.0 - cpu
            cpu = np.clip(cpu + np.random.uniform(-0.05, 0.15), 0, 1)
        elif action == 2:  # offload task
            reward += 1.5 - latency
            latency = np.clip(latency - np.random.uniform(0.05, 0.2), 0, 1)
        else:  # idle
            reward -= 0.1

        self.energy -= energy_cost
        self.steps += 1
        self.exec_time += exec_time
        self.energy_used += energy_cost
        self.memory_used += mem_usage
        self.total_tasks += 1
        if reward > 0:
            self.tasks_completed += 1

        done = self.steps >= 100 or self.energy <= 0
        self.state = np.array([cpu, mem, latency, self.energy], dtype=np.float32)
        return self.state, reward, done, False, {}
