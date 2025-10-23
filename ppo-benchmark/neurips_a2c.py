# -*- coding: utf-8 -*-
"""
Created on Fri May  9 20:23:36 2025

@author: User
"""

import numpy as np
import controlgym as gym
import pandas as pd
#import gym
from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results, X_TIMESTEPS
import matplotlib.pyplot as plt
import os

log_dir = "./logs/"
os.makedirs(log_dir, exist_ok=True)

env = Monitor(gym.make("lah"), log_dir)

model = A2C("MlpPolicy", env, verbose=1, device="cuda", tensorboard_log=log_dir)

# Train the model
model.learn(total_timesteps=100000)

monitor_file = os.path.join("./logs", [f for f in os.listdir("./logs") if f.startswith("monitor")][0])
df = pd.read_csv(monitor_file, skiprows=1)

# Plot raw and moving average rewards
plt.figure(figsize=(10,5))
plt.plot(df["r"], label="Episode Reward")
plt.plot(df["r"].rolling(window=50).mean(), label="Moving Average (50)", linewidth=2)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Episode Reward Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plotting results
plot_results([log_dir], 100000, X_TIMESTEPS, "TRPO on lah")
plt.show()

# Save the model
model.save("a2c_env")

# Delete and reload (optional, to show usage)
del model
model = A2C.load("a2c_env")

# Run the trained agent
obs, info = env.reset()
done = False
while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)