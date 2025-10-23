# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 12:00:46 2025

@author: User
"""

import numpy as np
import controlgym as gym
import pandas as pd
#import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results, X_TIMESTEPS
import matplotlib.pyplot as plt
import os

log_dir = "./logs/"
os.makedirs(log_dir, exist_ok=True)

env = Monitor(gym.make("lah"), log_dir)
# env = gym.make("lah")#, log_dir
env.env.init_state = np.ones(env.env.n_state)
env.env.random_init_state_cov = 0
env.env.action_limit = 1000000
# env.init_state = np.ones(env.n_state)
# env.random_init_state_cov = 0
# env.action_limit = 1000000
time_step = 20000
env.env.mat_fact = 0.01
model = PPO("MlpPolicy", 
            env,
            verbose=0, 
            learning_rate=3e-4,
            n_steps=8, #2048
            batch_size=2, #16
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.9,
            clip_range=0.2,
            ent_coef=0.0,
            device="cuda", 
            tensorboard_log=log_dir)

# Train the model
model.learn(total_timesteps=time_step)

monitor_file = os.path.join("./logs", [f for f in os.listdir("./logs") if f.startswith("monitor")][0])
df = pd.read_csv(monitor_file, skiprows=1)

# Plot raw and moving average rewards
plt.figure(figsize=(10,5))
#plt.plot(df["r"], label="Episode Reward")
plt.plot(df["r"].rolling(window=10).mean(), label="Moving Average (10)", linewidth=2)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Episode Reward Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("PPO-tuning-pid", dpi=800)
plt.show()

# Plotting results
plot_results([log_dir], time_step, X_TIMESTEPS, "PPO on lah")
plt.show()

# Save the model
model.save("ppo_env")

# Delete and reload (optional, to show usage)
del model
model = PPO.load("ppo_env")

# Run the trained agent
obs, info = env.reset()
done = False
while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    
env.close()