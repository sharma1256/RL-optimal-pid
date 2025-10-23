# -*- coding: utf-8 -*-
"""
Created on Mon May 12 21:11:52 2025

@author: User
"""

#import gymnasium as gym
import controlgym as gym
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
plt.style.use('ggplot')
# ==== CONFIGURATION ====
ENV_ID = "lah"
N_RUNS = 4  
TIMESTEPS = 80_000
WINDOW = 50  # For moving average
LOG_ROOT = "./ppo_run"

# Clean previous logs
if os.path.exists(LOG_ROOT):
    shutil.rmtree(LOG_ROOT)
os.makedirs(LOG_ROOT)

all_rewards = []

for run in range(N_RUNS):
    print(f"Training run {run + 1}/{N_RUNS}")
    run_log_dir = os.path.join(LOG_ROOT, f"run_{run}")
    os.makedirs(run_log_dir, exist_ok=True)

    env = Monitor(gym.make(ENV_ID), run_log_dir)
    env.env.init_state = np.ones(env.env.n_state)
    env.env.random_init_state_cov = 0
    env.env.action_limit = 1000000
    env.env.mat_fact = 0.01
#    model = PPO("MlpPolicy", env, verbose=0, device="cuda")
    model = PPO("MlpPolicy", 
                env,
                verbose=0, 
                learning_rate=3e-4,
                n_steps=8,
                batch_size=2,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.9,
                clip_range=0.2,
                ent_coef=0.0,
                device="cuda")
    model.learn(total_timesteps=TIMESTEPS)

    # Load monitor log
    monitor_file = [f for f in os.listdir(run_log_dir) if f.startswith("monitor")][0]
    df = pd.read_csv(os.path.join(run_log_dir, monitor_file), skiprows=1)
    all_rewards.append(df["r"].values)

# === ALIGN REWARD LENGTHS ===
max_len = max(len(r) for r in all_rewards)
for i in range(len(all_rewards)):
    all_rewards[i] = np.pad(all_rewards[i], (0, max_len - len(all_rewards[i])), constant_values=np.nan)

rewards_array = np.array(all_rewards)  # shape: (n_runs, max_len)

# === COMPUTE STATS ===
df_rewards = pd.DataFrame(rewards_array.T)  # shape: (episodes, runs)
rolling_mean = df_rewards.rolling(window=WINDOW, min_periods=1).mean()
rolling_std = df_rewards.rolling(window=WINDOW, min_periods=1).std()
rolling_se = rolling_std / np.sqrt(N_RUNS)

# === PLOT ===
# episodes = range(len(df))
episodes = range(1, len(df) + 1)
new_episodes = [100*(x) for x in episodes]
mean_reward = rolling_mean.mean(axis=1)
std_reward = rolling_std.mean(axis=1)
se_reward = rolling_se.mean(axis=1)
# x_ax = range(0, TIMESTEPS+1, 100)
plt.plot(figsize=(3.25, 2.5))
plt.plot(new_episodes, mean_reward, label="Mean Reward")
plt.fill_between(new_episodes,mean_reward.index, mean_reward - std_reward, mean_reward + std_reward,
                 alpha=0.3, label="±1 Std Dev")
# plt.fill_between(new_episodes,mean_reward.index, mean_reward - se_reward, mean_reward + se_reward,
#                  alpha=0.2, label="±1 Std Error", color="orange")
plt.xlabel("Number of Samples")
plt.ylabel("Reward")
# plt.title(f"PPO on {ENV_ID} – Avg Reward over {N_RUNS} Runs")
plt.title(f"PID Tuning using PPO on {ENV_ID}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("PPO-tuning-pid", dpi=800)
plt.show()
