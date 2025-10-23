# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 17:19:03 2025

@author: User
"""
import numpy as np
import controlgym as gym
import numpy as np
import controlgym as gym
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def collect_trajectories(state):
    states = []
    actions = []
    rewards = []
    states.append(state)
    K = env.K
    sigma = env.noise_cov
    sigma_k = np.dot(env._get_mod_obs(),env._get_mod_obs().T)
    mean = -np.dot(K,state)
    std_dev = sigma
    action = np.random.normal(mean, std_dev)
    for i in range(rollout_len):
        x, z, reward, done, _ = env.step(action)
        state = env._get_mod_obs()
        sigma_k = sigma_k + np.dot(state,state.T)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        mean = -np.dot(K,state)
        action = np.random.normal(mean, std_dev)
    return states, actions, rewards, sigma_k

def compute_returns(rewards):
    # Assume this function computes returns from rewards
    return sum(rewards)
sup_rew_arr =[]
rollout_len = 100
num_episodes = 10
lah=1
rea=0
for q in range(100):
    if lah == 1:
        env = gym.make("lah")
    if rea == 1:
        env = gym.make("rea")
    env.reset()
    tau = env.sample_time
    
    u_ext = 1*np.ones([1])
    u_ext_arr = []
    y_arr = []
    
    alpha=0.001 #step size
    
    #x_arr = []
    rew_arr = []
    diff_arr = []
    t_arr = []
    t=0
    env.target = 0*np.random.uniform(4, 7, size=env.n_state)
    env.P = 1
    env.I = 1
    env.D = 0
    env.mat_fact = 0.001
    for episode in range(num_episodes):
        #print(env._get_obs())
        g = env._get_mod_obs()
        g_temp = g
        #env.g = g
        env.states = g[:env.n_state,:]
        x= env.states
        env.z = g[env.n_state]
        y = np.dot(env.C,x).reshape(1,) + u_ext.reshape(1,)
        
        if episode == 0:
            e_agg = y #- np.dot(env.C,x_goal).reshape(1,)
            x_arr =  env._get_obs().reshape(env.n_state,1)
        else:
            e_agg = e_agg + y
            x_arr = np.hstack((x_arr, x.reshape(env.n_state,1)))
    
        diff_arr.append(u_ext-y)
        u_ext_arr.append(u_ext[0])
        y_arr.append(y[0])
        #x.append(g)
        states, actions, rewards, sigma_k = collect_trajectories(g)
        env.sigma_k = sigma_k
        
        controller = gym.controllers.LQR(env)
        a = controller.select_action(g) 
        
        env.g = g_temp
        env.states = g[:env.n_state,:]
        env.z = g[env.n_state]
        # Compute returns
        G = compute_returns(rewards)
        rew_arr.append(G)
        t_arr.append(t)

        x, z, reward, done, _ = env.step(a)
    
        t = t + 1

    if q == 0:
        sup_rew_arr = (1/1)*np.array(rew_arr)
    else: sup_rew_arr = np.vstack((sup_rew_arr,(1/1)*np.array(rew_arr)))

t_arr = [x + 1 for x in t_arr]
t_arr = [rollout_len*x for x in t_arr]


# #start lah
np.save("mdl_bld_lah.npy", sup_rew_arr)
np.save("t_arr.npy", t_arr)
y_mean = sup_rew_arr.mean(axis=0)            # Mean of trajectories
y_std = sup_rew_arr.std(axis=0)              # Standard deviation of trajectories
y_upper = y_mean + y_std                      # Upper band (mean + std)
y_lower = y_mean - y_std     
plt.figure(1)
for trajectory in sup_rew_arr:
    plt.plot(t_arr, trajectory, color='gray', alpha=0.3, linewidth=0.8)

plt.plot(t_arr, y_mean, label='Mean Trajectory', color='blue', linewidth=2)
# Plot the statistical band
plt.fill_between(t_arr, y_lower, y_upper, color='blue', alpha=0.2, label='1-SD Band')
plt.xlabel("Number of Samples")
plt.ylabel("Accumulated Reward")
plt.title("Model Building PID for LA Unversity hospital")
plt.legend()
plt.grid(True)
plt.savefig("mdl_bld_lah_band_rew.png", dpi=800, bbox_inches='tight')
plt.show()
# #end lah


# # #start rea
# np.save("mdl_bld_chem.npy", sup_rew_arr)
# # np.save("mdl_bld_lah.npy", sup_rew_arr)
# np.save("t_arr.npy", t_arr)
# y_mean = sup_rew_arr.mean(axis=0)            # Mean of trajectories
# y_std = sup_rew_arr.std(axis=0)              # Standard deviation of trajectories
# y_upper = y_mean + y_std                      # Upper band (mean + std)
# y_lower = y_mean - y_std     

# plt.figure(1)
# #plt.title("Model Free PI for LA Unversity hospital")
# #plt.title("Model Free PI for a Checmial Reactor")
# for trajectory in sup_rew_arr:
#     plt.plot(t_arr, trajectory, color='gray', alpha=0.3, linewidth=0.8)

# # Plot the central tendency
# plt.plot(t_arr, y_mean, label='Mean Trajectory', color='blue', linewidth=2)

# # Plot the statistical band
# plt.fill_between(t_arr, y_lower, y_upper, color='blue', alpha=0.2, label='1-SD Band')
# plt.xlabel("Number of Samples")
# plt.ylabel("Accumulated Reward")
# # plt.title("Model Building PID for LA Unversity hospital")
# plt.title("Model Building PID for Chemical Reactor")
# plt.legend()
# plt.grid(True)
# # plt.savefig("mdl_bld_lah_band_rew.png", dpi=800, bbox_inches='tight')
# plt.savefig("mdl_bld_rea_band_rew.png", dpi=800, bbox_inches='tight')
# # Show the plot
# plt.show()
# # #end rea
