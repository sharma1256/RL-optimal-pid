# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 17:19:03 2025

@author: User
"""
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

lah=0
rea=1
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
num_episodes = 10

rollout_len = 100
#x_arr = []
rew_arr = []
diff_arr = []
t_arr = []
t=0

env.target = 0*np.ones(env.n_state)
env.P = 1
env.I = 1
env.D = 0
env.mat_fact = 0.001
for episode in range(num_episodes):

    g = env._get_mod_obs()
    g_temp = g

    env.states = g[:env.n_state,:]
    x= env.states
    env.z = g[env.n_state]
    y = np.dot(env.C,x).reshape(1,) -np.dot(env.C,env.target).reshape(1,)  + u_ext.reshape(1,)
    
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
    
    #LQR function referenced below updates pid parameters
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

print(env.K_p,env.K_i,env.K_d)
t_arr = [x + 1 for x in t_arr]
t_arr = [rollout_len*x for x in t_arr]


# #start rea
plt.figure(1)
plt.plot(figsize=(3.25, 2.5))
# plt.title("PG4PID on an LA Unversity hospital")
plt.title("PG4PID on a Chemical Reactor")
plt.xlabel("Number of Samples")
plt.ylabel("Reward")
plt.plot(t_arr,rew_arr)
# plt.legend()
plt.savefig("mdl_bld_rea_rew", dpi=800, bbox_inches='tight')
# plt.savefig("mdl_bld_lah_rew.png", dpi=800)#, bbox_inches='tight')

plt.figure(2)
plt.plot(figsize=(3.25, 2.5))
for i in range(env.n_state):
    plt.plot(t_arr,x_arr[i])
# plt.title("PG4PID on an LA Unversity hospital")
plt.title("PG4PID on a Chemical Reactor")
plt.xlabel("Number of Samples")
plt.ylabel("States")
# plt.legend()
plt.savefig("mdl_bld_rea_state", dpi=800, bbox_inches='tight')
# plt.savefig("mdl_bld_lah_state.png", dpi=800)#, bbox_inches='tight')

plt.figure(3)
plt.plot(figsize=(3.25, 2.5))
# for i in range(env.n_state):
#     plt.plot(t_arr,x_arr[i]-env.target[i])
# plt.plot(t_arr,u_ext_arr,label='Unit Step Input')
# plt.plot(t_arr,y_arr,label='System Output')
plt.plot(t_arr,diff_arr,label='Tracking Error')
# plt.title("PG4PID on an LA Unversity hospital")
plt.title("PG4PID on a Chemical Reactor")
plt.xlabel("Number of Samples")
plt.ylabel("Tracking Error")
plt.legend()
# plt.savefig("mdl_bld_lah_track_error", dpi=800)#, bbox_inches='tight')
plt.savefig("mdl_bld_rea_track_error", dpi=800, bbox_inches='tight')


plt.figure(4)
# for i in range(env.n_state):
#     plt.plot(t_arr,x_arr[i]-env.target[i])
plt.plot(figsize=(3.25, 2.5))
plt.plot(t_arr,u_ext_arr,label='Unit Step Input')
plt.plot(t_arr,y_arr,label='System Output')
# plt.title("PG4PID on an LA Unversity hospital")
plt.title("PG4PID on a Chemical Reactor")
plt.xlabel("Number of Samples")
plt.ylabel("Tracking")
plt.legend()
# plt.savefig("mdl_bld_lah_track", dpi=800)#, bbox_inches='tight')
plt.savefig("mdl_bld_rea_track", dpi=800, bbox_inches='tight')
plt.show()
# #end rea


# #start lah
# plt.figure(1)
# plt.plot(figsize=(3.25, 2.5))
# plt.title("PG4PID on an LA Unversity hospital")
# # plt.title("PG4PID on a Chemical Reactor")
# plt.xlabel("Number of Samples")
# plt.ylabel("Reward")
# plt.plot(t_arr,rew_arr)
# # plt.legend()
# # plt.savefig("mdl_bld_rea_rew", dpi=800, bbox_inches='tight')
# plt.savefig("mdl_bld_lah_rew.png", dpi=800)#, bbox_inches='tight')

# plt.figure(2)
# plt.plot(figsize=(3.25, 2.5))
# for i in range(env.n_state):
#     plt.plot(t_arr,x_arr[i])
# plt.title("PG4PID on an LA Unversity hospital")
# # plt.title("PG4PID on a Chemical Reactor")
# plt.xlabel("Number of Samples")
# plt.ylabel("States")
# # plt.legend()
# # plt.savefig("mdl_bld_rea_state", dpi=800, bbox_inches='tight')
# plt.savefig("mdl_bld_lah_state.png", dpi=800)#, bbox_inches='tight')

# plt.figure(3)
# plt.plot(figsize=(3.25, 2.5))
# # for i in range(env.n_state):
# #     plt.plot(t_arr,x_arr[i]-env.target[i])
# # plt.plot(t_arr,u_ext_arr,label='Unit Step Input')
# # plt.plot(t_arr,y_arr,label='System Output')
# plt.plot(t_arr,diff_arr,label='Tracking Error')
# plt.title("PG4PID on an LA Unversity hospital")
# # plt.title("PG4PID on a Chemical Reactor")
# plt.xlabel("Number of Samples")
# plt.ylabel("Tracking Error")
# plt.legend()
# plt.savefig("mdl_bld_lah_track_error", dpi=800)#, bbox_inches='tight')
# # plt.savefig("mdl_bld_rea_track_error", dpi=800, bbox_inches='tight')


# plt.figure(4)
# # for i in range(env.n_state):
# #     plt.plot(t_arr,x_arr[i]-env.target[i])
# plt.plot(figsize=(3.25, 2.5))
# plt.plot(t_arr,u_ext_arr,label='Unit Step Input')
# plt.plot(t_arr,y_arr,label='System Output')
# plt.title("PG4PID on an LA Unversity hospital")
# # plt.title("PG4PID on a Chemical Reactor")
# plt.xlabel("Number of Samples")
# plt.ylabel("Tracking")
# plt.legend()
# plt.savefig("mdl_bld_lah_track", dpi=800)#, bbox_inches='tight')
# # plt.savefig("mdl_bld_rea_track", dpi=800, bbox_inches='tight')
# plt.show()

# #end lah