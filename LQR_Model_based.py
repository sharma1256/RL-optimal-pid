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
    for i in range(100):
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
env = gym.make("lah")
env.reset()
tau = env.sample_time

u_ext = 1*np.ones([1])
u_ext_arr = []
y_arr = []

alpha=0.001 #step size
num_episodes = 10

#x_arr = []
rew_arr = []
diff_arr = []
t_arr = []
t=0
#env.target = 1*np.random.uniform(4, 7, size=env.n_state)
env.target = 0*np.ones(env.n_state)

#Below 3 flags indicate that 
#only P controller has been implemented
env.P = 1 
env.I = 0 
env.D = 0

env.mat_fact = 0.001
env.K = -1*(np.linalg.inv(1+(1/env.sample_time)*env.K_d*np.dot(env.C,env.B)))[0,0]*np.hstack((env.K_p*env.C + env.K_d*(1/env.sample_time)*np.dot(env.C,env.A-np.eye(env.A.shape[0])).reshape(1,env.A.shape[0]), np.array([env.K_i]).reshape(1,1)))
for episode in range(num_episodes):
    #print(env._get_obs())
    g = env._get_mod_obs()
    g_temp = g
    #env.g = g
    env.states = g[:env.n_state,:]
    x= env.states
    env.z = g[env.n_state]
    if episode == 0: #adding step input at the beginning
        y = np.dot(env.C,x).reshape(1,)-np.dot(env.C,env.target).reshape(1,) + u_ext.reshape(1,)
    else: #at later time steps, evolution of output is part of the modified dynamics
        y = np.dot(env.C,x).reshape(1,)-np.dot(env.C,env.target).reshape(1,) #+ u_ext.reshape(1,)
    
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
    
    controller = gym.controllers.LQR_bench(env)
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
    if episode==0 or episode==num_episodes-1:
        print(env.K)
    #rew_arr.append(reward)


plt.figure(1)
plt.plot(figsize=(3.25, 2.5))
plt.title("Model Based LQR on an LA University hospital")
# plt.title("Model Based LQR on a Chemical Reactor")
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.plot(t_arr,rew_arr)
# plt.legend()
# plt.savefig("mdl_bld_rea_rew_lqr", dpi=800, bbox_inches='tight')
plt.savefig("mdl_bld_lah_rew_lqr.png", dpi=800)#, bbox_inches='tight')

plt.figure(2)
plt.plot(figsize=(3.25, 2.5))
for i in range(env.n_state):
    plt.plot(t_arr,x_arr[i])
plt.title("Model Based LQR on an LA University hospital")
# plt.title("Model Based LQR on a Chemical Reactor")
plt.xlabel("Episodes")
plt.ylabel("States")
# plt.legend()
# plt.savefig("mdl_bld_rea_state_lqr", dpi=800, bbox_inches='tight')
plt.savefig("mdl_bld_lah_state_lqr.png", dpi=800)#, bbox_inches='tight')

plt.figure(3)
plt.plot(figsize=(3.25, 2.5))
# for i in range(env.n_state):
#     plt.plot(t_arr,x_arr[i]-env.target[i])
# plt.plot(t_arr,u_ext_arr,label='Unit Step Input')
# plt.plot(t_arr,y_arr,label='System Output')
plt.plot(t_arr,diff_arr,label='Tracking Error')
plt.title("Model Based LQR on an LA University hospital")
# plt.title("Model Based LQR on a Chemical Reactor")
plt.xlabel("Episodes")
plt.ylabel("Tracking Error")
# plt.legend()
plt.savefig("mdl_bld_lah_track_error_lqr", dpi=800)#, bbox_inches='tight')
# plt.savefig("mdl_bld_rea_track_error_lqr", dpi=800, bbox_inches='tight')


plt.figure(4)
# for i in range(env.n_state):
#     plt.plot(t_arr,x_arr[i]-env.target[i])
plt.plot(figsize=(3.25, 2.5))
plt.plot(t_arr,u_ext_arr,label='Unit Step Input')
plt.plot(t_arr,y_arr,label='System Output')
plt.title("Model Based LQR on an LA University hospital")
# plt.title("Model Based LQR on a Chemical Reactor")
plt.xlabel("Episodes")
plt.ylabel("Tracking")
# plt.legend()
plt.savefig("mdl_bld_lah_track_lqr", dpi=800)#, bbox_inches='tight')
# plt.savefig("mdl_bld_rea_track_lqr", dpi=800, bbox_inches='tight')
plt.show()



