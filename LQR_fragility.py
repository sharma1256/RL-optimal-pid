# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 17:19:03 2025

@author: User
"""
import numpy as np
import controlgym as gym
import matplotlib.pyplot as plt

from numpy.linalg import inv
import logging
from scipy.linalg import solve_discrete_are, solve, LinAlgError
plt.style.use('ggplot')

def collect_trajectories(state):
    states = []
    actions = []
    rewards = []
    states.append(state)
    K = env.K
    sigma = env.noise_cov
    sigma_k = np.dot(env._get_mod_obs(),env._get_mod_obs().T)
    mean = np.dot(K,state)
    std_dev = sigma
    action = np.random.normal(mean, std_dev)
    for i in range(100):
        x, z, reward, done, _ = env.step(action)
        state = env._get_mod_obs()
        sigma_k = sigma_k + np.dot(state,state.T)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        mean = np.dot(K,state)
        action = np.random.normal(mean, std_dev)
    return states, actions, rewards, sigma_k

def compute_returns(rewards):
    # Assume this function computes returns from rewards
    return sum(rewards)
env = gym.make("rea")
env.reset()
tau = env.sample_time

u_ext = 1*np.ones([1])
u_ext_arr = []
y_arr = []

alpha=0.001 #step size
rollout_len = 100
num_episodes = 10

#x_arr = []
rew_arr = []
diff_arr = []
t_arr = []
t=0
env.target = 0*np.ones(env.n_state)
env.P = 1
env.I = 0
env.D = 0

env.mat_fact = 0.01

#system matrix error
eps = 0.01

#rea environment matrix imported manually
env.A = np.array([
    [0.5623, -0.01642,  0.01287, -0.0161,   0.02094, -0.02988,  0.0183,   0.00874],
    [0.102,   0.6114,  -0.02468,  0.02468, -0.03005,  0.04195, -0.02559,  0.03889],
    [0.1361,  0.2523,   0.641,   -0.03404,  0.03292, -0.04296,  0.02588,  0.08467],
    [0.09951, 0.2859,   0.3476,   0.6457,  -0.03249,  0.03316, -0.01913,  0.1103 ],
    [-0.04794, 0.08708,  0.3297,   0.3102,   0.6201,  -0.03015,  0.01547,  0.08457],
    [-0.1373, -0.1224,   0.1705,   0.3106,   0.191,    0.5815,  -0.01274,  0.05394],
    [-0.1497, -0.1692,   0.1165,   0.2962,   0.1979,   0.07631,  0.5242,   0.04702],
    [0,       0,        0,        0,        0,        0,        0,        0.6065]
])
env.A = env.A + eps*np.eye(env.A.shape[0])
A_bar = np.vstack((np.hstack((env.A, np.zeros([env.A.shape[0],env.C.shape[0]]))),
                  np.hstack((env.C,np.array([[1]])))))
A_bar_temp = A_bar + eps*np.eye(A_bar.shape[0])
B_bar = np.vstack((env.B,np.zeros([env.C.shape[0]])))
R = np.identity(env.n_action)
S = np.zeros([A_bar.shape[0],env.B.shape[1]])

Q_mod = np.vstack((np.hstack((np.dot(env.C.T,env.C),np.zeros([env.A.shape[0],1]))),np.hstack((np.zeros([1,env.A.shape[0]]),np.zeros([1,1])))))
P = solve_discrete_are(A_bar_temp, B_bar, Q_mod, R, e=None, s=S)
env.K = np.dot(np.linalg.inv(R+np.dot(B_bar.T,(np.dot(P,B_bar)))),np.dot(B_bar.T,(np.dot(P,A_bar))))
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
    
    #controller = gym.controllers.LQR_bench(env)
    #a = controller.select_action(g)
    mean = -np.dot(env.K,g)
    #std_dev = env.noise_cov
    a = mean
    # Collect trajectories  
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

t_arr = [x + 1 for x in t_arr]
t_arr = [rollout_len*x for x in t_arr]

plt.figure(1)
plt.plot(figsize=(3.25, 2.5))
# plt.title("LQR on an LA Unversity hospital")
plt.title("Model Based LQR on a Chemical Reactor")
plt.xlabel("Number of Samples")
plt.ylabel("Reward")
plt.plot(t_arr,rew_arr)
# plt.legend()
plt.savefig("rea_rew_lqr", dpi=800, bbox_inches='tight')

plt.figure(2)
plt.plot(figsize=(3.25, 2.5))
for i in range(env.n_state):
    plt.plot(t_arr,x_arr[i])
# plt.title("LQR on an LA Unversity hospital")
plt.title("Model Based LQR on a Chemical Reactor")
plt.xlabel("Number of Samples")
plt.ylabel("States")
# plt.legend()
plt.savefig("rea_state_lqr", dpi=800, bbox_inches='tight')
# plt.savefig("mdl_bld_lah_state_lqr.png", dpi=800)#, bbox_inches='tight')

plt.figure(3)
plt.plot(figsize=(3.25, 2.5))
plt.plot(t_arr,diff_arr,label='Tracking Error')
plt.title("Model Based LQR on a Chemical Reactor")
plt.xlabel("Number of Samples")
plt.ylabel("Tracking Error")
plt.legend()
plt.savefig("rea_track_error_lqr", dpi=800, bbox_inches='tight')


plt.figure(4)
plt.plot(figsize=(3.25, 2.5))
plt.plot(t_arr,u_ext_arr,label='Unit Step Reference')
plt.plot(t_arr,y_arr,label='System Output')
plt.title("Model Based LQR on a Chemical Reactor")
plt.xlabel("Number of Samples")
plt.ylabel("Tracking")
plt.legend()
plt.savefig("rea_track_lqr", dpi=800, bbox_inches='tight')
plt.show()