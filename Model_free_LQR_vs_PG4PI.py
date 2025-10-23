import numpy as np
import controlgym as gym
import matplotlib.pyplot as plt
plt.style.use('ggplot')
#seaborn to create statistical plots
# Initialization
def initialize_policy(num_actions):
    return np.random.rand(num_actions)

# Other functions
def collect_trajectories(state, perturbation):

    states = []
    actions = []
    rewards = []
    states.append(state)
    K = env.K
    sigma = env.noise_cov
    mean = -np.dot(K,state) 
    std_dev = sigma
    #action = np.random.normal(mean, std_dev)
    action = mean + perturbation
    for i in range(100):
        x, z, reward, done, _ = env.step(action)
        state = env._get_mod_obs()
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        mean = -np.dot(K,state)
        action = mean
    return states, actions, rewards

def compute_returns(rewards):
    # Assume this function computes returns from rewards
    return sum(rewards)

def compute_scores(env, states, actions, returns):
    # Assume this function computes the policy gradient
    G = returns
    tau = env.sample_time
    g = env._get_mod_obs()
    T_x = np.hstack((np.eye(env.A.shape[0]),np.zeros([env.A.shape[0],1])))
    T_z = np.hstack((np.zeros([1,env.A.shape[0]]).reshape(1,env.A.shape[0]), np.array([[1]])))
    score_p = -tau*np.dot(g.T,np.dot(T_x.T,env.C.T))/(tau + env.K_d*np.dot(env.C,env.B))
    score_i = -tau*np.dot(g.T,T_z.T)/(tau + env.K_p*np.dot(env.C,env.B))
    return score_p, score_i

def update_policy_parameters(policy_gradient):
    # Assume this function updates the policy parameters using the gradient
    pass

# Training loop
rollout_len = 100
num_episodes = 10
lah=0
rea=1
if lah == 1:
    env = gym.make("lah")
if rea == 1:
    env = gym.make("rea")
env.reset()
tau = env.sample_time
alpha=0.000001 #step size

u_ext = 1*np.ones([1])
u_ext_arr = []
y_arr = []
diff_arr=[]
x_arr = []
rew_arr = []
t_arr = []
t=0
env.target = 0*np.ones(env.n_state)
env.P = 0
env.I = 0
env.D = 0
#env.K_p = 0.001
env.K_d = 0.0
env.K_i = 0.0
env.mat_fact = 0.00001
perturbation = 0.1
env.K = -1*(np.linalg.inv(1+(1/tau)*env.K_d*np.dot(env.C,env.B)))[0,0]*np.hstack((env.K_p*env.C + env.K_d*(1/tau)*np.dot(env.C,env.A-np.eye(env.A.shape[0])).reshape(1,env.A.shape[0]), np.array([env.K_i]).reshape(1,1)))
for episode in range(num_episodes):
    if episode == 0:
        x_arr =  env._get_obs().reshape(env.n_state,1)
    else:
        x_arr = np.hstack((x_arr, x.reshape(env.n_state,1)))
    g = env._get_mod_obs()

    a = -np.dot(env.K,g) #+ np.dot(env.K,g)
    states, actions, rewards = collect_trajectories(g, 0)
    temp_1 ,temp_1, rewards_1 = collect_trajectories(g, perturbation)
    env.g = g
    env.states = g[:env.n_state,:]
    env.z = g[env.n_state]
    
    x=env.states
    if episode == 0: #adding step input at the beginning
        y = np.dot(env.C,x).reshape(1,)-np.dot(env.C,env.target).reshape(1,) + u_ext.reshape(1,)
    else: #at later time steps, evolution of output is part of the modified dynamics
        y = np.dot(env.C,x).reshape(1,)-np.dot(env.C,env.target).reshape(1,) #+ u_ext.reshape(1,)
    
    diff_arr.append(u_ext-y)
    y_arr.append(y)
    u_ext_arr.append(u_ext)
    # Compute returns
    G = compute_returns(rewards)
    G_1 = compute_returns(rewards_1)
    rew_arr.append(G)
    t_arr.append(t)

    temp_1 ,temp_1, rewards_1 = collect_trajectories(g, perturbation)
    grad_K = (G_1-G)/perturbation
    env.K = env.K - alpha*grad_K

    x, z, reward, done, _ = env.step(a)
    t = t + 1

t_arr = [x + 1 for x in t_arr]
t_arr = [rollout_len*x for x in t_arr]

#for rea environment, comment when using lah environment
# #start rea
plt.figure(1)
plt.plot(figsize=(3.25, 2.5))
plt.title("Model Free LQR on a Chemical Reactor")
plt.xlabel("Number of Samples")
plt.ylabel("Reward")
plt.plot(t_arr,rew_arr)
plt.savefig("mdl_free_rea_rew_lqr", dpi=800)#, bbox_inches='tight')

plt.figure(2)
plt.plot(figsize=(3.25, 2.5))
for i in range(env.n_state):
    plt.plot(t_arr,x_arr[i])
# plt.title("Model Free LQR on an LA University hospital")
plt.title("Model Free LQR on a Chemical Reactor")
plt.xlabel("Number of Samples")
plt.ylabel("States")
plt.savefig("mdl_free_rea_state_lqr", dpi=800)#, bbox_inches='tight')

plt.figure(3)
plt.plot(figsize=(3.25, 2.5))
plt.plot(t_arr,diff_arr,label='Tracking Error')
# plt.title("Model Free LQR on an LA University hospital")
plt.title("Model Free LQR on a Chemical Reactor")
plt.xlabel("Number of Samples")
plt.ylabel("Tracking Error")
plt.legend()
plt.savefig("mdl_free_rea_track_error_lqr")#, dpi=800, bbox_inches='tight')
y_arr_lah = np.load('mdl_free_lah_track.npy')
y_arr_rea = np.load('mdl_free_rea_track.npy')
plt.figure(4)
plt.plot(figsize=(3.25, 2.5))
plt.plot(t_arr,u_ext_arr,label='Unit Step Reference')
plt.plot(t_arr,y_arr,label='LQR Controller')
plt.plot(t_arr,y_arr_rea,label='PID Controller')
plt.title("Model Free LQR vs. PG4PI on a Chemical Reactor")
plt.xlabel("Number of Samples")
plt.ylabel("Tracking")
plt.legend()
plt.savefig("mdl_free_rea_vs_pg4pi_track_lqr", dpi=800)
plt.show()
# #end rea


# #for lah environment, comment when using rea environment
# #start lah
# plt.figure(1)
# plt.plot(figsize=(3.25, 2.5))
# plt.title("Model Free LQR on an LA University hospital")
# # plt.title("Model Free LQR on a Chemical Reactor")
# plt.xlabel("Number of Samples")
# plt.ylabel("Reward")
# plt.plot(t_arr,rew_arr)
# # plt.legend()
# # plt.savefig("mdl_free_rea_rew_lqr", dpi=800)#, bbox_inches='tight')
# plt.savefig("mdl_free_lah_rew_lqr.png", dpi=800)#, bbox_inches='tight')

# plt.figure(2)
# plt.plot(figsize=(3.25, 2.5))
# for i in range(env.n_state):
#     plt.plot(t_arr,x_arr[i])
# plt.title("Model Free LQR on an LA University hospital")
# # plt.title("Model Free LQR on a Chemical Reactor")
# plt.xlabel("Number of Samples")
# plt.ylabel("States")
# # plt.legend()
# # plt.savefig("mdl_free_rea_state_lqr", dpi=800)#, bbox_inches='tight')
# plt.savefig("mdl_free_lah_state_lqr.png", dpi=800)#, bbox_inches='tight')

# plt.figure(3)
# plt.plot(figsize=(3.25, 2.5))
# plt.plot(t_arr,diff_arr,label='Tracking Error')
# # plt.title("Model Free LQR on an LA University hospital")
# plt.title("Model Free LQR on a Chemical Reactor")
# plt.xlabel("Number of Samples")
# plt.ylabel("Tracking Error")
# plt.legend()
# plt.savefig("mdl_free_lah_track_error_lqr", dpi=800)#, bbox_inches='tight')
# # plt.savefig("mdl_free_rea_track_error_lqr")#, dpi=800, bbox_inches='tight')

# y_arr_lah = np.load('mdl_free_lah_track.npy')
# y_arr_rea = np.load('mdl_free_rea_track.npy')
# plt.figure(4)
# # for i in range(env.n_state):
# #     plt.plot(t_arr,x_arr[i]-env.target[i])
# plt.plot(figsize=(3.25, 2.5))
# plt.plot(t_arr,u_ext_arr,label='Unit Step Reference')
# plt.plot(t_arr,y_arr,label='LQR Controller')
# # plt.plot(t_arr,y_arr_lah,label='PID Controller')
# plt.plot(t_arr,y_arr_rea,label='PID Controller')
# # plt.title("Model Free LQR vs. PG4PI on an LA University hospital")
# plt.title("Model Free LQR vs. PG4PI on a Chemical Reactor")
# plt.xlabel("Number of Samples")
# plt.ylabel("Tracking")
# plt.legend()
# #plt.savefig("mdl_free_lah_track_lqr", dpi=800)#, bbox_inches='tight')
# plt.savefig("mdl_free_lah_vs_pg4pi_track_lqr", dpi=800)#, bbox_inches='tight')
# # plt.savefig("mdl_free_rea_vs_pg4pi_track_lqr", dpi=800)#, bbox_inches='tight')
# plt.show()
# #end lah
