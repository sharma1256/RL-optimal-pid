import numpy as np
import controlgym as gym
import matplotlib.pyplot as plt
plt.style.use('ggplot')
#seaborn to create statistical plots
# Initialization



def initialize_policy(num_actions):
    return np.random.rand(num_actions)

# Other functions
def collect_trajectories(state):

    states = []
    actions = []
    rewards = []
    states.append(state)
    K = env.K
    sigma = env.noise_cov
    mean = -np.dot(K,state)
    std_dev = sigma
    action = np.random.normal(mean, std_dev)
    for i in range(rollout_len):
        x, z, reward, done, _ = env.step(action)
        state = env._get_mod_obs()
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        mean = -np.dot(K,state)
        action = np.random.normal(mean, std_dev)
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
alpha=0.01 #step size

u_ext = 1*np.ones([1])
u_ext_arr = []
y_arr = []
diff_arr=[]
x_arr = []
rew_arr = []
t_arr = []
t=0
env.target = 0*np.random.uniform(4, 7, size=env.n_state)
env.P = 1
env.I = 1
env.D = 0
env.mat_fact = 0.01
k_p_arr=[]
k_i_arr=[]
k_d_arr=[]
u_arr=[]
for episode in range(num_episodes):
    if episode == 0:
        x_arr =  env._get_obs().reshape(env.n_state,1)
    else:
        x_arr = np.hstack((x_arr, x.reshape(env.n_state,1)))
    g = env._get_mod_obs()
    #x.append(g)
    mean = -np.dot(env.K,g)
    
    std_dev = env.noise_cov
    a = np.random.normal(mean, std_dev)
    # Collect trajectories
    
    states, actions, rewards = collect_trajectories(g)
    
    env.g = g
    env.states = g[:env.n_state,:]
    env.z = g[env.n_state]
    
    x=env.states
    y = np.dot(env.C,x).reshape(1,) + u_ext.reshape(1,)
    diff_arr.append(u_ext-y)
    y_arr.append(y)
    u_ext_arr.append(u_ext)
    k_p_arr.append(env.K_p)
    k_i_arr.append(env.K_i)
    k_d_arr.append(env.K_d)
    u_arr.append(a[0,0])
    # Compute returns
    G = compute_returns(rewards)
    rew_arr.append(G)
    t_arr.append(1*(t))
    # Compute policy gradient
    score_p, score_i = compute_scores(env, states, actions, G)
    
    env.K_p = (env.K_p - alpha*(a-mean)*G*score_p)[0,0]
    env.K_i = (env.K_i - alpha*(a-mean)*G*score_i)[0,0]
    #env.K = -(1+(1/tau)*env.K_d*(np.dot(env.C,env.B))[0])^(-1)*np.array([[env.K_p*env.C + env.K_d*(1/tau)*np.dot(env.C,env.A-np.eye([env.A.shape[0]])), env.K_i]])
    env.K = -1*(np.linalg.inv(1+(1/tau)*env.K_d*np.dot(env.C,env.B)))[0,0]*np.hstack((env.K_p*env.C + env.K_d*(1/tau)*np.dot(env.C,env.A-np.eye(env.A.shape[0])).reshape(1,env.A.shape[0]), np.array([env.K_i]).reshape(1,1)))
    
    x, z, reward, done, _ = env.step(a)
    t = t + 1

# t_arr = (t_arr + 1)*rollout_len
t_arr = [x + 1 for x in t_arr]
t_arr = [rollout_len*x for x in t_arr]

# #start lah

# plt.figure(1)
# plt.plot(figsize=(3.25, 2.5))
# plt.title("PG4PI on an LA Unversity hospital")
# # plt.title("PG4PI on a Chemical Reactor")
# plt.xlabel("Number of Samples")
# plt.ylabel("Reward")
# plt.plot(t_arr,rew_arr)
# # plt.legend()
# # plt.savefig("mdl_free_rea_rew", dpi=800)#, bbox_inches='tight')
# plt.savefig("mdl_free_lah_rew.png", dpi=800)#, bbox_inches='tight')

# plt.figure(2)
# plt.plot(figsize=(3.25, 2.5))
# for i in range(env.n_state):
#     plt.plot(t_arr,x_arr[i])
# plt.title("PG4PI on an LA Unversity hospital")
# # plt.title("PG4PI on a Chemical Reactor")
# plt.xlabel("Number of Samples")
# plt.ylabel("States")
# # plt.legend()
# # plt.savefig("mdl_free_rea_state", dpi=800)#, bbox_inches='tight')
# plt.savefig("mdl_free_lah_state.png", dpi=800)#, bbox_inches='tight')

# plt.figure(3)
# plt.plot(figsize=(3.25, 2.5))
# # for i in range(env.n_state):
# #     plt.plot(t_arr,x_arr[i]-env.target[i])
# # plt.plot(t_arr,u_ext_arr,label='Unit Step Input')
# # plt.plot(t_arr,y_arr,label='System Output')
# plt.plot(t_arr,diff_arr,label='Tracking Error')
# plt.title("PG4PI on an LA Unversity hospital")
# # plt.title("PG4PI on a Chemical Reactor")
# plt.xlabel("Number of Samples")
# plt.ylabel("Tracking Error")
# plt.legend()
# plt.savefig("mdl_free_lah_track_error", dpi=800)#, bbox_inches='tight')
# # plt.savefig("mdl_free_rea_track_error")#, dpi=800, bbox_inches='tight')

# # np.save('mdl_free_lah_track.npy', y_arr)
# plt.figure(4)
# #np.save('mdl_free_rea_track.npy', y_arr)
# # for i in range(env.n_state):
# #     plt.plot(t_arr,x_arr[i]-env.target[i])
# plt.plot(figsize=(3.25, 2.5))
# plt.plot(t_arr,u_ext_arr,label='Unit Step Input')
# plt.plot(t_arr,y_arr,label='System Output')
# plt.title("PG4PI on an LA University hospital")
# # plt.title("PG4PI on a Chemical Reactor")
# plt.xlabel("Number of Samples")
# plt.ylabel("Tracking")
# plt.legend()
# plt.savefig("mdl_free_lah_track", dpi=800)#, bbox_inches='tight')
# #plt.show()

# plt.figure(5)
# #np.save('mdl_free_rea_track.npy', y_arr)
# # for i in range(env.n_state):
# #     plt.plot(t_arr,x_arr[i]-env.target[i])
# plt.plot(figsize=(3.25, 2.5))
# plt.plot(t_arr,k_p_arr,label='$K_P$')
# plt.plot(t_arr,k_i_arr,label='$K_I$')
# plt.plot(t_arr,k_d_arr,label='$K_D$')
# # plt.plot(t_arr,u_arr,label='Control Input')
# plt.title("PG4PI on an LA University hospital")
# # plt.title("PG4PI on a Chemical Reactor")
# plt.xlabel("Number of Samples")
# plt.ylabel("PID Parameters")
# plt.legend()
# plt.savefig("mdl_free_lah_control_param", dpi=800)#, bbox_inches='tight')

# plt.figure(6)
# #np.save('mdl_free_rea_track.npy', y_arr)
# # for i in range(env.n_state):
# #     plt.plot(t_arr,x_arr[i]-env.target[i])
# plt.plot(figsize=(3.25, 2.5))
# plt.plot(t_arr,u_arr,label='Control Input')
# plt.title("PG4PI on an LA University hospital")
# # plt.title("PG4PI on a Chemical Reactor")
# plt.xlabel("Number of Samples")
# plt.ylabel("Control Values")
# plt.legend()
# plt.savefig("mdl_free_lah_control_param", dpi=800)#, bbox_inches='tight')
# plt.show()

# #end lah


# #start rea
plt.figure(1)
plt.plot(figsize=(3.25, 2.5))
# plt.title("PG4PI on an LA Unversity hospital")
plt.title("PG4PI on a Chemical Reactor")
plt.xlabel("Number of Samples")
plt.ylabel("Reward")
plt.plot(t_arr,rew_arr)
#plt.legend()
plt.savefig("mdl_free_rea_rew", dpi=800)#, bbox_inches='tight')
  # plt.savefig("mdl_free_lah_rew.png", dpi=800)#, bbox_inches='tight')

plt.figure(2)
plt.plot(figsize=(3.25, 2.5))
for i in range(env.n_state):
    plt.plot(t_arr,x_arr[i])
# plt.title("PG4PI on an LA Unversity hospital")
plt.title("PG4PI on a Chemical Reactor")
plt.xlabel("Number of Samples")
plt.ylabel("States")
# plt.legend()
plt.savefig("mdl_free_rea_state", dpi=800)#, bbox_inches='tight')
# plt.savefig("mdl_free_lah_state.png", dpi=800)#, bbox_inches='tight')

plt.figure(3)
plt.plot(figsize=(3.25, 2.5))
# for i in range(env.n_state):
#     plt.plot(t_arr,x_arr[i]-env.target[i])
# plt.plot(t_arr,u_ext_arr,label='Unit Step Input')
# plt.plot(t_arr,y_arr,label='System Output')
plt.plot(t_arr,diff_arr,label='Tracking Error')
# plt.title("PG4PI on an LA Unversity hospital")
plt.title("PG4PI on a Chemical Reactor")
plt.xlabel("Number of Samples")
plt.ylabel("Tracking Error")
plt.legend()
# plt.savefig("mdl_free_lah_track_error", dpi=800)#, bbox_inches='tight')
plt.savefig("mdl_free_rea_track_error")#, dpi=800, bbox_inches='tight')

# np.save('mdl_free_lah_track.npy', y_arr)
plt.figure(4)
np.save('mdl_free_rea_track.npy', y_arr)
# for i in range(env.n_state):
#     plt.plot(t_arr,x_arr[i]-env.target[i])
plt.plot(figsize=(3.25, 2.5))
plt.plot(t_arr,u_ext_arr,label='Unit Step Input')
plt.plot(t_arr,y_arr,label='System Output')
# plt.title("PG4PI on an LA University hospital")
plt.title("PG4PI on a Chemical Reactor")
plt.xlabel("Number of Samples")
plt.ylabel("Tracking")
plt.legend()
# plt.savefig("mdl_free_lah_track", dpi=800)
plt.savefig("mdl_free_rea_track", dpi=800)#, bbox_inches='tight')
plt.show()
# #end rea
