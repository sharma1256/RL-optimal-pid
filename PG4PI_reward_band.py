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
    # score_d = -1*(np.dot(np.dot(g.T,T_x.T),np.dot((env.A-np.eye(env.A.shape[0])).T,env.C.T))+
    #               np.dot(np.dot(g.T,env.K.T),np.dot(env.B.T,env.C.T)))/(tau + env.K_p*np.dot(env.C,env.B))
    return score_p, score_i

def update_policy_parameters(policy_gradient):
    # Assume this function updates the policy parameters using the gradient
    pass

# Training loop
rollout_len = 100
num_episodes = 10
lah=1
rea=0

sup_rew_arr =[]
for q in range(100): 
    if lah == 1:
        env = gym.make("lah")
    if rea == 1:
        env = gym.make("rea")
    env.reset()
    tau = env.sample_time
    alpha=0.0001 #step size
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
    env.mat_fact = 0.001
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
        # Compute returns
        G = compute_returns(rewards)
        rew_arr.append(G)
        t_arr.append(t)
        # Compute policy gradient
        score_p, score_i = compute_scores(env, states, actions, G)
        
        env.K_p = env.K_p - alpha*(a-mean)*G*score_p
        env.K_i = env.K_i - alpha*(a-mean)*G*score_i
        env.K = -1*(np.linalg.inv(1+(1/tau)*env.K_d*np.dot(env.C,env.B)))[0,0]*np.hstack((env.K_p*env.C + env.K_d*(1/tau)*np.dot(env.C,env.A-np.eye(env.A.shape[0])).reshape(1,env.A.shape[0]), np.array([env.K_i]).reshape(1,1)))
        
        x, z, reward, done, _ = env.step(a)
        t = t + 1
    if q == 0:
        sup_rew_arr = (1/1)*np.array(rew_arr)
    else: sup_rew_arr = np.vstack((sup_rew_arr,(1/1)*np.array(rew_arr)))

t_arr = [x + 1 for x in t_arr]
t_arr = [rollout_len*x for x in t_arr]

# #start lah
np.save("mdl_free_lah.npy", sup_rew_arr)
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
plt.title("Model Free PI for LA Unversity hospital")
plt.legend()
plt.grid(True)
plt.savefig("mdl_free_lah_band_rew.png", dpi=800, bbox_inches='tight')
plt.show()
# #end lah

# #start rea
# np.save("mdl_free_chem.npy", sup_rew_arr)
# # np.save("mdl_free_lah.npy", sup_rew_arr)
# y_mean = sup_rew_arr.mean(axis=0)            # Mean of trajectories
# y_std = sup_rew_arr.std(axis=0)              # Standard deviation of trajectories
# y_upper = y_mean + y_std                      # Upper band (mean + std)
# y_lower = y_mean - y_std 
# plt.figure(1)
# # plt.title("Model Free PI for LA Unversity hospital")
# # plt.title("Model Free PI for a Checmial Reactor")
# for trajectory in sup_rew_arr:
#     plt.plot(t_arr, trajectory, color='gray', alpha=0.3, linewidth=0.8)

# # Plot the central tendency
# plt.plot(t_arr, y_mean, label='Mean Trajectory', color='blue', linewidth=2)

# # Plot the statistical band
# plt.fill_between(t_arr, y_lower, y_upper, color='blue', alpha=0.2, label='1-SD Band')
# plt.xlabel("Number of Samples")
# plt.ylabel("Accumulated Reward")
# # plt.title("Model Free PI for LA Unversity hospital")
# plt.title("Model Free PI for Chemical Reactor")
# plt.legend()
# plt.grid(True)
# # plt.savefig("mdl_free_lah_band_rew.png", dpi=800, bbox_inches='tight')
# plt.savefig("mdl_free_rea_band_rew.png", dpi=800, bbox_inches='tight')
# # Show the plot
# plt.show()
# #end rea
