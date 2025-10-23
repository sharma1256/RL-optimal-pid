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
    for i in range(100):
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

env = gym.make("rea")

tau = env.sample_time
alpha=0.0001 #step size

#step input
u_ext = 1*np.ones([1])


env.target = 0*np.random.uniform(4, 7, size=env.n_state)
env.P = 1
env.I = 1
env.D = 0
env.mat_fact = 0.01
std_dev_arr = [0.1, 0.5, 1.0 , 1.5, 2.0, 2.5, 3.0]
big_rew_arr=[]
for std_dev in std_dev_arr:
    env.reset()
    t_arr = []
    t=0
    u_ext_arr = []
    y_arr = []
    diff_arr=[]
    x_arr = []
    rew_arr = []
    for episode in range(num_episodes):
        if episode == 0:
            x_arr =  env._get_obs().reshape(env.n_state,1)
        else:
            x_arr = np.hstack((x_arr, x.reshape(env.n_state,1)))
        g = env._get_mod_obs()
        mean = -np.dot(env.K,g)
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
    t_arr = [x + 1 for x in t_arr]
    t_arr = [rollout_len*x for x in t_arr]
    big_rew_arr.append(rew_arr)
    for array in big_rew_arr:
        plt.plot(t_arr, array)#, marker='o')  # Plot each array

   
    plt.figure(1)
    plt.plot(figsize=(3.25, 2.5))
    
    # #start rea
    plt.title("PG4PI on a Chemical Reactor")
    plt.xlabel("Number of Samples")
    plt.ylabel("Reward")
    plt.legend([f'Variance = {eps}' for eps in std_dev_arr], loc='lower right')
    plt.savefig("mdl_free_rea_rew_ablation_run=large", dpi=800)
    # #end rea
    
    
    # #start lah
    # plt.title("PG4PI on an LA University Hospital")
    # plt.xlabel("Number of Samples")
    # plt.ylabel("Reward")
    # plt.legend([f'Variance = {eps}' for eps in std_dev_arr], loc='lower right')
    # plt.savefig("mdl_free_lah_rew_ablation_run=large.png", dpi=800)
    # #end lah
plt.show()    