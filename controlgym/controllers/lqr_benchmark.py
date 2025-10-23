import numpy as np
from numpy.linalg import inv
import logging
from scipy.linalg import solve_discrete_are, solve, LinAlgError


def _lqr_gain(self, grad_p, grad_d, grad_i,grad_K):
    """Private function to compute the gain matrices of the LQR controller
        using algebraic Riccati equation (ARE).

    Args:
        A, B, Q, R, S: ndarray[float], system matrices.

    Returns:
        gain_lqr: ndarray[float], control gain matrix for regulation.
        gain_lqt: ndarray[float], additional control gain matrix for tracking.
        If the ARE fails to find a solution, gain is set to None.
    """
    #This is where we write the code for policy gradient steps
    #try:
    alpha_p = 0.1
    alpha_i = 0.001
    alpha_d = 0.001
    # alpha_p = 0.1
    # alpha_i = 0.001
    # alpha_d = 0.001
    
    if self.env.P == 1 & self.env.I == 1 & self.env.D == 1:
        self.env.K_p = self.env.K_p + alpha_p*grad_p[0,0]
        self.env.K_i = self.env.K_i + alpha_i*grad_i[0,0]
        self.env.K_d = self.env.K_d + alpha_d*grad_d[0,0]
    elif self.env.P == 1 & self.env.I == 1 & self.env.D == 0:
        self.env.K_p = self.env.K_p + alpha_p*grad_p[0,0]
        self.env.K_i = self.env.K_i + alpha_i*grad_i[0,0]
    elif self.env.P == 1 & self.env.I == 0 & self.env.D == 0:
        self.env.K_p = self.env.K_p + alpha_p*grad_p[0,0] 
    self.env.K = self.env.K + alpha_p*grad_K
    #THIS IS ONLY PI
    #K = -(1+(1/dt)*K_d*(np.dot(C,B))[0])^(-1)*np.array([[K_p*C + K_d*(1/dt)*np.dot(C,A-np.eye([A.shape[0]])), K_i]])
    #gain_lqr = np.array([np.dot(C,gain_lqr_p),gain_lqr_i])
    #self.env.K = -1*(np.linalg.inv(1+(1/self.env.sample_time)*self.env.K_d*np.dot(self.env.C,self.env.B)))[0,0]*np.hstack((self.env.K_p*self.env.C + self.env.K_d*(1/self.env.sample_time)*np.dot(self.env.C,self.env.A-np.eye(self.env.A.shape[0])).reshape(1,self.env.A.shape[0]), np.array([self.env.K_i]).reshape(1,1)))
    #gain_lqr = self.env.K
    #gain_lqt = 0*gain_lqr
    #return self.env.K_p, self.env.K_i, self.env.K_d


class LQR_bench:
    """
    ### Description

    This environment defines the LQR state-feedback controller for linear systems.
    The system dynamics is evolved based on the following discrete-time state-space model:
        state_{t+1} = A * state_t + B2 * action_t + B1 * noise_t
        output_t = C1 * state_t + D12 * action_t + D11 * noise_t
        noise_t = N(0, noise_cov * I)
    The LQR controller is computed as:
        action_t = gain_lqr * state_t + gain_lqt * target_state 
    where gain_lqr is the control gain matrix.

    ### Arguments
    For env_id in the following list:
    ["toy", "ac1", "ac2", "ac3", "ac4", "ac5", "ac6", "ac7", "ac8", "ac9", "ac10",
    "bdt1", "bdt2", "cbm", "cdp", "cm1", "cm2", "cm3", "cm4", "cm5",
    "dis1", "dis2", "dlr", "he1", "he2", "he3", "he4", "he5", "he6", "iss",
    "je1", "je2", "lah", "pas", "psm", "rea", "umv", "convection_diffusion_reaction",
    "wave", "schrodinger"]

    ```
    env = controlgym.make(env_id, **kwargs)
    controlgym.controllers.LQR(env)
    ```

    Argument:
        None.
    """

    def __init__(self, env):
        self.env = env

        # check whether the problem is linear
        is_linear = self.env.category == "linear" or self.env.id in [
            "convection_diffusion_reaction",
            "wave",
            "schrodinger",
        ]
        assert is_linear and all(
            hasattr(self.env, attr) for attr in ["A", "B2"]
        ), "The environment is not linear or system matrices do not exist. LQR is not applicable"

        #A, B2 = self.env.A, self.env.B2
        A, B = self.env.A, self.env.B2
        Q = self.env.Q if hasattr(self.env, "Q") else np.identity(self.env.n_state)
        R = self.env.R if hasattr(self.env, "R") else np.identity(self.env.n_action)
        S = self.env.S if hasattr(self.env, "S") else np.zeros((self.env.n_state, self.env.n_action))
        #check if the below line is correct
        C = self.env.C
        A_bar = np.vstack((np.hstack((A, np.zeros([A.shape[0],C.shape[0]]))),
                          np.hstack((C,np.array([[1]])))))
        B_bar = np.vstack((B,np.zeros([C.shape[0]])))
        C_bar = np.hstack((C,np.zeros(1).reshape(1,1)))
        S = np.zeros([A_bar.shape[0],B.shape[1]])
        
        Q_mod = np.vstack((np.hstack((np.dot(C.T,C),np.zeros([A.shape[0],1]))),np.hstack((np.zeros([1,A.shape[0]]),np.zeros([1,1])))))
        P = solve_discrete_are(A_bar, B_bar, Q_mod, R)
        #K = np.linalg.inv(C_bar @ P @ C_bar.T + R) @ (C_bar @ P @ A_bar)
        # P = solve_discrete_are(A_bar.T, C_bar.T, Q_mod, R, e=None, s=S)
        K_p = self.env.K_p
        K_i = self.env.K_i
        K_d = self.env.K_d
        
        #self.env.K = -1*(np.linalg.inv(1+(1/self.env.sample_time)*self.env.K_d*np.dot(self.env.C,self.env.B)))[0,0]*np.hstack((self.env.K_p*self.env.C + self.env.K_d*(1/self.env.sample_time)*np.dot(self.env.C,self.env.A-np.eye(self.env.A.shape[0])).reshape(1,self.env.A.shape[0]), np.array([self.env.K_i]).reshape(1,1)))
        
        #sigma_K = np.eye(A.shape[0]+1) #likely incorrect
        sigma_K = self.env.sigma_k
        E_K = np.dot((R+np.dot(np.dot(B_bar.T,P),B_bar)),self.env.K) 
        - np.dot(B_bar.T,np.dot(P,A_bar))
        
        # T_x = np.array([[np.eye([A.shape[0]])], np.zeros([A.shape[0],1])])
        # T_z = np.array([[np.zeros([1,A.shape[0]]), 1]])
        T_x = np.hstack((np.eye(env.A.shape[0]),np.zeros([env.A.shape[0],1])))
        T_z = np.hstack((np.zeros([1,env.A.shape[0]]).reshape(1,env.A.shape[0]), np.array([[1]])))
        grad_p = 2*np.dot(np.dot(np.dot(E_K,sigma_K),T_x.T),C.T)
        grad_i = 2*np.dot(np.dot(E_K,sigma_K),T_z.T)
        
        alpha_1 = 1/(self.env.sample_time + K_d*np.dot(C,B)[0])
        alpha_2 = np.dot(np.dot(T_x.T,C.T),K_p) 
        + K_d*(1/self.env.sample_time)*np.dot(np.dot(T_x.T,(A-np.eye(A.shape[0])).T),C.T) + K_d*T_z.T
        
        alpha_3 = np.dot(C,B)
        
        grad_d = 2*np.dot(np.dot(E_K,sigma_K), np.dot(np.dot(alpha_1[0]*T_x.T,(A-np.eye(A.shape[0])).T),C.T)-np.dot(alpha_2,alpha_3))
        
        grad_K = 2*np.dot(E_K,sigma_K)
        # compute the LQR gain
        #_lqr_gain(self, grad_p, grad_d, grad_i)
        #self.env.K = _lqr_gain(self, grad_p, grad_d, grad_i)

    def select_action(self, state: np.ndarray[float]):
        """Compute the LQR control input using state information.

        Args:
            state: ndarray[float], state information.

        Returns:
            action: ndarray[float], control input.
        """
        A, B = self.env.A, self.env.B2
        Q = self.env.Q if hasattr(self.env, "Q") else np.identity(self.env.n_state)
        R = self.env.R if hasattr(self.env, "R") else np.identity(self.env.n_action)
        S = self.env.S if hasattr(self.env, "S") else np.zeros((self.env.n_state, self.env.n_action))
        #check if the below line is correct
        C = self.env.C
        A_bar = np.vstack((np.hstack((A, np.zeros([A.shape[0],C.shape[0]]))),
                          np.hstack((C,np.array([[1]])))))
        B_bar = np.vstack((B,np.zeros([C.shape[0]])))
        
        S = np.zeros([A_bar.shape[0],B.shape[1]])
        
        Q_mod = np.vstack((np.hstack((np.dot(C.T,C),np.zeros([A.shape[0],1]))),np.hstack((np.zeros([1,A.shape[0]]),np.zeros([1,1])))))
        P = solve_discrete_are(A_bar, B_bar, Q_mod, R, e=None, s=S)
        self.env.K = np.dot(np.linalg.inv(R+np.dot(B_bar.T,(np.dot(P,B_bar)))),np.dot(B_bar.T,(np.dot(P,A_bar))))
        K_p = self.env.K_p
        K_i = self.env.K_i
        K_d = self.env.K_d
        

        sigma_K = np.eye(A.shape[0]+1) #likely incorrect
        
        E_K = np.dot((R+np.dot(np.dot(B_bar.T,P),B_bar)),self.env.K) 
        - np.dot(B_bar.T,np.dot(P,A_bar))
        

        T_x = np.hstack((np.eye(self.env.A.shape[0]),np.zeros([self.env.A.shape[0],1])))
        T_z = np.hstack((np.zeros([1,self.env.A.shape[0]]).reshape(1,self.env.A.shape[0]), np.array([[1]])))
        grad_p = 2*np.dot(np.dot(np.dot(E_K,sigma_K),T_x.T),C.T)
        grad_i = 2*np.dot(np.dot(E_K,sigma_K),T_z.T)
        
        alpha_1 = 1/(self.env.sample_time + K_d*np.dot(C,B)[0])
        alpha_2 = np.dot(np.dot(T_x.T,C.T),K_p) 
        + K_d*(1/self.env.sample_time)*np.dot(np.dot(T_x.T,(A-np.eye(A.shape[0])).T),C.T) + K_d*T_z.T
        
        alpha_3 = np.dot(C,B)
        
        grad_d = 2*np.dot(np.dot(E_K,sigma_K), np.dot(np.dot(alpha_1[0]*T_x.T,(A-np.eye(A.shape[0])).T),C.T)-np.dot(alpha_2,alpha_3))
        grad_K = 2*np.dot(E_K,sigma_K)

        if hasattr(self.env, "target_state"):
            return self.env.K @ state #+ self.env.gain_lqt @ self.env.target_state
        else:
            return self.env.K[:,:self.env.n_state] @ (state[:self.env.n_state,:] - self.env.target.reshape(self.env.n_state,1))

    def run(self, state: np.ndarray[float] = None, seed: int = None):
        """Run a trajectory of the environment using the LQR controller,
            calculate the H2 cost, and save the state trajectory to env.state_traj.
            The trajectory is terminated when the environment returns a done signal (most likely
            due to the exceedance of the maximum number of steps: env.n_steps)
        Args:
            state: (optional ndarray[float]), an user-defined initial state.
            seed: (optional int), random seed for the environment.

        Returns:
            total_reward: float, the accumulated reward of the trajectory,
                which is equal to the negative H2 cost.
        """
        # reset the environment
        _, info = self.env.reset(seed=seed, state=state)
        # run the simulated trajectory and calculate the h2 cost
        total_reward = 0
        state_traj = np.zeros((self.env.n_state, self.env.n_steps + 1))
        state_traj[:, 0] = info["state"]

        for t in range(self.env.n_steps):
            action = self.select_action(info["state"])
            observation, reward, terminated, truncated, info = self.env.step(action)
            state_traj[:, t + 1] = info["state"]
            if terminated or truncated:
                break
            total_reward += reward

        self.env.state_traj = state_traj
        return total_reward
