import numpy as np
import gymnasium
#import pdb
import importlib.resources as pkg_resources
import scipy.io as sio
from controlgym.envs import linear_control_src
from controlgym.envs.utils import c2d
from gym import spaces

class LinearControlEnv(gymnasium.Env):
    """
    ### Description

    This environment provides a template to model linear control problems, inheriting from gym.Env class.

    ### Action Space

    The action is a `ndarray` with shape `(n_action,)` which can take continuous values.

    ### Observation Space

    The observation is a `ndarray` with shape `(n_observation,) which can take continuous values.

    ### Rewards

    The default rewards are set to be the negative of the quadratic stage cost (regulation cost)

    ### Episode End

    The episode ends if any one of the following occurs:

    1. Termination: Episode length is greater than self.n_steps
    2. Truncation: reward goes beyond the range of (-self.reward_range, reward_range)

    ### Arguments
    For env_id in the following list:
    ["toy", "ac1", "ac2", "ac3", "ac4", "ac5", "ac6", "ac7", "ac8", "ac9", "ac10",
    "bdt1", "bdt2", "cbm", "cdp", "cm1", "cm2", "cm3", "cm4", "cm5",
    "dis1", "dis2", "dlr", "he1", "he2", "he3", "he4", "he5", "he6", "iss",
    "je1", "je2", "lah", "pas", "psm", "rea", "umv"]

    ```
    controlgym.make('env_id')
    ```

    optional arguments:
    [n_steps]: the number of discrete-time steps. Default is 1000.
    [sample_time]: each discrete-time step represents (ts) seconds. Default is 0.1.
    [noise_cov]: process noise covariance coefficient. Default is 0.1.
    [random_init_state_cov]: random initial state covariance coefficient. Default is 0.1.
    [init_state]: initial state. Default is 
        saved values in .mat file or np.zeros(self.n_state) + self.noise_cov * np.identity(self.n_state).
    [action_limit]: limit of action. Default is None.
    [observation_limit]: limit of observation. Default is None.
    [reward_limit]: limit of reward. Default is None.
    [seed]: random seed. Default is None.
    """
    def __init__(
        self,
        id: str,
        n_steps: int = 1000,
        sample_time: float = 0.1,
        noise_cov: float = 0.1,
        random_init_state_cov: float = 0.1,
        init_state: np.ndarray[float] = None,
        target: np.ndarray[float] = None,
        action_limit: float = None,
        observation_limit: float = None,
        reward_limit: float = None,
        seed: int = None,
        #for reactor
        K_p: float = 0.001,
        K_i: float = 0.001,
        #for lah
        # K_p: float = 0.1,
        # K_i: float = 0.1,
        K_d: float = 0,
        K: float = None,
        P: bool = None,
        I: bool = None,
        D: bool = None,
    ):
        self.n_steps = n_steps
        self.sample_time = sample_time
        self.noise_cov = noise_cov
        self.random_init_state_cov = random_init_state_cov
        self.seed = seed
        self.rng = np.random.default_rng(seed=self.seed)
        self.K_p = K_p
        self.K_i = K_i
        self.K_d = K_d
        #self.K = K
        self.z = np.zeros([1])

        # setup the problem settings
        self.id = id
        self.category = "linear"
        self.action_space = spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64)
        #high = 15.0 * np.ones(shape=(10,)) --original
        #high = np.array([15, 15, 0, 0.5, 0.5, 0, 2, 2, 0, 0])
        #low = np.array([0, 0, 0, -0.5, -0.5, 0, 2, 2, 0, 0])
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(5,), dtype=np.float32)

        # with pkg_resources.path(linear_control_src, f"{self.id}.mat") as mat_path:
        #     assert mat_path.exists(), "Environment id does not exist!"
        #     # load the environment mat file
        #     env = sio.loadmat(str(mat_path))
        
        mat_path = pkg_resources.files(linear_control_src) / f"{self.id}.mat"
        assert mat_path.exists(), f"Environment id {self.id} does not exist!"
        with mat_path.open('rb') as f:
            #assert f.exists(), "Environment id does not exist!"
            # load the environment mat file
            #env = sio.loadmat(str(mat_path))
            env = sio.loadmat(f)

        # compute the discrete-time linear system parameters
        self.A, self.B1, self.B2, self.C1, self.D11, self.D12 = c2d(env["A"],
            env["B1"], env["B2"], env["C1"], env["D11"], env["D12"], self.sample_time)
        self.C = env["C"]
        self.D21 = env["D21"]
        self.B = self.B2
        #self.D12 = 0* self.D12
        
        # set the dimension of the system
        self.n_state = self.A.shape[1]
        self.n_disturbance = self.B1.shape[1]
        self.n_action = self.B2.shape[1]
        self.n_observation = self.C.shape[0]

        # set the default weighting matrices of the linear control problem
        self.Q = np.eye(self.A.shape[0]) #self.C1.T @ self.C1
        if self.id in ("ac1", "ac6", "ac8"):
            self.R = max(np.linalg.eig(self.D12.T @ self.D12)[0]) * np.identity(
                self.n_action
            )
        else:
            self.R = np.eye(self.B2.shape[1]) #self.D12.T @ self.D12
        self.S = self.C1.T @ self.D12

        # set up the initial system state
        # the highest priority is to use the user-defined initial state
        # the second priority is to use the initial state defined in the mat file
        # the lowest priority is to use a random initial state
        init_state = -3*np.ones([self.A.shape[0],])
        #self.target = np.ones([self.A.shape[0],])
        if init_state is not None:
            self.init_state = init_state
        elif "x_0" in env.keys():
            self.init_state = env["x_0"].reshape(self.n_state)
        else:
            self.init_state = np.zeros(self.n_state)

        # add Gaussian random noise to the initial state with covaraince matrix
        # being self.random_init_state_cov * I
        self.state = self.rng.multivariate_normal(
            self.init_state,
            self.random_init_state_cov * np.identity(self.n_state),
        )
        #self.g = np.zeros([1])
        self.action_limit = np.inf if action_limit is None else action_limit
        self.observation_limit = np.inf if observation_limit is None else observation_limit
        self.reward_limit = np.inf if reward_limit is None else reward_limit

        self.action_space = gymnasium.spaces.Box(
            low=-self.action_limit, high=self.action_limit, shape=(self.n_action,), dtype=float
        )
        self.observation_space = gymnasium.spaces.Box(
            low=-self.observation_limit,
            high=self.observation_limit,
            shape=(self.n_observation,),
            dtype=float,
        )
        self.reward_range = (-self.reward_limit, self.reward_limit)

        self.step_count = 0
        self.P = P
        self.I = I
        self.D = D
        self.sigma_k = np.zeros([self.n_state,self.n_state])
        self.state_traj = None
        self.K = -1*(np.linalg.inv(1+(1/self.sample_time)*self.K_d*np.dot(self.C,self.B)))[0,0]*np.hstack((self.K_p*self.C + self.K_d*(1/self.sample_time)*np.dot(self.C,self.A-np.eye(self.A.shape[0])).reshape(1,self.A.shape[0]), np.array([self.K_i]).reshape(1,1)))
        #self.K = -(1+(1/self.sample_time)*self.K_d*(np.dot(self.C,self.B))[0])^(-1)*np.array([[self.K_p*self.C + self.K_d*(1/self.sample_time)*np.dot(self.C,self.A-np.eye([self.A.shape[0]])), self.K_i]])
    def step(self, action: np.ndarray[float]):
        """Run one timestep of the environment's dynamics using the agent actions and optional disturbance input.

        When the end of an episode is reached (``terminated or truncated``), it is necessary to call reset() to
        reset this environment's state for the next episode.

        Args:
            action (`ndarray` with shape `(n_action,)): an action provided by the agent to update the environment state.
            disturbance (optional `ndarray` with shape `(n_disturbance,))
                            : an disturbance provided by the agent to update the environment state.
                ** Dynamics is evolved based on: state_{t+1} = self.A * state_t + self.B1 * disturbance + self.B2 * action_t

        Returns:
            observation (`ndarray` with shape `(n_observation,)): 
                ** Observation is obtained based on: observation = self.C * state_t + self.D21 * disturbance
            reward (float): The reward as the negative quadratic H2 cost of the current stage:
                ** reward = - ||self.C1 @ self.state + self.D11 @ disturbance + self.D12 @ action||_2^2
                **        = - (state_t.T @ self.Q @ state_t + action_t.T @ self.R @ action_t + 2 * state_t.T @ self.S @ action_t)
            terminated (bool): Whether the agent reach the maximum length of the episode (defined in self.n_Steps).
                                If true, the user needs to call reset().
            truncated (bool): Whether the reward goes out of bound. If true, the user needs to call reset().
            info (dict): Contains auxillary information. In this case, it contains the state of the system to be utlized
                        for deploying state-feedback controllers. 
        """
        # check whether the input control is of the right dimension
        # assert action.shape[0] == (
        #     self.n_action,
        # ), "Input control has wrong dimension, the correct dimension is: " + str(
        #     (self.n_action,)
        # )

        # if disturbance is None:
        #     # sample the process noise, which is a Gaussian random vector with dimension n_disturbance
        #     disturbance = self.rng.multivariate_normal(
        #         np.zeros(self.n_disturbance),
        #         self.noise_cov * np.identity(self.n_disturbance),
        #     )
        # else:
        #     assert disturbance.shape == (self.n_disturbance,), (
        #         "Input disturbance has wrong dimension, the correct dimension is: "
        #         + str((self.n_disturbance,))
        #     )

        # generate the observation
        #distrubance = 0* disturbance
        observation = self._get_obs()
        output = self._get_output(action)
        self.z = self.z + output
        # step the system dynamics forward for one discrete step
        next_state = self.target + self.A @ (self.state-self.target) + self.B2 @ action.reshape(1,)

        # compute the reward, which happens before updating the environment state
        # because the reward (might) depends on both the current state and the next state.
        # * In the default reward function, the dependence on the current state is
        # through the self.state attribute, which will not be updated until the next line.
        reward = self.get_reward(action, observation, next_state)

        # update the environment
        self.state = next_state

        # terminated if the maximum episode length has been reached
        self.step_count += 1
        terminated = False if self.step_count < self.n_steps else True
        truncated = (
            False if self.reward_range[0] <= reward <= self.reward_range[1] else True
        )
        info = {"state": self.state, "output": output}

        # return the observation and stage cost
        return self.state, self.z, reward, terminated, truncated
    
    def set_target_state(self, state: np.ndarray[float]):
        """Set the target state of the system

        Args:
            target_state (`ndarray` with shape `(n_state,)`): the target state of the system

        Returns:
            None.
        """
        # check whether the input target state is of the right dimension
        assert state.shape == (
            self.n_state,
        ), "Input target state has wrong dimension, the correct dimension is: " + str(
            (self.n_state,)
        )
        self.target_state = state
    
    def reset(self, seed: int = None, state: np.ndarray[float] = None):
        """Resets the environment to an initial internal state, returning an initial observation and info.

        This method generates a new starting state often with some randomness to ensure that the agent explores the
        state space and learns a generalized policy about the environment. This randomness can be controlled
        with the ``seed`` parameter otherwise if the environment already has a random number generator and
        reset() is called with ``seed=None``, the RNG is not reset.

        Args:
            seed (optional int): The seed that is used to initialize the environment's PRNG (`np_random`).
            state (optional `ndarray` with shape `(n_state,)): An specific initial state to reset the environment to.

        Returns:
            observation (`ndarray` with shape `(n_observation,)): 
                ** Observation is obtained based on: observation = self.C * state_t + self.D21 * disturbance
            info (dict): Contains auxillary information. In this case, it contains the state of the system to be utlized
                        for deploying state-feedback controllers. 
        """
        # reset the random number generator if there is a new seed provided
        if seed is not None:
            self.rng = np.random.default_rng(seed=seed)

        # reset the system to a user-defined initial state if there is one
        if state is not None:
            assert state.shape == (
                self.n_state,
            ), "Input state has wrong dimension, the correct dimension is: " + str(
                (self.n_state,)
            )
            self.init_state = state

        # add Gaussian random noise to the initial state with covaraince matrix
        # being self.random_init_state_cov * I
        self.state = self.rng.multivariate_normal(
            self.init_state,
            self.random_init_state_cov * np.identity(self.n_state),
        )

        # w = self.rng.multivariate_normal(
        #     np.zeros(self.n_disturbance),
        #     self.noise_cov * np.identity(self.n_disturbance),
        # )
        # generate the observation
        state = self._get_obs()
        info = {"state": self.state}
        self.z = np.zeros([1])
        self.g = np.vstack((state.reshape(state.shape[0],1),self.z.reshape(1,1)))
        self.step_count = 0
        # return the observation
        return 0.1*state, info

    def _get_output(self, action: np.ndarray[float]):
        """private function to generate the output

        Args:
            action (`ndarray` with shape `(n_action,)): action provided by the agent to update the environment state.
            disturbance (`ndarray` with shape `(n_disturbance,)): either stochastic or deterministic disturbance input.

        Returns:
            output (`ndarray`): output = C1 * state + D11 * disturbance + D12 * action
        """
        output = self.C @ self.state #+ self.D11 @ disturbance + self.D12 @ action
        return output

    def _get_obs(self):#, disturbance: np.ndarray[float]):
        """private function to generate the observation for linear systems

        Args:
            disturbance (`ndarray` with shape `(n_disturbance,)): either stochastic or deterministic disturbance input.

        Returns:
            observation (`ndarray` with shape `(n_observation,)): observation = C * state + D21 * disturbance
        """
        #observation = self.C @ self.state + self.D21 @ disturbance
        #observation = self.A @ state + self.B2 @ action
        return self.state
    
    def _get_mod_obs(self):
        # assert self.state.shape[0] == (
        #     self.n_state,
        # ), "Input state has wrong dimension, the correct dimension is: " + str(
        #     (self.n_state,)
        # )
        if self.state.shape[0] == self.A.shape[0]:   
            #print(self.state.shape[0])
            x = self.state
            z = self.z
            self.g = np.vstack((x.reshape(self.A.shape[0],1),z.reshape(1,1)))
        else:
            import pdb
            pdb.set_trace()
        return self.g

    def get_reward(self, action: np.ndarray[float], observation: np.ndarray[float], 
                   next_state: np.ndarray[float]):
        """ function to generate the reward for the current time step

        Args:
            action (`ndarray` with shape `(n_action,)): action provided by the agent to update the environment state.
            observation (`ndarray` with shape `(n_observation,)): observation = C * state + D21 * disturbance 
            (not used in the default reward function)
            disturbance (`ndarray` with shape `(n_disturbance,)): either stochastic or deterministic disturbance input.
            (not used in the default reward function)
            next_state (`ndarray` with shape `(n_state,)): the next state of the system.
            (not used in the default reward function)

        Returns:
            reward (float): The reward as the negative quadratic H2 cost of the current stage:

        Example of constructing an environment with a custom reward function:
        ```
        def custom_get_reward(self, action, observation, disturbance, next_state):
            return - np.linalg.norm(self.state)**2 - np.linalg.norm(action)**2 
        
        if __name__ == "__main__":
            env = gym.make('env_id', **kwargs)
            env.get_reward = custom_get_reward.__get__(env)
        ```
        """
        reward = -1*((self.state-self.target).T @ (self.mat_fact*self.Q) @ (self.state-self.target) + action.T @ (self.mat_fact*self.R) @ action)[0,0]
                         #+ 2 * self.state.T @ self.S @ action)
        return reward

    def get_params_asdict(self):
        """save the parameters of the environment as a dictionary

        Args:
            None.

        Returns:
            a dictionary containing the parameters of the environment
        """
        env_params = {
            "id": self.id,
            "n_steps": self.n_steps,
            "sample_time": self.sample_time,
            "noise_cov": self.noise_cov,
            "random_init_state_cov": self.random_init_state_cov,
            "init_state": self.init_state,
            "action_limit": self.action_limit,
            "observation_limit": self.observation_limit,
            "reward_limit": self.reward_limit,
        }

        # Conditionally add "seed" if it's not None
        if self.seed is not None:
            env_params["seed"] = self.seed

        return env_params
