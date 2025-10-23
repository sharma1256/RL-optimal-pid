# Policy Gradient Reinforcement Learning with PID Control Policies
[![PyPI version](https://badge.fury.io/py/controlgym.svg)](https://pypi.org/project/controlgym/)

## Description 
This repository provides the framework used to conduct the experiments for our paper "Globally Optimal Policy Gradient Algorithms for Reinforcement Learning with PID Control Policies", appearing in 39th Conference on Neural Information Processing Systems (NeurIPS 2025). The paper link will be soon available here.

The model-based  `PG4PID` and model-free `PG4PI` policy gradient algorithm for tuning Optimal Proportional-integral-derivative (PID) controller on the LA University Hospital `lah` and the Chemical Reactor `rea` environemnt from `controlgym` suite of environments.

<p align="center">
  <img src="figures/gallery.jpeg" alt="" width="700px">
</p>

Experiments
1. Reward Bands : Figure 1
	a) PG4PI_reward_band.py
		i) For lah environment : Set flag lah = 1, rea =0; Uncomment from `'start lah' to 'end lah'` AND Comment from `'start rea' to 'end rea'`\\
		ii) For rea environment : Set flag lah = 0, rea =1; Uncomment from `'start rea' to 'end rea'` AND Comment from `'start lah' to 'end lah'`\\
	b) PG4PID_reward_band.py
		i) For lah environment : Set flag lah = 1, rea =0; Uncomment from `'start lah' to 'end lah'` AND Comment from `'start rea' to 'end rea'`
		ii) For rea environment : Set flag lah = 0, rea =1; Uncomment from `'start rea' to 'end rea'` AND Comment from `'start lah' to 'end lah'`
        c) PG4PI_vs_PG4PID.py
                i) For lah environment : Uncomment from `'start lah' to 'end lah'` AND Comment from `'start rea' to 'end rea'` for loading 
			'mdl_bld_lah.npy and mdl_free_lah.npy' AND lah related plotting
		ii) For rea environment : Uncomment from `'start rea' to 'end rea'` AND Comment from `'start lah' to 'end lah'` for loading 
			'mdl_bld_rea.npy and mdl_free_rea.npy' AND rea related plotting
	c) In 'ppo-benchmark folder', Run trial-stat-run.py
	
2. 	Tracking (Error) Performance and State Trajectories: Figures 2 and 3
	a) PG4PI.py
		i) For lah environment : Set flag lah = 1, rea =0; Uncomment from `'start lah' to 'end lah'` AND Comment from `'start rea' to 'end rea'`
		ii) For rea environment : Set flag lah = 0, rea =1; Uncomment from `'start rea' to 'end rea'` AND Comment from `'start lah' to 'end lah'`
	b) PG4PID.py
		i) For lah environment : Set flag lah = 1, rea =0; Uncomment from `'start lah' to 'end lah'` AND Comment from `'start rea' to 'end rea'`
		ii) For rea environment : Set flag lah = 0, rea =1; Uncomment from `'start rea' to 'end rea'` AND Comment from `'start lah' to 'end lah'`
		
3. 	Ablation Studies: Figure 4
	a) Small Variances - Run PG4PI_ablation_small_variance.py
		i) For lah environment : Set flag lah = 1, rea =0; Uncomment from `'start lah' to 'end lah'` AND Comment from `'start rea' to 'end rea'`
		ii) For rea environment : Set flag lah = 0, rea =1; Uncomment from `'start rea' to 'end rea'` AND Comment from `'start lah' to 'end lah'`
	b) Large Variances - Run PG4PI_ablation_large_variance.py
		i) For lah environment : Set flag lah = 1, rea =0; Uncomment from `'start lah' to 'end lah'` AND Comment from `'start rea' to 'end rea'`
		ii) For rea environment : Set flag lah = 0, rea =1; Uncomment from `'start rea' to 'end rea'` AND Comment from `'start lah' to 'end lah'`
		
4.  Model Free Policy Gradient LQR vs. PG4PI: Figure 5
	a) Run Model_free_LQR_vs_PG4PI.py
		i) For lah environment : Set flag lah = 1, rea =0; Uncomment from `'start lah' to 'end lah'` AND Comment from `'start rea' to 'end rea'`
		ii) For rea environment : Set flag lah = 0, rea =1; Uncomment from `'start rea' to 'end rea'` AND Comment from `'start lah' to 'end lah'`

5.  Fragility of LQR vs PG4PI: Figure 6
	a) LQR Fragility - Run LQR_fragility.py
	b) PG4PI Fragility - Run PID_fragility.py
