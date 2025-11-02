# Policy Gradient Reinforcement Learning with PID Control Policies
[![PyPI version](https://badge.fury.io/py/controlgym.svg)](https://pypi.org/project/controlgym/)

## Description 
This repository provides the framework used to conduct the experiments for our paper "Globally Optimal Policy Gradient Algorithms for Reinforcement Learning with PID Control Policies", appearing in 39th Conference on Neural Information Processing Systems (NeurIPS 2025). The paper can be found [here](https://openreview.net/pdf/b09c1e2835b8fe3cd762b69d2923bc48bb624343.pdf).

Spceifically, this repo contains the following:
1. RL with PID policy
2. LQR benchmarks
3. PPO benchmark
4. Instructions to reproduce each of the experiments in the paper

The files 'PG4PID.py' and 'PG4PI.py' contains the model-based  `PG4PID` and model-free `PG4PI` policy gradient algorithm for tuning Optimal Proportional-integral-derivative (PID) controller, implemented on the LA University Hospital `lah` and the Chemical Reactor `rea` environemnt from `controlgym` suite of environments. We also run ablation studies for different ranges of variance for the model-free `PG4PI`.

As part of benchmark experiments, we implement riccati equation based model based LQR and model free LQR, adapted from [Fazel etal., 2018](https://proceedings.mlr.press/v80/fazel18a.html). We illsutrate the fragility of LQR with respect to model errors in 'LQR_fragility.py' compared to robustness of proposed PID policies in 'PID_fragility.py'.

Then, in the folder 'ppo-benchmark', we implement `PPO` from [Stable Baselines 3](https://stable-baselines3.readthedocs.io/en/master/) to tune PID parameters in `neurips_ppo.py` and illustrate the larger number of samples required compared to proposed methods.

## Usage
1. Reward Bands : Figure 1
	a) PG4PI_reward_band.py
		i) For lah environment : Set flag lah = 1, rea =0; Uncomment from `'start lah' to 'end lah'` AND Comment from `'start rea' to 'end rea'`
		ii) For rea environment : Set flag lah = 0, rea =1; Uncomment from `'start rea' to 'end rea'` AND Comment from `'start lah' to 'end lah'`
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


 
<p align="center">
  <img src="figures/gallery.jpeg" alt="" width="700px">
</p>
