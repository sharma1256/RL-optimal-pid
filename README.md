# Policy Gradient Reinforcement Learning with PID Control Policies
[![PyPI version](https://badge.fury.io/py/controlgym.svg)](https://pypi.org/project/controlgym/)

## Description 
This repository provides the framework used to conduct the experiments for our paper "Globally Optimal Policy Gradient Algorithms for Reinforcement Learning with PID Control Policies", appearing in 39th Conference on Neural Information Processing Systems (NeurIPS 2025). The paper can be found [here](https://openreview.net/pdf/b09c1e2835b8fe3cd762b69d2923bc48bb624343.pdf).


<p align="center">
  <img src="figures/gallery.jpeg" alt="" width="700px">
</p>
Spceifically, this repo contains the following:\\
  1) RL with PID policy\\
  2) LQR benchmarks\\
  3) PPO benchmark\\
 
4) Instructions for reproduce each of the experiments in the paper

The files 'PG4PID.py' and 'PG4PI.py' contains the model-based  `PG4PID` and model-free `PG4PI` policy gradient algorithm for tuning Optimal Proportional-integral-derivative (PID) controller, implemented on the LA University Hospital `lah` and the Chemical Reactor `rea` environemnt from `controlgym` suite of environments.
 
