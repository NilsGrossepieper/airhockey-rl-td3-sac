import dreamer as dm
import numpy as np
import sys
import os

import gymnasium as gym
from importlib import reload
import time

sys.path.append(os.path.abspath('./hockey_env/hockey'))
import hockey_env as h_env

env = h_env.HockeyEnv()

d = dm.DreamerV3(
	env=env,
    obs_dim=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0] // 2,
    latent_dim=8,
    latent_categories_size=32,
    model_dim=256,
    imagination_horizon=15)

replay_ratio = 32
number_of_training_steps = 100
batch_size = 7
seq_len = 11
number_of_trajectories = 10
max_steps = 10

for step in range(number_of_training_steps):
    d.generate_trajectories(number_of_trajectories,max_steps)

    for _ in range(replay_ratio * number_of_trajectories * max_steps // batch_size):
        d.train(batch_size, seq_len)
        

    

pass