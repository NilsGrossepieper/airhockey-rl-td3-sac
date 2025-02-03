import dreamer as dm
import numpy as np
import sys
import os
import torch

import gymnasium as gym
from importlib import reload
import time

sys.path.append(os.path.abspath('./hockey_env/hockey'))
import hockey_env as h_env

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = h_env.HockeyEnv()

d = dm.DreamerV3(
	env=env,
    obs_dim=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0] // 2,
    latent_dim=32,
    latent_categories_size=32,
    model_dim=256,
    num_blocks=8,
    imagination_horizon=15,
    capacity=10000,
    replay_ratio=32,
    imag_horizon=15,
    bins=32,
    device=device)  

replay_ratio = 32
number_of_training_steps = 100
batch_size = 16
seq_len = 64
number_of_trajectories = 25
max_steps = 100

world_losses = []
critic_losses = []
for step in range(number_of_training_steps):
    d.generate_trajectories(number_of_trajectories,max_steps)

    for _ in range(replay_ratio * number_of_trajectories * max_steps):
        world_loss, critic_loss = d.train(batch_size, seq_len) # TODO: first get the data then calculate the latents
        world_losses.append(world_loss)
        critic_losses.append(critic_loss)
        print(f"Step {step} Losses: {world_loss[0].item():.2f} {world_loss[1].item():.2f} {world_loss[2].item():.2f}    {critic_loss:.2f}")

    

pass