import dreamer as dm
import numpy as np
import sys
import os
import torch

import gymnasium as gym
from importlib import reload
import time

torch.cuda.empty_cache()
torch.cuda.synchronize()
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

sys.path.append(os.path.abspath('./hockey_env/hockey'))
import hockey_env as h_env

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#env = h_env.HockeyEnv()
env = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_SHOOTING)
_ = env.render()

d = dm.DreamerV3(
	env=env,
    render=True,
    obs_dim=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0] // 2,
    latent_dim=16,
    latent_categories_size=16,
    model_dim=256,
    num_blocks=8,
    imagination_horizon=15,
    capacity=10000,
    replay_ratio=32,
    imag_horizon=15,
    bins=5,
    min_reward=-10,
    max_reward=10,
    device=device)  

replay_ratio = 128

number_of_training_steps = 10000
batch_size = 16
seq_len = 64
number_of_trajectories = 25
max_steps = 100

env_step_per_training_step = batch_size * seq_len // replay_ratio 
start_samples = 1200

world_losses = []
critic_losses = []
actor_losses = []
rewards = []

d.generate_samples(start_samples)
for step in range(number_of_training_steps):
    rew = d.generate_samples(env_step_per_training_step)

    world_loss, critic_loss, actor_loss = d.train(batch_size, seq_len)
    world_losses.append(world_loss)
    critic_losses.append(critic_loss)
    actor_losses.append(actor_loss)
    rewards.append(rew)
    print(f"Step {step} Reward:{rew: .4f} Losses: {world_loss[0].item():.2f} {world_loss[1].item():.2f} {world_loss[2].item():.2f}    {critic_loss:.2f}   {actor_loss:.2f}")

    

pass