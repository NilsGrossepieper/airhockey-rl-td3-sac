import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import actor 
from critic import Critic
from memory import Memory
from world_model import WorldModel

import numpy as np

class DreamerV3():
    def __init__(self,
                 env,
                 obs_dim, 
                 action_dim, 
                 latent_dim, 
                 latent_categories_size,
                 model_dim, 
                 imagination_horizon,
                 replay_ratio=32,
                 **kwargs):
        
        self.env = env
        self.obs_dim = obs_dim
        self.action_dim = action_dim 
        self.latent_dim = latent_dim
        self.latent_categories_size = latent_categories_size
        self.model_dim = model_dim
        self.imagination_horizon = imagination_horizon
        self.replay_ratio = replay_ratio


        self.world_model = WorldModel(latent_dim, action_dim, obs_dim, latent_categories_size, model_dim, num_blocks=8, embedding_dim=32)
        self.memory = Memory(100000, obs_dim, action_dim, latent_dim, latent_categories_size)
        #self.actor = Actor()
        #self.critic = Critic()

    def generate_trajectories(self, number_of_trajectories, max_steps):
        for traj in range(number_of_trajectories):
            obs_old, info = self.env.reset()
            h = self.world_model.get_default_hidden()
            for step in range(max_steps):
                a1 = (torch.rand(4) - 0.5) * 2  #Actor generates action
                a2 = (torch.rand(4) - 0.5) * 2   
                

                obs, r, d, t, info = self.env.step(np.hstack([a1,a2]))
                
                obs_t = torch.tensor(obs, dtype=torch.float32)
                 
                latents, h = self.world_model.get_encoding_and_recurrent_hidden(h, obs_t, a1)
                self.memory.add(obs_old, a1, r, d, obs, latents)
                obs_old = obs




