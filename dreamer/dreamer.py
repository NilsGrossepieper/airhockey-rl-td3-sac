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
            obs_0, info = self.env.reset()
            h_0 = self.world_model.get_default_hidden()
            for step in range(max_steps):
                a_0 = (torch.rand(4) - 0.5) * 2  #Actor generates action
                a_enemy = (torch.rand(4) - 0.5) * 2   
                

                obs_1, r, d, t, info = self.env.step(np.hstack([a_0,a_enemy]))
                
                obs_0_t = torch.tensor(obs_0, dtype=torch.float32)
                 
                latent, h_1 = self.world_model.get_encoding_and_recurrent_hidden(h_0, obs_0_t, a_0)
                self.memory.add(obs_0, a_0, r, d, latent)
                
                obs_0 = obs_1
                h_0 = h_1


    def train(self, batch_size, seq_len):
        #indices, (obs, actions, rewards, dones, obs_next, latents, recurrent_hiddens) = self.memory.sample(batch_size)
        start_indices, (obs, actions, rewards, dones, latents) = self.memory.sample(batch_size, seq_len)
        
        # Train the world model
        loss, new_latents = self.world_model.train(x=obs, a=actions, r=rewards, c=dones, z_memory=latents)
        self.memory.update(start_indices, latents=new_latents)
        return loss
        

