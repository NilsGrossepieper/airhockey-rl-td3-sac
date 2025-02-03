import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from actor import Actor 
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
                 num_blocks,
                 imagination_horizon,
                 capacity,
                 replay_ratio,
                 imag_horizon,
                 bins,
                 device,
                 **kwargs):
        
        self.env = env
        self.obs_dim = obs_dim
        self.action_dim = action_dim 
        self.latent_dim = latent_dim
        self.latent_categories_size = latent_categories_size
        self.model_dim = model_dim
        self.imagination_horizon = imagination_horizon
        self.replay_ratio = replay_ratio
        self.device = device
        self.num_blocks = num_blocks
        self.imag_horizon = imag_horizon
        self.bins = bins
        


        self.world_model = WorldModel(latent_dim, action_dim, obs_dim, latent_categories_size, model_dim, bins, num_blocks, device=device)
        self.memory = Memory(capacity, obs_dim, action_dim, latent_dim, latent_categories_size, device=device)
        self.critic = Critic(recurrent_hidden_dim=model_dim * num_blocks,
                           latent_dim=latent_dim,
                           latent_categories_size=latent_categories_size,
                           model_dim=model_dim,
                           bins=bins,
                           device=device)
        self.actor = Actor(
            recurrent_hidden_dim=model_dim * num_blocks,
            latent_dim=latent_dim,
            latent_categories_size=latent_categories_size,
            model_dim=model_dim,
            action_dim=action_dim,
            action_bins=bins,
            device=device)

    def generate_trajectories(self, number_of_trajectories, max_steps):
        for traj in range(number_of_trajectories):
            obs_0, info = self.env.reset()
            h_0 = self.world_model.get_default_hidden()
            for step in range(max_steps):
                #a_0 = (torch.rand(4) - 0.5) * 2  #Actor generates action
                #a_enemy = (torch.rand(4) - 0.5) * 2   
                a_0 = torch.tensor([0.1, 0.1, 0.1, 0.0], dtype=torch.float32)
                a_enemy = torch.tensor([-0.1, -0.1, -0.1, 0.0], dtype=torch.float32)

                obs_1, r, d, t, info = self.env.step(np.hstack([a_0,a_enemy]))
                
                obs_0_t = torch.tensor(obs_0, dtype=torch.float32, device=self.device)
                a_0 = torch.tensor(a_0, dtype=torch.float32, device=self.device)
                 
                latent, h_1 = self.world_model.get_encoding_and_recurrent_hidden(h_0, obs_0_t, a_0)
                self.memory.add(obs_0, a_0, r / 10, (1-d), latent) # r/10!
                
                obs_0 = obs_1
                h_0 = h_1
                if (d):
                    break
            print(f"Trajectory {traj} completed.")


    def train(self, batch_size, seq_len):
        # Train the world model
        start_indices, (obs, actions, rewards, dones, latents) = self.memory.sample_for_world_model(batch_size, seq_len)
        loss_world, new_latents = self.world_model.train(x=obs, a=actions, r=rewards, c=dones, z_memory=latents)
        self.memory.update_latents(start_indices, latents=new_latents)
        
        # Generate imagined trajections
        latents = self.memory.sample_for_imagination(batch_size)
        h_imag, z_imag, a_all_probs_imag,a_taken_probs, r_imag, c_imag = self.world_model.imagine(z_0=latents, 
                                                     actor=self.actor,
                                                     imag_horizon=self.imag_horizon)
        
        # Train the critic
        loss_critic, R_lambda, v = self.critic.train(z=z_imag.detach(),
                                        h=h_imag.detach(),
                                        r=r_imag.detach(),
                                        c=c_imag.detach(),
                                        gamma=0.997,
                                        lambda_=0.95)
        
        # Train the actor
        loss_actor = self.actor.train(R_lambda=R_lambda.detach(),
                         v=v.detach(),
                         all_probs=a_all_probs_imag,
                         taken_probs=a_taken_probs)


        return loss_world, loss_critic, loss_actor
        

