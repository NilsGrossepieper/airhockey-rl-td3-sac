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
                 render,
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
                 min_reward,
                 max_reward,
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
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.render = render


        self.world_model = WorldModel(latent_dim=latent_dim,
                                      action_dim=action_dim,
                                      obs_size=obs_dim,
                                      latent_categories_size=latent_categories_size,
                                      model_dim=model_dim,
                                      bins=bins,
                                      min_reward=min_reward,
                                      max_reward=max_reward,
                                      num_blocks=num_blocks,
                                      device=device)
        self.memory = Memory(capacity, obs_dim, action_dim, latent_dim, latent_categories_size, device=device)
        self.critic = Critic(recurrent_hidden_dim=model_dim * num_blocks,
                           latent_dim=latent_dim,
                           latent_categories_size=latent_categories_size,
                           model_dim=model_dim,
                           bins=bins,
                           min_reward=min_reward,
                           max_reward=max_reward,
                           device=device)
        self.actor = Actor(
            recurrent_hidden_dim=model_dim * num_blocks,
            latent_dim=latent_dim,
            latent_categories_size=latent_categories_size,
            model_dim=model_dim,
            action_dim=action_dim,
            action_bins=bins,
            device=device)

        self.env_reset = True
        self.h_0 = None
        self.obs_0 = None
    def generate_samples(self, number_of_samples):
        rs = []
        for _ in range(number_of_samples):
            if self.env_reset:
                self.obs_0, info = self.env.reset()
                self.h_0 = self.world_model.get_default_hidden()
                self.env_reset = False
            if self.render:
                self.env.render(mode="human")
            self.obs_0 = torch.tensor(self.obs_0, dtype=torch.float32, device=self.device)
            z_0 = self.world_model.get_latent(self.h_0, self.obs_0)
            
            # TODO: implement weak, strong and self-play also
            a_enemy = (torch.rand(4) - 0.5) * 2   
            a_0 = self.actor.get_action(self.h_0, z_0)
            a_0_numpy = a_0.cpu().detach().numpy()
            
            obs_1, r, d, t, info = self.env.step(np.hstack([a_0_numpy,a_enemy]))
                            
            self.memory.add(self.obs_0, a_0, r, (1-d), z_0) # r/100!
            
            self.h_0 = self.world_model.get_recurrent_hidden(self.h_0, z_0, a_0)
            self.obs_0 = obs_1
            if (d):
                self.env_reset = True
            rs.append(r)
        return np.mean(rs)
            


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
        

