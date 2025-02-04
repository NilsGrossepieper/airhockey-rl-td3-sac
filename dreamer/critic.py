import torch
import torch.nn as nn
import torch.nn.functional as F
from world_model import WorldModel
import utils

# Critic Model
class Critic(nn.Module):
    def __init__(self, recurrent_hidden_dim, latent_dim, latent_categories_size, model_dim, bins, min_reward, max_reward, device):
        super(Critic, self).__init__()
        self.recurrent_hidden_dim = recurrent_hidden_dim
        self.latent_dim = latent_dim
        self.latent_categories_size = latent_categories_size
        self.model_dim = model_dim
        self.bins = bins
        self.device = device
        self.min_reward = min_reward
        self.max_reward = max_reward

        input_dim = recurrent_hidden_dim + latent_dim * latent_categories_size
        output_dim = bins
        self.model = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, output_dim), 
        ).to(device)

        self.target = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, output_dim), 
        ).to(device)

        self.target.load_state_dict(self.model.state_dict())  

    
        self.optimizer = torch.optim.Adam(self.parameters(), lr=4e-4)
        self.tau = 0.98

    def forward(self, h, z, use_target=False, return_logits=False):
        """
        h: (batch_size, recurrent_hidden_dim)
        z: (batch_size, latent_dim * latent_categories_size)
        """
        s = torch.cat((h, z), dim=-1)
        if use_target:
            logits = self.target(s)
            if return_logits:
                return logits
            probs = F.softmax(logits, dim=-1)
        else:
            logits = self.model(s)
            if return_logits:
                return logits
            probs = F.softmax(logits, dim=-1)
        
        return probs
    
    def train(self, z, h, r, c, gamma, lambda_):
        """
        z: (batch_size, imag_horizon + 1, latent_dim * latent_categories_size)
        h: (batch_size, imag_horizon + 1, recurrent_hidden_dim * num_blocks)
        r: (batch_size, imag_horizon, 1)
        c: (batch_size, imag_horizon, 1)
        """
        batch_size, imag_horizon = r.shape[0], r.shape[1]
        with torch.no_grad():
            v_target = self(h.view(-1, self.recurrent_hidden_dim),
                     z.view(-1, self.latent_dim * self.latent_categories_size),
                     use_target=True)
        v_target = v_target.view(batch_size, imag_horizon + 1, -1)
        v_target_num = utils.get_value_from_distribution(v_target, self.min_reward, self.max_reward).unsqueeze(-1)
        R_lambda_num = self.compute_lambda_return(r, c, v_target_num, gamma, lambda_)
        R_lambda = utils.get_twohot_from_value(R_lambda_num.squeeze(dim=-1), self.bins, self.min_reward, self.max_reward)

        v_logits = self(h.view(-1, self.recurrent_hidden_dim),
                     z.view(-1, self.latent_dim * self.latent_categories_size),
                     use_target=False, return_logits=True)
        v = F.softmax(v_logits, dim=-1)
        v_num = utils.get_value_from_distribution(v.view(batch_size, imag_horizon + 1, -1), -1, 1).unsqueeze(-1)
        # Update critic network
        critic_loss = F.cross_entropy(v_logits.view(-1, self.bins), R_lambda.view(-1,self.bins))
        self.optimizer.zero_grad()
        critic_loss.backward()
        self.optimizer.step()

        # Apply EMA update to the target network
        self.update_target_critic()

        return critic_loss.item(), R_lambda_num[:,:-1,:], v_num[:,:-1,:] 

    def update_target_critic(self):
        """
        Updates the target critic using an exponential moving average (EMA).
        """
        for param, target_param in zip(self.model.parameters(), self.target.parameters()):
            target_param.data.copy_(self.tau * target_param.data + (1 - self.tau) * param.data)
    
    def compute_lambda_return(self, r, c, v, gamma, lambda_):
        """
        r: (batch_size, imag_horizon, 1)
        c: (batch_size, imag_horizon, 1)
        v: (batch_size, imag_horizon + 1, 1)
        """
        imag_horizon = r.shape[1]
        R_lambda = torch.zeros_like(v)
        R_lambda[:, -1] = v[:, -1]

        for t in reversed(range(imag_horizon)):
            R_lambda[:, t] = r[:, t] + gamma * c[:, t] * ((1 - lambda_) * v[:, t] + lambda_ * R_lambda[:, t + 1])

        return R_lambda
        
