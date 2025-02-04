import torch
import torch.nn as nn
import torch.nn.functional as F
import utils


# Actor Model
# a_t ∼ π_θ(a_t | h_t, z_t) 
class Actor(nn.Module):
    def __init__(self, recurrent_hidden_dim, latent_dim, latent_categories_size, model_dim, action_dim, action_bins, device):
        super().__init__()
        self.recurrent_hidden_dim = recurrent_hidden_dim
        self.latent_dim = latent_dim
        self.latent_categories_size = latent_categories_size
        self.model_dim = model_dim
        self.action_dim = action_dim
        self.action_bins = action_bins
        self.device = device
        self.eta = 3e-4
        
        input_dim = recurrent_hidden_dim + latent_dim * latent_categories_size
        output_dim = action_dim * action_bins
        self.model = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, output_dim), 
        ).to(device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=4e-4)

    def forward(self, h, z):
        """
        h: (batch_size, recurrent_hidden_dim)
        z: (batch_size, latent_dim * latent_categories_size)
        """
        s = torch.cat((h, z), dim=-1)
        logits = self.model(s).view(-1, self.action_dim, self.action_bins)
        probs = F.softmax(logits, dim=-1)
        
        dist = torch.distributions.Categorical(probs=probs)
        sampled = dist.sample()
        hard_sampled = F.one_hot(sampled, num_classes=self.action_bins).float()
        sampled_straight = hard_sampled + probs - probs.detach()

        return sampled_straight, probs
    
    def get_action(self, h, z):
        """
        h: (1, recurrent_hidden_dim)
        z: (1, latent_dim, latent_categories_size)
        """
        s = torch.cat((h, z.view(1, -1)), dim=-1)
        logits = self.model(s).view(-1, self.action_dim, self.action_bins)
        probs = F.softmax(logits, dim=-1)
        
        dist = torch.distributions.Categorical(probs=probs)
        sampled = dist.sample()
        hard_sampled = F.one_hot(sampled, num_classes=self.action_bins).float()
        a = utils.get_value_from_distribution(hard_sampled, -1, 1)

        return a.squeeze(0)

    def train(self, R_lambda, v, all_probs, taken_probs):
        advantage = R_lambda - v
        #advantage *= -1
        log_probs = torch.log(taken_probs + 1e-6)  # Avoid log(0) 
        entropy = -torch.sum(all_probs * torch.log(all_probs + 1e-6), dim=-1)

        policy_loss = - (advantage * log_probs).sum(dim=-1) + self.eta * torch.sum(entropy, dim=-1)
        
        self.optimizer.zero_grad()
        policy_loss.mean().backward()
        self.optimizer.step()

        return policy_loss.mean().item()
