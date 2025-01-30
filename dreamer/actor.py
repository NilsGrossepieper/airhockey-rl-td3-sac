import torch
import torch.nn as nn
import torch.nn.functional as F



# Actor Model
# a_t ∼ π_θ(a_t | h_t, z_t) 
class Actor(nn.Module):
    def __init__(self, recurrent_hidden_dim, latent_dim, latent_categories_size, model_dim, action_dim, action_bins):
        super().__init__()
        self.recurrent_hidden_dim = recurrent_hidden_dim
        self.latent_dim = latent_dim
        self.latent_categories_size = latent_categories_size
        self.model_dim = model_dim
        self.action_dim = action_dim
        self.action_bins = action_bins

        input_dim = recurrent_hidden_dim + latent_dim * latent_categories_size
        output_dim = action_dim * action_bins
        self.model = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, output_dim), 
        )

    
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
        hard_sampled = F.one_hot(sampled, num_classes=self.latent_categories_size).float()
        sampled_straight = hard_sampled + probs - probs.detach()

        return sampled_straight, probs