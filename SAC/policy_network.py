import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, max_action, lr, device):
        super().__init__()
        self.input_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.max_action = max_action
        self.lr = lr
        self.device = device
        self.min_sigma = 1e-6

        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        #self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_sigma = nn.Linear(hidden_dim, action_dim)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        self.to(device)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        mu = self.mu(x)
        log_sigma = self.log_sigma(x)

        log_sigma = torch.clamp(log_sigma, min=-20, max=10)

        return mu, log_sigma
    
    def sample_actions(self, obs, stochastic):
        mu, log_sigma = self.forward(obs)
        sigma = torch.exp(log_sigma)
        probs = torch.distributions.Normal(mu, sigma)
        if stochastic:
            u = probs.rsample()
        else:
            u = mu
        # (20)
        a = torch.tanh(u)
        real_a = a * self.max_action
        # (21)
        log_prob = probs.log_prob(u) - torch.log(1 - a.pow(2) + self.min_sigma)
        log_prob = log_prob.sum(-1, keepdim=True)

        return real_a, log_prob