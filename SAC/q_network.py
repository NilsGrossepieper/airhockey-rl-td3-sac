import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, lr, device):
        super().__init__()
        self.input_dim = obs_dim + action_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.device = device

        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        #self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.dropout = nn.Dropout(0.2) 
        self.to(device)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        x = self.output(x)
        return x