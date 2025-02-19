import torch
import torch.nn as nn
import torch.nn.functional as F

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.max_action = max_action
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)

        self.apply(init_weights)  # Apply weight initialization

        self.to(self.device)

    def forward(self, state):
        state = state.to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x)) * self.max_action
        return action

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        """
        TD3 Critic: Two separate Q-networks for stability.
        """
        super(Critic, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # First Q-network (Q1)
        self.fc1_1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2_1 = nn.Linear(256, 256)
        self.fc3_1 = nn.Linear(256, 1)

        # Second Q-network (Q2)
        self.fc1_2 = nn.Linear(state_dim + action_dim, 256)
        self.fc2_2 = nn.Linear(256, 256)
        self.fc3_2 = nn.Linear(256, 1)

        self.to(self.device)

    def forward(self, state, action):
        """
        Forward pass for both Q-networks.
        Returns Q1 and Q2 values.
        """
        state = state.to(self.device)
        action = action.to(self.device)
        sa = torch.cat([state, action], dim=1)

        # Q1 computation
        q1 = F.relu(self.fc1_1(sa))
        q1 = F.relu(self.fc2_1(q1))
        q1 = self.fc3_1(q1)

        # Q2 computation
        q2 = F.relu(self.fc1_2(sa))
        q2 = F.relu(self.fc2_2(q2))
        q2 = self.fc3_2(q2)

        return q1, q2
