import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        """
        Actor network: Maps states to actions.

        Parameters:
        - state_dim (int): Dimension of the state space
        - action_dim (int): Dimension of the action space
        - max_action (float): Maximum action value for scaling
        """
        super(Actor, self).__init__()
        self.max_action = max_action

        self.fc1 = nn.Linear(state_dim, 256)  # Input layer
        self.fc2 = nn.Linear(256, 256)  # Hidden layer
        self.fc3 = nn.Linear(256, action_dim)  # Output layer

    def forward(self, state):
        """
        Forward pass of the Actor network.

        Parameters:
        - state (torch.Tensor): The current state

        Returns:
        - action (torch.Tensor): The action, scaled between [-max_action, max_action]
        """
        x = F.relu(self.fc1(state))  # Apply ReLU activation
        x = F.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x)) * self.max_action  # Scale output with tanh
        return action


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        """
        Critic network: Maps states and actions to Q-values.
        """
        
        super(Critic, self).__init__()
        
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)  # Outputs a single Q-value

    def forward(self, state, action):
        """
        Critic forward pass.
        """
        
        sa = torch.cat([state, action], dim=1)

        q = F.relu(self.fc1(sa))
        q = F.relu(self.fc2(q))
        q = self.fc3(q)

        return q
