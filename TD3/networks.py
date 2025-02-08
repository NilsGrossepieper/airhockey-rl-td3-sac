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
        Critic network: Estimates Q-values for (state, action) pairs.

        Parameters:
        - state_dim (int): Dimension of the state space
        - action_dim (int): Dimension of the action space
        """
        super(Critic, self).__init__()

        # First Q-network
        self.fc1_1 = nn.Linear(state_dim + action_dim, 256)
        self.fc1_2 = nn.Linear(256, 256)
        self.fc1_3 = nn.Linear(256, 1)  # Outputs Q-value

        # Second Q-network (TD3 requires two critics)
        self.fc2_1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2_2 = nn.Linear(256, 256)
        self.fc2_3 = nn.Linear(256, 1)

    def forward(self, state, action):
        """
        Forward pass of the Critic network.

        Parameters:
        - state (torch.Tensor): The current state
        - action (torch.Tensor): The selected action

        Returns:
        - Q1 (torch.Tensor): Q-value from first critic
        - Q2 (torch.Tensor): Q-value from second critic
        """
        # Combine state and action as input
        sa = torch.cat([state, action], dim=1)

        # Q1 forward pass
        q1 = F.relu(self.fc1_1(sa))
        q1 = F.relu(self.fc1_2(q1))
        q1 = self.fc1_3(q1)

        # Q2 forward pass
        q2 = F.relu(self.fc2_1(sa))
        q2 = F.relu(self.fc2_2(q2))
        q2 = self.fc2_3(q2)

        return q1, q2
