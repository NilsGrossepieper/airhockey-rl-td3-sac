import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """
    Q-Network for TD3 algorithm.
    This network estimates the Q-value for a given state-action pair.
    """
    def __init__(self, obs_dim, action_dim, hidden_dim, lr, device):
        """
        Initializes the Q-Network.

        Args:
            obs_dim (int): Dimension of the observation space.
            action_dim (int): Dimension of the action space.
            hidden_dim (int): Number of neurons in the hidden layers.
            lr (float): Learning rate for the optimizer.
            device (str): Device to run the model on ("cpu" or "cuda").
        """
        super().__init__()
        self.input_dim = obs_dim + action_dim  # Total input size (state + action)
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.device = device

        # Define the network layers
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)  # First hidden layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # Second hidden layer
        self.output = nn.Linear(hidden_dim, 1)  # Output layer (Q-value)

        # Optimizer (Adam for stability and efficiency)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        # Move model to specified device (CPU/GPU)
        self.to(device)

    def forward(self, obs, action):
        """
        Forward pass through the network.
        
        Args:
            obs (Tensor): Observation (state) tensor.
            action (Tensor): Action tensor.
        
        Returns:
            Tensor: Estimated Q-value for the given state-action pair.
        """
        obs = obs.to(self.device)  # Ensure obs is on the correct device
        action = action.to(self.device)  # Ensure action is on the correct device
        
        # Concatenate state and action as input
        x = torch.cat([obs, action], dim=-1)
        
        # Pass through the network with ReLU activations
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output(x)  # Output Q-value
        
        return x