import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    """
    Policy network for the TD3 algorithm.
    
    This network takes an observation as input and outputs an action in a 
    continuous action space using a fully connected neural network.
    """

    def __init__(self, obs_dim, action_dim, hidden_dim, max_action, lr, device):
        """
        Initializes the policy network.
        
        Args:
            obs_dim (int): Dimension of the observation space.
            action_dim (int): Dimension of the action space.
            hidden_dim (int): Number of neurons in hidden layers.
            max_action (float): Maximum action value (for scaling outputs).
            lr (float): Learning rate for the optimizer.
            device (str): Device to run the network on ('cpu' or 'cuda').
        """
        super().__init__()
        self.device = device
        self.input_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.max_action = max_action
        self.lr = lr

        # Define the fully connected layers
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)  # First hidden layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # Second hidden layer
        self.mu = nn.Linear(hidden_dim, action_dim)  # Output layer

        # Optimizer (Adam)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # Move model to the specified device
        self.to(device)

    def forward(self, obs):
        """
        Forward pass through the network.
        
        Args:
            obs (torch.Tensor): The input observation.
        
        Returns:
            torch.Tensor: The computed action values, scaled by max_action.
        """
        obs = obs.to(self.device)  # Ensure the input is on the correct device
        x = F.relu(self.fc1(obs))  # Apply ReLU activation to the first layer
        x = F.relu(self.fc2(x))  # Apply ReLU activation to the second layer
        action = torch.tanh(self.mu(x)) * self.max_action  # Output action (scaled)
        return action