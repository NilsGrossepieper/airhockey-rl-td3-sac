import torch
import numpy as np

class Memory:
    """
    Experience Replay Buffer for storing and sampling transitions.
    """
    def __init__(self, capacity, obs_dim, action_dim, device='cpu'):
        """
        Initializes the memory buffer.
        
        Args:
            capacity (int): Maximum number of transitions to store.
            obs_dim (int): Dimension of the observation space.
            action_dim (int): Dimension of the action space.
            device (str): Device where tensors will be stored ('cpu' or 'cuda').
        """
        self.capacity = capacity  # Max buffer size
        self.index = 0  # Current index for storing new transitions
        self.full = False  # Flag indicating if buffer is full
        self.device = device  # Device allocation

        # Preallocate memory for experiences to optimize performance
        self.obs = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=device)
        self.obs_next = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((capacity, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.dones = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        
    def add(self, obs, obs_next, action, reward, done):
        """
        Stores a new transition in the memory buffer.
        
        Args:
            obs (torch.Tensor): Current observation.
            obs_next (torch.Tensor): Next observation.
            action (torch.Tensor or np.ndarray): Action taken.
            reward (float): Reward received.
            done (bool): Whether the episode has ended.
        """
        # Store transition components
        self.obs[self.index] = obs.clone().detach().to(self.device)
        self.obs_next[self.index] = obs_next.clone().detach().to(self.device)

        # Ensure action is converted to tensor if provided as NumPy array
        if isinstance(action, np.ndarray):
            action = torch.tensor(action, dtype=torch.float32, device=self.device)
        self.actions[self.index] = action.clone().detach()
        
        self.rewards[self.index] = torch.tensor(reward, dtype=torch.float32, device=self.device)
        self.dones[self.index] = torch.tensor(done, dtype=torch.float32, device=self.device)

        # Update index and check if buffer is full
        self.index = (self.index + 1) % self.capacity
        self.full = self.full or self.index == 0

    def sample(self, batch_size):
        """
        Randomly samples a batch of transitions from the memory buffer.
        
        Args:
            batch_size (int): Number of samples to retrieve.
        
        Returns:
            Tuple of torch.Tensors: (obs, obs_next, actions, rewards, dones)
        """
        indices = torch.randint(0, self.capacity if self.full else self.index, (batch_size,), device=self.device)
        
        return (
            self.obs[indices],
            self.obs_next[indices],
            self.actions[indices],
            self.rewards[indices],
            self.dones[indices]
        )
        
    @property
    def max_index(self):
        """
        Returns the current size of the memory buffer.
        
        Returns:
            int: Number of stored transitions.
        """
        return self.capacity if self.full else self.index