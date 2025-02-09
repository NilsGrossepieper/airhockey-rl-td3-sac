import torch

class Memory: 
    def __init__(self, capacity, obs_dim, action_dim, device='cpu'):
        self.capacity = capacity
        self.index = 0
        self.full = False
        self.device = device

        self.obs = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=device)
        self.obs_next = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((capacity, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.dones = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        
    def add(self, obs, obs_next, action, reward, done):
        self.obs[self.index] = obs.clone().detach()
        self.obs_next[self.index] = obs_next.clone().detach()
        self.actions[self.index] = action.clone().detach()
        self.rewards[self.index] = reward
        self.dones[self.index] = done
        
        self.index = (self.index + 1) % self.capacity
        self.full = self.full or self.index == 0

    def sample(self, batch_size):
        indices  = torch.randperm(self.max_index)[:batch_size]
        
        obs = self.obs[indices]
        obs_next = self.obs_next[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        dones = self.dones[indices]

        return obs, obs_next, actions, rewards, dones
        
    @property
    def max_index(self):
        return self.capacity if self.full else self.index
    