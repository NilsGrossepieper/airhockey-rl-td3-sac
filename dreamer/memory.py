import torch

class Memory:
    def __init__(self, capacity, obs_dim, action_dim, latent_dim, latent_categories_dim, device='cpu'):
        self.capacity = capacity
        self.index = 0
        self.full = False
        self.device = device

        self.obs = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((capacity, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.dones = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.latents = torch.zeros((capacity, latent_dim, latent_categories_dim), dtype=torch.float32, device=device)
        self.next_obs = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=device)

    def add(self, obs, action, reward, done, next_obs, latent):
        self.obs[self.index] = torch.tensor(obs, device=self.device).clone().detach()
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.dones[self.index] = done
        self.next_obs[self.index] = torch.tensor(next_obs, device=self.device).clone().detach()
        self.latents[self.index] = torch.tensor(latent, device=self.device).clone().detach()
        
        self.index = (self.index + 1) % self.capacity
        self.full = self.full or self.index == 0

    def sample(self, batch_size):
        max_index = self.capacity if self.full else self.index
        indices = torch.randint(0, max_index, (batch_size,), device=self.device)
        return (
            self.obs[indices],
            self.actions[indices],
            self.rewards[indices],
            self.dones[indices],
            self.next_obs[indices],
        )
