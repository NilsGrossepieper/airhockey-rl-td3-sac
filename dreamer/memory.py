import torch

class Memory: # TODO: Make into dataloader
    def __init__(self, capacity, obs_dim, action_dim, latent_dim, latent_categories_dim, recurrent_dim, num_blocks, device='cpu'):
        self.capacity = capacity
        self.index = 0
        self.full = False
        self.device = device

        self.obs = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((capacity, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.dones = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.latents = torch.zeros((capacity, latent_dim, latent_categories_dim), dtype=torch.float32, device=device)
        #self.next_obs = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=device)
        #self.recurrent_hidden = torch.zeros((capacity, recurrent_dim * num_blocks), dtype=torch.float32, device=device)

    def add(self, obs, action, reward, done, latent):
        self.obs[self.index] = torch.tensor(obs, device=self.device).clone().detach()
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.dones[self.index] = done
        self.latents[self.index] = torch.tensor(latent, device=self.device).clone().detach()
        #self.next_obs[self.index] = torch.tensor(next_obs, device=self.device).clone().detach()
        #self.recurrent_hidden[self.index] = torch.tensor(recurrent_hidden, device=self.device).clone().detach()
        
        self.index = (self.index + 1) % self.capacity
        self.full = self.full or self.index == 0

    def sample(self, batch_size):
        max_index = self.capacity if self.full else self.index
        start_index = torch.randint(0, max_index - batch_size + 1, (1,), device=self.device).item()
        indices = torch.arange(start_index, start_index + batch_size, device=self.device) # TODO: change to slicing
        return indices, (
            self.obs[indices],
            self.actions[indices],
            self.rewards[indices],
            self.dones[indices],
            self.latents[indices],
            #self.next_obs[indices],
            #self.recurrent_hidden[indices]
        )
    
    def update(self, indices, obs=None, actions=None, rewards=None, dones=None, next_obs=None, latents=None, recurrent_hidden=None):
        #if obs is not None:
        #    self.obs[indices] = obs
        #if actions is not None:
        #    self.actions[indices] = actions
        #if rewards is not None:
        #    self.rewards[indices] = rewards
        #if dones is not None:
        #    self.dones[indices] = dones
        if latents is not None:
            self.latents[indices] = latents.clone().detach()
        #if recurrent_hidden is not None:
        #    self.recurrent_hidden[indices] = recurrent_hidden
        #if next_obs is not None:
        #    self.next_obs[indices] = next_obs
