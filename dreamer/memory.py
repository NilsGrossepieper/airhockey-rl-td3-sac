import torch

class Memory: # TODO: Make into dataloader
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
        
    def add(self, obs, action, reward, done, latent):
        self.obs[self.index] = torch.tensor(obs, device=self.device).clone().detach()
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.dones[self.index] = done
        self.latents[self.index] = torch.tensor(latent, device=self.device).clone().detach()
        
        self.index = (self.index + 1) % self.capacity
        self.full = self.full or self.index == 0

    def sample(self, batch_size, seq_len=1):
        max_index = self.capacity if self.full else self.index

        number_of_blocks = max_index // seq_len
        if batch_size > number_of_blocks:
            raise ValueError(f"Cannot pick {batch_size * seq_len} data out of {max_index} available data.")

        chosen_blocks = torch.randperm(number_of_blocks)[:batch_size]
        start_indices = chosen_blocks * seq_len
        
        batches = {"obs":[], "actions":[], "rewards":[], "dones":[], "latents":[]}
        for start_idx in start_indices:
            batches["obs"].append(self.obs[start_idx:start_idx + seq_len])
            batches["actions"].append(self.actions[start_idx:start_idx + seq_len])
            batches["rewards"].append(self.rewards[start_idx:start_idx + seq_len])
            batches["dones"].append(self.dones[start_idx:start_idx + seq_len])  
            batches["latents"].append(self.latents[start_idx:start_idx + seq_len])
        
        
        obs = torch.stack(batches["obs"])
        actions = torch.stack(batches["actions"])
        rewards = torch.stack(batches["rewards"])
        dones = torch.stack(batches["dones"])
        latents = torch.stack(batches["latents"])

        # x: (batch_size, seq_len, obs_size)
        # a: (batch_size, seq_len, action_size)
        # r: (batch_size, seq_len)
        # c: (batch_size, seq_len)
        # z_memory: (batch_size, seq_len, latent_size, latent_categories_size)
        return start_indices, (
            obs,
            actions,
            rewards,
            dones,
            latents,
        )
    
    def update(self, start_indices, latents):
        seq_len = latents.shape[1]
        for i, start_idx in enumerate(start_indices):
            self.latents[start_idx: start_idx + seq_len] = latents[i].clone().detach()
