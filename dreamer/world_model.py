import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# Sequence model:
#   h_t = f_ϕ(h_{t-1}, z_{t-1}, a_{t-1})
class SequenceModel(nn.Module):
    def __init__(self, latent_size, latent_categories_size, action_size, model_dim, num_blocks=8):
        super().__init__()
        self.latent_size = latent_size
        self.action_size = action_size
        self.input_size = latent_size * latent_categories_size + action_size
        self.model_dim = model_dim
        self.num_blocks = num_blocks
        self.latent_categories_size = latent_categories_size
        
        # Each block has hidden_size = D, and we have 8 of these
        self.blocks = nn.ModuleList([
            nn.GRU(
                input_size=self.input_size,
                hidden_size=model_dim,
                batch_first=True
            )
            for _ in range(num_blocks)
        ])
    
    def get_default_hidden(self, batch_size=1):
        return torch.zeros(batch_size, self.model_dim * self.num_blocks)
        #return [None] * self.num_blocks # TODO: The initialization of h as [None] * self.num_blocks in the absence of hidden states may cause issues if the GRU expects tensor inputs. Explicitly initialize h with tensors of appropriate dimensions.
    
    def forward(self, z, a, h=None):
        """
        z: (batch_size, seq_len, latent_size * latent_categories_size)
        a: (batch_size, seq_len, action_size)
        h: optional list/tuple of hidden states, 
           one for each block, each of shape (batch_size, model_dim * num_blocks)
        """
        assert z.size(0) == a.size(0), "Batch size of z and a must match"
        assert z.size(1) == a.size(1), "Sequence length of z and a must match"
        assert z.size(2) == self.latent_size * self.latent_categories_size, "Size[-1] of z must match model's latent size"
        assert a.size(2) == self.action_size, "Size[-1] of a must match model's action size"

        # Combine a and z
        x = torch.cat((a, z), dim=-1)  # shape: (batch_size, seq_len, a_dim + z_dim)

        if h is None:
            # If no hidden states are given, default to None for each block
            h = self.get_default_hidden()        
        
        batch_size = z.shape[0]
        h = h.view(batch_size, self.num_blocks, self.model_dim)
        h = h.permute(1, 0, 2)

        outputs = []        #TODO: check what happens if seq_len is not 1
        #new_h = []
        #Forward pass through each GRU block
        for i, block in enumerate(self.blocks):
            out_i, h_i = block(x, h[i:i+1])
            outputs.append(out_i)
            #new_h.append(h_i)
        
        # Concatenate all the block outputs along the last dimension
        # out_i: (batch_size, seq_len, model_dim)
        # concatenated: (batch_size, seq_len, num_blocks * model_dim)
        combined_output = torch.cat(outputs, dim=-1)            # TODO: Check if this is correct and find out how I can use it simply (get back h_t)
                                                                # TODO: If any GRU changes the sequence length due to different handling of padding or inputs, this will fail
        
        return combined_output#, new_h
    
# Encoder:
#   z_t ~ q_ϕ(z_t | h_t, x_t)
class Encoder(nn.Module):
    def __init__(self, recurrent_size, obs_size, latent_size, latent_categories_size, model_dim):
        super().__init__()
        self.recurrent_size = recurrent_size
        self.obs_size = obs_size
        self.latent_size = latent_size
        self.model_dim = model_dim
        self.latent_categories_size = latent_categories_size
        
        self.model = nn.Sequential(
            nn.Linear(obs_size + recurrent_size, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, latent_size * latent_categories_size)
        )

    def forward(self, h, x):
        """
        z: (batch_size, latent_size)
        a: (batch_size, action_size)
        h: optional list/tuple of hidden states, 
           one for each block, each of shape (batch_size, model_dim * num_blocks)
        """
        _x = torch.cat((h, x), dim=-1)
        # Forward pass through the MLP
        logits = self.model(_x).view(-1, self.latent_size, self.latent_categories_size)
        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1)       # TODO: Check dimensions
        
        # Sample from the categorical distribution
        dist = torch.distributions.Categorical(probs=probs)
        sampled = dist.sample()  # Sample indices from the categorical distribution
        hard_sampled = F.one_hot(sampled, num_classes=self.latent_categories_size).float()  # TODO: this assumes that sampled is integer tensor

        # Straight-through trick
        sampled_straight = hard_sampled + probs - probs.detach()
        return sampled_straight, probs
    
# Dynamics predictor:
#   ẑ_t ~ p_ϕ(ẑ_t | h_t)
class DynamicsPredictor(nn.Module):
    def __init__(self, recurrent_size, latent_size, latent_categories_size, model_dim):
        super().__init__()
        self.recurrent_size = recurrent_size
        self.latent_size = latent_size
        self.model_dim = model_dim
        
        self.model = nn.Sequential(
            nn.Linear(recurrent_size, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, latent_size * latent_categories_size)
        )

    def forward(self, h):
        # Forward pass through the MLP
        logits = self.model(h).view(-1, self.latent_size, self.latent_categories_size)
        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1)       # TODO: Check dimensions
        
        # Sample from the categorical distribution
        dist = torch.distributions.Categorical(probs=probs)
        sampled = dist.sample()  # Sample indices from the categorical distribution
        hard_sampled = F.one_hot(sampled, num_classes=self.latent_categories_size).float()

        # Straight-through trick
        sampled_straight = hard_sampled + probs - probs.detach()
        return sampled_straight, probs

# r̂_t ~ p_ϕ(r̂_t | h_t, z_t)
class RewardPredictor(nn.Module):   # TODO: initially predicts 0
    def __init__(self, hidden_dim, latent_dim, embedding_dim, output_dim, model_dim):
        """
        Parameters:
        - hidden_dim: Dimension of the recurrent hidden state h_t
        - latent_dim: Number of categorical latent variables (each with its own classes)
        - embedding_dim: Embedding size for each categorical variable
        - output_dim: Dimension of the predicted output (e.g., reward distribution)
        - model_dim: Number of units in each hidden layer
        """
        super().__init__()

        self.latent_dim = latent_dim

        # Input dimension is hidden state + all embedded latent dimensions
        input_dim = hidden_dim + embedding_dim * latent_dim

        self.model = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, output_dim)
        )

    def forward(self, h, z):    # TODO: Check if z is a list of tensors
        """
        h: Tensor of shape (batch_size, hidden_dim)
        z: List of Tensors, each of shape (batch_size,) representing categorical indices
        """
        embedded_z = torch.cat(z, dim=-1)  # Concatenate embeddings
        
        # Concatenate h with embedded latent variables
        input_tensor = torch.cat([h, embedded_z], dim=-1)
        return self.model(input_tensor)

# Continue predictor:
# ĉ_t ~ p_ϕ(ĉ_t | h_t, z_t)
class ContinuePredictor(nn.Module):
    def __init__(self, hidden_dim, latent_dim, embedding_dim, output_dim, model_dim):
        """
        Parameters:
        - hidden_dim: Dimension of the recurrent hidden state h_t
        - latent_dim: Number of categorical latent variables (each with its own classes)
        - embedding_dim: Embedding size for each categorical variable
        - output_dim: Dimension of the predicted output (e.g., reward distribution)
        - model_dim: Number of units in each hidden layer
        """
        super().__init__()

        self.latent_dim = latent_dim

        # Input dimension is hidden state + all embedded latent dimensions
        input_dim = hidden_dim + embedding_dim * latent_dim

        self.model = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, h, z):    # TODO: Check if z is a list of tensors
        """
        h: Tensor of shape (batch_size, hidden_dim)
        z: List of Tensors, each of shape (batch_size,) representing categorical indices
        """
        embedded_z = torch.cat(z, dim=-1)  # Concatenate embeddings
        
        # Concatenate h with embedded latent variables
        input_tensor = torch.cat([h, embedded_z], dim=-1)
        return self.model(input_tensor)


class Decoder(nn.Module):
    def __init__(self, hidden_dim, latent_dim, embedding_dim, output_dim, model_dim):
        """
        Parameters:
        - hidden_dim: Dimension of the recurrent hidden state h_t
        - latent_dim: Number of categorical latent variables (each with its own classes)
        - embedding_dim: Embedding size for each categorical variable
        - output_dim: Dimension of the predicted output (e.g., reward distribution)
        - model_dim: Number of units in each hidden layer
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.model_dim = model_dim
        
        input_dim = hidden_dim + embedding_dim * latent_dim

        self.model = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, output_dim),
        )

    def forward(self, h, z):
        """
        h: Tensor of shape (batch_size, hidden_dim)
        z: List of Tensors, each of shape (batch_size,) representing categorical indices
        """
        embedded_z = torch.cat(z, dim=-1)  # Concatenate embeddings
        
        # Concatenate h with embedded latent variables
        input_tensor = torch.cat([h, embedded_z], dim=-1)
        return self.model(input_tensor)
    
class WorldModel():
    def __init__(self, latent_size, action_size, obs_size, latent_categories_size, model_dim, num_blocks=8, embedding_dim=32):
        super().__init__()
        self.latent_size = latent_size
        self.action_size = action_size
        self.obs_size = obs_size
        self.latent_categories_size = latent_categories_size
        self.model_dim = model_dim
        self.num_blocks = num_blocks
        self.embedding_dim = embedding_dim
        
        
        self.sequence_model = SequenceModel(latent_size, latent_categories_size, action_size, model_dim, num_blocks)
        self.encoder = Encoder(model_dim * num_blocks, obs_size, latent_size, latent_categories_size, model_dim)
        self.dynamics_predictor = DynamicsPredictor(model_dim * num_blocks, latent_size, latent_categories_size, model_dim)
        self.reward_predictor = RewardPredictor(model_dim * num_blocks, latent_size, embedding_dim, 1, model_dim)
        self.continue_predictor = ContinuePredictor(model_dim * num_blocks, latent_size, embedding_dim, 1, model_dim)
        self.decoder = Decoder(model_dim * num_blocks, latent_size, embedding_dim, obs_size, model_dim)

    def get_encoding_and_recurrent_hidden(self, h, x, a):
        # TODO: currently only supports a single sequence
        # h = (batch_size, model_dim * num_blocks)
        # x = (obs_size)
        # a = (action_size)
        # but (batch_size, seq_len, ...) is needed for sequence_model
        if x.dim() == 1:
            x = x.unsqueeze(dim=0)        
        z, _ = self.encoder(h, x)


        z_t = z.view(-1, 1, self.latent_size * self.latent_categories_size)
        h_t = h.view(-1, 1, self.model_dim * self.num_blocks)
        a_t = a.view(-1, 1, self.action_size)
        new_h = self.sequence_model(z_t, a_t, h_t) # (batch_size, seq_len, num_blocks * model_dim), but seq_len is 1
        new_h = new_h.squeeze(dim=1)
        return z, new_h
    
    def get_default_hidden(self):
        return self.sequence_model.get_default_hidden()

    




