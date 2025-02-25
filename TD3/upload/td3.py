import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from memory import Memory
from policy_network import PolicyNetwork
from q_network import QNetwork

class TD3():
    """Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm implementation."""
    
    def __init__(self,
                args,
                obs_dim=18,
                action_dim=4,
                max_action=1,
                device="cpu"):
        """Initialize TD3 with hyperparameters, networks, and replay memory."""
        
        self.device = torch.device(device if torch.cuda.is_available() else "cpu") 
    
        # Environment parameters
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.max_action = max_action
        
        # Training Hyperparameters
        self.gamma = args.gamma # Discount factor
        self.model_dim = args.model_dim # Model layer dimensions
        self.capacity = args.capacity # Replay memory capacity
        self.batch_size = args.batch_size # Batch size
        self.lr = args.lr # Learning rate
        self.policy_delay = args.policy_delay  # Delay policy updates
        
        # Noise parameter for policy smoothing
        self.policy_noise_method = args.policy_noise_method # "constant" or "decay"
        self.policy_noise = args.policy_noise
        self.noise_clip = args.noise_clip
        self.initial_policy_noise = args.initial_policy_noise
        self.policy_noise_decay_rate = args.policy_noise_decay_rate

        # Noise parameter for exploration
        self.exploration_noise_method = args.exploration_noise_method # "constant" or "decay"
        self.exploration_noise = args.exploration_noise
        self.initial_exploration_noise = args.initial_exploration_noise
        self.exploration_noise_decay_rate = args.exploration_noise_decay_rate
        
        # Target update parameters
        self.update_type = args.update_type  # "soft" or "hard"
        self.tau = args.tau # Soft update parameter
        self.hard_update_frequency = args.hard_update_frequency  # hard update frequency
        
        # Tracking noise values for logging
        self.episode_noise_values_absolute = []
        self.episode_noise_values = []
        self.episode_policy_noise_values_absolute = []
        self.episode_policy_noise_values = []

        # Initialize replay memory
        self.memory = Memory(self.capacity, obs_dim, action_dim, device)

        # Initialize Q-networks (for value function approximation)
        self.q1_net = QNetwork(obs_dim, action_dim, self.model_dim, self.lr, device)
        self.q2_net = QNetwork(obs_dim, action_dim, self.model_dim, self.lr, device)
        self.q1_target_net = QNetwork(obs_dim, action_dim, self.model_dim, self.lr, device)
        self.q2_target_net = QNetwork(obs_dim, action_dim, self.model_dim, self.lr, device)
        self.q1_target_net.load_state_dict(self.q1_net.state_dict())
        self.q2_target_net.load_state_dict(self.q2_net.state_dict())

        # Initialize policy network (actor)
        self.policy_net = PolicyNetwork(obs_dim, action_dim, self.model_dim, max_action, self.lr, device)
        self.policy_target_net = PolicyNetwork(obs_dim, action_dim, self.model_dim, max_action, self.lr, device)
        self.policy_target_net.load_state_dict(self.policy_net.state_dict())

        self.total_iterations = 0  # Track updates for delayed policy updates

    def update_target_networks(self):
        """Update target networks using either soft or hard updates."""
        if self.update_type == "soft":
            for param, target_param in zip(self.q1_net.parameters(), self.q1_target_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.q2_net.parameters(), self.q2_target_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.policy_net.parameters(), self.policy_target_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        elif self.update_type == "hard" and self.total_iterations % self.hard_update_frequency == 0:
            self.q1_target_net.load_state_dict(self.q1_net.state_dict())
            self.q2_target_net.load_state_dict(self.q2_net.state_dict())
            self.policy_target_net.load_state_dict(self.policy_net.state_dict())
    
    def add_transition(self, obs, obs_next, action, reward, done):
        """Add experience to replay memory."""
        self.memory.add(obs, obs_next, action, reward, done)

    def get_action(self, obs, episode_count):
        """Compute action using policy network and add exploration noise."""
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        obs = obs.clone().detach().to(self.device)
        action = self.policy_net.forward(obs).detach()

        # Choose the method for exploration noise
        if self.exploration_noise_method == "decay":
            decayed_noise = self.initial_exploration_noise * (self.exploration_noise_decay_rate ** episode_count)
            noise = (torch.randn_like(action, device=self.device) * decayed_noise).to(self.device)
        else:
            noise = (torch.randn_like(action, device=self.device) * self.exploration_noise).to(self.device)

        action = (action + noise).clamp(-self.max_action, self.max_action).cpu().numpy()  # Convert to CPU before returning  

        # Track exploration noise magnitude
        self.latest_noise = noise.mean().item()
        self.latest_noise_absolute = noise.abs().mean().item()
        self.episode_noise_values_absolute.append(self.latest_noise_absolute)
        self.episode_noise_values.append(self.latest_noise)

        return action
    
    def learn(self, episode_count):
        """Sample a batch from replay memory and perform a learning step."""
        if self.memory.max_index < self.batch_size:
            return None

        self.total_iterations += 1

        # Sample a batch from replay memory
        obs, obs_next, actions, rewards, dones = self.memory.sample(self.batch_size)
        obs, obs_next, actions, rewards, dones = (
            obs.to(self.device),
            obs_next.to(self.device),
            actions.to(self.device),
            rewards.to(self.device),
            dones.to(self.device),
        )

        # Compute target Q-values
        with torch.no_grad():
            next_actions = self.policy_target_net.forward(obs_next)
    
            # Choose the method for policy noise
            if self.policy_noise_method == "decay":
                decayed_policy_noise = self.initial_policy_noise * (self.policy_noise_decay_rate ** episode_count)
                noise = (torch.randn_like(next_actions, device=self.device) * decayed_policy_noise).clamp(-self.noise_clip, self.noise_clip)
            else:
                noise = (torch.randn_like(next_actions, device=self.device) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)

            # Track policy noise magnitude
            self.latest_policy_noise = noise.mean().item()
            self.latest_policy_noise_absolute = noise.abs().mean().item()
            self.episode_policy_noise_values.append(self.latest_policy_noise)
            self.episode_policy_noise_values_absolute.append(self.latest_policy_noise_absolute) 

            # Calculate target Q-values
            next_actions = (next_actions + noise).clamp(-self.max_action, self.max_action)
            q_target_min = torch.min(
                self.q1_target_net.forward(obs_next, next_actions),
                self.q2_target_net.forward(obs_next, next_actions)
            )
            q_target = rewards + self.gamma * (1 - dones) * q_target_min

        # Update Q-networks
        q_1 = self.q1_net.forward(obs, actions)
        q_2 = self.q2_net.forward(obs, actions)
        
        q_1_loss = F.mse_loss(q_1, q_target)
        q_2_loss = F.mse_loss(q_2, q_target)

        self.q1_net.optimizer.zero_grad()
        q_1_loss.backward()
        self.q1_net.optimizer.step()

        self.q2_net.optimizer.zero_grad()
        q_2_loss.backward()
        self.q2_net.optimizer.step()

        # Update policy network
        if self.total_iterations % self.policy_delay == 0:
            policy_loss = -self.q1_net.forward(obs, self.policy_net.forward(obs)).mean()
            self.policy_net.optimizer.zero_grad()
            policy_loss.backward()
            self.policy_net.optimizer.step()
    
            # Ensure hard updates are applied only when required
            if self.update_type == "soft" or (self.update_type == "hard" and self.total_iterations % self.hard_update_frequency == 0):
                self.update_target_networks()
                
        # Return losses for logging
        if self.total_iterations % self.policy_delay == 0:
            return q_1_loss.item(), q_2_loss.item(), policy_loss.item()
        else:
            return q_1_loss.item(), q_2_loss.item(), None
    
    def save(self, pth):
        """Save model parameters to a file."""
        checkpoint = {
            "q1_state_dict": self.q1_net.state_dict(),
            "q2_state_dict": self.q2_net.state_dict(),
            "policy_state_dict": self.policy_net.state_dict(),
        }
        torch.save(checkpoint, pth)

    def load(self, pth):
        """Load model parameters from a file."""
        checkpoint = torch.load(pth, map_location=self.device)
        self.q1_net.load_state_dict(checkpoint["q1_state_dict"])
        self.q2_net.load_state_dict(checkpoint["q2_state_dict"])
        self.policy_net.load_state_dict(checkpoint["policy_state_dict"])
        self.policy_target_net.load_state_dict(checkpoint["policy_state_dict"])