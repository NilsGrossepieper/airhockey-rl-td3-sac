import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from networks import Actor, Critic
from replay_buffer import ReplayBuffer
import copy

class TD3Agent:
    def __init__(self, state_dim, action_dim, max_action, lr=3e-4, gamma=0.99, tau=0.005, buffer_size=1e6):
        """
        Initialize TD3 agent with actor, critics, target networks, and replay buffer.

        Parameters:
        - state_dim: int -> Dimension of state space
        - action_dim: int -> Dimension of action space
        - max_action: float -> Maximum action value (for scaling)
        - lr: float -> Learning rate
        - gamma: float -> Discount factor for future rewards
        - tau: float -> Soft update rate for target networks
        - buffer_size: int -> Maximum size of replay buffer
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau

        # Actor and Critic Networks
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = copy.deepcopy(self.actor)

        self.critic1 = Critic(state_dim, action_dim)
        self.critic2 = Critic(state_dim, action_dim)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(int(buffer_size))

        # Set target networks to evaluation mode (no gradient updates)
        self.actor_target.eval()
        self.critic1_target.eval()
        self.critic2_target.eval()
