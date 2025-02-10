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
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau  # Soft update rate
        self.train_step = 0  # Tracks training steps

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
        
    def select_action(self, state, explore=True, noise_std=0.1):
        """
        Selects an action using the Actor network.
        """
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).cpu().data.numpy().flatten()

        if explore:
            action += np.random.normal(0, noise_std, size=self.action_dim)

        return np.clip(action, -self.max_action, self.max_action)

    def soft_update(self, target, source):
        """
        Perform a soft update of the target network parameters.
        """
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1 - self.tau) * target_param.data)
            
    def train(self, batch_size=100, policy_noise=0.2, noise_clip=0.5, policy_delay=2):
        """
        Train the TD3 agent using a batch of experiences from the replay buffer.
        """
        if self.replay_buffer.size() < batch_size:
            return  

        # Sample a batch of experiences from the replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Select action according to target policy with added noise
        next_actions = self.actor_target(next_states)
        noise = torch.clamp(torch.randn_like(next_actions) * policy_noise, -noise_clip, noise_clip)
        next_actions = (next_actions + noise).clamp(-self.max_action, self.max_action)

        # Compute target Q-values using both target critics
        target_q1, _ = self.critic1_target(next_states, next_actions)
        target_q2, _ = self.critic2_target(next_states, next_actions)
        target_q1, target_q2 = target_q1.squeeze(-1), target_q2.squeeze(-1)
        target_q = torch.min(target_q1, target_q2)
        target_q = rewards.squeeze(-1) + (self.gamma * target_q * (1 - dones.squeeze(-1)))


        # Unpack the critic outputs correctly
        current_q1, _ = self.critic1(states, actions) 
        current_q2, _ = self.critic2(states, actions)  

        # Compute critic loss
        critic1_loss = nn.MSELoss()(current_q1, target_q.detach().unsqueeze(1))
        critic2_loss = nn.MSELoss()(current_q2, target_q.detach().unsqueeze(1))


        # Update critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Delayed actor updates
        if self.train_step % policy_delay == 0:
            q1_value, _ = self.critic1(states, self.actor(states))
            actor_loss = -q1_value.mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft update target networks
            self.soft_update(self.actor_target, self.actor)
            self.soft_update(self.critic1_target, self.critic1)
            self.soft_update(self.critic2_target, self.critic2)

        self.train_step += 1  # Increase training step counter   