import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
import random
from networks import Actor, Critic
from replay_buffer import ReplayBuffer
import copy

class TD3Agent:
    def __init__(self, state_dim, action_dim, max_action, lr, gamma, tau, buffer_size, explore_noise, seed=None, device="cpu"):
        """
        Initialize TD3 agent with actor, critics, target networks, and replay buffer.
        """
        
        # Set device (either GPU or CPU)
        self.device = torch.device(device)  # Use user-provided device
        print(f"Using device: {self.device}")  # Confirm GPU/CPU usage
        
        # Set random seed for reproducibility
        if seed is not None:
            self.set_random_seed(seed)
        
        self.state_dim = state_dim # Number of state variables
        self.action_dim = action_dim # Number of actions
        self.max_action = max_action # Maximum action value
        self.gamma = gamma # Discount factor
        self.tau = tau  # Soft update rate
        self.train_step = 0  # Tracks training steps
        self.explore_noise = explore_noise # Noise level for exploration vs. exploitation
        
        # Initialize loss attribute
        self.actor_loss = 0.0

        # Actor and Critic Networks (Moved to correct device)
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).to(self.device)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)


        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(int(buffer_size), self.state_dim, self.action_dim)

        # Set target networks to evaluation mode (no gradient updates)
        self.actor_target.eval()
        self.critic_target.eval()
        
    def set_random_seed(self, seed):
        """
        Set random seed for reproducibility.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    def select_action(self, state, explore=True):
        """
        Selects an action using the Actor network.
        """
        # Convert state to tensor and move to correct device
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        # Compute action and move it back to CPU for environment compatibility
        with torch.no_grad():
            action = self.actor(state).cpu().numpy().flatten()

        # Add exploration noise (on CPU, since env expects numpy array)
        if explore:
            noise = self.explore_noise * max(0.05, 1 - (self.train_step / 50000))
            action += np.random.normal(0, noise, size=self.action_dim)



        return np.clip(action, -self.max_action, self.max_action)  # Ensure action remains within valid bounds

    def soft_update(self, target, source):
        with torch.no_grad():  # Prevents autograd from tracking updates
            for target_param, source_param in zip(target.parameters(), source.parameters()):
                target_param.copy_(self.tau * source_param + (1 - self.tau) * target_param)


    def train(self, batch_size, policy_noise, noise_clip, policy_delay):
        """
        Train the TD3 agent using a batch of experiences from the replay buffer.
        """
        if self.replay_buffer.size() < batch_size:
            return  

        # Sample a batch of experiences from the replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Select action according to target policy with added noise
        next_actions = self.actor_target(next_states)
        policy_noise = 0.15 * max(0.05, 1 - (self.train_step / 50000))
        noise = torch.clamp(torch.randn_like(next_actions) * policy_noise, -noise_clip, noise_clip)
        next_actions = (next_actions + noise).clamp(-self.max_action, self.max_action)

        # Compute target Q-values using both target critics
        target_q1, target_q2 = self.critic_target(next_states, next_actions)
        target_q = torch.min(target_q1, target_q2)
        target_q = rewards.view(-1, 1) + (1 - dones.view(-1, 1)) * (self.gamma * target_q.detach())

        # Unpack the critic outputs correctly
        current_q1, current_q2 = self.critic(states, actions)

        # Compute critic loss
        critic_loss = nn.MSELoss()(current_q1, target_q.detach()) + nn.MSELoss()(current_q2, target_q.detach())

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # Delayed actor updates
        if self.train_step % max(2, policy_delay) == 0:
            q1_value, _ = self.critic(states, self.actor(states))
            self.actor_loss = -q1_value.mean()
            self.actor_optimizer.zero_grad()
            self.actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()
            
            # Debug log
            print(f"Actor updated at train step {self.train_step}")
            
            # Soft update target networks
            self.soft_update(self.actor_target, self.actor)
            self.soft_update(self.critic_target, self.critic)

            # Log training information
            if self.train_step % 10 == 0:  # Log every 10 steps
                avg_q1 = current_q1.mean().item()
                avg_q2 = current_q2.mean().item()

                wandb.log({
                    "Train Step": self.train_step,
                    "Actor Loss": self.actor_loss.item(),
                    "Critic Loss": critic_loss.item(),
                    "Average Q1": avg_q1,
                    "Average Q2": avg_q2,
                    "Exploration Noise": self.explore_noise,
                    "Policy Noise": policy_noise,
                    "Batch Reward Mean": rewards.mean().item(),  # Logs average reward per batch
                    "Batch Done Rate": dones.mean().item()  # Tracks how often episodes terminate
                })


        # Increase training step counter 
        self.train_step += 1