import torch
import torch.nn as nn
import torch.nn.functional as F

from memory import Memory
from policy_network import PolicyNetwork
from q_network import QNetwork

class SAC():
    def __init__(self,
                 args,
                 obs_dim=18,
                 action_dim=4,
                 max_action=1,
                 device="cpu"):
    
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.gamma = args.gamma
        self.model_dim = args.model_dim
        self.capacity = args.capacity
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.tau = args.tau
        self.alpha = args.alpha
        self.entropy_tuning = args.entropy_tuning
        self.device = device

        # Memory
        self.memory = Memory(self.capacity, obs_dim, action_dim, device)

        if self.entropy_tuning=="adaptive":
            self.target_entropy = -action_dim
            self.log_alpha = torch.tensor(torch.log(torch.tensor(self.alpha, dtype=torch.float32)), requires_grad=True, device=device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.lr)
        

        # Q networks
        self.q1_net = QNetwork(obs_dim, action_dim, self.model_dim, self.lr, device)
        self.q2_net = QNetwork(obs_dim, action_dim, self.model_dim, self.lr, device)
        self.q1_target_net = QNetwork(obs_dim, action_dim, self.model_dim, self.lr, device)
        self.q2_target_net = QNetwork(obs_dim, action_dim, self.model_dim, self.lr, device)
        self.q1_target_net.load_state_dict(self.q1_net.state_dict())
        self.q2_target_net.load_state_dict(self.q2_net.state_dict())

        # Policy network
        self.policy_net = PolicyNetwork(obs_dim, action_dim, self.model_dim, max_action, self.lr, device)


    def update_target_networks(self):
        for param, target_param in zip(self.q1_net.parameters(), self.q1_target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.q2_net.parameters(), self.q2_target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        

    def add_transition(self, obs, obs_next, action, reward, done):
        self.memory.add(obs, obs_next, action, reward, done)

    
    def get_action(self, obs, stochastic=True):
        actions, _ = self.policy_net.sample_actions(obs, stochastic=stochastic)

        return actions
    
    def act(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        actions, _ = self.policy_net.sample_actions(obs, stochastic=False)

        return actions.cpu().detach().numpy()

    
    def learn(self, episode_end=False):
        if self.memory.max_index < self.batch_size:
            return None

        obs, obs_next, actions, rewards, dones = self.memory.sample(self.batch_size)

        # Train Q networks
        with torch.no_grad():
            sampled_next_actions, log_probs = self.policy_net.sample_actions(obs_next, stochastic=True)
        q_target_min = torch.min(self.q1_target_net.forward(obs_next, sampled_next_actions), \
                          self.q2_target_net.forward(obs_next, sampled_next_actions))
        
        q_target = rewards + self.gamma*(1-dones) * (q_target_min - self.alpha * log_probs)
        q_1 = self.q1_net.forward(obs, actions)
        q_2 = self.q2_net.forward(obs, actions)
        q_1_loss = F.mse_loss(q_1, q_target)
        q_2_loss = F.mse_loss(q_2, q_target)

        self.q1_net.optimizer.zero_grad()
        q_1_loss.backward(retain_graph=True)
        self.q1_net.optimizer.step()

        self.q2_net.optimizer.zero_grad()
        q_2_loss.backward()
        self.q2_net.optimizer.step()

        # Train policy network
        sampled_actions, log_probs = self.policy_net.sample_actions(obs, stochastic=True)
        q_min = torch.min(self.q1_net.forward(obs, sampled_actions), \
                          self.q2_net.forward(obs, sampled_actions))
        
        policy_loss = (self.alpha * log_probs - q_min).mean()
        self.policy_net.optimizer.zero_grad()
        policy_loss.backward()
        self.policy_net.optimizer.step()
        
        # Update alpha
        if self.entropy_tuning=="adaptive":
            alpha_loss = -(self.log_alpha.exp() * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        if episode_end and self.entropy_tuning == "fixed":
            self.alpha = self.alpha * 0.9996

        self.update_target_networks()

        return q_1_loss.item(), q_2_loss.item(), policy_loss.item(), self.alpha
    
    def save(self, pth):
        checkpoint = {
            "q1_state_dict": self.q1_net.state_dict(),
            "q2_state_dict": self.q2_net.state_dict(),
            "q1_target_state_dict": self.q1_target_net.state_dict(),
            "q2_target_state_dict": self.q2_target_net.state_dict(),
            "policy_state_dict": self.policy_net.state_dict(),

            "q1_optimizer_state_dict": self.q1_net.optimizer.state_dict(),
            "q2_optimizer_state_dict": self.q2_net.optimizer.state_dict(),
            "q1_target_optimizer_state_dict": self.q1_target_net.optimizer.state_dict(),
            "q2_target_optimizer_state_dict": self.q2_target_net.optimizer.state_dict(),
            "policy_optimizer_state_dict": self.policy_net.optimizer.state_dict(),
            "hyperparameters": {
                "gamma": self.gamma,
                "tau": self.tau,
                "alpha": self.alpha,
                "learning_rate": self.lr
            }
        }
        torch.save(checkpoint, pth)

    def load(self, pth):
        checkpoint = torch.load(pth, map_location=self.device)
        self.q1_net.load_state_dict(checkpoint["q1_state_dict"])
        self.q2_net.load_state_dict(checkpoint["q2_state_dict"])
        self.q1_target_net.load_state_dict(checkpoint["q1_target_state_dict"])
        self.q2_target_net.load_state_dict(checkpoint["q2_target_state_dict"])
        self.policy_net.load_state_dict(checkpoint["policy_state_dict"])

        self.q1_net.optimizer.load_state_dict(checkpoint["q1_optimizer_state_dict"])
        self.q2_net.optimizer.load_state_dict(checkpoint["q2_optimizer_state_dict"])
        self.q1_target_net.optimizer.load_state_dict(checkpoint["q1_target_optimizer_state_dict"])
        self.q2_target_net.optimizer.load_state_dict(checkpoint["q2_target_optimizer_state_dict"])
        self.policy_net.optimizer.load_state_dict(checkpoint["policy_optimizer_state_dict"])

        self.gamma = checkpoint["hyperparameters"]["gamma"]
        self.tau = checkpoint["hyperparameters"]["tau"]
        self.alpha = checkpoint["hyperparameters"]["alpha"]
        self.lr = checkpoint["hyperparameters"]["learning_rate"]
