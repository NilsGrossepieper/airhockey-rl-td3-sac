from td3 import TD3
import numpy as np
import datetime as datetime
import os
import torch
import sys
import argparse
sys.path.append(os.path.abspath('./hockey_env/hockey'))
import hockey_env as h_env

# Arguments parser to handle user-friendly training parameters
parser = argparse.ArgumentParser(description="Arguments for TD3")
parser.add_argument("--training_episodes", type=int, default=30000, help="Number of training episodes")
parser.add_argument("--only_self_play", action="store_true", help="Train TD3 only against itself")
parser.add_argument("--render", action="store_true", help="Render the environment")
parser.add_argument("--save_every", type=int, default=100, help="Save the model every n episodes")
parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
parser.add_argument("--model_dim", type=int, default=256, help="Model dimension")
parser.add_argument("--capacity", type=int, default=100_000, help="Capacity of the memory")
parser.add_argument("--save_folder", type=str, default=None, help="Folder to save the models")

# Reward-related arguments
parser.add_argument("--win_reward", type=float, default=10, help="Reward for winning")
parser.add_argument("--lose_reward", type=float, default=-10, help="Reward for losing")
parser.add_argument("--draw_reward", type=float, default=0, help="Reward for drawing")
parser.add_argument("--puck_closeness_reward_multiplier", type=float, default=1, help="Multiplier for puck closeness")
parser.add_argument("--puck_touch_reward_multiplier", type=float, default=1, help="Multiplier for puck touch")
parser.add_argument("--puck_direction_reward_multiplier", type=float, default=1, help="Multiplier for puck direction")
parser.add_argument("--puck_not_touched_yet_reward", type=float, default=0, help="Reward for every step the puck is not YET touched")

# Training hyperparameters
parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
parser.add_argument("--policy_delay", type=int, default=2, help="Policy delay")

# Policy delay-related arguments
parser.add_argument("--policy_noise_method", type=str, default="decay", choices=["decay", "constant"],
                    help="Method for policy noise: 'decay' (default) or 'constant'")
parser.add_argument("--policy_noise", type=float, default=0.2, help="Policy noise")
parser.add_argument("--noise_clip", type=float, default=0.5, help="Noise clip")
parser.add_argument("--initial_policy_noise", type=float, default=0.4, help="Initial policy noise")
parser.add_argument("--policy_noise_decay_rate", type=float, default=0.9997, help="Decay rate for policy noise")

# Exploration noise-related arguments
parser.add_argument("--exploration_noise_method", type=str, default="decay", choices=["decay", "constant"],
                    help="Method for exploration noise: 'decay' (default) or 'constant'")
parser.add_argument("--exploration_noise", type=float, default=0.1, help="Exploration noise")
parser.add_argument("--initial_exploration_noise", type=float, default=0.4, help="Initial exploration noise")
parser.add_argument("--exploration_noise_decay_rate", type=float, default=0.9997, help="Decay rate for exploration noise")

# Target update-related arguments
parser.add_argument("--update_type", type=str, default="soft", choices=["soft", "hard"],
                    help="Type of target update: 'soft' (default) or 'hard'")
parser.add_argument("--tau", type=float, default=0.005, help="Tau for updating the target networks")
parser.add_argument("--hard_update_frequency", type=int, default=10000,
                    help="Number of steps between hard updates")
args = parser.parse_args()
args.render = True

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = h_env.HockeyEnv()  

# Initialize the agent
agent = TD3(
    args,
	obs_dim=env.observation_space.shape[0],
    action_dim= env.action_space.shape[0] // 2,
    device=device) 

# Initialize the opponent
agent2 = h_env.BasicOpponent(weak=True)

# Load the model
agent.load("td3/checkpoints/td3_10000.pth") #this was good

#agent2.load("td3/checkpoints/td3_10000_delay_1.pth")
#agent2.load("td3/checkpoints/self_play_4/sac_3000.pth")
outcomes = {"win": 0, "lose": 0, "draw": 0}

# Play against the opponent
done = True
info = {"winner":-2}
while True:
    if done:
        if info["winner"] == 0:
            outcomes["draw"] += 1
        elif info["winner"] == 1:
            outcomes["win"] += 1
        elif info["winner"] == -1:
            outcomes["lose"] += 1
        print(outcomes)
        obs, info = env.reset()
    
    # Get the action from the agent
    action = agent.get_action(obs, episode_count=10000)
    obs_agent2 = env.obs_agent_two()
    a_enemy = agent2.act(obs_agent2)
    obs, reward, done, t, info = env.step(np.hstack([action,a_enemy]))
    env.render()