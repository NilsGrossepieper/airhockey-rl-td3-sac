from sac import SAC
import numpy as np
import datetime as datetime
import os
import torch
import sys
import argparse
sys.path.append(os.path.abspath('./hockey_env/hockey'))
import hockey_env as h_env

parser = argparse.ArgumentParser(description="Arguments for SAC")
parser.add_argument("--training_episodes", type=int, default=10000, help="Number of training episodes")
parser.add_argument("--only_self_play", action="store_true", help="Train SAC only against itself")
parser.add_argument("--render", action="store_true", help="Render the environment")
parser.add_argument("--save_every", type=int, default=100, help="Save the model every n episodes")
parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
parser.add_argument("--model_dim", type=int, default=256, help="Model dimension")
parser.add_argument("--alpha", type=float, default=0.2, help="Alpha for entropy calculation")
parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
parser.add_argument("--lr_policy", type=float, default=3e-4, help="Learning rate of policy network")
parser.add_argument("--lr_alpha", type=float, default=3e-4, help="Learning rate of enropy tuning")
parser.add_argument("--capacity", type=int, default=100_000, help="Capacity of the memory")
parser.add_argument("--entropy_tuning", type=str, default=None, help="Entropy tuning method")
parser.add_argument("--save_folder", type=str, default=None, help="Folder to save the models")
parser.add_argument("--win_reward", type=float, default=10, help="Reward for winning")
parser.add_argument("--lose_reward", type=float, default=-10, help="Reward for losing")
parser.add_argument("--draw_reward", type=float, default=0, help="Reward for drawing")
parser.add_argument("--puck_closeness_reward_multiplier", type=float, default=1, help="Multiplier for puck closeness")
parser.add_argument("--puck_not_touched_yet_reward", type=float, default=-0.1, help="Reward for every step the puck is not YET touched")
parser.add_argument("--tau", type=float, default=0.005, help="Tau for updating the target networks")
parser.add_argument("--update_target_every", type=int, default=1, help="Update the target networks every n steps")
parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
parser.add_argument("--continue_training", action="store_true", help="Continue training from a saved model")
parser.add_argument("--training_number", type=int, default=2, help="Number to display in WandB")

args = parser.parse_args()
args.render = True


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = h_env.HockeyEnv()  


agent = SAC(
    args,
	obs_dim=env.observation_space.shape[0],
    action_dim= env.action_space.shape[0] // 2,
    device=device) 

agent2 = SAC(
    args,
	obs_dim=env.observation_space.shape[0],
    action_dim= env.action_space.shape[0] // 2,
    device=device) 

agent.load("checkpoints/non_self_play_10/sac_9800.pth")
#agent.load("checkpoints/2025-02-18_23-51-27/sac_10000.pth")
#agent.load("checkpoints/2025-02-19_19-02-08/sac_10000.pth") #this was good
# best : 2025-02-18_23-51-27
agent2.load("checkpoints/non_self_play_10/sac_10700.pth")
#agent2.load("checkpoints/self_play_4/sac_3000.pth")
outcomes = {"win": 0, "lose": 0, "draw": 0}


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
    
    action = agent.act(obs)
    obs_agent2 = env.obs_agent_two()
    a_enemy = agent2.act(obs_agent2)
    obs, reward, done, t, info = env.step(np.hstack([action,a_enemy]))
    if args.render:
        env.render()
