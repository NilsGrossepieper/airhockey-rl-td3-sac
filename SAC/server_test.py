from sac import SAC
import numpy as np
import datetime as datetime
import os
import torch
import sys
import argparse
sys.path.append(os.path.abspath('./hockey_env/hockey'))
import hockey_env as h_env
import pickle
import time

FPS = 50
SCALE = 60.0  # affects how fast-paced the game is, forces should be adjusted as well (Don't touch)

VIEWPORT_W = 600
VIEWPORT_H = 480
W = VIEWPORT_W / SCALE
H = VIEWPORT_H / SCALE
CENTER_X = W / 2
CENTER_Y = H / 2

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


with open(r"C:\Users\pmaty\Downloads\d1b72add-eba8-452e-b3b3-ca748d595f62.pkl", "rb") as f:
    data = pickle.load(f)



obs = data["observations_round_0"]
actions = data["actions_round_0"]
agent = SAC(
    args,
	obs_dim=env.observation_space.shape[0],
    action_dim= env.action_space.shape[0] // 2,
    device=device) 
agent.load("checkpoints/non_self_play_10/sac_4160.pth")

agent2 = SAC(
    args,
	obs_dim=env.observation_space.shape[0],
    action_dim= env.action_space.shape[0] // 2,
    device=device) 
agent2.load("checkpoints/non_self_play_10/sac_24960.pth")
second = True
def print_info(state,action):
    return
    print(state)
    print(env.obs_agent_two())
    print("actions: done", action)
    print("agent 1 obs 1:", agent.act(state))
    print("agent 1 obs 2:", agent.act(env.obs_agent_two()))
    print("\n")
    if second:
        print("agent 2 obs 1:", agent2.act(state))
        print("agent 2 obs 2:", agent2.act(env.obs_agent_two()))

    print("\n\n")

def set_obs(state):
    env.player1.position = (state[[0, 1]] + [CENTER_X, CENTER_Y]).tolist()
    env.player1.angle = state[2]
    env.player1.linearVelocity = [state[3], state[4]]
    env.player1.angularVelocity = state[5]
    env.player2.position = (state[[6, 7]] + [CENTER_X, CENTER_Y]).tolist()
    env.player2.angle = state[8]
    env.player2.linearVelocity = [state[9], state[10]]
    env.player2.angularVelocity = state[11]
    env.puck.position = (state[[12, 13]] + [CENTER_X, CENTER_Y]).tolist()
    env.puck.linearVelocity = [state[14], state[15]]

set_obs(data[f"observations_round_0"][0])
print_info(data[f"observations_round_0"][0], data[f"actions_round_0"][0])
set_obs(data[f"observations_round_1"][0])
print_info(data[f"observations_round_1"][0], data[f"actions_round_1"][0])
set_obs(data[f"observations_round_2"][0])
print_info(data[f"observations_round_2"][0], data[f"actions_round_2"][0])
set_obs(data[f"observations_round_3"][0])
print_info(data[f"observations_round_3"][0], data[f"actions_round_3"][0])

for i in range(4):
    for x in range(len(data[f"observations_round_{i}"]) - 1):
        state = data[f"observations_round_{i}"][x]
        action = data[f"actions_round_{i}"][x]
        set_obs(state)
        env.render()
        agent_action = agent.act(state)
        #obs, reward, done, t, info = env.step(action)
        #if (x < 30):
        #    print(obs)
        #    print(data[f"observations_round_{i}"][x + 1])
        #    print()
        print_info(state, action)
        time.sleep(0.02)
    print("end of round", i)
    pass