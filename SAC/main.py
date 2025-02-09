
from sac import SAC
import numpy as np
import datetime as datetime
import os
import torch
from dynamic_env import DynamicEnvironment
import argparse

torch.cuda.empty_cache()
torch.cuda.synchronize()
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

parser = argparse.ArgumentParser(description="Arguments for SAC")
parser.add_argument("--training_episodes", type=int, default=10000, help="Number of training episodes")
parser.add_argument("--only_self_play", action="store_true", help="Train SAC only against itself")
parser.add_argument("--render", action="store_true", help="Render the environment")
parser.add_argument("--save_every", type=int, default=100, help="Save the model every n episodes")
parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
parser.add_argument("--model_dim", type=int, default=256, help="Model dimension")
parser.add_argument("--alpha", type=float, default=0.2, help="Alpha for entropy calculation")
parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
parser.add_argument("--capacity", type=int, default=100_000, help="Capacity of the memory")
parser.add_argument("--entropy_tuning", type=str, default=None, help="Entropy tuning method")
parser.add_argument("--save_folder", type=str, default=None, help="Folder to save the models")
parser.add_argument("--win_reward", type=float, default="10", help="Reward for winning")
parser.add_argument("--lose_reward", type=float, default="-10", help="Reward for losing")
parser.add_argument("--draw_reward", type=float, default="0", help="Reward for drawing")
parser.add_argument("--puck_closeness_reward_multiplier", type=float, default="1", help="Multiplier for puck closeness")
parser.add_argument("--puck_not_touched_yet_reward", type=float, default="-0.1", help="Reward for every step the puck is not YET touched")
parser.add_argument("--tau", type=float, default=0.005, help="Tau for updating the target networks")
parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")

args = parser.parse_args()

if args.save_folder is None:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args.save_folder = f"checkpoints/{timestamp}"
os.makedirs(args.save_folder, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dyn_env = DynamicEnvironment(args)

agent = SAC(
    args,
	obs_dim=dyn_env.env.observation_space.shape[0],
    action_dim=dyn_env.env.action_space.shape[0] // 2,
    device=device)  

dyn_env.add_agent(agent)


agent.save(f"{args.save_folder}/sac_0.pth")



rews = []
q1_losses = []
q2_losses = []
policy_losses = []
alphas = []

d = False
episode = 0
obs, info = dyn_env.reset()
obs = torch.tensor(obs, dtype=torch.float32, device=device)
while episode < args.training_episodes:
    if d:
        episode += 1
        if episode % args.save_every == 0:
            agent.save(f"checkpoints/sac_{episode}.pth")
        obs, info = dyn_env.reset()
        obs = torch.tensor(obs, dtype=torch.float32, device=device)
        evaluation_string = dyn_env.get_evaluation()
        print(f"Step {episode: 5} Reward:{np.sum(rews): .4f} Losses: {np.mean(q1_losses): .2f} {np.mean(q2_losses): .2f} {np.mean(policy_losses): .2f} {np.mean(alphas): .2f} {evaluation_string}")
        rews = []
        q1_losses = []
        q2_losses = []
        policy_losses = []
        alphas = []
    

    
    
    a = agent.get_action(obs) 
    
    obs_next, r, d, t, info = dyn_env.step(a)
    
    if args.render:
        dyn_env.env.render(mode="human")
    obs_next = torch.tensor(obs_next, dtype=torch.float32, device=device)

    agent.add_transition(obs, obs_next, a, r, d) 
    obs = obs_next
    
    

    ret = agent.learn(d)
    if ret is not None:
        q1_loss, q2_loss, policy_loss, alpha = ret
        q1_losses.append(q1_loss)
        q2_losses.append(q2_loss)
        policy_losses.append(policy_loss)
        alphas.append(alpha)
    rews.append(r)

    

    

pass