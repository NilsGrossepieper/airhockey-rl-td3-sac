from sac import SAC
import numpy as np
import sys
import os
import torch
from dynamic_env import DynamicEnvironment

torch.cuda.empty_cache()
torch.cuda.synchronize()
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dyn_env = DynamicEnvironment(only_self=True)

agent = SAC(
	obs_dim=dyn_env.env.observation_space.shape[0],
    action_dim=dyn_env.env.action_space.shape[0] // 2,
    batch_size=256,
    model_dim=256,
    alpha=0.2,
    lr=3e-4,
    capacity=100_000,
    entropy_tuning="fixed",
    device=device)  

dyn_env.add_agent(agent)

number_of_training_episodes = 10000

render = True


rews = []
q1_losses = []
q2_losses = []
policy_losses = []
alphas = []

d = False
episode = 0
obs, info = dyn_env.reset()
obs = torch.tensor(obs, dtype=torch.float32, device=device)
while episode < number_of_training_episodes:
    if d:
        obs, info = dyn_env.reset()
        episode += 1
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
    
    if render:
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