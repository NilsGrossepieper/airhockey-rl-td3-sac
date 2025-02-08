import torch
import numpy as np
import gymnasium as gym
import time
import sys
import os
from td3_agent import TD3Agent
from replay_buffer import ReplayBuffer
from utils import save_model
sys.path.append(os.path.abspath('./hockey_env/hockey'))
import hockey_env as h_env

# Hyperparameters
ENV_NAME = "HockeyEnv"  # Custom environment
MAX_EPISODES = 1000  # Total training episodes
MAX_TIMESTEPS = 250  # Max steps per episode
BATCH_SIZE = 100  # Mini-batch size
GAMMA = 0.99  # Discount factor
TAU = 0.005  # Soft update rate
POLICY_DELAY = 2  # Actor update frequency
EXPLORE_NOISE = 0.1  # Exploration noise
POLICY_NOISE = 0.2  # Noise for target policy smoothing
NOISE_CLIP = 0.5  # Noise clipping
EVAL_FREQ = 10  # Evaluate every X episodes
SAVE_FREQ = 100  # Save model every X episodes
BUFFER_SIZE = 1e6  # Replay buffer size
LR = 3e-4  # Learning rate

# Initialize environment
env = h_env.HockeyEnv()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

# Initialize agent and replay buffer
agent = TD3Agent(state_dim, action_dim, max_action, lr=LR, gamma=GAMMA, tau=TAU, buffer_size=BUFFER_SIZE)
replay_buffer = agent.replay_buffer

# Training loop
total_timesteps = 0
for episode in range(1, MAX_EPISODES + 1):
    state, _ = env.reset()
    episode_reward = 0
    for t in range(MAX_TIMESTEPS):
        total_timesteps += 1

        # Select action with exploration noise
        action = agent.select_action(np.array(state))
        action = (action + np.random.normal(0, EXPLORE_NOISE, size=action_dim)).clip(-max_action, max_action)

        # Step environment
        next_state, reward, done, _, _ = env.step(action)
        replay_buffer.add(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward

        # Train agent after collecting enough experience
        if total_timesteps > BATCH_SIZE:
            agent.train(BATCH_SIZE, POLICY_NOISE, NOISE_CLIP, POLICY_DELAY)

        if done:
            break

    print(f"Episode {episode}, Reward: {episode_reward:.2f}")

    # Evaluate agent
    if episode % EVAL_FREQ == 0:
        eval_reward = 0
        for _ in range(5):  # Run 5 evaluation episodes
            state, _ = env.reset()
            for _ in range(MAX_TIMESTEPS):
                action = agent.select_action(np.array(state), explore=False)
                next_state, reward, done, _, _ = env.step(action)
                eval_reward += reward
                state = next_state
                if done:
                    break
        avg_eval_reward = eval_reward / 5
        print(f"Evaluation Reward: {avg_eval_reward:.2f}")

    # Save model
    if episode % SAVE_FREQ == 0:
        save_model(agent.actor, agent.critic1, agent.critic2, filename=f"TD3_Hockey_{episode}.pth")

# Close environment
env.close()
