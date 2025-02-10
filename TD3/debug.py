import os
import numpy as np
import torch
from td3_agent import TD3Agent
import sys
sys.path.append(os.path.abspath('./hockey_env/hockey'))
import hockey_env as h_env
from train_td3 import train_td3

# Get original reward from environment
next_state, reward, done, _, info = env.step(action)

# Modify the reward manually
custom_reward = reward  # Start with the base reward

# Increase reward for touching the puck
custom_reward += 2.0 * info["reward_touch_puck"]

# Penalize for moving away from the puck
custom_reward -= 0.1 * abs(info["reward_closeness_to_puck"])

# Reward for moving the puck in the correct direction
custom_reward += 5.0 * info["reward_puck_direction"]

# Extra bonus for winning
if reward == 10:
    custom_reward += 5  # Extra bonus for scoring a goal

# Larger penalty for losing
if reward == -10:
    custom_reward -= 5  # Extra penalty for conceding a goal

# Save modified reward in the replay buffer instead of the original
replay_buffer.add(state, action, custom_reward, next_state, done)
