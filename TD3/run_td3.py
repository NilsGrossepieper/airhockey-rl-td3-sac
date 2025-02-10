import os
import sys
import numpy as np
import torch

# Ensure Python recognizes the correct folders
sys.path.append(os.path.abspath("../hockey_env"))  # Move up to access hockey_env
sys.path.append(os.path.abspath("."))  # Stay inside TD3 to access its modules

# Import TD3 agent and training script
from td3_agent import TD3Agent  
from train_td3 import train_td3  

# Import Hockey Environment Correctly
from hockey.hockey_env import HockeyEnv, HockeyEnv_BasicOpponent  # Correct path

print("✅ All imports successful! 🚀")

# Start training TD3
train_td3(num_episodes=1000, save_every=100, opponent="weak")
