import os
import sys
import numpy as np
import torch

# Ensure Python recognizes the correct folders
sys.path.append(os.path.abspath("."))

# Import training function
from train_td3_dynamic import train_td3_dynamic 

# Start dynamic training with user-defined settings
train_td3_dynamic(
    num_episodes=100,
    save_every=10,
    render=True,
    load_existing_agent=None,
    experiment_name="basic",
    seed=42
)