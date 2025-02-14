import os
import sys
import numpy as np
import torch

# Ensure Python recognizes the correct folders
sys.path.append(os.path.abspath("."))

# Import training function
from train_td3 import train_td3  
from train_td3_dynamic import train_td3_dynamic 

# Start training with user-defined settings
""""
train_td3(
    num_episodes=10000,
    save_every=100,
    opponent="weak",
    render=True,
    load_existing_model=False,
    model_filename="",
    experiment_name="basic"
)
"""

# Start dynamic training with user-defined settings
train_td3_dynamic(
    num_episodes=5,
    save_every=1,
    render=True,
    load_existing_model=False,
    experiment_name="basic"
)