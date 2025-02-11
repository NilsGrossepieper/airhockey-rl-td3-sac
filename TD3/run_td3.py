import os
import sys
import numpy as np
import torch

# Ensure Python recognizes the correct folders
sys.path.append(os.path.abspath("."))

# Import training function
from train_td3 import train_td3  

# User-defined settings
render = True  # Set to False to disable rendering
load_existing_model = False  # Set to True to load a saved model
model_filename = ""  # Change this to match your saved model
opponent = "weak"  # Choose between "weak", "strong", "td3"

# Start training with user-defined settings
train_td3(
    num_episodes=1000,
    save_every=100,
    opponent=opponent,
    render=render,
    load_existing_model=load_existing_model,
    model_filename=model_filename
)
