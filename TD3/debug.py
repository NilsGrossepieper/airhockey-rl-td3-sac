import os
import numpy as np
import torch
from td3_agent import TD3Agent
import sys
sys.path.append(os.path.abspath('./hockey_env/hockey'))
import hockey_env as h_env
from train_td3 import train_td3