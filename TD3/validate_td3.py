import torch
import numpy as np
import gymnasium as gym
import sys
import os
import re
import random
from td3_agent import TD3Agent
from utils import load_model
from hockey.hockey_env import HockeyEnv

# Ensure Python finds the hockey environment
sys.path.append(os.path.abspath("../hockey_env"))

# Constants
BUFFER_DIR = "./TD3_models"
MAX_TIMESTEPS = 251
NUM_EVAL_EPISODES = 100  # Number of evaluation games
RENDER = True  # Set to False to disable rendering

def select_model_from_buffer():
    """
    Allow user to select a trained TD3 model from the BUFFER_DIR.
    Returns the selected model filename.
    """
    models = [f for f in os.listdir(BUFFER_DIR) if f.endswith(".pth")]
    
    if not models:
        print("No trained models found in the buffer directory.")
        return None
    
    print("\nAvailable models:")
    for i, model in enumerate(models):
        print(f"{i+1}. {model}")

    while True:
        choice = input("\nSelect a model by number: ")
        if choice.isdigit() and 1 <= int(choice) <= len(models):
            return models[int(choice) - 1]
        print("Invalid choice. Please select a valid number.")

def validate_td3(model1_filename, model2_filename, render=RENDER):
    """
    Validate two trained TD3 agents against each other in the Hockey environment.
    """

    # Initialize environment
    env = HockeyEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] // 2
    max_action = float(env.action_space.high[0])

    # Load first agent (Player 1)
    agent1 = TD3Agent(state_dim, action_dim, max_action, lr=0, gamma=0, tau=0, buffer_size=0, explore_noise=0)
    load_model(agent1.actor, agent1.critic1, agent1.critic2, 
               agent1.actor_target, agent1.critic1_target, agent1.critic2_target, 
               filename=os.path.join(BUFFER_DIR, model1_filename))

    # Load second agent (Player 2)
    agent2 = TD3Agent(state_dim, action_dim, max_action, lr=0, gamma=0, tau=0, buffer_size=0, explore_noise=0)
    load_model(agent2.actor, agent2.critic1, agent2.critic2, 
               agent2.actor_target, agent2.critic1_target, agent2.critic2_target, 
               filename=os.path.join(BUFFER_DIR, model2_filename))

    # Initialize tracking stats
    win_counts = {1: 0, 0: 0, -1: 0}  # Win/Loss/Draw

    print(f"\nValidating {model1_filename} (Agent 1) vs {model2_filename} (Agent 2) over {NUM_EVAL_EPISODES} episodes...\n")

    for episode in range(NUM_EVAL_EPISODES):
        state, _ = env.reset()
        done = False
        t = 0

        while not done and t < MAX_TIMESTEPS:
            if render:
                env.render()

            # Get actions from both agents
            action1 = agent1.select_action(state, explore=False)
            action2 = agent2.select_action(state, explore=False)

            # Combine actions and step environment
            next_state, reward, done, _, info = env.step(np.hstack([action1, action2]))

            state = next_state
            t += 1

        # Track game results
        winner = info["winner"]
        win_counts[winner] += 1

        print(f"Game {episode + 1}: Winner -> {'Agent 1' if winner == 1 else 'Agent 2' if winner == -1 else 'Draw'}")

    # Print final results
    print("\nValidation Complete!")
    print(f"Agent 1 ({model1_filename}) Wins: {win_counts[1]}")
    print(f"Agent 2 ({model2_filename}) Wins: {win_counts[-1]}")
    print(f"Draws: {win_counts[0]}")
    env.close()

# Allow user to pick two models to compete
model1 = select_model_from_buffer()
if model1:
    model2 = select_model_from_buffer()
    if model2:
        validate_td3(model1, model2, render=RENDER)
