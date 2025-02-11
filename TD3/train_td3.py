import torch
import numpy as np
import gymnasium as gym
import time
import sys
import os
from td3_agent import TD3Agent  
from replay_buffer import ReplayBuffer 
from utils import save_model, load_model  

# Ensure Python finds the hockey environment
sys.path.append(os.path.abspath("../hockey_env"))

# Import Hockey Environment Correctly
from hockey.hockey_env import HockeyEnv, HockeyEnv_BasicOpponent, BasicOpponent

# Hyperparameters
ENV_NAME = "HockeyEnv"
MAX_EPISODES = 1000
MAX_TIMESTEPS = 251
BATCH_SIZE = 100
GAMMA = 0.99
TAU = 0.005
POLICY_DELAY = 2
EXPLORE_NOISE = 0.1
POLICY_NOISE = 0.2
NOISE_CLIP = 0.5
EVAL_FREQ = 10
SAVE_FREQ = 100
BUFFER_DIR = "./TD3"  # Directory where models are saved
BUFFER_SIZE = int(1e6)
LR = 3e-4
WINNER_REWARD = 0
DRAW_REWARD = 0
LOSER_REWARD = 0
CLOSENESS_REWARD = 1
TOUCH_REWARD = 1
DIRECTION_REWARD = 1

def train_td3(num_episodes=1000, save_every=100, opponent="weak", render=False, load_existing_model=False, model_filename=None,
            winner_reward=WINNER_REWARD, draw_reward=DRAW_REWARD, loser_reward=LOSER_REWARD, closeness_reward=CLOSENESS_REWARD,
            touch_reward=TOUCH_REWARD, direction_reward=DIRECTION_REWARD):
    """
    Train TD3 agent with options for rendering and loading an existing model.
    When training in self-play (opponent="td3"), the latest model is loaded as the opponent.

    Parameters:
    - num_episodes: Number of training episodes
    - save_every: Save model every X episodes
    - opponent: "weak", "strong", or "td3"
    - render: If True, renders the game
    - load_existing_model: If True, loads a pre-trained model
    - model_filename: File to load (manually entered by user)
    """

    # Initialize environment with opponent
    if opponent == "weak":
        env = HockeyEnv_BasicOpponent(weak_opponent=True)
        env.opponent = BasicOpponent(weak=True, keep_mode=False)  # Ensure keep_mode=False
    elif opponent == "strong":
        env = HockeyEnv_BasicOpponent(weak_opponent=False)
        env.opponent = BasicOpponent(weak=False, keep_mode=False)  # Ensure keep_mode=False
    elif opponent == "td3":
        env = HockeyEnv()  # Self-play
    else:
        raise ValueError("Invalid opponent type. Choose 'weak', 'strong', or 'td3'.")

    # Ensure keep_mode is disabled in the environment
    env.keep_mode = False

    state, _ = env.reset()  # Get initial state
    state_dim = len(state)  # Ensure state_dim matches actual state size
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Initialize TD3 agent
    agent = TD3Agent(state_dim, action_dim, max_action, lr=LR, gamma=GAMMA, tau=TAU, buffer_size=BUFFER_SIZE)

    # If loading an existing model, manually enter filename
    if load_existing_model:
        if model_filename is None:
            model_filename = input("Enter the name of the model file to load (e.g., TD3_Hockey_500.pth): ").strip()
        
        if os.path.exists(os.path.join(BUFFER_DIR, model_filename)):
            load_model(agent.actor, agent.critic1, agent.critic2, filename=os.path.join(BUFFER_DIR, model_filename))
            print(f"Loaded model from {model_filename}")
        else:
            print(f"Model file '{model_filename}' not found! Training from scratch...")

    replay_buffer = agent.replay_buffer
    total_timesteps = 0

    # Set up opponent TD3 agent if in self-play
    opponent_agent = None
    if opponent == "td3":
        opponent_agent = TD3Agent(state_dim, action_dim, max_action, lr=LR, gamma=GAMMA, tau=TAU, buffer_size=BUFFER_SIZE)
        
        # Load the latest model if available
        model_files = sorted([f for f in os.listdir(BUFFER_DIR) if f.startswith("TD3_Hockey_") and f.endswith(".pth")])
        if model_files:
            latest_model = os.path.join(BUFFER_DIR, model_files[-1])
            load_model(opponent_agent.actor, opponent_agent.critic1, opponent_agent.critic2, filename=latest_model)
            print(f"Loaded latest opponent model: {latest_model}")

    # Training loop
    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0

        for t in range(MAX_TIMESTEPS):
            total_timesteps += 1

            if render:
                env.render(mode="human")  # Render the game if enabled

            # Select action with exploration noise
            action = agent.select_action(state, explore=True, noise_std=EXPLORE_NOISE)

            # Get opponent's action
            if opponent == "td3":
                opponent_action = opponent_agent.select_action(np.array(state), explore=True, noise_std=EXPLORE_NOISE)
            else:
                opponent_action = env.opponent.act(env.obs_agent_two())

            # Step environment
            next_state, reward, done, _, info = env.step(np.hstack([action, opponent_action]))
            
            # Reward for end outcome
            if info["winner"] == 1:
                reward += winner_reward
            elif info["winner"] == 0:
                reward -= draw_reward
            elif info["winner"] == -1:
                reward -= loser_reward
                
            # Reward for close to puck
            reward += closeness_reward * info["reward_closeness_to_puck"]
            
            # Reward for touching puck
            reward += touch_reward * info["reward_touch_puck"]
            
            # Reward for moving puck in correct direction
            reward += direction_reward * info["reward_puck_direction"]
            
            # Add to replay buffer
            replay_buffer.add(state, action, reward, next_state, done)            

            state = next_state
            episode_reward += reward

            # Train agent after collecting enough experience
            if total_timesteps > BATCH_SIZE:
                agent.train(BATCH_SIZE, POLICY_NOISE, NOISE_CLIP, POLICY_DELAY)

            if done:
                break  # Exit if the episode is done

        # Evaluate agent
        if episode % EVAL_FREQ == 0:
            eval_reward = 0
            for _ in range(5):  # Run 5 evaluation episodes
                state, _ = env.reset()
                for _ in range(MAX_TIMESTEPS):
                    if render:
                        env.render(mode="human")  # Render during evaluation if enabled
                    action = agent.select_action(np.array(state), explore=False, noise_std=0.0)
                    if opponent == "td3":
                        opponent_action = opponent_agent.select_action(np.array(state), explore=False, noise_std=0.0)
                    else:
                        opponent_action = env.opponent.act(env.obs_agent_two())

                    next_state, reward, done, _, _ = env.step(np.hstack([action, opponent_action]))
                    eval_reward += reward
                    state = next_state
                    if done:
                        break
            avg_eval_reward = eval_reward / 5
            print(f"Evaluation Reward: {avg_eval_reward:.2f}")

        # Save model
        if episode % save_every == 0:
            model_path = os.path.join(BUFFER_DIR, f"TD3_Hockey_{episode}.pth")
            save_model(agent.actor, agent.critic1, agent.critic2, filename=model_path)

            # Update opponent if in self-play mode
            if opponent == "td3":
                load_model(opponent_agent.actor, opponent_agent.critic1, opponent_agent.critic2, filename=model_path)
                print(f"🔥 Updated opponent to latest model: {model_path}")

    # Close environment
    env.close()