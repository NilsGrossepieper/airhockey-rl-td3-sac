import torch
import numpy as np
import gymnasium as gym
import sys
import os
import re
import random
import wandb
from datetime import datetime
from td3_agent import TD3Agent  
from replay_buffer import ReplayBuffer 
from utils import save_model, load_model  

# Ensure Python finds the hockey environment
sys.path.append(os.path.abspath("../hockey_env"))

# Import Hockey Environment Correctly
from hockey.hockey_env import HockeyEnv, HockeyEnv_BasicOpponent, BasicOpponent

# Hyperparameters
MAX_EPISODES = 10000
MAX_TIMESTEPS = 251
BATCH_SIZE = 256
GAMMA = 0.99
TAU = 0.005
POLICY_DELAY = 2
EXPLORE_NOISE = 0.1
POLICY_NOISE = 0.2
NOISE_CLIP = 0.5
EVAL_FREQ = 10
SAVE_FREQ = 100
BUFFER_DIR = "./TD3_models"  
BUFFER_SIZE = int(1e6)
LR = 3e-4
WINNER_REWARD = 0
DRAW_REWARD = 0
LOSER_REWARD = 0
CLOSENESS_REWARD = 1
TOUCH_REWARD = 1
DIRECTION_REWARD = 1

def train_td3_dynamic(num_episodes=10000, save_every=100, render=False, load_existing_model=False, experiment_name="basic"):
    """
    Train TD3 agent with a dynamically changing opponent.
    """

        # Initialize Weights & Biases logging inside the function
    if wandb.run is None:  # ✅ Prevents re-initialization if already running
        wandb.init(project="td3_hockey", name=f"TD3_{experiment_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}", config={
            "learning_rate": LR,
            "batch_size": BATCH_SIZE,
            "gamma": GAMMA,
            "tau": TAU,
            "policy_delay": POLICY_DELAY,
            "explore_noise": EXPLORE_NOISE,
            "policy_noise": POLICY_NOISE,
            "noise_clip": NOISE_CLIP,
            "eval_freq": EVAL_FREQ,
            "save_freq": SAVE_FREQ,
            "buffer_size": BUFFER_SIZE
        },
        mode="online"  # Ensure results are stored on WandB servers           
        )

    # Initialize environment (only for getting dimensions)
    env = HockeyEnv(keep_mode=False)  # Initialize default environment
    env.keep_mode = False  # Ensure correct feature count before reset
    state, _ = env.reset()  # Get initial state
    state_dim = state.shape[0]  # Get correct feature size dynamically
    action_dim = env.action_space.shape[0] // 2
    max_action = float(env.action_space.high[0])

    # Initialize TD3 Agent
    agent = TD3Agent(state_dim, action_dim, max_action, lr=LR, gamma=GAMMA, tau=TAU, buffer_size=BUFFER_SIZE)

    # Opponent selection options
    opponents = {
        "weak": None,
        "strong": None,
        "td3": None  # Placeholder for dynamic self-play
    }

    # Initialize training loop variables
    replay_buffer = agent.replay_buffer
    total_timesteps = 0
    win_loss_history = {"weak": [], "strong": [], "td3": []}  # Track win/loss history

    # Training Loop
    for episode in range(1, num_episodes + 1):
        
        # Select opponent
        opponent_name = random.choice(list(opponents.keys()))
        print(f"Episode {episode}/{num_episodes} | Selected Opponent: {opponent_name}")

        # Initialize the environment with the selected opponent
        if opponent_name == "weak":
            env = HockeyEnv_BasicOpponent(weak_opponent=True)
            env.opponent = BasicOpponent(weak=True, keep_mode=False)  # Ensure keep_mode=False
            
        elif opponent_name == "strong":
            env = HockeyEnv_BasicOpponent(weak_opponent=False)
            env.opponent = BasicOpponent(weak=False, keep_mode=False)  # Ensure keep_mode=False
            
        elif opponent_name == "td3":
            env = HockeyEnv()  # Self-play
            
        else:
            raise ValueError(f"Invalid opponent type: {opponent_name}")  # Catch errors early

        env.reset(one_starting=episode % 2 == 0)  # Ensure alternating starts

        # Explicitly Set keep_mode=False Before Resetting the Environment
        env.keep_mode = False  
        state, _ = env.reset()

        # Handle Self-Play Model Selection with Weighted Probability
        if opponent_name == "td3":
            if not os.path.exists(BUFFER_DIR):
                os.makedirs(BUFFER_DIR)

            filenames = [f for f in os.listdir(BUFFER_DIR) if f.startswith("TD3_Hockey_")]
            if filenames:
                # Use regex to extract episode numbers safely
                def extract_episode_number(filename):
                    match = re.search(r"TD3_Hockey_\w+_(\d+).pth", filename)  
                    return int(match.group(1)) if match and match.group(1).isdigit() else -1


                # Sorting models based on extracted episode numbers
                sorted_filenames = sorted(filenames, key=extract_episode_number)

                # Apply exponential weighting to prioritize recent models
                weights = np.exp(np.linspace(-1.0, 0, len(sorted_filenames)))  
                weights /= weights.sum()  # Normalize to make it a probability distribution
                
                model_filename = np.random.choice(sorted_filenames, p=weights)  # Weighted selection
                print(f"Using self-play opponent: {model_filename}")
                
                opponents["td3"] = TD3Agent(state_dim, action_dim, max_action, lr=LR, gamma=GAMMA, tau=TAU, buffer_size=BUFFER_SIZE)
                load_model(opponents["td3"].actor, opponents["td3"].critic1, opponents["td3"].critic2,
                            opponents["td3"].actor_target, opponents["td3"].critic1_target, opponents["td3"].critic2_target,
                            filename=os.path.join(BUFFER_DIR, model_filename))
            else:
                print("No previous self-play models found. Training opponent from scratch.")
                opponents["td3"] = TD3Agent(state_dim, action_dim, max_action, lr=LR, gamma=GAMMA, tau=TAU, buffer_size=BUFFER_SIZE)

        episode_reward = 0

        for t in range(MAX_TIMESTEPS):
            total_timesteps += 1

            if render:
                env.render(mode="human")

            # Select action
            action = agent.select_action(state, explore=True, noise_std=EXPLORE_NOISE)

            # Get opponent's action
            if opponent_name == "td3":
                opponent_action = opponents["td3"].select_action(np.array(state), explore=True, noise_std=EXPLORE_NOISE)
            else:
                opponent_action = env.opponent.act(env.obs_agent_two())

            # Step environment
            next_state, reward, done, _, info = env.step(np.hstack([action, opponent_action]))

            # Reward adjustments
            if info["winner"] == 1:
                reward += WINNER_REWARD
            elif info["winner"] == 0:
                reward -= DRAW_REWARD
            elif info["winner"] == -1:
                reward -= LOSER_REWARD

            reward += CLOSENESS_REWARD * info["reward_closeness_to_puck"]
            reward += TOUCH_REWARD * info["reward_touch_puck"]
            reward += DIRECTION_REWARD * info["reward_puck_direction"]

            # Add to replay buffer
            replay_buffer.add(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            # Train agent after collecting enough experience
            if total_timesteps > BATCH_SIZE:
                agent.train(BATCH_SIZE, POLICY_NOISE, NOISE_CLIP, POLICY_DELAY)

            if done:
                break
            
        # ✅ Track win/loss/draw rate in WandB
        win_loss_history[opponent_name].append(info["winner"])
        if len(win_loss_history[opponent_name]) > 50:
            win_loss_history[opponent_name].pop(0)

        win_rates = {opp: sum([1 for outcome in win_loss_history[opp] if outcome == 1]) / max(1, len(win_loss_history[opp])) for opp in win_loss_history}
        loss_rates = {opp: sum([1 for outcome in win_loss_history[opp] if outcome == -1]) / max(1, len(win_loss_history[opp])) for opp in win_loss_history}
        draw_rates = {opp: sum([1 for outcome in win_loss_history[opp] if outcome == 0]) / max(1, len(win_loss_history[opp])) for opp in win_loss_history}

        # ✅ Log everything to WandB
        wandb.log({
            "Episode": episode,
            "Total Timesteps": total_timesteps,
            "Episode Reward": episode_reward,
            f"win_rate_{opponent_name}": win_rates[opponent_name],
            f"loss_rate_{opponent_name}": loss_rates[opponent_name],
            f"draw_rate_{opponent_name}": draw_rates[opponent_name]
        })

        # Save model
        if episode % save_every == 0:
            if not os.path.exists(BUFFER_DIR):
                os.makedirs(BUFFER_DIR)

            # ✅ Save and Log Model Checkpoint
            model_path = os.path.join(BUFFER_DIR, f"TD3_Hockey_{experiment_name}_{datetime.now().strftime('%Y-%m-%d')}_{episode}.pth")
            save_model(agent.actor, agent.critic1, agent.critic2, agent.actor_target, agent.critic1_target, agent.critic2_target, filename=model_path)

            # ✅ Log model save event in WandB
            wandb.log({"Saved Model": model_path})

    env.close()
wandb.finish()