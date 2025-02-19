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

# Device selection: Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# Hyperparameters
MAX_EPISODES = 10000
MAX_TIMESTEPS = 251
BATCH_SIZE = 256
GAMMA = 0.99
TAU = 0.01
POLICY_DELAY = 2
EXPLORE_NOISE = 0.1
POLICY_NOISE = 0.15
NOISE_CLIP = 0.5
EVAL_FREQ = 10
SAVE_FREQ = 100
BUFFER_DIR = "./TD3_models"  
BUFFER_SIZE = int(1e6)
LR = 3e-4
SEED = None
WINNER_REWARD = 10
DRAW_REWARD = 0
LOSER_REWARD = 0
CLOSENESS_REWARD = 0
TOUCH_REWARD = 1
DIRECTION_REWARD = 1
PUCK_NOT_TOUCHED_YET_PENALTY = 0

def train_td3_dynamic(num_episodes=10000, save_every=100, render=False, load_existing_agent=None, experiment_name="basic", seed=SEED):
    """
    Train TD3 agent with a dynamically changing opponent.
    """
    
    if seed is not None:
        # Set the random seed globally
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Initialize Weights & Biases logging inside the function
    if wandb.run is None:  # Prevents re-initialization if already running
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
    env = HockeyEnv(keep_mode=True)  # Initialize default environment
    env.keep_mode = True  # Ensure correct feature count before reset
    state, _ = env.reset()  # Get initial state
    state_dim = state.shape[0]  # Get correct feature size dynamically
    action_dim = env.action_space.shape[0] // 2
    max_action = float(env.action_space.high[0])

    # Initialize TD3 Agent
    agent = TD3Agent(state_dim, action_dim, max_action, lr=LR, gamma=GAMMA, tau=TAU, buffer_size=BUFFER_SIZE, explore_noise=EXPLORE_NOISE, seed=seed, device=device)

    # Load existing agent if specified
    if load_existing_agent:
        model_path = os.path.join(BUFFER_DIR, load_existing_agent)
        if os.path.exists(model_path):
            print(f"Loading existing agent from {model_path}")
            load_model(agent.actor, agent.critic, agent.actor_target, agent.critic_target, filename=model_path)
        else:
            print(f"Warning: Specified model {model_path} not found. Training from scratch.")

    # Opponent selection options
    opponents = {
        "weak": "basic_weak",
        "strong": "basic_strong",
        "td3": "td3_self_play"
    }


    # Initialize training loop variables
    replay_buffer = agent.replay_buffer
    total_timesteps = 0
    win_loss_history = {"weak": [], "strong": [], "td3": []}  # Track win/loss history

    # Training Loop
    for episode in range(1, num_episodes + 1):
        
        if episode % 3 == 0:
            opponent_name = "td3"  # Every 3rd episode, use self-play
        elif episode % 2 == 0:
            opponent_name = "weak"  # Every even non-TD3 episode uses weak
        else:
            opponent_name = "strong"  # Every odd non-TD3 episode uses strong


        #opponent_name = random.choice(list(opponents.keys()))
        print(f"Episode {episode}/{num_episodes} | Selected Opponent: {opponent_name}")

        # Initialize the environment with the selected opponent
        if opponent_name == "weak":
            env = HockeyEnv_BasicOpponent(weak_opponent=True)
            env.opponent = BasicOpponent(weak=True, keep_mode=True)
            
        elif opponent_name == "strong":
            env = HockeyEnv_BasicOpponent(weak_opponent=False)
            env.opponent = BasicOpponent(weak=False, keep_mode=True)
            
        elif opponent_name == "td3":
            env = HockeyEnv()  # Self-play
            
        else:
            raise ValueError(f"Invalid opponent type: {opponent_name}")  # Catch errors early

        # Explicitly Set keep_mode=False Before Resetting the Environment
        env.keep_mode = True  
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32)  # Ensure it's a CPU-based NumPy array

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
                
                opponents["td3"] = TD3Agent(state_dim, action_dim, max_action, lr=LR, gamma=GAMMA, tau=TAU, buffer_size=BUFFER_SIZE, explore_noise=EXPLORE_NOISE, seed=seed, device=device)
                load_model(opponents["td3"].actor, opponents["td3"].critic, opponents["td3"].actor_target, opponents["td3"].critic_target,
                            filename=os.path.join(BUFFER_DIR, model_filename))

                
            else:
                print("No previous self-play models found. Training opponent from scratch.")
                opponents["td3"] = TD3Agent(state_dim, action_dim, max_action, lr=LR, gamma=GAMMA, tau=TAU, buffer_size=BUFFER_SIZE, explore_noise=EXPLORE_NOISE, seed=seed, device=device)

        # Ensure alternating starts
        env.reset(one_starting=episode % 2 == 0)
        state = np.array(state, dtype=np.float32)  # Ensure it's a CPU-based NumPy array
        episode_reward = 0
        episode_result_reward = 0
        episode_reward_closeness = 0
        episode_reward_touch = 0
        episode_reward_direction = 0

        for t in range(MAX_TIMESTEPS):
            total_timesteps += 1

            if render:
                env.render(mode="human")

            # Select action
            state_tensor = torch.FloatTensor(state).to(device)
            action = agent.select_action(state_tensor, explore=True)
            
            # Convert action to NumPy array
            if isinstance(action, torch.Tensor):  
                action = action.cpu().numpy()  # Move back to CPU only if it's a PyTorch tensor

            #print(f"Action before scaling: {action}")

            # Get opponent's action
            if opponent_name == "td3":
                opponent_action = opponents["td3"].select_action(np.array(state), explore=True)
                # Only move to CPU if it's a PyTorch tensor
                if isinstance(opponent_action, torch.Tensor):  
                    opponent_action = opponent_action.cpu().numpy()
                
            else:
                opponent_action = np.array(env.opponent.act(env.obs_agent_two()), dtype=np.float32)

            #print(f"Agent action: {action} | Opponent action: {opponent_action}")
            
            #print(f"Critic1 Loss: {agent.critic1_loss:.4f} | Critic2 Loss: {agent.critic2_loss:.4f} | Actor Loss: {agent.actor_loss:.4f}")
            
            #print("Action Space:", env.action_space)


            # Step environment
            next_state, reward, done, _, info = env.step(np.hstack([action, opponent_action]))

            # Ensure next_state is a NumPy array for environment compatibility
            next_state = np.array(next_state, dtype=np.float32)

            # Reward adjustments
            if info["winner"] == 1:
                reward += WINNER_REWARD
            elif info["winner"] == 0:
                reward += DRAW_REWARD
            elif info["winner"] == -1:
                reward += LOSER_REWARD

            reward += CLOSENESS_REWARD * info["reward_closeness_to_puck"] # 0
            reward += TOUCH_REWARD * info["reward_touch_puck"] # 0
            reward += DIRECTION_REWARD * info["reward_puck_direction"] # 0
                
            reward = reward / 10.0  # Scale rewards down to prevent instability    
                
            episode_reward_closeness += info["reward_closeness_to_puck"]
            episode_reward_closeness += CLOSENESS_REWARD * info["reward_closeness_to_puck"]
            episode_reward_touch += TOUCH_REWARD * info["reward_touch_puck"]
            episode_reward_direction += DIRECTION_REWARD * info["reward_puck_direction"]

            # Add to replay buffer
            replay_buffer.add(state, action, float(reward), next_state, float(done))

            state = next_state
            episode_reward += reward

            # Train agent after collecting enough experience
            if total_timesteps > BATCH_SIZE:
                agent.train(BATCH_SIZE, POLICY_NOISE, NOISE_CLIP, POLICY_DELAY)

            if done:
                break
            
        # Track win/loss/draw rate in WandB
        win_loss_history[opponent_name].append(info["winner"])
        if len(win_loss_history[opponent_name]) > 50:
            win_loss_history[opponent_name].pop(0)

        win_rates = {opp: sum([1 for outcome in win_loss_history[opp] if outcome == 1]) / max(1, len(win_loss_history[opp])) for opp in win_loss_history}
        loss_rates = {opp: sum([1 for outcome in win_loss_history[opp] if outcome == -1]) / max(1, len(win_loss_history[opp])) for opp in win_loss_history}
        draw_rates = {opp: sum([1 for outcome in win_loss_history[opp] if outcome == 0]) / max(1, len(win_loss_history[opp])) for opp in win_loss_history}
        
        win_loss_component = 0  # Default value (if no win/loss occurs)
        if info["winner"] == 1:
            win_loss_component = WINNER_REWARD  # Add win reward
        elif info["winner"] == 0:
            win_loss_component = DRAW_REWARD  # Add draw reward
        elif info["winner"] == -1:
            win_loss_component = LOSER_REWARD  # Add loss penalty

        episode_length = t + 1  # Track episode length
        
        if done:  # Only apply at the end of the episode
            if info["winner"] == 1:
                episode_result_reward += 10 + WINNER_REWARD
            elif info["winner"] == 0:
                episode_result_reward += DRAW_REWARD
            elif info["winner"] == -1:
                episode_result_reward += -10 + LOSER_REWARD
                
        # Compute cumulative win, loss, and draw rates
        cumulative_wins = sum([1 for outcome in win_loss_history[opponent_name] if outcome == 1])
        cumulative_losses = sum([1 for outcome in win_loss_history[opponent_name] if outcome == -1])
        cumulative_draws = sum([1 for outcome in win_loss_history[opponent_name] if outcome == 0])
        total_games = len(win_loss_history[opponent_name])
        
        cumulative_win_rate = cumulative_wins / max(1, total_games)
        cumulative_loss_rate = cumulative_losses / max(1, total_games)
        cumulative_draw_rate = cumulative_draws / max(1, total_games)
        
        # Ensure info dictionary is handled on CPU (for logging purposes)
        # reward_touch_puck = float(info["reward_touch_puck"])  # Ensure float conversion
        # reward_closeness = float(info["reward_closeness_to_puck"])
        # reward_direction = float(info["reward_puck_direction"])

        # Log everything to WandB
        wandb.log({
            f"win_rate_{opponent_name}": win_rates[opponent_name],
            f"loss_rate_{opponent_name}": loss_rates[opponent_name],
            f"draw_rate_{opponent_name}": draw_rates[opponent_name],
            f"cumulative_win_rate_{opponent_name}": cumulative_win_rate,
            f"cumulative_loss_rate_{opponent_name}": cumulative_loss_rate,
            f"cumulative_draw_rate_{opponent_name}": cumulative_draw_rate,
            "Episode": episode,
            "Total Timesteps": total_timesteps,
            "Episode Reward": episode_reward,
            "Episode Length": episode_length,  # Tracks how long each game lasts
            "Episode Result Reward": episode_result_reward,
            "Episode Closeness Reward": episode_reward_closeness,
            "Episode Touch Reward": episode_reward_touch,
            "Episode Direction Reward": episode_reward_direction,
        })

        # Save model
        if episode % save_every == 0:
            if not os.path.exists(BUFFER_DIR):
                os.makedirs(BUFFER_DIR)

            # Save and Log Model Checkpoint
            model_path = os.path.join(BUFFER_DIR, f"TD3_Hockey_{experiment_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{episode}.pth")
            save_model(agent.actor, agent.critic, agent.actor_target, agent.critic_target, filename=model_path)

            # Log model save event in WandB
            wandb.log({"Saved Model": model_path})

        print("Total Episode Reward",episode_reward)
        print("End Results Reward", episode_result_reward)
        print("End Closeness Reward", episode_reward_closeness)
        print("End Touch Reward", episode_reward_touch)
        print("End Direction Reward", episode_reward_direction)
    env.close()
wandb.finish()