from td3 import TD3
import numpy as np
import datetime as datetime
import os
import torch
from dynamic_env import DynamicEnvironment
import argparse
import wandb

# Arguments parser to handle user-friendly training parameters
parser = argparse.ArgumentParser(description="Arguments for TD3")
parser.add_argument("--training_episodes", type=int, default=10000, help="Number of training episodes")
parser.add_argument("--only_self_play", action="store_true", help="Train TD3 only against itself")
parser.add_argument("--render", action="store_true", help="Render the environment")
parser.add_argument("--save_every", type=int, default=100, help="Save the model every n episodes")
parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
parser.add_argument("--model_dim", type=int, default=256, help="Model dimension")
parser.add_argument("--capacity", type=int, default=100_000, help="Capacity of the memory")
parser.add_argument("--save_folder", type=str, default=None, help="Folder to save the models")

# Reward-related arguments
parser.add_argument("--win_reward", type=float, default=10, help="Reward for winning")
parser.add_argument("--lose_reward", type=float, default=-10, help="Reward for losing")
parser.add_argument("--draw_reward", type=float, default=0, help="Reward for drawing")
parser.add_argument("--puck_closeness_reward_multiplier", type=float, default=1, help="Multiplier for puck closeness")
parser.add_argument("--puck_touch_reward_multiplier", type=float, default=0, help="Multiplier for puck touch")
parser.add_argument("--puck_direction_reward_multiplier", type=float, default=0, help="Multiplier for puck direction")
parser.add_argument("--puck_not_touched_yet_reward", type=float, default=0, help="Reward for every step the puck is not YET touched")

# Training hyperparameters
parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
parser.add_argument("--policy_delay", type=int, default=2, help="Policy delay")

# Policy delay-related arguments
parser.add_argument("--policy_noise_method", type=str, default="constant", choices=["decay", "constant"],
                    help="Method for policy noise: 'decay' (default) or 'constant'")
parser.add_argument("--policy_noise", type=float, default=0.2, help="Policy noise")
parser.add_argument("--noise_clip", type=float, default=0.5, help="Noise clip")
parser.add_argument("--initial_policy_noise", type=float, default=0.4, help="Initial policy noise")
parser.add_argument("--policy_noise_decay_rate", type=float, default=0.9997, help="Decay rate for policy noise")

# Exploration noise-related arguments
parser.add_argument("--exploration_noise_method", type=str, default="constant", choices=["decay", "constant"],
                    help="Method for exploration noise: 'decay' (default) or 'constant'")
parser.add_argument("--exploration_noise", type=float, default=0.1, help="Exploration noise")
parser.add_argument("--initial_exploration_noise", type=float, default=0.4, help="Initial exploration noise")
parser.add_argument("--exploration_noise_decay_rate", type=float, default=0.9997, help="Decay rate for exploration noise")

# Target update-related arguments
parser.add_argument("--update_type", type=str, default="soft", choices=["soft", "hard"],
                    help="Type of target update: 'soft' (default) or 'hard'")
parser.add_argument("--tau", type=float, default=0.005, help="Tau for updating the target networks")
parser.add_argument("--hard_update_frequency", type=int, default=10000,
                    help="Number of steps between hard updates")

args = parser.parse_args()

# Create save folder if not provided
if args.save_folder is None:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args.save_folder = f"checkpoints/{timestamp}"
os.makedirs(args.save_folder, exist_ok=True)

# Initialize Weights and Biases (WandB) for experiment tracking
wandb.init(project="Hockey-RL", config=vars(args), name=timestamp)  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create environment and agent
dyn_env = DynamicEnvironment(args)
agent = TD3(
    args,
    obs_dim=dyn_env.env.observation_space.shape[0],
    action_dim=dyn_env.env.action_space.shape[0] // 2,
    device=device)  

# Print configuration summary
print("=" * 50)
print("TD3 Training Configuration:")
print(f"Training Episodes: {args.training_episodes}")
print(f"Batch Size: {args.batch_size}")
print(f"Learning Rate: {args.lr}")
print(f"Gamma (Discount Factor): {args.gamma}")
print(f"Replay Buffer Capacity: {args.capacity}")
print("=" * 50)

# Target Network Updates
print("=" * 50)
print(f"Target Update Type: {args.update_type}")
if args.update_type == "hard":
    print(f"Hard Update Frequency: {args.hard_update_frequency} steps")
else:
    print(f"Soft Update Tau: {args.tau}")
print(f"Policy Delay: {args.policy_delay}")
print("=" * 50)

# Policy Noise
print("=" * 50)
print(f"Policy Noise Method: {args.policy_noise_method}")
if args.policy_noise_method == "decay":
    print(f"  - Initial Policy Noise: {args.initial_policy_noise}")
    print(f"  - Policy Noise Decay Rate: {args.policy_noise_decay_rate}")
else:
    print(f"  - Constant Policy Noise: {args.policy_noise}")
print(f"Policy Noise Clipping: {args.noise_clip}")
print("=" * 50)

# Exploration Noise
print("=" * 50)
print(f"Exploration Noise Method: {args.exploration_noise_method}")
if args.exploration_noise_method == "decay":
    print(f"  - Initial Exploration Noise: {args.initial_exploration_noise}")
    print(f"  - Exploration Noise Decay Rate: {args.exploration_noise_decay_rate}")
else:
    print(f"  - Constant Exploration Noise: {args.exploration_noise}")
print("=" * 50)

# Attach agent to the environment
dyn_env.add_agent(agent)
agent.save(f"{args.save_folder}/td3_0.pth")

# Initialize tracking variables
rews = []
q1_losses = []
q2_losses = []
policy_losses = []
agent.episode_noise_values_absolute = []
agent.episode_noise_values = []
agent.episode_policy_noise_values_absolute = []
agent.episode_policy_noise_values = []

d = False
episode = 0
episode_count = 0
obs, info = dyn_env.reset(episode_count)
obs = torch.tensor(obs, dtype=torch.float32, device=device)

# Training loop
while episode < args.training_episodes:
    if d:
        episode += 1
        episode_count += 1
        
        # Save model periodically
        if episode % args.save_every == 0:
            agent.save(f"{args.save_folder}/td3_{episode}.pth")
        
        obs, info = dyn_env.reset(episode_count)
        obs = torch.tensor(obs, dtype=torch.float32, device=device)
        
        # Log episode stats
        evaluation_string = dyn_env.get_evaluation()
        
        # Remove None values from policy_losses before computing mean
        filtered_policy_losses = [loss for loss in policy_losses if loss is not None]
        avg_policy_loss = np.mean(filtered_policy_losses) if len(filtered_policy_losses) > 0 else float('nan')
        
        # Compute mean exploration noise per episode
        mean_exploration_noise_absolute = np.mean(agent.episode_noise_values_absolute) if len(agent.episode_noise_values_absolute) > 0 else float('nan')
        mean_exploration_noise = np.mean(agent.episode_noise_values) if len(agent.episode_noise_values) > 0 else float('nan')
        
        # Compute mean policy noise per episode
        mean_policy_noise_absolute = np.mean(agent.episode_policy_noise_values_absolute) if len(agent.episode_policy_noise_values_absolute) > 0 else float('nan')
        mean_policy_noise = np.mean(agent.episode_policy_noise_values) if len(agent.episode_policy_noise_values) > 0 else float('nan')

        # Log episode stats to console and Weights and Biases
        print(f"Step {episode: 5} Reward:{np.sum(rews): .4f} Losses: {np.mean(q1_losses): .2f} {np.mean(q2_losses): .2f} {avg_policy_loss: .2f} {evaluation_string}")

        # Log episode stats to Weights and Biases
        wandb.log({
            "Reward": np.sum(rews), # Logs total reward for the episode
            "q1_loss": np.mean(q1_losses), # WandB handles NaN values properly
            "q2_loss": np.mean(q2_losses), # WandB handles NaN values properly
            "policy_loss": avg_policy_loss,  # WandB handles NaN values properly
            "mean_exploration_noise_absolute": mean_exploration_noise_absolute,  # Logs absolute mean exploration noise
            "mean_exploration_noise": mean_exploration_noise,  # Logs raw mean exploration noise
            "mean_policy_noise_absolute": mean_policy_noise_absolute,  # Logs absolute mean policy noise
            "mean_policy_noise": mean_policy_noise,  # Logs raw mean policy noise
        }, step=episode)
        
        # Calculate last 50 games stats and all-time stats
        for name, outcome in dyn_env.outcomes.items():
            total_games_last_50 = len(outcome)
            if total_games_last_50 > 0:
                win_rate_last_50 = outcome.count(1) / total_games_last_50
                draw_rate_last_50 = outcome.count(0) / total_games_last_50
                loss_rate_last_50 = outcome.count(-1) / total_games_last_50
            else:
                win_rate_last_50, draw_rate_last_50, loss_rate_last_50 = float('nan'), float('nan'), float('nan')

            # Calculate all-time stats
            total_games_all_time = (
                dyn_env.total_outcomes[name]["wins"] +
                dyn_env.total_outcomes[name]["draws"] +
                dyn_env.total_outcomes[name]["losses"]
            )

            # Calculate all-time win, draw, and loss rates
            if total_games_all_time > 0:
                win_rate_all_time = dyn_env.total_outcomes[name]["wins"] / total_games_all_time
                draw_rate_all_time = dyn_env.total_outcomes[name]["draws"] / total_games_all_time
                loss_rate_all_time = dyn_env.total_outcomes[name]["losses"] / total_games_all_time
            else:
                win_rate_all_time, draw_rate_all_time, loss_rate_all_time = float('nan'), float('nan'), float('nan')

            # Log win, draw, and loss rates to Weights and Biases
            wandb.log({
                # Last 50 Games Stats
                f"outcomes/{name}_wins_last_50_games": outcome.count(1),
                f"outcomes/{name}_draws_last_50_games": outcome.count(0),
                f"outcomes/{name}_losses_last_50_games": outcome.count(-1),
                f"outcomes/{name}_win_rate_last_50": win_rate_last_50,
                f"outcomes/{name}_draw_rate_last_50": draw_rate_last_50,
                f"outcomes/{name}_loss_rate_last_50": loss_rate_last_50,

                # All-Time Stats
                f"outcomes/{name}_total_wins": dyn_env.total_outcomes[name]["wins"],
                f"outcomes/{name}_total_draws": dyn_env.total_outcomes[name]["draws"],
                f"outcomes/{name}_total_losses": dyn_env.total_outcomes[name]["losses"],
                f"outcomes/{name}_win_rate_all_time": win_rate_all_time,
                f"outcomes/{name}_draw_rate_all_time": draw_rate_all_time,
                f"outcomes/{name}_loss_rate_all_time": loss_rate_all_time
            }, step=episode)
        
        # Reset tracking variables
        rews = []
        q1_losses = []
        q2_losses = []
        policy_losses = []
        agent.episode_noise_values_absolute = []
        agent.episode_noise_values = []
        agent.episode_policy_noise_values_absolute = []
        agent.episode_policy_noise_values = []
        
    # Get action from agent
    a = agent.get_action(obs, episode_count) 
    obs_next, r, d, t, info = dyn_env.step(a, episode_count)
        
    if args.render:
        dyn_env.env.render(mode="human")

    # Convert observation to tensor and send to device
    obs_next = torch.tensor(obs_next, dtype=torch.float32).clone().detach().to(device)

    # Store transition in replay buffer
    agent.add_transition(obs, obs_next, a, r, d) 
    obs = obs_next
    
    # Update agent with experience replay
    ret = agent.learn(episode_count)
    if ret is not None:
        q1_loss, q2_loss, policy_loss = ret
        q1_losses.append(q1_loss)
        q2_losses.append(q2_loss)
        policy_losses.append(policy_loss)
    
    rews.append(r)