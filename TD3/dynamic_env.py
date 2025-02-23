import sys
import os
import numpy as np
import random
from td3 import TD3

# Ensure the path is correctly added
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add the full hockey_env path
sys.path.append(os.path.join(ROOT_DIR, "hockey_env", "hockey"))

import hockey_env as h_env

class RandomAgent():
    """A simple random agent that selects actions uniformly."""
    def act(self, obs):
        return np.random.uniform(-1, 1, 4)

class DynamicEnvironment:
    """This class manages the dynamic training environment for a reinforcement learning agent."""
    
    def __init__(self, args):
        """Initialize the environment and tracking structures."""
        self.counter = 0
        self.only_self = args.only_self_play
        self.save_folder = args.save_folder
        
        # Reward settings
        self.win_reward = args.win_reward
        self.lose_reward = args.lose_reward
        self.draw_reward = args.draw_reward
        self.puck_closeness_reward_multiplier = args.puck_closeness_reward_multiplier
        self.puck_touch_reward_multiplier = args.puck_touch_reward_multiplier
        self.puck_direction_reward_multiplier = args.puck_direction_reward_multiplier
        self.puck_not_touched_yet_reward = args.puck_not_touched_yet_reward

        # Initialize tracking structures for outcomes
        self.outcomes = {}  # Stores last 50 games
        self.total_outcomes = {  # Stores all-time results
            "weak": {"wins": 0, "draws": 0, "losses": 0},
            "strong": {"wins": 0, "draws": 0, "losses": 0},
            "self": {"wins": 0, "draws": 0, "losses": 0}
        }

        # Define opponents
        self.opponents = {
            "weak": h_env.BasicOpponent(weak=True),
            "strong": h_env.BasicOpponent(weak=False),
            "self": TD3(args)
        }
        
        # Initialize outcome tracking for each opponent
        for name in self.opponents.keys():
            self.outcomes[name] = []  # Stores last 50 outcomes

        self.env = h_env.HockeyEnv()
        self.current_player2_name = None

    def add_outcome(self, enemy_agent_name, info):
        """Update game outcomes tracking structures."""
        result = info["winner"]
        
        # Track last 50 games
        self.outcomes[enemy_agent_name].append(result)
        if len(self.outcomes[enemy_agent_name]) > 50:
            self.outcomes[enemy_agent_name].pop(0)

        # Track overall statistics
        if result == 1:
            self.total_outcomes[enemy_agent_name]["wins"] += 1
        elif result == 0:
            self.total_outcomes[enemy_agent_name]["draws"] += 1
        elif result == -1:
            self.total_outcomes[enemy_agent_name]["losses"] += 1

    def reset(self, episode_count):
        """Reset the environment and select an opponent."""
        self.counter += 1
        self.env_step = 0
        self.already_touched_puck = False

        # Select opponent
        if self.only_self:
            self.current_player2_name = "self"
            self.evaluate(episode_count)
        else:
            self.current_player2_name = random.choice(list(self.opponents.keys()))
        
        # Load trained policy for self-play
        if self.current_player2_name == "self":
            filenames = os.listdir(self.save_folder)
            file_dict = {fname: int(fname.split('_')[1].split('.')[0]) for fname in filenames}
            sorted_filenames = sorted(file_dict.keys(), key=lambda x: file_dict[x])
            episode_numbers = np.array([file_dict[fname] for fname in sorted_filenames])
            if len(sorted_filenames) == 1:
                random_filename = sorted_filenames[0]
            else:
                weights = np.exp(episode_numbers / episode_numbers.max())  
                weights /= weights.sum()  
                random_filename = np.random.choice(sorted_filenames, p=weights)
            self.opponents["self"].load(f"{self.save_folder}/{random_filename}")
        
        return self.env.reset(one_starting=self.counter % 2 == 0)
    
    def add_agent(self, agent):
        """Add an agent to the environment."""
        self.agent = agent
    
    def step(self, action, episode_count):
        """Execute an environment step with the given action."""
        obs_agent2 = self.env.obs_agent_two()
        opponent = self.opponents[self.current_player2_name]
        
        # Get action from opponent
        if isinstance(opponent, TD3):  # Use get_action() for TD3
            a_enemy = opponent.get_action(obs_agent2, episode_count)
        else:  # Use act() for BasicOpponent
            a_enemy = opponent.act(obs_agent2)
        
        # Convert action to NumPy array if necessary
        action_np = action if isinstance(action, np.ndarray) else action.cpu().detach().numpy()
        
        # Step environment
        obs_next, r, d, t, info = self.env.step(np.hstack([action_np, a_enemy]))
        
        if d:
            self.add_outcome(self.current_player2_name, info)
        
        mod_r = self.calculate_reward(info, d)
        self.env_step += 1
        return obs_next, mod_r, d, t, info
    
    def calculate_reward(self, info, d):
        """Calculate the reward based on environment info and game state."""
        winner_r = 0
        if d:
            match info["winner"]:
                case 1:
                    winner_r = self.win_reward
                case 0:
                    winner_r = self.draw_reward
                case -1:
                    winner_r = self.lose_reward

        # Puck touch reward logic
        if info["reward_touch_puck"] and not self.already_touched_puck:
            touch_r = - self.puck_not_touched_yet_reward * self.env_step
            self.already_touched_puck = True
        elif not self.already_touched_puck:
            touch_r = self.puck_not_touched_yet_reward * self.env_step
        else:
            touch_r = 0

        # Additional reward components
        closeness_r = self.puck_closeness_reward_multiplier * info["reward_closeness_to_puck"]
        touch_bonus = self.puck_touch_reward_multiplier * info["reward_touch_puck"]
        puck_direction_r = self.puck_direction_reward_multiplier * info["reward_puck_direction"]
        
        return winner_r + touch_r + touch_bonus + closeness_r + puck_direction_r

    def evaluate(self, episode_count):
        """Evaluate the agent's performance against different opponents."""
        player2_idx = self.counter % (len(self.opponents) - 1)
        one_starting = self.counter // (len(self.opponents) - 1) % 2 == 0
        player2_name = list(self.opponents.keys())[player2_idx]
        player2 = self.opponents[player2_name]
        obs, info = self.env.reset(one_starting=one_starting)
        d = False
        while not d:
            obs_agent2 = self.env.obs_agent_two()
            a_enemy = player2.act(obs_agent2)
            a = self.agent.get_action(obs, episode_count)
            obs, r, d, t, info = self.env.step(np.hstack([a, a_enemy]))
        self.add_outcome(player2_name, info)

    def get_evaluation(self):
        """Retrieve formatted evaluation results for each opponent."""
        strings = []
        for name in self.opponents.keys():
            outcomes = self.outcomes[name]
            strings.append(f"{name}: {outcomes.count(1)}-{outcomes.count(0)}-{outcomes.count(-1)}\t")
        return "".join(strings)