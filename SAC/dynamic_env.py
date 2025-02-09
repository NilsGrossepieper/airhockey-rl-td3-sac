import sys
import os
import numpy as np
import random
from sac import SAC

sys.path.append(os.path.abspath('./hockey_env/hockey'))
import hockey_env as h_env

class RandomAgent():
    def act(self, obs):
        return np.random.uniform(-1,1,4)

class DynamicEnvironment():
    def __init__(self, args):
        self.counter = 0
        self.only_self = args.only_self_play
        self.save_folder = args.save_folder
        self.win_reward = args.win_reward
        self.lose_reward = args.lose_reward
        self.draw_reward = args.draw_reward
        self.puck_closeness_reward_multiplier = args.puck_closeness_reward_multiplier
        self.puck_not_touched_yet_reward = args.puck_not_touched_yet_reward

        self.outcomes = {}
        self.opponents = {"weak": h_env.BasicOpponent(weak=True),
                          #"random": RandomAgent(),
                          "strong": h_env.BasicOpponent(weak=False),
                          "self": SAC(args)}
        for name in self.opponents.keys():
            self.outcomes[name] = []
        
        self.env = h_env.HockeyEnv()
        
        self.current_player2_name = None


    def reset(self):
        self.counter += 1
        self.env_step = 0
        self.already_touched_puck = False

        if self.only_self:
            self.current_player2_name = "self"
            self.evaluate()
        else:
            self.current_player2_name = random.choice(list(self.opponents.keys()))
        
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
        self.agent = agent
    
    def step(self, action):
        obs_agent2 = self.env.obs_agent_two()
        a_enemy = self.opponents[self.current_player2_name].act(obs_agent2)
        obs_next, r, d, t, info = self.env.step(np.hstack([action.cpu().detach().numpy(),a_enemy]))
        if d:
            self.add_outcome(self.current_player2_name, info)
        mod_r = self.calculate_reward(info, d)
        self.env_step += 1
        return obs_next, mod_r, d, t, info
    
    def calculate_reward(self, info, d):
        winner_r = 0
        if d:
            match info["winner"]:
                case 1:
                    winner_r = self.win_reward
                case 0:
                    winner_r = self.draw_reward
                case -1:
                    winner_r = self.lose_reward

        if info["reward_touch_puck"] and not self.already_touched_puck:
            touch_r = - self.puck_not_touched_yet_reward * self.env_step
            self.already_touched_puck = True
        elif not self.already_touched_puck:
            touch_r = self.puck_not_touched_yet_reward
        else:
            touch_r = 0

        closeness_r = self.puck_closeness_reward_multiplier * info["reward_closeness_to_puck"]
        
        r = winner_r + touch_r +  closeness_r

        return r

    def evaluate(self):
        player2_idx = self.counter % (len(self.opponents) - 1)
        one_starting = self.counter // (len(self.opponents) - 1) % 2 == 0

        player2_name = list(self.opponents.keys())[player2_idx]
        player2 = self.opponents[player2_name]
        obs, info = self.env.reset(one_starting=one_starting)
        d = False
        while not d:
            obs_agent2 = self.env.obs_agent_two()
            a_enemy = player2.act(obs_agent2)
            a = self.agent.act(obs)

            obs, r, d, t, info = self.env.step(np.hstack([a, a_enemy]))
        self.add_outcome(player2_name, info)
            


    
    def add_outcome(self, enemy_agent_name, info):
        self.outcomes[enemy_agent_name].append(info["winner"])
        if len(self.outcomes[enemy_agent_name]) > 50:
            self.outcomes[enemy_agent_name].pop(0)

    def get_evaluation(self):
        strings = []
        for name in self.opponents.keys():
            outcomes = self.outcomes[name]
            strings.append(f"{name}: {outcomes.count(1)}-{outcomes.count(0)}-{outcomes.count(-1)}\t")
        return "".join(strings)