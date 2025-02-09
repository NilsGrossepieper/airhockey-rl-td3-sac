import sys
import os
import numpy as np
import random

sys.path.append(os.path.abspath('./hockey_env/hockey'))
import hockey_env as h_env

class RandomAgent():
    def act(self, obs):
        return np.random.uniform(-1,1,4)

class DynamicEnvironment():
    def __init__(self, only_self):
        self.counter = 0
        self.only_self = only_self
        self.outcomes = {}
        self.opponents = {"weak": h_env.BasicOpponent(weak=True),
                          #"random": RandomAgent(),
                          "strong": h_env.BasicOpponent(weak=False)}
        for name in self.opponents.keys():
            self.outcomes[name] = []
        
        self.env = h_env.HockeyEnv()
        
        self.current_player2_name = None

        
    def add_agent(self, agent):
        self.opponents["self"] = agent
        self.outcomes["self"] = []

    def reset(self):
        self.counter += 1
        self.env_step = 0
        self.already_touched_puck = False

        if self.only_self:
            self.current_player2_name = "self"
            self.evaluate()
        else:
            self.current_player2_name = random.choice(list(self.opponents.keys()))
        
        return self.env.reset(one_starting=self.counter % 2 == 0)
    
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
                    winner_r = 10
                case 0:
                    winner_r = 0
                case -1:
                    winner_r = -10

        if info["reward_touch_puck"] and not self.already_touched_puck:
            touch_r = 0.1 * self.env_step
            self.already_touched_puck = True
        elif not self.already_touched_puck:
            touch_r = -0.1
        else:
            touch_r = 0

        closeness_r = info["reward_closeness_to_puck"]
        
        r = winner_r + 1 * touch_r + 5 * closeness_r

        return r

    def evaluate(self):
        player2_idx = self.counter % (len(self.opponents) - 1)
        one_starting = self.counter // (len(self.opponents) - 1) % 2 == 0

        player2_name = list(self.opponents.keys())[player2_idx]
        player2 = self.opponents[player2_name]
        player_self = self.opponents["self"]
        obs, info = self.env.reset(one_starting=one_starting)
        d = False
        while not d:
            obs_agent2 = self.env.obs_agent_two()
            a_enemy = player2.act(obs_agent2)
            a = player_self.act(obs)

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