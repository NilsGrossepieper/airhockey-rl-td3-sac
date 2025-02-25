from __future__ import annotations

import argparse
import uuid

import hockey.hockey_env as h_env
import numpy as np
from td3 import TD3 

from comprl.client import Agent, launch_client


class RandomAgent(Agent):
    """A hockey agent that simply uses random actions."""

    def get_step(self, observation: list[float]) -> list[float]:
        return np.random.uniform(-1, 1, 4).tolist()

    def on_start_game(self, game_id) -> None:
        print("game started")

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"
        print(
            f"game ended: {text_result} with my score: "
            f"{stats[0]} against the opponent with score: {stats[1]}"
        )


class HockeyAgent(Agent):
    """A hockey agent that can be weak or strong."""

    def __init__(self, weak: bool) -> None:
        super().__init__()

        self.hockey_agent = h_env.BasicOpponent(weak=weak)

    def get_step(self, observation: list[float]) -> list[float]:
        # NOTE: If your agent is using discrete actions (0-7), you can use
        # HockeyEnv.discrete_to_continous_action to convert the action:
        #
        # from hockey.hockey_env import HockeyEnv
        # env = HockeyEnv()
        # continuous_action = env.discrete_to_continous_action(discrete_action)

        action = self.hockey_agent.act(observation).tolist()
        return action

    def on_start_game(self, game_id) -> None:
        game_id = uuid.UUID(int=int.from_bytes(game_id))
        print(f"Game started (id: {game_id})")

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"
        print(
            f"Game ended: {text_result} with my score: "
            f"{stats[0]} against the opponent with score: {stats[1]}"
        )

class TD3Agent(Agent):
    def __init__(self, args):
        super().__init__()
        self.td3 = TD3(args)
        self.td3.load("td3_model.pth")
        
    def get_step(self, observation: list[float]) -> list[float]:
        observation = np.array(observation)
        action = self.td3.get_action(observation, episode_count=1000000).tolist()
        return action

    def on_start_game(self, game_id) -> None:
        game_id = uuid.UUID(int=int.from_bytes(game_id))
        print(f"Game started (id: {game_id})")

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"
        print(
            f"Game ended: {text_result} with my score: "
            f"{stats[0]} against the opponent with score: {stats[1]}"
        )
        
# Function to initialize the agent.  This function is used with `launch_client` below,
# to lauch the client and connect to the server.
def initialize_agent(agent_args: list[str]) -> Agent:
    # Use argparse to parse the arguments given in `agent_args`.
    args = argparse.Namespace(
    gamma = 0.99,
    model_dim = 256,
    capacity = 10000,
    batch_size = 256,
    lr = 3e-4,
    policy_delay = 2,
    policy_noise_method = "decay",
    policy_noise = 0.2,
    noise_clip = 0.5,
    initial_policy_noise = 0.4,
    policy_noise_decay_rate = 0.9997,
    exploration_noise_method = "decay",
    exploration_noise = 0.1,
    initial_exploration_noise = 0.4,
    exploration_noise_decay_rate = 0.9997,
    update_type = "soft",
    tau = 0.005,
    hard_update_frequency = 10000,
)

    agent = TD3Agent(args)

    # And finally return the agent.
    return agent


def main() -> None:
    launch_client(initialize_agent)


if __name__ == "__main__":
    main()
