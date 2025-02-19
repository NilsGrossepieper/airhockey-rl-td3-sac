import numpy as np
import random

class ReplayBuffer:
    def __init__(self, max_size, state_dim, action_dim):
        self.max_size = max_size

        self.buffer = {
            "states": np.zeros((max_size, state_dim), dtype=np.float32),
            "actions": np.zeros((max_size, action_dim), dtype=np.float32),
            "rewards": np.zeros((max_size, 1), dtype=np.float32),
            "next_states": np.zeros((max_size, state_dim), dtype=np.float32),
            "dones": np.zeros((max_size, 1), dtype=np.float32),
        }

        self.position = 0  # Tracks where to insert the next experience

    def add(self, state, action, reward, next_state, done):       
        idx = self.position  # Current position in circular buffer

        self.buffer["states"][idx] = state
        self.buffer["actions"][idx] = action
        self.buffer["rewards"][idx] = reward
        self.buffer["next_states"][idx] = next_state
        self.buffer["dones"][idx] = done

        self.position = (self.position + 1) % self.max_size  # Move insert position

    def sample(self, batch_size):
        idxs = np.random.randint(0, len(self.buffer["states"]), batch_size)

        return (
            self.buffer["states"][idxs],
            self.buffer["actions"][idxs],
            self.buffer["rewards"][idxs].reshape(-1, 1),
            self.buffer["next_states"][idxs],
            self.buffer["dones"][idxs].reshape(-1, 1),
        )

    def size(self):
        """
        Returns the current size of the buffer.
        """
        return len(self.buffer)
