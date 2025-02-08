import numpy as np
import random

class ReplayBuffer:
    def __init__(self, max_size):
        """
        Initialize the Replay Buffer.

        Parameters:
        - max_size (int): Maximum number of experiences the buffer can store.
        """
        self.max_size = max_size
        self.buffer = []
        self.position = 0  # Tracks where to insert the next experience

    def add(self, state, action, reward, next_state, done):
        """
        Add a new experience to the buffer.

        Parameters:
        - state (np.array): The current state (s).
        - action (np.array): The action taken (a).
        - reward (float): The reward received (r).
        - next_state (np.array): The next state (s').
        - done (bool): Whether the episode has ended.
        """
        experience = (state, action, reward, next_state, done)

        # If buffer is not full, just append
        if len(self.buffer) < self.max_size:
            self.buffer.append(experience)
        else:
            # Overwrite the oldest experience (circular buffer)
            self.buffer[self.position] = experience

        self.position = (self.position + 1) % self.max_size  # Move insert position

    def sample(self, batch_size):
        """
        Sample a random batch of experiences from the buffer.

        Parameters:
        - batch_size (int): Number of experiences to sample.

        Returns:
        - Tuple of (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)  # Randomly pick batch_size elements

        # Unpack the batch into separate numpy arrays
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            np.array(rewards, dtype=np.float32).reshape(-1, 1),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32).reshape(-1, 1)
        )

    def size(self):
        """
        Returns the current size of the buffer.
        """
        return len(self.buffer)
