"""
The code here is courtesy of Dr. Georg Martius, Max Planck Institute for Intelligent Systems, Tübingen, Germany. 
As part of the course: "Reinforcement Learning".
The code is used for educational purposes only.
"""

import numpy as np

# class to store transitions
class Memory:
    def __init__(self, max_size=100000, state_dim=8, action_dim=4):
        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)
        self.max_size = max_size
        self.size = 0
        self.current_idx = 0

    def add_transition(self, transition):
        state, action, reward, next_state, done = transition
        self.states[self.current_idx] = state
        self.actions[self.current_idx] = action
        self.rewards[self.current_idx] = reward
        self.next_states[self.current_idx] = next_state
        self.dones[self.current_idx] = done
        self.size = min(self.size + 1, self.max_size)
        self.current_idx = (self.current_idx + 1) % self.max_size

    def sample(self, batch=1):
        batch = min(batch, self.size)
        indices = np.random.choice(self.size, size=batch, replace=False)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )