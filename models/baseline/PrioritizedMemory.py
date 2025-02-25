import numpy as np

class PrioritizedMemory():
    def __init__(self, max_size=100000, alpha=0.6):
        """
        max_size: maximum number of transitions
        alpha: how much prioritization is used (0 corresponds to uniform sampling)
        """
        self.max_size = max_size
        self.alpha = alpha
        self.size = 0
        self.current_idx = 0

        # Pre-allocate arrays for transitions and priorities.
        self.transitions = np.empty(max_size, dtype=object)
        self.priorities = np.zeros(max_size, dtype=np.float32)

    def add_transition(self, transition):
        """
        Store a new transition with maximum priority so that it is sampled soon.
        transition: typically a tuple (s, a, r, s_next, done)
        """
        self.transitions[self.current_idx] = transition
        
        # Assign the new transition a maximum priority (or default value if empty)
        max_priority = self.priorities.max() if self.size > 0 else 1.0
        self.priorities[self.current_idx] = max_priority
        
        # Update counters
        self.size = min(self.size + 1, self.max_size)
        self.current_idx = (self.current_idx + 1) % self.max_size

    def sample(self, batch=1, beta=0.4):
        """
        Sample a batch of transitions.
        beta: used to control how much the IS weights correct for bias (anneal beta towards 1)
        Returns:
            samples: array of sampled transitions
            indices: indices of the sampled transitions in the buffer
            is_weights: importance-sampling weights for bias correction
        """
        # Make sure we do not sample more than we have
        batch = min(batch, self.size)
        
        # Compute the probability distribution using priorities
        priorities = self.priorities[:self.size]
        scaled_priorities = priorities ** self.alpha
        sample_probabilities = scaled_priorities / scaled_priorities.sum()

        # Sample indices according to the probabilities
        indices = np.random.choice(self.size, batch, p=sample_probabilities, replace=False)
        samples = self.transitions[indices]
        
        # Compute importance-sampling (IS) weights
        is_weights = (self.size * sample_probabilities[indices]) ** (-beta)
        # Normalize weights
        is_weights /= is_weights.max()
        
        return samples, indices, is_weights

    def update_priorities(self, indices, new_priorities):
        """
        Update the priorities of sampled transitions after learning.
        indices: indices of the transitions that were sampled.
        new_priorities: new priority values (e.g., |TD error| + epsilon)
        """
        for idx, priority in zip(indices, new_priorities):
            self.priorities[idx] = priority
