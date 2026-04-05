import numpy as np
import random

class PrioritizedReplayBuffer:
    """
    Implementation of Proportional Prioritized Experience Replay (PER).
    Stores transitions with priorities based on absolute TD-error.
    """
    def __init__(self, capacity: int = 20000, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha  # Prioritization strength (0 = random sampling)
        self.beta = beta    # Importance sampling weight correction
        self.beta_increment = 0.001
        
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        """
        Stores a transition. Initial priority is set to the maximum existing priority.
        """
        max_prio = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int):
        """
        Samples a batch of experiences with probability proportional to their priority.
        Returns: samples, indices, importance sampling weights.
        """
        if len(self.buffer) == 0:
            return [], [], []
            
        # Calculate current probabilities
        current_priorities = self.priorities[:len(self.buffer)]
        probs = current_priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices based on probabilities
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        # Calculate importance sampling weights
        # IS_weight = (1/N * 1/P(i))^beta / max_weight
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        # Anneal beta towards 1.0 (Full correction)
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return samples, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        """
        Updates priorities of sampled transitions based on calculated TD-error.
        """
        for idx, prio in zip(batch_indices, batch_priorities):
            # Priority = |TD_error| + small epsilon to avoid zero selection
            self.priorities[idx] = float(abs(prio)) + 1e-6

    def __len__(self):
        return len(self.buffer)
