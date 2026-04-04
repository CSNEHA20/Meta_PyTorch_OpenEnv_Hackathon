import numpy as np
import random

class PrioritizedReplayBuffer:
    """
    Implementation of Prioritized Experience Replay using reward magnitude 
    as the priority metric.
    """
    def __init__(self, capacity: int = 10000, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        """
        Stores a transition and assigns initial priority based on reward magnitude.
        """
        # Priority based on absolute reward magnitude + small epsilon
        priority = abs(reward) + 0.1
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        
        # Store scaled priority
        self.priorities[self.pos] = priority ** self.alpha
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int):
        """
        Samples a batch of experiences with probability proportional to their priority.
        """
        if len(self.buffer) == 0:
            return []
            
        # Calculate current probabilities
        current_priorities = self.priorities[:len(self.buffer)]
        probabilities = current_priorities / current_priorities.sum()
        
        # Sample indices based on probabilities
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        
        # Return samples
        samples = [self.buffer[idx] for idx in indices]
        return samples

    def __len__(self):
        return len(self.buffer)
