import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    """
    Deep Q-Network for ambulance dispatch reinforcement learning.
    """
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        
        # Dense layers
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        """
        Forward pass to compute Q-values for all possible actions.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    import numpy as np

    # Example sizes
    state_size = 50
    action_size = 10

    # Initialize model
    model = DQN(state_size, action_size)

    # Create dummy input (batch of 1)
    sample_state = np.random.rand(state_size).astype(np.float32)
    sample_state = torch.tensor(sample_state)

    # Forward pass
    output = model(sample_state)

    print("Input shape:", sample_state.shape)
    print("Output Q-values shape:", output.shape)
    print("Q-values:", output)
