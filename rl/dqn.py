import torch
import torch.nn as nn
import torch.nn.functional as F


class StandardDQN(nn.Module):
    """
    Standard (non-dueling) DQN baseline.
    Architecture: Input -> 512 -> 256 -> 128 -> action_size
    Used as fallback when use_dueling=False.
    """
    def __init__(self, state_size, action_size):
        super(StandardDQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.01),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        return self.net(x)


class DQN(nn.Module):
    """
    Dueling DQN for improved learning stability and performance.
    Architecture: Input -> 512 -> 256 -> (Value & Advantage streams)
    """
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        
        # Shared Feature Extractor
        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.01),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.01)
        )
        
        # Value Stream V(s) - 1 output
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1)
        )
        
        # Advantage Stream A(s, a) - action_size outputs
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        """
        Forward pass using Dueling architecture: Q(s,a) = V(s) + (A(s,a) - mean(A))
        """
        features = self.feature_layer(x)
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine Value and Advantage streams
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        
        return q_values


class DuelingDQN(DQN):
    """
    DuelingDQN: Explicit named class for the dueling architecture.
    Inherits DQN which already implements Dueling DQN with:
      - Shared feature extractor (512 -> 256)
      - Separate value stream V(s)
      - Separate advantage stream A(s,a)
      - Q(s,a) = V(s) + (A(s,a) - mean(A))
    Used when use_dueling=True (default).
    """
    pass


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
