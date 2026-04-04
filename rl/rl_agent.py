import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl.dqn import DQN
from env.models import Severity
from rl.replay_buffer import PrioritizedReplayBuffer

# ReplayBuffer class removed (now using external PrioritizedReplayBuffer)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=5e-4)

        self.memory = PrioritizedReplayBuffer()

        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.999
        self.batch_size = 64
        self.update_target_steps = 50
        self.step_count = 0

    def act(self, state, mask):
        if random.random() < self.epsilon:
            valid_indices = np.where(mask == 1)[0]
            return int(np.random.choice(valid_indices))

        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        q_values = self.policy_net(state).detach().cpu().numpy()

        # Apply mask
        q_values[mask == 0] = -1e9

        return int(np.argmax(q_values))

    def get_coordinated_reward(self, observation, action_model, base_reward):
        """
        Calculates a shaped reward considering coordination between ambulances.
        """
        penalty = 0.0
        bonus = 0.0

        # 1. Prevent Conflict: Multiple ambulances going to same emergency
        # (Though mask handles assignments, this penalizes redundant movement logic)
        other_targets = [a.target_emg_id for a in observation.ambulances if a.id != action_model.ambulance_id]
        if action_model.emergency_id in other_targets and action_model.emergency_id != "":
            penalty -= 15.0

        # 2. Encourage Coverage Diversity: Distribute across city
        # Reward for spatial variance among all ambulance nodes
        positions = [a.node for a in observation.ambulances]
        if len(positions) > 1:
            spread = np.std(positions)
            bonus += (spread / 100.0) * 2.0  # Encourage dispersion

        # 3. Penalty for Duplicate Assignment logic
        # (Checks if the current action is redundant with current field states)
        if action_model.ambulance_id is not None and action_model.emergency_id:
            amb = next((a for a in observation.ambulances if a.id == action_model.ambulance_id), None)
            if amb and amb.state != "idle":
                penalty -= 5.0 # Penalty for trying to dispatch a non-idle ambulance

        return base_reward + penalty + bonus

    def get_priority_weighted_reward(self, observation, action_model, base_reward):
        """
        Scales the reward based on the severity of the emergency.
        """
        # Find the emergency to get its severity
        emg = next((e for e in observation.emergencies if e.id == action_model.emergency_id), None)
        if not emg:
            return base_reward
            
        multipliers = {
            Severity.CRITICAL: 2.0,
            Severity.HIGH: 1.5,
            Severity.NORMAL: 1.0
        }
        
        return base_reward * multipliers.get(emg.severity, 1.0)

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        current_q = self.policy_net(states).gather(1, actions)
        
        # Double DQN: Select best action using policy_net, evaluate using target_net
        with torch.no_grad():
            best_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, best_actions).detach()
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # Bias Q-learning: Weight loss by reward magnitude (proxy for priority)
        # Rescale rewards to use as importance weights for the loss
        weights = torch.clamp(rewards.abs(), min=1.0)
        loss = (weights * (current_q - target_q)**2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient Clipping (Stability Boost)
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        
        self.optimizer.step()

        # Update target network
        self.step_count += 1
        if self.step_count % self.update_target_steps == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self, episode=None):
        """
        Decays exploration rate once per episode.
        Optional episode parameter to force epsilon values at milestones.
        """
        if episode is not None and episode >= 1000:
            self.epsilon = 0.05
            return

        if episode is not None and episode >= 500:
            self.epsilon = 0.1
            return

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Ensure epsilon does NOT go below 0.05
        self.epsilon = max(self.epsilon, 0.05)

if __name__ == "__main__":
    state_size = 50
    action_size = 10

    agent = DQNAgent(state_size, action_size)

    # Dummy data
    state = np.random.rand(state_size).astype(np.float32)
    next_state = np.random.rand(state_size).astype(np.float32)

    mask = np.ones(action_size)

    action = agent.act(state, mask)

    agent.memory.push(state, action, 1.0, next_state, False)

    # Fill memory enough to train
    for _ in range(agent.batch_size):
        agent.memory.push(state, action, 1.0, next_state, False)

    for _ in range(100):
        agent.train_step()

    print("Test completed successfully")
