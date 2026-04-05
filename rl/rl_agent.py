import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from rl.dqn import DQN, DuelingDQN, StandardDQN
from env.models import Severity
# Import both buffer types; rl_agent selects based on use_per flag
from rl.prioritized_replay_buffer import PrioritizedReplayBuffer, SimpleReplayBuffer


class DQNAgent:
    def __init__(
        self,
        state_size,
        action_size,
        # --- Feature flags (all True = fully-upgraded path) ---
        use_dueling: bool = True,      # Use DuelingDQN instead of StandardDQN
        use_per: bool = True,           # Use Prioritized Experience Replay
        use_soft_update: bool = True,  # Soft target-network update each step
        normalize_rewards: bool = False # Z-score normalize rewards before storing
    ):
        self.state_size = state_size
        self.action_size = action_size

        # Store feature flags for runtime inspection / easy toggling
        self.use_dueling = use_dueling
        self.use_per = use_per
        self.use_soft_update = use_soft_update
        self.normalize_rewards = normalize_rewards

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Network selection: DuelingDQN (default) or StandardDQN (fallback) ---
        NetworkClass = DuelingDQN if use_dueling else StandardDQN
        self.policy_net = NetworkClass(state_size, action_size).to(self.device)
        self.target_net = NetworkClass(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=5e-4)

        # --- Replay buffer selection: PER (default) or SimpleReplayBuffer (fallback) ---
        if use_per:
            self.memory = PrioritizedReplayBuffer(capacity=20000)
        else:
            self.memory = SimpleReplayBuffer(capacity=20000)

        # Running stats for optional reward normalisation (Welford online update)
        self._reward_mean = 0.0
        self._reward_var = 1.0
        self._reward_count = 0

        self.gamma = 0.99
        self.tau = 0.005  # Soft target update rate
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9992  # Slower decay for more exploration
        self.batch_size = 128
        self.step_count = 0

    def _normalize_reward(self, reward: float) -> float:
        """
        Welford online z-score normalisation.
        Keeps mean ≈ 0, std ≈ 1 so the agent sees a stable reward scale.
        Only active when normalize_rewards=True.
        """
        self._reward_count += 1
        delta = reward - self._reward_mean
        self._reward_mean += delta / self._reward_count
        delta2 = reward - self._reward_mean
        self._reward_var += (delta * delta2 - self._reward_var) / self._reward_count
        std = max(float(self._reward_var ** 0.5), 1e-8)
        return (reward - self._reward_mean) / std

    def act(self, state, mask):
        if random.random() < self.epsilon:
            valid_indices = np.where(mask == 1)[0]
            return int(np.random.choice(valid_indices))

        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        self.policy_net.eval()
        with torch.no_grad():
            q_values = self.policy_net(state).cpu().numpy()[0]
        self.policy_net.train()

        # Apply mask
        q_values[mask == 0] = -1e9

        return int(np.argmax(q_values))

    def soft_update(self, target, online, tau):
        """
        Soft update for target network weights: target = τ * online + (1 - τ) * target
        """
        for target_param, online_param in zip(target.parameters(), online.parameters()):
            target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)

    def get_coordinated_reward(self, observation, action_model, base_reward):
        """
        Calculates a shaped reward considering coordination between ambulances.
        - Coordination Penalty: -50 if multiple ambulances assigned to same emergency.
        - Resource Awareness: Bonus for remaining hospital capacity, penalty for excessive idle.
        """
        penalty = 0.0
        bonus = 0.0

        # 1. COORDINATION PENALTY (Strict -50)
        # Check if another ambulance already has this emergency as target
        other_targets = [a.target_emg_id for a in observation.ambulances if a.id != action_model.ambulance_id]
        if action_model.emergency_id in other_targets and action_model.emergency_id != "":
            penalty -= 50.0

        # 2. FUTURE-AWARE RESOURCE SIGNALS
        # Bonus for remaining hospital capacity (Resource balancing)
        total_capacity = sum([h.capacity for h in observation.hospitals])
        bonus += (total_capacity * 0.1)

        # Penalty for excessive idle ambulances (Efficiency)
        idle_count = len([a for a in observation.ambulances if a.state == "idle"])
        penalty -= (idle_count * 0.2)

        # 3. Encourage Coverage Diversity
        positions = [a.node for a in observation.ambulances]
        if len(positions) > 1:
            spread = np.std(positions)
            bonus += (spread / 100.0) * 2.0

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

    def store(self, state, action, reward, next_state, done):
        """
        Stores a transition in the replay buffer.
        Applies z-score reward normalization when normalize_rewards=True.
        Use this instead of agent.memory.push() so the flag is honoured.
        """
        if self.normalize_rewards:
            reward = self._normalize_reward(reward)
        self.memory.push(state, action, reward, next_state, done)

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        # 1. Sample from PER
        batch, indices, weights = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # 2. Convert to tensors
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)
        weights = torch.tensor(weights, dtype=torch.float32).unsqueeze(1).to(self.device)

        # 3. Compute Current Q (Online Net)
        current_q = self.policy_net(states).gather(1, actions)
        
        # 4. Compute Target Q (Double DQN: Use Online Net to select action, Target Net to evaluate)
        with torch.no_grad():
            best_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, best_actions).detach()
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # 5. Calculate TD-errors for PER priority update
        td_errors = torch.abs(target_q - current_q).detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors)

        # 6. Optimized Loss (Smooth L1 + PER Weights)
        loss = (weights * F.smooth_l1_loss(current_q, target_q, reduction='none')).mean()

        # 7. Step Optimizer
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient Clipping (max norm = 1.0)
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        
        self.optimizer.step()

        # 8. Target network update
        if self.use_soft_update:
            # Soft update every step: target = τ*online + (1-τ)*target
            self.soft_update(self.target_net, self.policy_net, self.tau)
        else:
            # Hard update fallback: copy weights every 1000 steps
            if self.step_count % 1000 == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

        self.step_count += 1

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
