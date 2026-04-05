import torch
import numpy as np
import time
from env.environment import AmbulanceEnvironment
from rl.state_encoder import StateEncoder
from rl.action_mapper import ActionMapper
from rl.action_mask import ActionMask
from rl.dqn import DQN

def main():
    # 1. INITIALIZE COMPONENTS
    env = AmbulanceEnvironment()
    encoder = StateEncoder()
    mapper = ActionMapper()
    mask_builder = ActionMask()

    # 2. LOAD TRAINED MODEL
    obs = env.reset()
    mapper.build_action_space(obs)
    state_size = len(encoder.encode(obs))
    action_size = mapper.size()

    model = DQN(state_size, action_size)
    try:
        model.load_state_dict(torch.load("dqn_model.pth", map_location=torch.device('cpu')))
        model.eval()
    except Exception as e:
        print(f"Warning: dqn_model.pth could not be loaded ({e}). Using random agent for demo.")

    # 3. RUN DEMO LOOP
    print("Starting Advanced Simulation Demo...")
    obs = env.reset()
    step_count = 0
    done = False
    episode_reward = 0.0

    while not done:
        step_count += 1
        
        # Prepare state and mask
        state = encoder.encode(obs)
        mapper.build_action_space(obs)
        mask = mask_builder.build_mask(mapper)
        
        # Select action
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            try:
                q_values = model(state_tensor).numpy()[0]
                q_values[mask == 0] = -1e9
                action_index = int(np.argmax(q_values))
            except:
                # Fallback to random if model fails or is uninitialized
                valid_indices = np.where(mask == 1)[0]
                action_index = int(np.random.choice(valid_indices))
        
        action = mapper.decode(action_index)
        
        # Step environment
        obs = env.step(action)
        reward = obs.reward
        done = obs.done
        episode_reward += reward
        
        # Print decision & rubric
        print(f"Step {step_count}:")
        print(f"Action -> amb: {action.ambulance_id}, emg: {action.emergency_id}, hosp: {action.hospital_id}")
        if obs.rubric:
            r = obs.rubric
            print(f"Rubric -> Served: {r.emergency_served:+.1f}, Sev: {r.severity_bonus:+.1f}, Disp: {r.dispatch_speed:+.1f}, Time: {r.timeout_penalty:+.1f}")
        print(f"Step Reward -> {reward:+.2f} | Total: {episode_reward:.2f}")
        print("-" * 40)
        
        # Slow down for visualization if needed
        # time.sleep(0.05)

    print(f"Simulation complete. Final Score: {episode_reward:.2f}")

if __name__ == "__main__":
    main()
