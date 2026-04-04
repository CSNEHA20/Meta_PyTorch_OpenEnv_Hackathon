import torch
import numpy as np
from env.environment import AmbulanceEnv
from rl.state_encoder import StateEncoder
from rl.action_mapper import ActionMapper
from rl.action_mask import ActionMask
from rl.dqn import DQN

def main():
    # 1. INITIALIZE COMPONENTS
    env = AmbulanceEnv()
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
    except FileNotFoundError:
        print("Error: dqn_model.pth not found. Please train the model first.")
        return

    # 3. RUN DEMO LOOP
    print("Starting Inference Demo...")
    obs = env.reset()
    step_count = 0
    done = False

    while not done:
        step_count += 1
        
        # Prepare state and mask
        state = encoder.encode(obs)
        mapper.build_action_space(obs)
        mask = mask_builder.build_mask(mapper)
        
        # Select action
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = model(state_tensor).numpy()[0]
        
        q_values[mask == 0] = -1e9
        action_index = int(np.argmax(q_values))
        action = mapper.decode(action_index)
        
        # Step environment
        obs, reward, done, info = env.step(action)
        
        # Print decision
        print(f"Step {step_count}:")
        print(f"Action \u2192 ambulance_id: {action.ambulance_id}, emergency_id: {action.emergency_id}, hospital_id: {action.hospital_id}")
        print(f"Reward \u2192 {reward:.2f}")
        print("-" * 30)

    print("Demo complete.")

if __name__ == "__main__":
    main()
