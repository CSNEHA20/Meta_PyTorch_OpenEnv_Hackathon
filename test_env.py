from env.environment import AmbulanceEnvironment as AmbulanceEnv
from env.models import ActionModel, AmbulanceState

def test_ambulance_env():
    # 1. Initialize environment
    config = {
        "n_ambulances": 3,
        "n_hospitals": 2,
        "max_steps": 100,
        "seed": 42
    }
    env = AmbulanceEnv(config)
    
    # 2. Reset
    obs = env.reset(seed=42)
    print(f"Initial Observation: {len(obs.ambulances)} ambulances, {len(obs.hospitals)} hospitals")
    
    total_reward = 0
    done = False
    
    # 3. Simple Dispatch Strategy for Testing
    for step in range(100):
        # Default action: do nothing
        action = ActionModel(ambulance_id=None, emergency_id="", hospital_id=0)
        
        # Simple Logic: If there is an idle ambulance and an emergency, dispatch it.
        # Note: in models.py, state is an Enum member but acts like a string.
        idle_ambs = [a for a in obs.ambulances if a.state == AmbulanceState.IDLE]
        live_emgs = [e for e in obs.emergencies if not e.assigned]
        
        if idle_ambs and live_emgs:
            action.ambulance_id = idle_ambs[0].id
            action.emergency_id = live_emgs[0].id
            action.hospital_id = obs.hospitals[0].id
            print(f"[{step}] Step: Dispatching {action.ambulance_id} to {action.emergency_id}")
            
        obs = env.step(action)
        total_reward += obs.reward
        
        if obs.done:
            break
            
    print(f"\nFinal State Info:")
    print(env.metrics)
    print(f"Total Reward: {total_reward}")
    
    # 4. Determinism Check
    env2 = AmbulanceEnv(config)
    obs2 = env2.reset(seed=42)
    # Check if first ambulance node is the same
    assert obs2.ambulances[0].node == env.reset(seed=42).ambulances[0].node
    print("\nDeterminism check passed.")

if __name__ == "__main__":
    test_ambulance_env()
