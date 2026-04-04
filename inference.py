import json
import sys
import logging
import torch
from env.environment import AmbulanceEnv
from env.models import ActionModel
from rl.dqn import DQN
from rl.state_encoder import StateEncoder
from rl.action_mapper import ActionMapper
from tasks.easy import EasyConfig
from tasks.medium import MediumConfig
from tasks.hard import HardConfig
from grader_easy import grade_easy
from grader_medium import grade_medium
from grader_hard import grade_hard

# Ensure all logging goes to stderr so stdout remains pure JSON
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)

# Global initialization of RL components
encoder = StateEncoder(max_nodes=100)
mapper = ActionMapper(max_ambulances=5, max_emergencies=10, max_hospitals=5)

# Dimensions are fixed based on encoder/mapper configurations
# state_dim = 55, action_dim = 251
model = DQN(state_size=55, action_size=251)
try:
    model.load_state_dict(torch.load("ambulance_dqn.pth", map_location=torch.device('cpu')))
    logging.info("Trained DQN model loaded successfully.")
except FileNotFoundError:
    logging.warning("ambulance_dqn.pth not found. Using untrained model.")
model.eval()

def run_task(task_name: str, config_class, grader_func):
    config = config_class()
    env = AmbulanceEnv(config.to_dict())
    
    # START LOG
    print(json.dumps({"type": "START", "task": task_name}))
    sys.stdout.flush()
    
    obs = env.reset(seed=config.seed)
    done = False
    step_count = 0
    
    try:
        while not done:
            # 1. Encode observation
            state = encoder.encode(obs)
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # 2. Identify currently valid actions
            valid_indices = mapper.get_action_space(obs)
            
            # 3. Model Inference (Greedy Selection)
            with torch.no_grad():
                q_values = model(state_tensor)
                
                # Mask invalid actions with negative infinity
                mask = torch.full_like(q_values, float('-inf'))
                mask[0, valid_indices] = 0
                masked_q = q_values + mask
                
                action_idx = int(masked_q.argmax().item())
                action = mapper.decode(action_idx)
            
            # 4. Environment Step
            obs, reward, done, info = env.step(action)
            
            # STEP LOG
            action_dict = action.model_dump() if hasattr(action, "model_dump") else action.dict()
            print(json.dumps({
                "type": "STEP", 
                "step": step_count, 
                "action": action_dict, 
                "reward": reward, 
                "done": done
            }))
            sys.stdout.flush()
            step_count += 1
            
        # END LOG
        score = grader_func(info["metrics"], info)
        clamped_score = float(max(0.0, min(1.0, score)))
            
        print(json.dumps({
            "type": "END", 
            "task": task_name, 
            "score": clamped_score
        }))
        sys.stdout.flush()
        
    except Exception as e:
        logging.critical(f"Task {task_name} crashed: {e}")
        print(json.dumps({
            "type": "END", 
            "task": task_name, 
            "score": 0.0,
            "error": str(e)
        }))
        sys.stdout.flush()

if __name__ == "__main__":
    # Sequentially run evaluation battery using the trained RL model
    run_task("easy", EasyConfig, grade_easy)
    run_task("medium", MediumConfig, grade_medium)
    run_task("hard", HardConfig, grade_hard)
