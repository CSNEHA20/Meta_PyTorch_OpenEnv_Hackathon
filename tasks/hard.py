from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class HardConfig:
    n_ambulances: int = 5
    n_hospitals: int = 3
    max_steps: int = 240
    lambda_param: float = 0.2
    traffic_range: tuple = (1.3, 2.5)
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_ambulances": self.n_ambulances,
            "n_hospitals": self.n_hospitals,
            "max_steps": self.max_steps,
            "lambda_param": self.lambda_param,
            "seed": self.seed
        }
