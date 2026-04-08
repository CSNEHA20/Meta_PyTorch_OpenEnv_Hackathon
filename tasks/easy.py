from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class EasyConfig:
    n_ambulances: int = 2
    n_hospitals: int = 2
    max_steps: int = 30
    lambda_param: float = 0.3
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_ambulances": self.n_ambulances,
            "n_hospitals": self.n_hospitals,
            "max_steps": self.max_steps,
            "lambda_param": self.lambda_param,
            "seed": self.seed,
        }
