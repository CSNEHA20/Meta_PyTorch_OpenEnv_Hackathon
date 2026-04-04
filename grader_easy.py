from typing import Dict, Any
from env.models import ObservationModel

def grade_easy(metrics: Dict[str, Any], trajectory: list) -> float:
    """
    Grader for the Easy level.
    Score = normalized response efficiency vs optimal.
    """
    # Assuming one emergency always exists or is spawned.
    response_time = metrics.get("avg_response_time", 60)
    optimal_time = max(3.0, response_time * 0.7)
    
    if metrics["served"] == 0:
        return 0.0
        
    efficiency = optimal_time / max(response_time, optimal_time)
    return float(min(1.0, max(0.0, efficiency)))
