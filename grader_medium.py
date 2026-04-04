from typing import Dict, Any

def grade_medium(metrics: Dict[str, Any], state_info: Dict[str, Any]) -> float:
    """
    Grader for the Medium level.
    Score = (Served %) + (Response Efficiency) - (Idle Penalty).
    """
    total_emg = max(metrics.get("total_emergencies", 0), 1)
    served_pct = metrics.get("served", 0) / total_emg
    
    # Efficiency: ratio of baseline time to actual response time
    baseline_time = 15.0
    avg_resp = max(metrics.get("avg_response_time", 60), baseline_time)
    efficiency = baseline_time / avg_resp
    
    # Idle Penalty: subtracted from the score (0.1 for every 10% idle)
    # Here we simplify idle as (1 - served/total) for the purpose of the grader
    idle_factor = max(0.0, 1.0 - (metrics.get("served", 0) / total_emg))
    
    score = (0.5 * served_pct) + (0.4 * efficiency) - (0.1 * idle_factor)
    return float(min(1.0, max(0.0, score)))
