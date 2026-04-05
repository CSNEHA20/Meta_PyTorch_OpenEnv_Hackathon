from typing import Dict, Any


def grade_medium(episode_info: Dict[str, Any]) -> float:
    """Score = 0.5*(served%) + 0.4*(response efficiency) - 0.1*(idle factor).

    Args:
        episode_info: The info dict returned by AmbulanceEnv.state().
    Returns:
        Float in [0.0, 1.0].
    """
    metrics = episode_info.get("metrics", {})
    total_emg = max(metrics.get("total_emergencies", 0), 1)
    served_pct = metrics.get("served", 0) / total_emg

    baseline_time = 15.0
    avg_resp = max(metrics.get("avg_response_time", 60), baseline_time)
    efficiency = baseline_time / avg_resp

    idle_factor = metrics.get("idle_fraction", 0.0)
    score = (0.5 * served_pct) + (0.4 * efficiency) - (0.1 * idle_factor)
    return float(min(1.0, max(0.0, score)))
