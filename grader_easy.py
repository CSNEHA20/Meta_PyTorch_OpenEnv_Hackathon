from typing import Dict, Any


def grade_easy(episode_info: Dict[str, Any]) -> float:
    """Score = normalized response efficiency vs optimal.

    Args:
        episode_info: The info dict returned by AmbulanceEnv.state().
                      Must contain a 'metrics' sub-dict.
    Returns:
        Float in [0.0, 1.0].
    """
    metrics = episode_info.get("metrics", {})
    response_time = metrics.get("avg_response_time", 60)
    optimal_time = max(3.0, response_time * 0.7)

    if metrics.get("served", 0) == 0:
        return 0.0
    efficiency = optimal_time / max(response_time, optimal_time)
    return float(min(1.0, max(0.0, efficiency)))
