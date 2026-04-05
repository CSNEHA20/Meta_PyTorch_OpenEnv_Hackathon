from typing import Dict, Any


def grade_hard(episode_info: Dict[str, Any]) -> float:
    """Score = 0.4*(critical rate) + 0.5*(priority accuracy) - 0.1*(overflow).

    Weights:
        critical_case_service_rate : 0.40
        priority_accuracy          : 0.50
        hospital_overflow_penalty  : 0.10

    Args:
        episode_info: The info dict returned by AmbulanceEnv.state().
    Returns:
        Float in [0.0, 1.0].
    """
    metrics = episode_info.get("metrics", {})
    total_served = max(metrics.get("served", 0), 1)

    critical_served = metrics.get("critical_served", 0)
    critical_score = min(1.0, (critical_served * 2.0) / max(total_served, 1))

    priority_accuracy = (
        (critical_served + metrics.get("high_served", 0)) / total_served
    )

    max_steps = episode_info.get("step", 240) or 240
    overflow_penalty = min(0.5, metrics.get("hospital_overflow", 0) / max(max_steps, 1))

    score = (0.4 * critical_score) + (0.5 * priority_accuracy) - (0.1 * overflow_penalty)
    return float(min(1.0, max(0.0, score)))

