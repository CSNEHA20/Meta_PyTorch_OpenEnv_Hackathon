from typing import Dict, Any


def grade_medium(episode_info: Dict[str, Any]) -> float:
    """Grade the Medium task.

    Formula (per spec):
        served_percentage = served / max(total_emergencies, 1)
        response_score = max(0.0, 1.0 - avg_response_time / 15.0)
        idle_fraction = idle_steps / max(total_steps, 1)
        score = 0.50 * served_percentage + 0.35 * response_score - 0.15 * idle_fraction
        clamped to [0.0, 1.0]

    Args:
        episode_info: dict with required keys or 'metrics' sub-dict.
    Returns:
        Float in [0.0, 1.0].
    """
    # Try direct keys first (spec-compliant format)
    total_emg = episode_info.get("total_emergencies")
    served = episode_info.get("served")
    avg_response_time = episode_info.get("avg_response_time")
    idle_steps = episode_info.get("idle_steps")
    total_steps = episode_info.get("total_steps")

    # Fallback to nested metrics dict
    if total_emg is None:
        metrics = episode_info.get("metrics", {})
        total_emg = int(metrics.get("total_emergencies", 0))
        served = int(metrics.get("served", 0))
        avg_response_time = float(metrics.get("avg_response_time", 0.0))
        idle_steps = int(metrics.get("idle_steps", 0))
        total_steps = int(episode_info.get("step", max(total_emg, 1)))

    total_emg = max(int(total_emg), 1)
    served = int(served or 0)
    avg_response_time = float(avg_response_time or 0.0)
    idle_steps = int(idle_steps or 0)
    total_steps = max(int(total_steps or 1), 1)

    served_percentage = served / total_emg
    response_score = max(0.0, 1.0 - avg_response_time / 15.0)
    idle_fraction = idle_steps / total_steps

    score = 0.50 * served_percentage + 0.35 * response_score - 0.15 * idle_fraction
    return float(min(1.0, max(0.0, score)))
