from typing import Dict, Any, List


def grade_easy(episode_info: Dict[str, Any]) -> float:
    """Grade the Easy task.

    Formula (per spec):
        For each served emergency: ratio = optimal_time / actual_response_time
        score = mean(ratios), clamped to [0.0, 1.0]

    Args:
        episode_info: dict with 'response_times' and 'optimal_times' lists.
            Falls back to 'metrics' sub-dict for backwards compatibility.
    Returns:
        Float in [0.0, 1.0].
    """
    # Extract directly or from nested metrics
    response_times: List[float] = episode_info.get("response_times") or []
    optimal_times: List[float] = episode_info.get("optimal_times") or []

    # Fallback: derive from metrics sub-dict
    if not response_times:
        metrics = episode_info.get("metrics", {})
        served = int(metrics.get("served", 0))
        if served == 0:
            return 0.0
        avg_rt = float(metrics.get("avg_response_time", 0.0))
        total = max(int(metrics.get("total_emergencies", 0)), 1)
        served_fraction = served / total
        if avg_rt <= 0:
            return float(min(1.0, served_fraction))
        optimal_time = 5.0
        response_eff = min(1.0, optimal_time / avg_rt)
        return float(min(1.0, max(0.0, 0.5 * served_fraction + 0.5 * response_eff)))

    # Spec formula: mean of (optimal / actual) ratios
    if len(response_times) == 0:
        return 0.0

    ratios = []
    for actual, optimal in zip(response_times, optimal_times):
        if actual <= 0:
            ratios.append(1.0)  # zero actual time = instant response = perfect
        else:
            ratios.append(min(1.0, optimal / actual))

    score = sum(ratios) / len(ratios)
    return float(min(1.0, max(0.0, score)))
