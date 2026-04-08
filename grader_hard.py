import math
from typing import Dict, Any


def grade_hard(episode_info: Dict[str, Any]) -> float:
    """Grade the Hard task.

    Formula (per spec):
        critical_rate = critical_served / max(critical_total, 1)
        overall_rate = served / max(total_emergencies, 1)
        weighted_served = 0.7 * critical_rate + 0.3 * overall_rate
        priority_accuracy = priority_correct / max(priority_total, 1)
        fairness_score = 1 - normalised_std_dev of zone service rates
        overload_penalty = 0.05 * capacity_violations
        score = 0.50 * weighted_served + 0.30 * priority_accuracy
                + 0.15 * fairness_score - overload_penalty
        clamped to [0.0, 1.0]

    Args:
        episode_info: dict with required keys or 'metrics' sub-dict.
    Returns:
        Float in [0.0, 1.0].
    """
    # Try direct keys first (spec-compliant format)
    critical_total = episode_info.get("critical_total")
    critical_served = episode_info.get("critical_served")
    total_emergencies = episode_info.get("total_emergencies")
    served = episode_info.get("served")
    priority_correct = episode_info.get("priority_correct")
    priority_total = episode_info.get("priority_total")
    capacity_violations = episode_info.get("capacity_violations")
    fairness_zone_counts = episode_info.get("fairness_zone_counts")

    # Fallback to nested metrics dict
    if critical_total is None:
        metrics = episode_info.get("metrics", {})
        critical_served = int(metrics.get("critical_served", 0))
        critical_total = max(critical_served, int(metrics.get("critical_total", critical_served)))
        total_emergencies = int(metrics.get("total_emergencies", 0))
        served = int(metrics.get("served", 0))
        high_served = int(metrics.get("high_served", 0))
        priority_correct = int(metrics.get("priority_correct", critical_served + high_served))
        priority_total = max(int(metrics.get("priority_total", served)), 1)
        capacity_violations = int(metrics.get("hospital_overflow", 0))
        fairness_zone_counts = {
            "zone_served": metrics.get("zone_served", {}),
            "zone_total": metrics.get("zone_total", {}),
        }

    critical_total = max(int(critical_total or 0), 1)
    critical_served = int(critical_served or 0)
    total_emergencies = max(int(total_emergencies or 0), 1)
    served = int(served or 0)
    priority_correct = int(priority_correct or 0)
    priority_total = max(int(priority_total or 1), 1)
    capacity_violations = int(capacity_violations or 0)

    critical_rate = critical_served / critical_total
    overall_rate = served / total_emergencies
    weighted_served = 0.7 * critical_rate + 0.3 * overall_rate

    priority_accuracy = priority_correct / priority_total

    # Fairness: 1 - normalised std dev of zone service rates
    zone_served = {}
    zone_total = {}
    if isinstance(fairness_zone_counts, dict):
        zone_served = fairness_zone_counts.get("zone_served", {})
        zone_total = fairness_zone_counts.get("zone_total", {})

    if zone_total and len(zone_total) >= 2:
        rates = []
        for z in zone_total:
            z_total = max(int(zone_total.get(z, 0)), 1)
            z_served = int(zone_served.get(z, 0))
            rates.append(z_served / z_total)
        mean_rate = sum(rates) / len(rates)
        variance = sum((r - mean_rate) ** 2 for r in rates) / len(rates)
        std_dev = math.sqrt(variance)
        # Normalise: perfect equity = 1.0, max imbalance ~ 0.0
        fairness_score = max(0.0, 1.0 - 2.0 * std_dev)
    else:
        fairness_score = served / total_emergencies  # fallback: overall service rate

    overload_penalty = 0.05 * capacity_violations

    score = (
        0.50 * weighted_served
        + 0.30 * priority_accuracy
        + 0.15 * fairness_score
        - overload_penalty
    )
    return float(min(1.0, max(0.0, score)))

