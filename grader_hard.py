from typing import Dict, Any

def grade_hard(metrics: Dict[str, Any], state_info: Dict[str, Any]) -> float:
    """
    Grader for the Hard level.
    Score = (Weighted Critical Served) + (Priority Accuracy) - (Hospital Overflow Penalty).
    """
    total_served = max(metrics.get("served", 0), 1)
    
    # 1. Weighted Critical Served: emphasis on critical cases
    # We use a proportion normalized by an expected max
    critical_served = metrics.get("critical_served", 0)
    critical_score = min(1.0, (critical_served * 2.0) / 10.0)
    
    # 2. Priority Accuracy: ratio of high/critical cases to total cases served
    priority_accuracy = (critical_served + metrics.get("high_served", 0)) / total_served
    
    # 3. Hospital Overflow Penalty
    # We penalize each overflow event (normalized by max steps)
    overflow_penalty = min(0.5, metrics.get("hospital_overflow", 0) / 240.0)
    
    # Weighted average for final score in [0, 1]
    score = (0.4 * critical_score) + (0.5 * priority_accuracy) - (0.1 * overflow_penalty)
    return float(min(1.0, max(0.0, score)))
