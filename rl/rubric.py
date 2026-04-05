"""
RFC 004 — Rubric-Based Reward System
=====================================
A Rubric is a named collection of reward components.  Each component computes
a sub-reward and exposes its last_score for training-loop introspection:

    for name, component in env.rubric.named_rubrics():
        print(f"{name}: {component.last_score}")

The Rubric integrates with openenv's Environment._apply_rubric() hook so it is
called automatically on every step.
"""

from __future__ import annotations
from typing import Any, Dict, Generator, Sequence, Tuple


class RubricComponent:
    """Single named reward component."""

    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight
        self.last_score: float = 0.0

    def compute(self, env_state: Dict[str, Any]) -> float:  # noqa: ARG002
        """Override in subclasses to compute the component reward.

        Args:
            env_state: Dict passed by the Rubric during scoring; contains
                       all relevant internal environment state.
        Returns:
            Unweighted reward for this component.
        """
        return 0.0

    def reset(self) -> None:
        self.last_score = 0.0


# ---------------------------------------------------------------------------
# Concrete ambulance-dispatch rubric components
# ---------------------------------------------------------------------------

class EmergencyServedRubric(RubricComponent):
    """Reward whenever an ambulance reaches an emergency scene (+10 per event)."""

    def __init__(self):
        super().__init__("emergency_served", weight=1.0)

    def compute(self, env_state: Dict[str, Any]) -> float:
        return float(env_state.get("served_this_step", 0)) * 10.0


class SeverityBonusRubric(RubricComponent):
    """Extra reward based on the severity of emergencies served this step."""

    def __init__(self):
        super().__init__("severity_bonus", weight=1.0)

    def compute(self, env_state: Dict[str, Any]) -> float:
        total = 0.0
        for sev in env_state.get("severities_served_this_step", []):
            if sev == "CRITICAL":
                total += 5.0
            elif sev == "HIGH":
                total += 2.0
            else:
                total += 0.5
        return total


class DispatchSpeedRubric(RubricComponent):
    """Partial reward for dispatching quickly (inverse of response-time lag)."""

    def __init__(self):
        super().__init__("dispatch_speed", weight=1.0)

    def compute(self, env_state: Dict[str, Any]) -> float:
        resp_times: list = env_state.get("response_times_this_step", [])
        if not resp_times:
            return 0.0
        avg = sum(resp_times) / len(resp_times)
        # Score decays as response time grows; 0 steps → 0.5, 10 steps → ~0.
        return max(0.0, 0.5 - avg * 0.05)


class HospitalDeliveryRubric(RubricComponent):
    """Reward per successful hospital delivery (+2 per delivery)."""

    def __init__(self):
        super().__init__("hospital_delivery", weight=1.0)

    def compute(self, env_state: Dict[str, Any]) -> float:
        return float(env_state.get("deliveries_this_step", 0)) * 2.0


class DistancePenaltyRubric(RubricComponent):
    """Small penalty per ambulance step of travel (-0.1 per en-route amb)."""

    def __init__(self):
        super().__init__("distance_penalty", weight=1.0)

    def compute(self, env_state: Dict[str, Any]) -> float:
        return -0.1 * float(env_state.get("en_route_count", 0))


class TrafficPenaltyRubric(RubricComponent):
    """Penalty when traffic multiplier is above baseline (-0.2 per step in rush)."""

    def __init__(self):
        super().__init__("traffic_penalty", weight=1.0)

    def compute(self, env_state: Dict[str, Any]) -> float:
        tm = env_state.get("traffic_multiplier", 1.0)
        return -0.2 if tm > 1.5 else 0.0


class IdlePenaltyRubric(RubricComponent):
    """Penalty for idle ambulances when emergencies are pending (-0.3 each)."""

    def __init__(self):
        super().__init__("idle_penalty", weight=1.0)

    def compute(self, env_state: Dict[str, Any]) -> float:
        pending = env_state.get("pending_emergencies", 0)
        if pending == 0:
            return 0.0
        return -0.3 * float(env_state.get("idle_ambulances", 0))


class CapacityViolationRubric(RubricComponent):
    """Penalty for routing to an overloaded hospital (-4 per violation)."""

    def __init__(self):
        super().__init__("capacity_violation", weight=1.0)

    def compute(self, env_state: Dict[str, Any]) -> float:
        return -4.0 * float(env_state.get("overflow_this_step", 0))


class TimeoutRubric(RubricComponent):
    """Heavy penalty per emergency that expires without service (-15 each)."""

    def __init__(self):
        super().__init__("timeout_penalty", weight=1.0)

    def compute(self, env_state: Dict[str, Any]) -> float:
        return -15.0 * float(env_state.get("missed_this_step", 0))


# ---------------------------------------------------------------------------
# Rubric container
# ---------------------------------------------------------------------------

class Rubric:
    """RFC 004 Rubric — aggregates reward components and exposes named scores.

    Usage in an openenv Environment subclass::

        self.rubric = AmbulanceRubric()

        # In step():
        env_state = self._build_rubric_state(...)
        observation.reward = self.rubric.score(env_state)

        # Training loop introspection:
        for name, component in self.rubric.named_rubrics():
            print(f"{name}: {component.last_score:.3f}")
    """

    def __init__(self, components: Sequence[RubricComponent]):
        self._components = list(components)

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def score(self, env_state: Dict[str, Any]) -> float:
        """Compute total reward from all components and cache per-component scores."""
        total = 0.0
        for comp in self._components:
            raw = comp.compute(env_state)
            comp.last_score = raw * comp.weight
            total += comp.last_score
        return total

    def __call__(self, action: Any, observation: Any) -> float:  # noqa: ARG002
        """openenv hook: called by Environment._apply_rubric(action, obs).

        The ambulance environment calls rubric.score(env_state) directly
        during step() because the required context exceeds (action, obs).
        This hook exists for framework compatibility; the env injects
        env_state via score() before _apply_rubric() is called and returns
        the already-computed last total.
        """
        return self._last_total

    def named_rubrics(self) -> Generator[Tuple[str, RubricComponent], None, None]:
        """Yield (name, component) pairs for training infrastructure."""
        for comp in self._components:
            yield comp.name, comp

    def reset(self) -> None:
        """Reset all component scores; call at episode start."""
        self._last_total = 0.0
        for comp in self._components:
            comp.reset()

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def scores_dict(self) -> Dict[str, float]:
        """Return {component_name: last_score} for logging."""
        return {comp.name: comp.last_score for comp in self._components}

    # Internal cache used by the __call__ hook.
    _last_total: float = 0.0


# ---------------------------------------------------------------------------
# Pre-built ambulance rubric
# ---------------------------------------------------------------------------

def make_ambulance_rubric() -> Rubric:
    """Factory that returns the standard ambulance-dispatch Rubric."""
    return Rubric(
        components=[
            EmergencyServedRubric(),
            SeverityBonusRubric(),
            DispatchSpeedRubric(),
            HospitalDeliveryRubric(),
            DistancePenaltyRubric(),
            TrafficPenaltyRubric(),
            IdlePenaltyRubric(),
            CapacityViolationRubric(),
            TimeoutRubric(),
        ]
    )
