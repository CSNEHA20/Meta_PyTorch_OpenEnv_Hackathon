"""
AmbulanceEnvironment — proper openenv.core.env_server.Environment subclass.

This wraps AmbulanceEnv (the standalone RL training env) and exposes the
exact interface required by the OpenEnv framework:
  - reset() returns ObservationModel (reward/done embedded)
  - step() returns ObservationModel (NOT a tuple)
  - state is a @property returning AmbulanceEnvState
  - get_metadata() returns rich EnvironmentMetadata
  - SUPPORTS_CONCURRENT_SESSIONS = True (session-isolated state)
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional, Any

from openenv.core.env_server import Environment
from openenv.core.env_server.types import EnvironmentMetadata

from env.environment import AmbulanceEnvironment as AmbulanceEnv
from env.models import ObservationModel, ActionModel, AmbulanceEnvState
from rl.rubric import make_ambulance_rubric

# README content loaded once at import time for EnvironmentMetadata
_README_PATH = Path(__file__).parent.parent / "README.md"
_README_CONTENT: str = _README_PATH.read_text(encoding="utf-8") if _README_PATH.exists() else ""


class AmbulanceEnvironment(Environment[ActionModel, ObservationModel, AmbulanceEnvState]):
    """
    Production OpenEnv environment for city-scale ambulance dispatch optimisation.

    Each WebSocket session gets its own AmbulanceEnvironment instance, providing
    complete state isolation between concurrent training runs.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, config: dict | None = None):
        rubric = make_ambulance_rubric()
        super().__init__(rubric=rubric)
        self._inner = AmbulanceEnv(config or {})
        self._episode_id: str = str(uuid.uuid4())
        self._last_info: dict = {}

    # ------------------------------------------------------------------
    # Required overrides
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> ObservationModel:
        self._episode_id = episode_id or str(uuid.uuid4())
        self._last_info = {}
        self.rubric.reset()  # type: ignore[union-attr]
        obs_model = self._inner.reset(seed=seed)
        return self._wrap_obs(obs_model, reward=0.0, done=False)

    def step(
        self,
        action: ActionModel,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> ObservationModel:
        # _inner.step() returns a single ObservationModel (reward/done embedded)
        obs_model = self._inner.step(action)
        raw_reward = obs_model.reward
        done = obs_model.done

        # Build rubric env_state for RFC 004 named reward introspection.
        rubric_state = _extract_rubric_state(obs_model, self._inner.metrics)
        rubric_reward = self.rubric.score(rubric_state)  # type: ignore[union-attr]
        self.rubric._last_total = rubric_reward           # type: ignore[union-attr]

        # Use rubric reward (richer signal) but fall back to raw if rubric is zero.
        reward = rubric_reward if rubric_reward != 0.0 else raw_reward
        obs_model.reward = reward

        self._last_info = {"metrics": self._inner.metrics}
        return self._wrap_obs(obs_model, reward=reward, done=done)

    @property
    def state(self) -> AmbulanceEnvState:
        # _inner.state is a @property — access it as a property, not as a method
        inner = self._inner.state
        return AmbulanceEnvState(
            episode_id=self._episode_id,
            step_count=self._inner.step_count,
            metrics=inner.metrics,
            ambulances=inner.ambulances,
            hospitals=inner.hospitals,
            emergencies=inner.emergencies,
            traffic_multiplier=inner.traffic_multiplier,
            rubric=inner.rubric,
        )

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="ambulance-dispatch",
            description=(
                "Multi-constraint reinforcement learning environment for city-scale "
                "ambulance dispatch optimisation. Agents manage a fleet of ambulances "
                "across a Barabasi-Albert road network, routing to emergencies of three "
                "severity tiers while respecting hospital capacity limits and dynamic "
                "rush-hour traffic. Reward is computed via an RFC 004 Rubric with nine "
                "named components enabling per-component introspection by training infrastructure."
            ),
            readme_content=_README_CONTENT,
            version="1.0.0",
            author="Ambulance-OpenENV Team",
            documentation_url="https://github.com/your-org/ambulance-openenv",
        )

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _wrap_obs(
        self,
        obs_model: ObservationModel,
        reward: float,
        done: bool,
    ) -> ObservationModel:
        """Embed reward and done into the observation (OpenEnv style)."""
        obs_model.reward = reward
        obs_model.done = done
        obs_model.metadata = {
            "episode_id": self._episode_id,
            "step": obs_model.step,
            "metrics": self._last_info.get("metrics", {}),
        }
        return obs_model


# ---------------------------------------------------------------------------
# Rubric state extractor
# ---------------------------------------------------------------------------

def _extract_rubric_state(obs: ObservationModel, metrics: dict) -> dict:
    """Build the env_state dict consumed by each RubricComponent.compute().

    We extract per-step event signals directly from the observation's embedded
    rubric (set by AmbulanceEnvironment.step), giving the RFC 004 rubric real
    signal rather than all-zero placeholders.
    """
    rubric = obs.rubric
    # Per-step events derived from the inline Pydantic rubric on the observation
    served_this_step = 1 if (rubric and rubric.emergency_served > 0) else 0
    severities: list = []
    if rubric and rubric.severity_bonus >= 30:
        severities = ["CRITICAL"]
    elif rubric and rubric.severity_bonus >= 10:
        severities = ["HIGH"]
    elif rubric and rubric.emergency_served > 0:
        severities = ["NORMAL"]

    deliveries = 1 if (rubric and rubric.hospital_delivery > 0) else 0
    overflow = 1 if (rubric and rubric.capacity_violation < -4) else 0
    missed = 1 if (rubric and rubric.timeout_penalty < 0) else 0

    return {
        "served_this_step": served_this_step,
        "severities_served_this_step": severities,
        "response_times_this_step": [metrics.get("avg_response_time", 10.0)] if served_this_step else [],
        "deliveries_this_step": deliveries,
        "overflow_this_step": overflow,
        "missed_this_step": missed,
        # Instantaneous state
        "en_route_count": sum(1 for a in obs.ambulances if a.state == "en_route"),
        "idle_ambulances": sum(1 for a in obs.ambulances if a.state == "idle"),
        "pending_emergencies": len([e for e in obs.emergencies if not e.assigned]),
        "traffic_multiplier": obs.traffic.get("global", 1.0),
        # Cumulative
        "metrics": metrics,
    }
