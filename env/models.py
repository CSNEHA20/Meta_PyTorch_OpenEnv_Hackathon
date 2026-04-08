from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from openenv.core.env_server import Observation, Action, State


class Rubric(BaseModel):
    """Named reward components for fine-grained diagnostic logging."""
    emergency_served: float = 0.0
    severity_bonus: float = 0.0
    dispatch_speed: float = 0.0
    hospital_delivery: float = 0.0
    distance_penalty: float = 0.0
    traffic_penalty: float = 0.0
    idle_penalty: float = 0.0
    capacity_violation: float = 0.0
    timeout_penalty: float = 0.0
    fairness_score: float = 0.0

    def total(self) -> float:
        return sum(self.model_dump().values())


class AmbulanceState(str, Enum):
    IDLE = "idle"
    DISPATCHED = "dispatched"
    EN_ROUTE = "en_route"
    AT_SCENE = "at_scene"
    TRANSPORTING = "transporting"
    RETURNING = "returning"
    REPOSITIONING = "repositioning"


class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    NORMAL = "NORMAL"


class AmbulanceInfo(BaseModel):
    id: int
    node: int
    state: AmbulanceState
    eta: int
    target_emg_id: Optional[str] = None
    target_hosp_id: Optional[int] = None


class EmergencyInfo(BaseModel):
    id: str
    node: int
    severity: Severity
    time_remaining: int
    max_time_remaining: int = 90  # set at spawn time for frontend progress bars
    assigned: bool
    spawn_time: int = 0


class HospitalInfo(BaseModel):
    id: int
    node: int
    capacity: int
    current_patients: int
    specialty: str = "General"


# --- OpenEnv base-type subclasses ---

class ObservationModel(Observation):
    """Full environment observation returned from reset() and step()."""

    ambulances: List[AmbulanceInfo] = Field(default_factory=list)
    emergencies: List[EmergencyInfo] = Field(default_factory=list)
    hospitals: List[HospitalInfo] = Field(default_factory=list)
    traffic: Dict[str, float] = Field(default_factory=dict)
    step: int = 0
    reward: float = 0.0
    reward_model: Optional["RewardModel"] = None  # typed reward — populated each step()
    done: bool = False
    rubric: Optional[Rubric] = None


class ActionModel(Action):
    """Dispatch action sent by the agent each step.

    extra='forbid' is inherited from Action base — any unknown field raises
    a Pydantic validation error, catching typos in inference.py immediately.
    """

    ambulance_id: Optional[int] = None
    emergency_id: str = ""
    hospital_id: Optional[int] = None
    reposition_node: Optional[int] = None
    is_noop: bool = False


class AmbulanceEnvState(State):
    """Extended env state exposed via the state property."""

    episode_id: str
    step_count: int
    metrics: Dict[str, Any] = Field(default_factory=dict)
    ambulances: List[Dict[str, Any]] = Field(default_factory=list)
    hospitals: List[Dict[str, Any]] = Field(default_factory=list)
    emergencies: List[Dict[str, Any]] = Field(default_factory=list)
    traffic_multiplier: float = 1.0
    rubric: Optional[Rubric] = None


class RewardModel(BaseModel):
    """Typed reward returned per step — exposes scalar value and all named components."""
    value: float
    components: Dict[str, float]
    emergencies_served: int
    emergencies_missed: int

    @classmethod
    def from_rubric(cls, rubric: "Rubric", served: int, missed: int) -> "RewardModel":
        return cls(
            value=rubric.total(),
            components=rubric.model_dump(),
            emergencies_served=served,
            emergencies_missed=missed,
        )
