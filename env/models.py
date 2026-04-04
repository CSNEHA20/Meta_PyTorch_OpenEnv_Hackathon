from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

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
    assigned: bool

class HospitalInfo(BaseModel):
    id: int
    node: int
    capacity: int
    current_patients: int

class ObservationModel(BaseModel):
    ambulances: List[AmbulanceInfo]
    emergencies: List[EmergencyInfo]
    hospitals: List[HospitalInfo]
    traffic: Dict[str, float]
    step: int

class ActionModel(BaseModel):
    ambulance_id: Optional[int] = None
    emergency_id: str = ""
    hospital_id: Optional[int] = None
    reposition_node: Optional[int] = None

class RewardModel(BaseModel):
    value: float
    components: Dict[str, float]
    emergencies_served: int
    emergencies_missed: int
