import uuid
import numpy as np
import random
import asyncio
from typing import Tuple, Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor

from env.models import (
    ObservationModel, ActionModel, RewardModel,
    AmbulanceState, Severity, AmbulanceInfo, EmergencyInfo, HospitalInfo,
    AmbulanceEnvState, Rubric
)
from env.simulator import CityGraph, TrafficEngine, AmbulanceFleet, EmergencyGenerator, Hospital

# Thread pool for CPU-bound Dijkstra computations
_executor = ThreadPoolExecutor(max_workers=4)

class AmbulanceEnvironment:
    """
    Production-grade Ambulance Dispatch Environment following OpenEnv RFC standards.
    Supports concurrent sessions, async I/O, and named reward rubrics.
    """
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.n_ambulances = self.config.get("n_ambulances", 5)
        self.n_hospitals = self.config.get("n_hospitals", 3)
        self.max_steps = self.config.get("max_steps", 1440)
        self.graph_size = self.config.get("graph_size", 100)
        
        self.seed_val = self.config.get("seed", 42)
        self.reset(seed=self.seed_val)

    def reset(self, seed: Optional[int] = None) -> ObservationModel:
        if seed is not None:
            self.seed_val = seed

        self.rng = np.random.default_rng(self.seed_val)
        random.seed(self.seed_val)
        np.random.seed(self.seed_val)

        self.episode_id = str(uuid.uuid4())
        self.city_graph = CityGraph(n=self.graph_size)
        self.nodes = list(self.city_graph.graph.nodes())
        self.fleet = AmbulanceFleet(n=self.n_ambulances, nodes=self.nodes)
        self.generator = EmergencyGenerator(nodes=self.nodes, lambda_param=self.config.get("lambda_param", 0.15))
        self.traffic = TrafficEngine()
        
        # Specialties for Rule #8
        specs = ["Trauma", "Cardiac", "General", "Paediatric"]
        self.hospitals = {}
        for i in range(self.n_hospitals):
            node = int(self.rng.choice(self.nodes))
            spec = specs[i % len(specs)]
            self.hospitals[i] = Hospital(hosp_id=i, node=node, capacity=8)
            self.hospitals[i].specialty = spec

        self.step_count = 0
        self.active_emergencies: List[EmergencyInfo] = []
        self.last_rubric = Rubric()

        self.metrics = {
            "served": 0, "missed": 0, "total_emergencies": 0,
            "critical_served": 0, "high_served": 0, "normal_served": 0,
            "avg_response_time": 0.0, "hospital_overflow": 0
        }
        self._dispatch_times = {}
        self._response_times = []

        return self._get_observation()

    async def reset_async(self, seed: Optional[int] = None, **kwargs) -> ObservationModel:
        return self.reset(seed=seed)

    @property
    def state(self) -> AmbulanceEnvState:
        tm = self.traffic.get_multiplier(self.step_count)
        return AmbulanceEnvState(
            episode_id=self.episode_id,
            step_count=self.step_count,
            metrics=self.metrics,
            ambulances=[a.to_info().model_dump() for a in self.fleet.ambulances],
            hospitals=[h.to_info().model_dump() for h in self.hospitals.values()],
            emergencies=[e.model_dump() for e in self.active_emergencies],
            traffic_multiplier=tm,
            rubric=self.last_rubric
        )

    async def step_async(self, action: ActionModel, **kwargs) -> ObservationModel:
        return self.step(action)

    def step(self, action: ActionModel) -> ObservationModel:
        self.step_count += 1
        rubric = Rubric()
        tm = self.traffic.get_multiplier(self.step_count)

        # 1. Process Dispatch Action
        if action.ambulance_id is not None and not action.is_noop:
            amb = next((a for a in self.fleet.ambulances if a.id == action.ambulance_id), None)
            emg = next((e for e in self.active_emergencies if e.id == action.emergency_id and not e.assigned), None)
            hosp = self.hospitals.get(action.hospital_id)

            if amb and emg and hosp and amb.state == AmbulanceState.IDLE:
                if hosp.is_available():
                    self.fleet.dispatch(amb.id, emg.id, emg.node, hosp.id, hosp.node)
                    emg.assigned = True
                    self._dispatch_times[emg.id] = self.step_count
                    
                    # RFC Feature: Dispatch Speed Rubric
                    wait_time = self.step_count - emg.spawn_time
                    rubric.dispatch_speed = max(0, 10.0 - wait_time * 0.5)
                else:
                    rubric.capacity_violation -= 5.0
                    self.metrics["hospital_overflow"] += 1
            else:
                rubric.idle_penalty -= 2.0

        # 2. Simulation Physics
        pre_states = {a.id: a.state for a in self.fleet.ambulances}
        self.fleet.step_update(self.city_graph, tm)
        post_states = {a.id: a.state for a in self.fleet.ambulances}

        # 3. Emergency Dynamics
        new_emgs = self.generator.generate(self.step_count)
        # Convert internal emgs to EmergencyInfo
        for ne in new_emgs:
            ne.spawn_time = self.step_count
        self.active_emergencies.extend(new_emgs)
        self.metrics["total_emergencies"] += len(new_emgs)

        missed = 0
        rem = []
        for e in self.active_emergencies:
            if not e.assigned:
                e.time_remaining -= 1
                if e.time_remaining <= 0:
                    missed += 1
                    continue
            rem.append(e)
        self.active_emergencies = rem
        rubric.timeout_penalty -= missed * 15.0
        self.metrics["missed"] += missed

        # 4. Success Event Processing
        for amb in self.fleet.ambulances:
            if pre_states[amb.id] == AmbulanceState.EN_ROUTE and post_states[amb.id] == AmbulanceState.AT_SCENE:
                rubric.emergency_served += 20.0
                self.metrics["served"] += 1
                
                emg = next((e for e in self.active_emergencies if e.id == amb.target_emg_id), None)
                if emg:
                    if emg.severity == Severity.CRITICAL:
                        rubric.severity_bonus += 30.0
                        self.metrics["critical_served"] += 1
                    elif emg.severity == Severity.HIGH:
                        rubric.severity_bonus += 10.0
                        self.metrics["high_served"] += 1
                    else:
                        self.metrics["normal_served"] += 1
                    self.active_emergencies = [e for e in self.active_emergencies if e.id != amb.target_emg_id]

            if pre_states[amb.id] == AmbulanceState.TRANSPORTING and post_states[amb.id] == AmbulanceState.RETURNING:
                rubric.hospital_delivery += 10.0
                hosp = self.hospitals.get(amb.target_hosp_id)
                if hosp: hosp.admit()

        # 5. Penalties & Discharge
        idle_count = len([a for a in self.fleet.ambulances if a.state == AmbulanceState.IDLE])
        if self.active_emergencies and idle_count > 0:
            rubric.idle_penalty -= idle_count * 1.0

        if self.step_count % 10 == 0:
            for h in self.hospitals.values(): h.release()

        self.last_rubric = rubric
        done = self.step_count >= self.max_steps
        obs = self._get_observation(tm)
        obs.reward = rubric.total()
        obs.done = done
        obs.rubric = rubric
        return obs

    def _get_observation(self, tm: Optional[float] = None) -> ObservationModel:
        if tm is None: tm = self.traffic.get_multiplier(self.step_count)
        return ObservationModel(
            ambulances=[amb.to_info() for amb in self.fleet.ambulances],
            emergencies=[e for e in self.active_emergencies if not e.assigned],
            hospitals=[h.to_info() for h in self.hospitals.values()],
            traffic={"global": tm},
            step=self.step_count
        )
