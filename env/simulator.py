import networkx as nx
import numpy as np
import uuid
from typing import List, Dict, Optional, Tuple
from env.models import AmbulanceState, Severity, AmbulanceInfo, EmergencyInfo, HospitalInfo

class CityGraph:
    """Graph representation of the city with time-dependent travel costs."""
    def __init__(self, n: int = 100, m: int = 2):
        # Ensure m < n for Barabasi-Albert model
        m = max(1, min(m, n - 1)) if n > 1 else 0
        if n > 1:
            self.graph = nx.barabasi_albert_graph(n, m, seed=42)
        else:
            self.graph = nx.Graph()
            self.graph.add_node(0)
            
        # Assign base weights representing minutes under perfect conditions
        if self.graph.edges():
            for u, v in self.graph.edges():
                self.graph[u][v]['base_weight'] = float(np.random.randint(2, 9))
        
        # Optimize: Precompute all-pairs shortest paths once at initialization
        self._all_pairs_len = dict(nx.all_pairs_dijkstra_path_length(self.graph, weight='base_weight'))

    def shortest_path_time(self, source: int, target: int, traffic_multiplier: float) -> int:
        """Calculate shortest travel time between two nodes under current traffic."""
        if source == target:
            return 0
        
        # Optimized: O(1) dictionary lookup replacing O(E log V) Dijkstra calculation
        base_length = self._all_pairs_len.get(source, {}).get(target)
        if base_length is None:
            return 9999
        return max(1, int(np.ceil(base_length * traffic_multiplier)))

class TrafficEngine:
    """Simulates dynamic traffic with rush hours and deterministic incidents."""

    def __init__(self, rng: np.random.Generator = None):
        self._rng = rng if rng is not None else np.random.default_rng(42)
        # Incident events: edge -> steps_remaining_blocked
        self._incidents: Dict[tuple, int] = {}
        self._incident_prob = 0.02  # 2% chance per step of a new incident

    def set_incident_prob(self, prob: float):
        self._incident_prob = prob

    def maybe_spawn_incident(self, graph_edges: list, step: int):
        """Randomly block an edge for 5 steps (Hard task dynamic incidents)."""
        if self._rng.random() < self._incident_prob and graph_edges:
            edge = graph_edges[int(self._rng.integers(0, len(graph_edges)))]
            self._incidents[edge] = 5  # block for 5 steps

    def tick_incidents(self):
        """Decrement incident timers; remove expired ones."""
        expired = [k for k, v in self._incidents.items() if v <= 1]
        for k in expired:
            del self._incidents[k]
        for k in self._incidents:
            self._incidents[k] -= 1

    def get_multiplier(self, step: int) -> float:
        hour = (step % 1440) // 60
        if (7 <= hour < 9) or (17 <= hour < 20):
            base = float(np.clip(self._rng.normal(1.6, 0.2), 1.2, 2.5))
        else:
            base = float(np.clip(self._rng.normal(1.0, 0.05), 0.9, 1.2))
        # Incidents raise multiplier to 10.0 on blocked edges (averaged globally as +0.5)
        if self._incidents:
            base += 0.5 * min(len(self._incidents), 2)
        return float(np.clip(base, 0.9, 10.0))

class Ambulance:
    """Ambulance unit with transition logic (Feature 6)."""
    def __init__(self, amb_id: int, start_node: int, rng: np.random.Generator = None):
        self.id = amb_id
        self.node = start_node
        self.state = AmbulanceState.IDLE
        self.eta = 0
        self._rng = rng if rng is not None else np.random.default_rng(amb_id)
        self.target_emg_id: Optional[str] = None
        self.target_emg_node: Optional[int] = None
        self.target_hosp_id: Optional[int] = None
        self.target_hosp_node: Optional[int] = None

    def update(self, city: CityGraph, tm: float):
        if self.eta > 0:
            self.eta -= 1
            return

        if self.state == AmbulanceState.DISPATCHED:
            self.state = AmbulanceState.EN_ROUTE
            if self.target_emg_node is not None:
                self.eta = city.shortest_path_time(self.node, self.target_emg_node, tm)

        elif self.state == AmbulanceState.EN_ROUTE:
            self.state = AmbulanceState.AT_SCENE
            self.node = self.target_emg_node if self.target_emg_node is not None else self.node
            self.eta = int(self._rng.integers(2, 5))  # Load time 2-4 steps

        elif self.state == AmbulanceState.AT_SCENE:
            self.state = AmbulanceState.TRANSPORTING
            if self.target_hosp_node is not None:
                self.eta = city.shortest_path_time(self.node, self.target_hosp_node, tm)

        elif self.state == AmbulanceState.TRANSPORTING:
            self.state = AmbulanceState.RETURNING
            self.node = self.target_hosp_node if self.target_hosp_node is not None else self.node
            self.target_emg_id = None
            self.target_emg_node = None
            self.eta = int(self._rng.integers(2, 4))  # Handover delay

        elif self.state == AmbulanceState.RETURNING:
            self.state = AmbulanceState.IDLE

        elif self.state == AmbulanceState.REPOSITIONING:
            self.state = AmbulanceState.IDLE
            if self.target_emg_node is not None:
                self.node = self.target_emg_node

    def to_info(self) -> AmbulanceInfo:
        return AmbulanceInfo(
            id=self.id,
            node=self.node,
            state=self.state,
            eta=self.eta,
            target_emg_id=self.target_emg_id,
            target_hosp_id=self.target_hosp_id
        )

class AmbulanceFleet:
    """Manages all ambulance units as a fleet."""
    def __init__(self, n: int, nodes: List[int], rng: np.random.Generator = None):
        self._rng = rng if rng is not None else np.random.default_rng(42)
        self.ambulances = [
            Ambulance(i, int(self._rng.choice(nodes)) if nodes else 0)
            for i in range(n)
        ]

    def get_idle(self) -> List[Ambulance]:
        return [a for a in self.ambulances if a.state == AmbulanceState.IDLE]

    def dispatch(self, amb_id: int, emg_id: str, emg_node: int, hosp_id: int, hosp_node: int):
        amb = next((a for a in self.ambulances if a.id == amb_id), None)
        if amb and amb.state == AmbulanceState.IDLE:
            amb.state = AmbulanceState.DISPATCHED
            amb.target_emg_id = emg_id
            amb.target_emg_node = emg_node
            amb.target_hosp_id = hosp_id
            amb.target_hosp_node = hosp_node

    def reposition(self, amb_id: int, node: int, city_graph: CityGraph, traffic_multiplier: float):
        amb = next((a for a in self.ambulances if a.id == amb_id), None)
        if amb and amb.state == AmbulanceState.IDLE:
            amb.state = AmbulanceState.REPOSITIONING
            amb.target_emg_node = node
            amb.target_emg_id = None
            amb.eta = city_graph.shortest_path_time(amb.node, node, traffic_multiplier)

    def step_update(self, city_graph: CityGraph, traffic_multiplier: float):
        for amb in self.ambulances:
            amb.update(city_graph, traffic_multiplier)

import random

class EmergencyGenerator:
    """Generates new emergencies following a Poisson process."""
    def __init__(self, nodes: List[int], lambda_param: float = 0.05, rng: np.random.Generator = None):
        self.nodes = nodes
        self.lambda_param = lambda_param
        self._rng = rng if rng is not None else np.random.default_rng(42)
        self.config = {
            Severity.CRITICAL: {"prob": 0.25, "timeout": 20},
            Severity.HIGH: {"prob": 0.35, "timeout": 45},
            Severity.NORMAL: {"prob": 0.40, "timeout": 90}
        }

    def generate(self, step: int) -> List[EmergencyInfo]:
        if not self.nodes:
            return []
        num_new = int(self._rng.poisson(self.lambda_param))
        new_emergencies = []
        severities = list(self.config.keys())
        probs = [self.config[s]["prob"] for s in severities]
        for _ in range(num_new):
            r = self._rng.random()
            cumulative = 0.0
            sev = severities[-1]
            for sv, p in zip(severities, probs):
                cumulative += p
                if r <= cumulative:
                    sev = sv
                    break
            timeout = self.config[sev]["timeout"]
            new_emergencies.append(EmergencyInfo(
                id=str(uuid.uuid4())[:8],
                node=int(self._rng.choice(self.nodes)),
                severity=sev,
                time_remaining=timeout,
                max_time_remaining=timeout,
                assigned=False
            ))
        return new_emergencies

class Hospital:
    """Hospital facility with limited patient capacity and specialty routing (Feature 8)."""
    def __init__(self, hosp_id: int, node: int, capacity: int):
        self.id = hosp_id
        self.node = node
        self.capacity = capacity
        self.current_patients = 0
        self.specialty = "General"

    def is_available(self) -> bool:
        return self.current_patients < self.capacity

    def admit(self):
        if self.is_available():
            self.current_patients += 1
            return True
        return False

    def release(self):
        if self.current_patients > 0:
            self.current_patients -= 1

    def to_info(self) -> HospitalInfo:
        return HospitalInfo(
            id=self.id,
            node=self.node,
            capacity=self.capacity,
            current_patients=self.current_patients,
            specialty=self.specialty
        )
