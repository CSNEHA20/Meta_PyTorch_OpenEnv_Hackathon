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
    """Simulates real-world traffic patterns based on the time of day."""
    def get_multiplier(self, step: int) -> float:
        minute_of_day = step % 1440
        hour = minute_of_day // 60
        
        # Rush hours: 7-9, 17-20
        if (7 <= hour < 9) or (17 <= hour < 20):
            return np.random.uniform(1.3, 2.5)
        return np.random.uniform(0.9, 1.1)

class Ambulance:
    """Individual ambulance unit with its own state machine."""
    def __init__(self, amb_id: int, start_node: int):
        self.id = amb_id
        self.node = start_node
        self.state = AmbulanceState.IDLE
        self.eta = 0
        self.target_emg_id: Optional[str] = None
        self.target_emg_node: Optional[int] = None
        self.target_hosp_id: Optional[int] = None
        self.target_hosp_node: Optional[int] = None

    def update(self, city_graph: CityGraph, traffic_multiplier: float):
        if self.eta > 0:
            self.eta -= 1
            return

        if self.state == AmbulanceState.DISPATCHED:
            self.state = AmbulanceState.EN_ROUTE
            if self.target_emg_node is not None:
                self.eta = city_graph.shortest_path_time(self.node, self.target_emg_node, traffic_multiplier)
        
        elif self.state == AmbulanceState.EN_ROUTE:
            # Arrived at scene
            self.state = AmbulanceState.AT_SCENE
            self.node = self.target_emg_node
            self.eta = np.random.randint(5, 12) # Time at scene
        
        elif self.state == AmbulanceState.AT_SCENE:
            # Moving to hospital
            self.state = AmbulanceState.TRANSPORTING
            if self.target_hosp_node is not None:
                self.eta = city_graph.shortest_path_time(self.node, self.target_hosp_node, traffic_multiplier)
        
        elif self.state == AmbulanceState.TRANSPORTING:
            # Arrived at hospital
            self.state = AmbulanceState.RETURNING
            self.node = self.target_hosp_node
            self.target_emg_id = None
            self.target_emg_node = None
            self.eta = np.random.randint(5, 10) # Handover time + base return time
        
        elif self.state == AmbulanceState.RETURNING:
            self.state = AmbulanceState.IDLE
        
        elif self.state == AmbulanceState.REPOSITIONING:
            self.state = AmbulanceState.IDLE
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
    def __init__(self, n: int, nodes: List[int]):
        # Safety check for empty node list
        self.ambulances = [
            Ambulance(i, int(np.random.choice(nodes)) if nodes else 0) 
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
    def __init__(self, nodes: List[int], lambda_param: float = 0.05):
        self.nodes = nodes
        self.lambda_param = lambda_param
        self.config = {
            Severity.CRITICAL: {"prob": 0.25, "timeout": 20},
            Severity.HIGH: {"prob": 0.35, "timeout": 45},
            Severity.NORMAL: {"prob": 0.40, "timeout": 90}
        }

    def generate(self, step: int) -> List[EmergencyInfo]:
        if not self.nodes:
            return []
            
        num_new = np.random.poisson(self.lambda_param)
        new_emergencies = []
        
        # Extract keys and probabilities for random.choices
        severities = list(self.config.keys())
        probs = [self.config[s]["prob"] for s in severities]
        
        for _ in range(num_new):
            sev = random.choices(severities, weights=probs, k=1)[0]
            new_emergencies.append(EmergencyInfo(
                id=str(uuid.uuid4())[:8],
                node=int(np.random.choice(self.nodes)),
                severity=sev,
                time_remaining=self.config[sev]["timeout"],
                assigned=False
            ))
        return new_emergencies

class Hospital:
    """Hospital facility with limited patient capacity."""
    def __init__(self, hosp_id: int, node: int, capacity: int):
        self.id = hosp_id
        self.node = node
        self.capacity = capacity
        self.current_patients = 0

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
            current_patients=self.current_patients
        )
