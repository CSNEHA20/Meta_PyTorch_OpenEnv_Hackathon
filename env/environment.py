import numpy as np
import random
from typing import Tuple, Dict, List, Any, Optional
from env.models import (
    ObservationModel, ActionModel, RewardModel, 
    AmbulanceState, Severity, AmbulanceInfo, EmergencyInfo, HospitalInfo
)
from env.simulator import CityGraph, TrafficEngine, AmbulanceFleet, EmergencyGenerator, Hospital

class AmbulanceEnv:
    """
    OpenEnv implementation for the Ambulance Dispatch Reinforcement Learning Environment.
    """
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.n_ambulances = self.config.get("n_ambulances", 5)
        self.n_hospitals = self.config.get("n_hospitals", 3)
        self.max_steps = self.config.get("max_steps", 1440) # One full day
        self.graph_size = self.config.get("graph_size", 100)
        
        self.seed_val = self.config.get("seed", 42)
        self.reset(seed=self.seed_val)

    def reset(self, seed: Optional[int] = None) -> ObservationModel:
        """Reset the environment to its initial state."""
        if seed is not None:
            self.seed_val = seed
            random.seed(seed)
            np.random.seed(seed)

        # Initialize Simulation Components
        self.city_graph = CityGraph(n=self.graph_size)
        self.nodes = list(self.city_graph.graph.nodes())
        self.fleet = AmbulanceFleet(n=self.n_ambulances, nodes=self.nodes)
        self.generator = EmergencyGenerator(nodes=self.nodes)
        self.traffic = TrafficEngine()
        
        # Initialize Hospitals with random locations
        self.hospitals = {
            i: Hospital(hosp_id=i, node=int(np.random.choice(self.nodes)), capacity=15)
            for i in range(self.n_hospitals)
        }

        self.step_count = 0
        self.active_emergencies: List[EmergencyInfo] = []
        
        # Track distances for shaping rewards
        self._prev_amb_emg_distances = {} # {amb_id: distance}

        # Epidsode Metrics
        self.metrics = {
            "served": 0,
            "missed": 0,
            "total_emergencies": 0,
            "critical_served": 0,
            "high_served": 0,
            "normal_served": 0,
            "successful_dispatches": 0,
            "avg_response_time": 0.0,
            "hospital_overflow": 0.0
        }
        self._dispatch_times = {}
        self._response_times = []
        self.predicted_hotspots = []

        return self._get_observation()

    def set_predicted_hotspots(self, nodes: List[int]):
        """Update the environment's list of predicted hotspots."""
        self.predicted_hotspots = nodes

    def step(self, action: ActionModel) -> Tuple[ObservationModel, float, bool, Dict[str, Any]]:
        """Advance the environment by one step using the provided action."""
        self.step_count += 1
        
        reward_components = {
            "served_bonus": 0.0,
            "missed_penalty": 0.0,
            "idle_penalty": 0.0,
            "step_penalty": -1.0, # Small step penalty
            "reposition_bonus": 0.0,
            "overflow_penalty": 0.0,
            "invalid_action_penalty": 0.0,
            "moving_towards_bonus": 0.0,
            "shaping_distance_bonus": 0.0
        }

        tm = self.traffic.get_multiplier(self.step_count)

        # 0. Calculate pre-update distances for shaping
        # This reflects the distance BEFORE simulator step update (to compare against previous step)
        for amb in self.fleet.ambulances:
            if amb.state == AmbulanceState.EN_ROUTE and amb.target_emg_node is not None:
                # Current distance to objective
                current_dist = self.city_graph.shortest_path_time(amb.node, amb.target_emg_node, tm)
                
                # If we have a previous distance for THIS ambulance's current mission
                prev_dist = self._prev_amb_emg_distances.get(amb.id)
                if prev_dist is not None:
                    if current_dist < prev_dist:
                        # Find the emergency to check severity for shaping
                        emg_obj = next((e for e in self.active_emergencies if e.id == amb.target_emg_id), None)
                        if emg_obj and emg_obj.severity == Severity.CRITICAL:
                            reward_components["shaping_distance_bonus"] += 10.0
                        else:
                            reward_components["shaping_distance_bonus"] += 3.0
                    
                    # Reward for being "close" (e.g., within 5 minutes)
                    if current_dist <= 5:
                        reward_components["shaping_distance_bonus"] += 5.0
                
                # Update tracking
                self._prev_amb_emg_distances[amb.id] = current_dist
            else:
                # Clear tracking if not en-route
                self._prev_amb_emg_distances.pop(amb.id, None)

        # 0. Repositioning Processing
        if action.reposition_node is not None and action.ambulance_id is not None:
            amb = next((a for a in self.fleet.ambulances if a.id == action.ambulance_id), None)
            if amb and amb.state == AmbulanceState.IDLE:
                # Execute repositioning
                self.fleet.reposition(amb.id, action.reposition_node, self.city_graph, tm)
                
                # Proactive Repositioning Bonus (Stronger incentive)
                if action.reposition_node in self.predicted_hotspots:
                    reward_components["reposition_bonus"] += 2.0
            else:
                reward_components["invalid_action_penalty"] -= 10.0

        if action.ambulance_id is not None and action.emergency_id:
            amb = next((a for a in self.fleet.ambulances if a.id == action.ambulance_id), None)
            emg = next((e for e in self.active_emergencies if e.id == action.emergency_id and not e.assigned), None)
            hosp = self.hospitals.get(action.hospital_id)

            if amb and emg and hosp and amb.state == AmbulanceState.IDLE:
                if hosp.is_available():
                    self.fleet.dispatch(amb.id, emg.id, emg.node, hosp.id, hosp.node)
                    emg.assigned = True
                    self._dispatch_times[emg.id] = self.step_count
                    self.metrics["successful_dispatches"] += 1
                else:
                    reward_components["overflow_penalty"] -= 10.0 # Standard invalid action penalty
                    self.metrics["hospital_overflow"] += 1
            else:
                reward_components["invalid_action_penalty"] -= 10.0

        # 2. Advanced Simulation
        tm = self.traffic.get_multiplier(self.step_count)
        
        # Track pre-update states for event detection
        pre_states = {a.id: a.state for a in self.fleet.ambulances}
        self.fleet.step_update(self.city_graph, tm)
        post_states = {a.id: a.state for a in self.fleet.ambulances}

        # 3. Emergency Generation & Timeouts
        new_emgs = self.generator.generate(self.step_count)
        self.active_emergencies.extend(new_emgs)
        self.metrics["total_emergencies"] += len(new_emgs)

        # Handle timeouts for unassigned emergencies
        missed_count = 0
        current_active = []
        for emg in self.active_emergencies:
            if not emg.assigned:
                emg.time_remaining -= 1
                if emg.time_remaining <= 0:
                    missed_count += 1
                    continue
            current_active.append(emg)
        self.active_emergencies = current_active
        
        reward_components["missed_penalty"] -= missed_count * 20.0
        # reward_components["delay_penalty"] -= len(self.active_emergencies) * 1.0 # Removed large penalties
        self.metrics["missed"] += missed_count

        # 4. Process State Transitions & Event Rewards
        for amb in self.fleet.ambulances:
            # Moving towards emergency bonus
            if amb.state == AmbulanceState.EN_ROUTE:
                reward_components["moving_towards_bonus"] += 2.0

            # Event: Reached Scene (Pickup)
            if pre_states[amb.id] == AmbulanceState.EN_ROUTE and post_states[amb.id] == AmbulanceState.AT_SCENE:
                reward_components["served_bonus"] += 50.0 # Emergency served
                self.metrics["served"] += 1
                
                # Tracking response time
                emg_id = amb.target_emg_id
                resp_time = self.step_count - self._dispatch_times.get(emg_id, self.step_count)
                self._response_times.append(resp_time)
                
                # Update avg_response_time metric
                if self._response_times:
                    self.metrics["avg_response_time"] = sum(self._response_times) / len(self._response_times)

                # Fetch original emergency for severity bonus and metrics
                emg_obj = next((e for e in self.active_emergencies if e.id == emg_id), None)
                if emg_obj:
                    if emg_obj.severity == Severity.CRITICAL:
                        reward_components["served_bonus"] += 80.0
                        self.metrics["critical_served"] += 1
                    elif emg_obj.severity == Severity.HIGH:
                        reward_components["served_bonus"] += 30.0
                        self.metrics["high_served"] += 1
                    else:
                        reward_components["served_bonus"] += 10.0
                        self.metrics["normal_served"] += 1
                    
                    # Remove from active list as it's now being processed at scene
                    self.active_emergencies = [e for e in self.active_emergencies if e.id != emg_id]

            # Event: Delivered to Hospital
            if pre_states[amb.id] == AmbulanceState.TRANSPORTING and post_states[amb.id] == AmbulanceState.RETURNING:
                hosp = self.hospitals.get(amb.target_hosp_id)
                if hosp:
                    hosp.admit()

        # Removed idle penalty to focus on positive reinforcement
        # if len(self.active_emergencies) > 0:
        #     num_idle = len(self.fleet.get_idle())
        #     reward_components["idle_penalty"] -= 5.0 * num_idle
        
        total_reward = float(sum(reward_components.values()))
        
        # Clamp reward to prevent extreme spikes and stabilize learning
        total_reward = float(max(min(total_reward, 50.0), -50.0))
        
        done = self.step_count >= self.max_steps
        obs = self._get_observation(tm)
        
        # Update running metrics
        # (Already updated during scene arrival)

        return obs, total_reward, done, self.state(tm)

    def state(self, tm: Optional[float] = None) -> Dict[str, Any]:
        """Return full internal state/metrics of the environment."""
        if tm is None:
            tm = self.traffic.get_multiplier(self.step_count)
            
        return {
            "step": self.step_count,
            "metrics": self.metrics,
            "ambulances": [a.to_info().dict() for a in self.fleet.ambulances],
            "hospitals": [h.to_info().dict() for h in self.hospitals.values()],
            "emergencies": [e.dict() for e in self.active_emergencies],
            "traffic_multiplier": tm
        }

    def _get_observation(self, tm: Optional[float] = None) -> ObservationModel:
        """Construct the observation model for the current step."""
        if tm is None:
            tm = self.traffic.get_multiplier(self.step_count)
            
        return ObservationModel(
            ambulances=[amb.to_info() for amb in self.fleet.ambulances],
            emergencies=[e for e in self.active_emergencies if not e.assigned],
            hospitals=[h.to_info() for h in self.hospitals.values()],
            traffic={"global": tm},
            step=self.step_count
        )
