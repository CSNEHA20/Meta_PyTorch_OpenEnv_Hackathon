from typing import List, Tuple, Optional
from env.models import ObservationModel, ActionModel, AmbulanceState, Severity, AmbulanceInfo, EmergencyInfo, HospitalInfo

class ActionMapper:
    """
    Action mapper for reinforcement learning ambulance dispatch system.
    Maps (ambulance_id, emergency_id, hospital_id) combinations to a focused discrete action space.
    Action Space: Index 0 (No-Op), Indices 1-12 (2x3x2 combinations)

    Constructor accepts optional keyword overrides so legacy callers like inference.py
    (which may pass max_ambulances=5, etc.) do not raise TypeError.
    The class-level constants remain canonical for the training action space.
    """
    
    # Reduced Action Space: 1 + (2 x 3 x 2) = 13 actions
    MAX_AMBULANCES = 2
    MAX_EMERGENCIES = 3
    MAX_HOSPITALS = 2
    
    def __init__(self, max_ambulances: int = None, max_emergencies: int = None, max_hospitals: int = None):
        # Allow callers to override the class constants (e.g., inference.py compatibility).
        # Training uses class-level defaults; inference.py passes larger values.
        if max_ambulances is not None:
            self.MAX_AMBULANCES = max_ambulances
        if max_emergencies is not None:
            self.MAX_EMERGENCIES = max_emergencies
        if max_hospitals is not None:
            self.MAX_HOSPITALS = max_hospitals

        # The actions list is now fixed and represents all potential dispatch combinations.
        self.actions: List[Tuple[Optional[int], str, Optional[int]]] = []
        self._initialize_actions()
        self.current_ambulances: List = []
        self.current_emergencies: List = []
        self.current_hospitals: List = []

    def _initialize_actions(self):
        """Initializes the fixed-size action list with 251 slots."""
        # Slot 0: No-Op
        self.actions = [(None, "", None)]
        
        # Slots 1-250: All potential ambulance-emergency-hospital index combinations
        for i in range(self.MAX_AMBULANCES):
            for j in range(self.MAX_EMERGENCIES):
                for k in range(self.MAX_HOSPITALS):
                    # These are placeholder IDs; IDs will be resolved during build_action_space.
                    self.actions.append((i, str(j), k))

    def build_action_space(self, observation: ObservationModel):
        """
        Filters and ranks entities carefully (Rule #7) to reduce the action space to a focused subset.
        1. TOP 3 Emergencies: Filtered by Severity (Critical > High > Normal) and time_remaining.
        2. TOP 2 Nearest Idle Ambulances: Sorted by distance to the focus area.
        3. TOP 2 Nearest Available Hospitals: Filtered by hospital capacity and distance.
        """
        # --- SMART EMERGENCY FILTERING ---
        sev_values = {Severity.CRITICAL: 100, Severity.HIGH: 50, Severity.NORMAL: 10}
        unassigned_emgs = [e for e in observation.emergencies if not e.assigned]
        
        # Rank by Severity priority and urgency (less time remaining = more urgent)
        # Note: We want higher severity first, then shorter time remaining.
        unassigned_emgs.sort(key=lambda e: (sev_values.get(e.severity, 0), -e.time_remaining), reverse=True)
        self.current_emergencies = unassigned_emgs[:self.MAX_EMERGENCIES]
        
        if not self.current_emergencies:
            self.current_ambulances = []
            self.current_hospitals = []
            return

        # Use the highest priority emergency location as the primary focus area
        focus_node = self.current_emergencies[0].node
        
        # --- SMART AMBULANCE FILTERING ---
        # Find nearest IDLE ambulances to the focus area (Improvement Rule #7)
        idle_ambs = [a for a in observation.ambulances if a.state == AmbulanceState.IDLE]
        idle_ambs.sort(key=lambda a: abs(a.node - focus_node))
        self.current_ambulances = idle_ambs[:self.MAX_AMBULANCES]
        
        # --- SMART HOSPITAL FILTERING ---
        # Find nearest hospitals with remaining CAPACITY (Improvement Rule #7)
        available_hospitals = [h for h in observation.hospitals if h.capacity > 0]
        if not available_hospitals:
            # Fallback to all if everyone is full (to avoid crash, though env handles overflow)
            available_hospitals = observation.hospitals
            
        available_hospitals.sort(key=lambda h: abs(h.node - focus_node))
        self.current_hospitals = available_hospitals[:self.MAX_HOSPITALS]

    def decode(self, index: int) -> ActionModel:
        """
        Converts a discrete index into a structured ActionModel based on the current observation.
        """
        if index == 0 or index >= len(self.actions):
            return ActionModel(ambulance_id=None, emergency_id="", hospital_id=None)
        
        # Calculate indices from fixed slot (1-based index)
        offset = index - 1
        amb_idx = offset // (self.MAX_EMERGENCIES * self.MAX_HOSPITALS)
        rem = offset % (self.MAX_EMERGENCIES * self.MAX_HOSPITALS)
        emg_idx = rem // self.MAX_HOSPITALS
        hosp_idx = rem % self.MAX_HOSPITALS
        
        # Resolve to current IDs if available
        ambulance_id = None
        if amb_idx < len(self.current_ambulances):
            ambulance_id = self.current_ambulances[amb_idx].id
            
        emergency_id = ""
        if emg_idx < len(self.current_emergencies):
            emergency_id = self.current_emergencies[emg_idx].id
            
        hospital_id = None
        if hosp_idx < len(self.current_hospitals):
            hospital_id = self.current_hospitals[hosp_idx].id
            
        return ActionModel(
            ambulance_id=ambulance_id,
            emergency_id=emergency_id,
            hospital_id=hospital_id
        )

    def size(self) -> int:
        """Returns the fixed number of actions (251)."""
        return 1 + (self.MAX_AMBULANCES * self.MAX_EMERGENCIES * self.MAX_HOSPITALS)

    def get_action_space(self, observation: ObservationModel) -> List[int]:
        """
        Builds the action space from `observation` and returns a list of VALID action indices.
        This is the inference.py-compatible API: it combines build_action_space + mask in one call.
        Fallback: returns [0] (No-Op) if no valid dispatches exist.
        """
        self.build_action_space(observation)
        valid = []

        # Slot 0 (No-Op) is always potentially valid
        ambulances = self.current_ambulances
        emergencies = self.current_emergencies
        hospitals = self.current_hospitals

        for i, amb in enumerate(ambulances):
            if amb.state != AmbulanceState.IDLE:
                continue
            for j, emg in enumerate(emergencies):
                if emg.assigned:
                    continue
                for k, hosp in enumerate(hospitals):
                    if hosp.current_patients >= hosp.capacity:
                        continue
                    idx = 1 + (i * self.MAX_EMERGENCIES * self.MAX_HOSPITALS) + (j * self.MAX_HOSPITALS) + k
                    if idx < self.size():
                        valid.append(idx)

        if not valid:
            valid = [0]  # Fallback to No-Op

        return valid
