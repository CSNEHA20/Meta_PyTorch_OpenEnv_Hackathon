from typing import List, Tuple, Optional
from env.models import ObservationModel, ActionModel, AmbulanceState, Severity, AmbulanceInfo, EmergencyInfo, HospitalInfo

class ActionMapper:
    """
    Action mapper for reinforcement learning ambulance dispatch system.
    Maps (ambulance_id, emergency_id, hospital_id) combinations to a focused discrete action space.
    Action Space: Index 0 (No-Op), Indices 1-12 (2x3x2 combinations)
    """
    
    # Reduced Action Space: 1 + (2 x 3 x 2) = 13 actions
    MAX_AMBULANCES = 2
    MAX_EMERGENCIES = 3
    MAX_HOSPITALS = 2
    
    def __init__(self):
        # The actions list is now fixed and represents all potential dispatch combinations.
        self.actions: List[Tuple[Optional[int], str, Optional[int]]] = []
        self._initialize_actions()

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
        Filters and ranks entities to reduce the action space to a focused subset.
        1. Top 3 emergencies by severity and urgency.
        2. Nearest 2 idle ambulances to the highest priority emergency.
        3. Nearest 2 hospitals to the highest priority emergency.
        """
        # 1. Select TOP 3 emergencies
        # Severity priority: CRITICAL(3), HIGH(2), NORMAL(1)
        sev_values = {Severity.CRITICAL: 3, Severity.HIGH: 2, Severity.NORMAL: 1}
        
        # Sort unassigned by (Severity Descending, time_remaining Ascending)
        unassigned = [e for e in observation.emergencies if not e.assigned]
        unassigned.sort(key=lambda e: (sev_values.get(e.severity, 0), -e.time_remaining), reverse=True)
        self.current_emergencies = unassigned[:self.MAX_EMERGENCIES]
        
        if not self.current_emergencies:
            self.current_ambulances = []
            self.current_hospitals = []
            return

        # Use the highest priority emergency as focus point
        ref_node = self.current_emergencies[0].node
        
        # 2. Select nearest 2 IDLE ambulances
        idle_ambs = [a for a in observation.ambulances if a.state == AmbulanceState.IDLE]
        idle_ambs.sort(key=lambda a: abs(a.node - ref_node))
        self.current_ambulances = idle_ambs[:self.MAX_AMBULANCES]
        
        # 3. Select nearest 2 hospitals
        hospitals = observation.hospitals[:]
        hospitals.sort(key=lambda h: abs(h.node - ref_node))
        self.current_hospitals = hospitals[:self.MAX_HOSPITALS]

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
