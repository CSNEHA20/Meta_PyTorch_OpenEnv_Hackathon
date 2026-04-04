from env.models import ObservationModel, ActionModel, AmbulanceState
from typing import Optional

class GreedyAgent:
    """A baseline greedy agent for ambulance dispatch."""
    
    def act(self, observation: ObservationModel) -> ActionModel:
        # 1. Filter idle ambulances
        idle_ambs = [a for a in observation.ambulances if a.state == AmbulanceState.IDLE]

        # 2. Handle missing resources
        if not idle_ambs or not observation.emergencies:
            return ActionModel(ambulance_id=None, emergency_id="", hospital_id=None)

        # 3. Choose first idle ambulance
        selected_amb = idle_ambs[0]

        # 4. Choose nearest emergency to that ambulance
        selected_emg = min(
            observation.emergencies,
            key=lambda e: abs(e.node - selected_amb.node)
        )

        # 5. Choose nearest hospital to the emergency
        selected_hosp = min(
            observation.hospitals,
            key=lambda h: abs(h.node - selected_emg.node)
        )

        # 6. Return action
        return ActionModel(
            ambulance_id=selected_amb.id,
            emergency_id=selected_emg.id,
            hospital_id=selected_hosp.id
        )
