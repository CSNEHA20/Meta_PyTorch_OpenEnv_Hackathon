from env.models import ObservationModel, ActionModel, AmbulanceState, Severity

class SmartDispatchAgent:
    """Enhanced agent with multi-criteria dispatch prioritization."""
    
    def act(self, observation: ObservationModel) -> ActionModel:
        # 1. Extract components
        ambulances = observation.ambulances
        emergencies = observation.emergencies
        hospitals = observation.hospitals

        # 2. Filter idle ambulances
        idle_ambs = [a for a in ambulances if a.state == "idle"]

        # 3. Handle edge cases
        if not idle_ambs or not emergencies:
            return ActionModel(ambulance_id=None, emergency_id="", hospital_id=None)

        # 4. Prioritize emergencies
        # CRITICAL > HIGH > NORMAL, then lower time_remaining
        priority_map = {Severity.CRITICAL: 0, Severity.HIGH: 1, Severity.NORMAL: 2}
        
        sorted_emgs = sorted(
            emergencies,
            key=lambda e: (priority_map.get(e.severity, 3), e.time_remaining)
        )
        selected_emg = sorted_emgs[0]

        # 5. Select nearest idle ambulance
        best_amb = min(
            idle_ambs,
            key=lambda a: abs(a.node - selected_emg.node)
        )

        # 6. Select nearest available hospital
        available_hosps = [h for h in hospitals if h.current_patients < h.capacity]
        if not available_hosps:
            return ActionModel(ambulance_id=None, emergency_id="", hospital_id=None)

        selected_hosp = min(
            available_hosps,
            key=lambda h: abs(h.node - selected_emg.node)
        )

        # 7. Return selected IDs
        return ActionModel(
            ambulance_id=best_amb.id,
            emergency_id=selected_emg.id,
            hospital_id=selected_hosp.id
        )
