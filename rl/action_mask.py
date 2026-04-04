import numpy as np

class ActionMask:
    def build_mask(self, action_mapper):
        """
        Builds a fixed-size numpy mask array of shape (13,) based on current observation.
        """
        num_actions = action_mapper.size()
        mask = np.zeros(num_actions, dtype=np.float32)
        
        # Access original entities from built mapper
        ambulances = action_mapper.current_ambulances
        emergencies = action_mapper.current_emergencies
        hospitals = action_mapper.current_hospitals
        
        # 1. Slot 0: No-Op
        # Allowed if ONLY it exists or handled by the edge case (all invalid -> all valid).
        # We start with it 1.0 but will handle dispatch logic below.
        mask[0] = 1.0
        
        # 2. Iterate fixed combinations 1-250
        for i, amb in enumerate(ambulances):
            # Condition 1: Ambulance exists and is idle
            is_amb_valid = (amb.state == "idle" or amb.state == 0) # Fallback to numerical state if enum mismatch
            
            if is_amb_valid:
                for j, emg in enumerate(emergencies):
                    # Condition 2: Emergency exists and is not assigned
                    is_emg_valid = not emg.assigned
                    
                    if is_emg_valid:
                        for k, hosp in enumerate(hospitals):
                            # Condition 3: Hospital exists and has capacity
                            is_hosp_valid = (hosp.current_patients < hosp.capacity)
                            
                            if is_hosp_valid:
                                # Map back to fixed action index using action_mapper's logic
                                n_emg_max = action_mapper.MAX_EMERGENCIES
                                n_hosp_max = action_mapper.MAX_HOSPITALS
                                idx = 1 + (i * n_emg_max * n_hosp_max) + (j * n_hosp_max) + k
                                if idx < num_actions:
                                    mask[idx] = 1.0
                                    
        # Edge weight logic: if valid dispatches exist, No-Op should be 0.0 (per common RL mask patterns)
        # unless specifically allowed only when it is the only action.
        if np.sum(mask[1:]) > 0:
            mask[0] = 0.0
        else:
            mask[0] = 1.0

        return mask

    def apply_mask(self, q_values, mask):
        """
        Replaces invalid actions in q_values with a very low value.
        """
        for i in range(len(q_values)):
            if mask[i] == 0:
                q_values[i] = -1e9
        return q_values
