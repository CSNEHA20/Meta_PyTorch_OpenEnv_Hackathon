import sys
import os
import numpy as np
from env.models import ObservationModel, AmbulanceInfo, EmergencyInfo, HospitalInfo, AmbulanceState, Severity
from rl.action_mapper import ActionMapper
from rl.action_mask import ActionMask

def test_reduction():
    mapper = ActionMapper()
    masker = ActionMask()
    
    print(f"Action Space Size: {mapper.size()}")
    assert mapper.size() == 13
    
    # Mock observation
    # 5 ambulances, 10 emergencies, 5 hospitals (as in original)
    ambulances = [
        AmbulanceInfo(id=i, node=i*10, state=AmbulanceState.IDLE, eta=0)
        for i in range(5)
    ]
    emergencies = [
        EmergencyInfo(id=str(j), node=j*5, severity=Severity.NORMAL, time_remaining=10+j, assigned=False)
        for j in range(10)
    ]
    # Make one emergency critical
    emergencies[5].severity = Severity.CRITICAL
    emergencies[5].node = 55
    
    hospitals = [
        HospitalInfo(id=k, node=k*20, capacity=10, current_patients=0)
        for k in range(5)
    ]
    
    obs = ObservationModel(
        ambulances=ambulances,
        emergencies=emergencies,
        hospitals=hospitals,
        traffic={"global": 1.0},
        step=0
    )
    
    mapper.build_action_space(obs)
    mask = masker.build_mask(mapper)
    
    print(f"Mask shape: {mask.shape}")
    print(f"Mask sum: {np.sum(mask)}")
    
    print("Selected Emergencies IDs:")
    print([e.id for e in mapper.current_emergencies])
    
    print("Selected Ambulances IDs:")
    print([a.id for a in mapper.current_ambulances])
    
    print("Selected Hospitals IDs:")
    print([h.id for h in mapper.current_hospitals])
    
    # The first emergency should be the critical one (ID '5')
    assert mapper.current_emergencies[0].id == '5'
    
    # Nearest ambulance to node 55 should be id 5 (node 50) or 6 (node 60)?
    # Wait, node 55 is closest to node 50 (amb id 5) and node 60 (not in list of 5? wait, I made 5 ambs)
    # Ambs nodes: 0, 10, 20, 30, 40.
    # Nearest to 55 in 0-40 is 40 (id 4).
    assert mapper.current_ambulances[0].id == 4
    
    print("Test passed!")

if __name__ == "__main__":
    test_reduction()
