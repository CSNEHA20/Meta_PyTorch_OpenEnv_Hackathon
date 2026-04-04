from env.simulator import CityGraph, TrafficEngine, AmbulanceFleet, EmergencyGenerator, Hospital
from env.models import AmbulanceState
import numpy as np

def test_simulation():
    # 1. Setup
    nodes = list(range(100))
    city = CityGraph(n=100)
    traffic = TrafficEngine()
    fleet = AmbulanceFleet(n=5, nodes=nodes)
    generator = EmergencyGenerator(nodes=nodes, lambda_param=0.1)
    hospitals = {i: Hospital(hosp_id=i, node=int(np.random.choice(nodes)), capacity=10) for i in range(3)}

    print("Initial Fleet State:")
    for amb in fleet.ambulances:
        print(f"Ambulance {amb.id} at node {amb.node} state {amb.state}")

    # 2. Simulate 100 steps
    active_emergencies = []
    
    for step in range(50): # 50 steps is enough for a quick check
        # Generate new emergencies
        new_emgs = generator.generate(step)
        active_emergencies.extend(new_emgs)
        
        # Traffic multiplier
        tm = traffic.get_multiplier(step)
        
        # Simple Dispatch Logic
        idle_ambs = fleet.get_idle()
        for amb in idle_ambs:
            if active_emergencies:
                # Find an unassigned emergency
                for emg in active_emergencies:
                    if not emg.assigned:
                        emg.assigned = True
                        hosp = hospitals[0] # Just use the first hospital for testing
                        fleet.dispatch(amb.id, emg.id, emg.node, hosp.id, hosp.node)
                        print(f"[{step}] Dispatched Amb {amb.id} to Emg {emg.id} severity {emg.severity}")
                        break

        # Update Fleet
        fleet.step_update(city, tm)
        
        # Progress Emergencies
        for emg in active_emergencies:
            if not emg.assigned:
                emg.time_remaining -= 1
        
        # Clean up expired emergencies
        active_emergencies = [e for e in active_emergencies if e.time_remaining > 0 or e.assigned]

    print("\nFinal Fleet State:")
    for amb in fleet.ambulances:
        print(f"Ambulance {amb.id} state {amb.state} at node {amb.node} ETA {amb.eta}")

if __name__ == "__main__":
    test_simulation()
