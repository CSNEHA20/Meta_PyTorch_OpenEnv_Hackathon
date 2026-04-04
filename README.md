# OpenEnv Ambulance Dispatch

This repository contains a production-grade reinforcement learning environment for city-scale ambulance dispatch optimization, built specifically for the OpenEnv platform.

## Problem Description
Efficiently dispatching emergency medical resources is a critical challenge for urban infrastructure. This project aims to optimize response times and resource allocation using reinforcement learning. The objective is to serve high-priority emergencies as quickly as possible while managing a fleet of ambulances across a dynamic city network with fluctuating traffic conditions and limited hospital capacity.

## System Design
The environment comprises several interconnected simulation components:
- **City Graph**: A complex urban network modeled using a Barabasi-Albert graph via `networkx`. Travel times are determined by base edge weights and dynamic traffic multipliers.
- **Ambulances**: A fleet of units operating through a finite state machine: `IDLE` → `DISPATCHED` → `EN_ROUTE` → `AT_SCENE` → `TRANSPORTING` → `RETURNING`.
- **Emergencies**: Calls generated via a Poisson process with varying severities (`CRITICAL`, `HIGH`, `NORMAL`) and specific timeout windows.
- **Hospitals**: Medical facilities with fixed occupancy capacities. Overflows occur when ambulances deliver patients to full hospitals.
- **Traffic Engine**: Simulates realistic city traffic patterns, including significant rush-hour multipliers (7-9 AM and 5-8 PM).

## Action Space
Agents interact with the environment using the `ActionModel`:
- **ambulance_id**: The unique identifier of the idle ambulance to be dispatched.
- **emergency_id**: The unique identifier of the unassigned emergency to be served.
- **hospital_id**: The target hospital for patient delivery.

## Observation Space
The `ObservationModel` provides a complete snapshot of the system state:
- **ambulances**: Current location, state, and ETA for all units.
- **emergencies**: List of all unassigned emergencies and their remaining time.
- **hospitals**: Current occupancy and capacity for all facilities.
- **traffic**: Current global traffic multiplier.
- **step**: Current simulation timestamp.

## Reward Function
The environment provides a dense reward signal composed of the following components:
- **Served Bonus**: +10.0 for reaching an emergency scene.
- **Severity Bonus**: +5.0 for `CRITICAL` cases, +2.0 for `HIGH` cases.
- **Delivery Reward**: +5.0 for successful hospital admission.
- **Distance Penalty**: -0.05 per unit of distance between ambulance and emergency.
- **Idle Penalty**: -0.5 per step for each ambulance left in an `IDLE` state.
- **Overflow Penalty**: -5.0 for attempting to deliver to a full hospital.
- **Missed Penalty**: -20.0 for each emergency that times out.

## Tasks
The simulation supports three difficulty levels:
- **Easy**: 1 ambulance, no traffic, single emergency focus. Ideal for basic functional testing.
- **Medium**: 3 ambulances, mild traffic variations, 3-5 concurrent emergencies. Tests coordination and prioritization.
- **Hard**: 5 ambulances, high dynamic traffic, and strict hospital capacity constraints. Tests edge-case management under heavy load.

## Agents

### SmartDispatchAgent
A high-performance heuristic agent that prioritizes emergencies by severity and distance. It serves as a robust baseline for evaluation.

### PriorityAgent
An LLM-driven dispatch coordinator that uses structured prompting to make informed allocation decisions. It includes a mission-critical heuristic fallback to ensure 100% availability during API latency or outages.

## Setup

### Installation
```bash
pip install -r requirements.txt
```

### Execution
Run the full evaluation cycle across all task levels:
```bash
python inference.py
```

## Environment Variables
The following variables are used for API integration and LLM agent connectivity:
- `API_BASE_URL`: The endpoint for OpenEnv performance logging.
- `MODEL_NAME`: The identifier for the RL policy.
- `HF_TOKEN`: HuggingFace token for secure model and API access.

## Example Output Logs
The system produces strict JSON telemetry for automated parsing:

```json
{"type": "START", "task": "medium"}
{"type": "STEP", "step": 1, "action": {"ambulance_id": 0, "emergency_id": "a1b2", "hospital_id": 1}, "reward": 8.4, "done": false}
{"type": "END", "task": "medium", "score": 0.82}
```

## Deployment
- **Docker**: Containerize the environment for reliable execution in any infrastructure.
- **HuggingFace**: Deploy as a Space for real-time monitoring and evaluation of public RL policies.

## Future Improvements
- **Real-World Topographies**: Integration of OpenStreetMap data for specific city simulations.
- **Multi-Agent Coordination**: Transitioning from centralized dispatching to decentralized agent cooperation.
- **Deep Reinforcement Learning**: Training Proximal Policy Optimization (PPO) models using the dense reward signals provided.
