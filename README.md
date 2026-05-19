# Drone-Swarm-Defence
 
2D simulation environment for simulating a 
decentralized drone swarms and integrated air defense networks.
 
Built just as challenge for myself, as I  wanted to apply the knowledge 
I have aquired through self-study on this field.
 
## What this is
 
The goal is to have a Python simulator in which a decentralized swarm of attacking
drones engages a defended area protected by a radar-tracking and interceptor
network. 
 
## Stack
- Python 3.12, NumPy, SciPy, Matplotlib, Pygame
- Filters implemented from scratch 
- pytest for testing, ruff for linting

## Setup
 
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
 
## Running the tests
 
```bash
pytest tests/ -v
```
 
## Repository structure
 
```
sim/
├── core/        # Clock, World, Entity protocol, shared types
├── entities/    # Target, Radar, Interceptor, Drone
├── estimation/  # Kalman filter variants (KF, EKF, UKF, IMM)
├── control/     # Guidance laws, weapon-target assignment
├── comms/       # Swarm communication models
├── scenarios/   # YAML-defined experiment configurations
└── viz/         # Pygame renderer
tests/           # Mirrors sim/ structure, one file per module
notebooks/       # Monte Carlo analysis, parameter sweeps
docs/            #technical report
```