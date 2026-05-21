# Drone-Swarm-Defence
 
2D simulation environment for simulating decentralized drone swarms versus
integrated air defense networks.
 
Built as a personal challenge to apply the knowledge I have acquired through
self-study on this field.
 
## What this is
 
A Python simulator in which a decentralized swarm of attacking drones engages
a defended area protected by a radar-tracking and interceptor network.
This is the plan that I aim to fulfil, hopefully :)
 
| Phase | Description | Status |
|-------|-------------|--------|
| 1 — Single-radar tracking | EKF tracker, one radar tracking one maneuvering target, live visualization | ✅ Working demo |
| 2 — Multi-radar network | Distributed multi-target tracking, sensor fusion, Covariance Intersection | ⬜ Planned |
| 3 — Interceptors | Proportional navigation guidance, engagement loop, miss-distance analysis | ⬜ Planned |
| 4 — Drone swarm | Decentralized swarm, saturation attacks, cost-exchange curves | ⬜ Planned |
 
## Current Phase 1 demo
 
A single ground-based radar tracks a maneuvering drone flying a square
waypoint pattern. The EKF maintains a position/velocity estimate from noisy
range/bearing measurements, with a 3 std covariance ellipse showing tracking
uncertainty in real time.
 
- **White circle + line** — ground truth target and heading.
- **Cyan triangle + faint ring** — radar position and ~50% detection range.
- **Red crosses** — noisy raw range/bearing measurements.
- **Green circle + ellipse** — EKF state estimate and 3σ uncertainty.
Run with:
 
\`\`\`bash
python run_phase1.py
\`\`\`
 
## Stack
 
- Python 3.12, NumPy, Pygame
- Kalman filters implemented from scratch (no FilterPy in production code)
- pytest for testing, ruff for linting
## Setup
 
\`\`\`bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
\`\`\`
 
## Running the tests
 
\`\`\`bash
pytest tests/ -v
\`\`\`
 
 
## Repository structure
 
\`\`\`
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
docs/            # Technical report
run_phase1.py    # Phase 1 demo entry point
\`\`\`
 