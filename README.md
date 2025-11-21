# Wildfire Firefighter Multiagent Simulation
## Overview

This project implements a multiagent simulation of wildfire containment using the Repast4Py framework.
A set of autonomous firefighter agents operate in a 2D grid environment where fire spreads stochastically.
Each firefighter makes decisions locally—using limited perception, bounded-range radio communication, and internal state (water level, behavioral mode, and nearby fire).
Together, they produce emergent global containment behavior without any centralized controller.

## Running the Simulation
1. Install Repast4Py
```bash
pip install repast4py
```
2. Activate virtual environment
``` bash
source ~/.venvs/repast4py/bin/activate
```

3. Install dependencies
``` bash
pip install matplotlib
```

4. Run the model
```bash
./run.sh
```

## Fire/Environment Model
The environment is represented using two numpy grids:

- fire_state[x, y]
    - SAFE (0)
    - BURNING (1)
    - BURNT (2)
    - EXTINGUISHED (3)
- fire_intensity[x, y] ∈ [0, 1]

Fire behavior includes:
- Random ignitions
- Intensity-based spread probability (base_spread_p)
- Burn-down over time
- Optional wind bias parameters
All fire dynamics occur independently of the firefighters.

## Firefighter Agent
Each firefighter behaves autonomously using only local info
#### Local Perception
- Detects burning cells within radius perception_r.
#### Local Communication
- Periodacally broadcasts local fire coordinates to a shared radio board
- Only recieves messages within comm_r
#### Internal State
- water supply and max_water capacity
- Must visit border to refill
- Finite-state behavioral modes:
  - ATTACK: move toward fire
  - REFILL: move toward border
  - SCOUT: random patrol
#### Movement & Extinguishing
- Moves one cell per tick toward target fire or radio report
- Extinguishes a burning cell if standing on it and has water
#### Local Coordination via Emergence
- Uses a claimed_targets set to avoid duplication and spread agents across the fire front
- No global planner or central control

## Output/Metrics
During each run, the system logs:
- Tick
- Burning count
- Burnt count
- Extinguished count
- Total messages sent
- Number of firefighters
- Containment time
Which are recorded in 
```
output/metrics.csv
```bash
We also record visual snapshots which are saved in 
```bash
output/snapshots
```

## Configuration (params.yaml)
All high-level controls are in params.yaml. These include the params we will be changing for our expirements.

