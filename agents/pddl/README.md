# PDDL Agent Module

A physics-based AI agent for the Science Birds (Angry Birds) domain that uses PDDL (Planning Domain Definition Language) planning combined with online physics learning to solve levels.

## Overview

The PDDL Agent is part of the Strike-Eagle project and implements an intelligent agent that:
1. Learns physics parameters (gravity, velocity) from observed trajectories
2. Uses PDDL planning to determine optimal shooting angles
3. Adapts its world model through online learning from game observations

## Architecture

```
agents/pddl/
├── pddl_agent.py          # Main agent class
├── optimizer.py           # Physics parameter optimization & polynomial fitting
├── trajectory_parser.py   # Trajectory extraction and simulation
├── metrics.py             # RMSE calculation and trajectory comparison
├── visualiator.py         # Visualization utilities
└── pddl_files/
    ├── pddl_parser.py     # PDDL problem/domain file generation
    ├── pddl_objects.py    # Game object extraction (birds, pigs, blocks)
    ├── segments.py        # Trajectory segmentation and event detection
    ├── world_model/
    │   ├── world_model.py # Physics world model
    │   ├── params.py      # Parameter definitions
    │   └── process.py     # Process definitions
    └── events/
        ├── event_conditions.py  # Event detection (collisions, hits)
        └── learn_events.py      # Event effect learning
```

## Core Components

### PDDLAgent (`pddl_agent.py`)

The main agent class that orchestrates the planning and learning pipeline.

**Key Attributes:**
- `world_model`: Physics parameters (gravity: ~90, velocity: ~200)
- `learned_transitions`: State transition functions for x, y, xdot, ydot, xddot, yddot
- `kb`: Knowledge base storing collision data and past trajectories
- `game_results`: Win/loss tracking per level

**Main Methods:**

| Method | Description |
|--------|-------------|
| `solve()` | Main game-solving loop - shoots, observes, learns |
| `get_action_to_perform()` | Uses PDDL planner to determine shooting angle |
| `learn_process()` | Learns gravity/velocity from single trajectory |
| `learn_process_transitions()` | Learns state transition functions from KB trajectories |

**Learning Pipeline:**
1. Shoot at computed angle → Record ground truth trajectory
2. Segment trajectory by events (ground collision, hits, platform collision)
3. Update collision effects in knowledge base
4. Learn state transition functions (frozen after 8 games)
5. Update world model with learned parameters

### Optimizer (`optimizer.py`)

Physics parameter optimization and polynomial fitting utilities.

**Key Functions:**

| Function | Description |
|----------|-------------|
| `get_poly_rank(x, y)` | Auto-select optimal polynomial degree using residual analysis |
| `grid_search()` | Search parameter space to minimize trajectory error |
| `compute_derivatives()` | Compute xdot, xddot, ydot, yddot from polynomials |
| `fit_state_transition()` | Fit polynomial transition functions for state prediction |
| `get_params_sensitivity()` | Sobol sensitivity analysis for parameter importance |

**Polynomial Fitting:**
- Uses residual improvement threshold (default: 1.0)
- Reports R² score for fit quality
- Supports multi-feature polynomial regression via sklearn

### Trajectory Parser (`trajectory_parser.py`)

Handles trajectory extraction from game observations and simulated trajectory construction.

**Key Functions:**

| Function | Description |
|----------|-------------|
| `extract_real_trajectory()` | Parse ground truth data into entity trajectories |
| `construct_trajectory()` | Simulate trajectory using world model |
| `euler_step()` | Simple Euler integration O(Δt²) |
| `midpoint_step()` | Midpoint method O(Δt³) |
| `rk4_step()` | Runge-Kutta 4th order O(Δt⁵) |

**Integration Methods:**
- `'euler'`: Fast, lowest accuracy
- `'midpoint'`: Good balance of speed/accuracy
- `'rk4'`: Most accurate (default)

### Metrics (`metrics.py`)

Trajectory comparison and RMSE calculation utilities.

**Key Functions:**

| Function | Description |
|----------|-------------|
| `calculate_rmse()` | RMSE between observed and estimated trajectories |
| `compare_truncation_methods()` | Compare X-range vs minimum overlap methods |
| `compare_interpolation_methods()` | Linear vs cubic spline interpolation |
| `analyze_error_by_position()` | Segment-by-segment error analysis |
| `analyze_launch_angle()` | Instantaneous angle analysis for slingshot effects |
| `get_robust_launch_angle()` | Robust angle estimation via polynomial fitting |

**RMSE Configuration:**
- `RMSE_ANGLE_BIAS_DEGREES`: Compensate for slingshot bias (default: 0)
- Supports trimming start/end percentages
- Uses x-coordinate based resampling for alignment

### Visualizer (`visualiator.py`)

Comprehensive visualization tools for debugging and analysis.

**Key Functions:**

| Function | Description |
|----------|-------------|
| `visualize_compare()` | Compare observed vs estimated trajectories |
| `visualize_rmse()` | Plot RMSE over time (optionally compare with suggested values) |
| `visuallize_wins_percentage()` | Cumulative win rate visualization |
| `full_trajectory_comparison()` | Frame-by-frame trajectory animation |
| `debug_is_hit_last_frames()` | Debug collision detection |
| `debug_all_events_full_trajectory()` | Visualize all detected events |
| `visualize_post_collision_trajectory_v2()` | Post-collision trajectory analysis (direct velocity) |

### World Model (`pddl_files/world_model/`)

Physics simulation model with learnable parameters.

**Parameters (Params enum):**
- `gravity`: Gravitational acceleration (~87-90)
- `velocity`: Initial launch velocity (~175-200)
- `velocity_x`, `velocity_y`: Component velocities
- `x`, `y`: Position coordinates

**Default Values:**
```python
{
    Params.gravity: 87.2,
    Params.velocity: 175.9259
}
```

### Event Detection (`pddl_files/events/`)

Event detection for trajectory segmentation.

**Supported Events:**

| Event | Function | Description |
|-------|----------|-------------|
| Ground Collision | `is_ground_collision()` | Bird touches ground (y ≤ ε) |
| Hit | `is_hit()` | Bird-pig collision (radius overlap) |
| Platform Collision | `is_platform_collision()` | Bird-platform collision |

### PDDL Parser (`pddl_files/pddl_parser.py`)

PDDL problem and domain file generation.

**Key Functions:**

| Function | Description |
|----------|-------------|
| `write_problem_file()` | Generate PDDL problem from game state |
| `inject_domain_file()` | Inject learned collision effects into domain |
| `parse_solution_to_actions()` | Parse planner output to shooting angles |

**PDDL Structure:**
- Domain: `angry_birds_scaled`
- Actions: `pa-twang` (shooting action)
- Planner: ENHSP (Expressive Numeric Heuristic Search Planner)

## Learning System

### State Transition Learning

The agent learns state transition functions of the form:
```
x(t) = f(x(t-1), xdot(t-1), xddot(t-1))
y(t) = f(y(t-1), ydot(t-1), yddot(t-1))
xdot(t) = f(xdot(t-1), xddot(t-1))
ydot(t) = f(ydot(t-1), yddot(t-1))
xddot(t) = constant (0)
yddot(t) = constant (gravity)
```

**Learning Process:**
1. Fit polynomials to each trajectory's x(t) and y(t)
2. Compute derivatives using numerical differentiation
3. Extract state transition pairs (prev_state → curr_state)
4. Aggregate pairs from all trajectories in KB
5. Fit polynomial regression models

### Collision Effect Learning

Collision effects are learned from pre/post-collision state observations:
- Velocity components (v_x, v_y)
- Position (y)

Learned models are injected into the PDDL domain file for planning.

### Model Freezing

Learning is frozen after 8 games (`self.games_played < 8`) to stabilize the world model.

## Configuration

**Agent Parameters:**
```python
min_deg = -4          # Minimum shooting angle
max_deg = 78          # Maximum shooting angle
deg_step = 1          # Angle step size
sim_speed = 20        # Simulation speed
start_counting_from_game = 8  # Games before counting results
```

**PDDL Configuration:**
- Planner: `enhsp-20.jar`
- Timeout: 200 seconds
- Mode: SAT (satisficing)

## Usage

```python
from agents.pddl.pddl_agent import PDDLAgent

agent = PDDLAgent(
    agent_ind=0,
    agent_configs=configs,
    min_deg=-4,
    max_deg=78,
    deg_step=1,
    learn=True,
    start_counting_from_game=8
)

# Solve current level
agent.solve()
```

## Dependencies

- numpy
- scipy
- sklearn (scikit-learn)
- matplotlib
- ruptures (change point detection)
- SALib (sensitivity analysis)

## File Formats

### PDDL Problem File
```pddl
(define (problem sample_problem)
    (:domain angry_birds_scaled)
    (:objects
        bird_0 - bird
        pig_0 - pig
    )
    (:init
        (= (angle) 90)
        (= (gravity) 90)
        ...
    )
    (:goal
        (and (pig_dead pig_0))
    )
)
```

### Solution File
```
0.0: (pa-twang bird_0)  ; angle = 90 - 0 * rate = 90°
```

## Metrics Tracked

- `rmse`: RMSE between observed and estimated trajectories
- `suggested_rmse`: RMSE using learned world model
- `wins`: Boolean array of game outcomes
- `game_results`: (level, "win"/"loss") tuples
- `aggravate_score`: Cumulative score over games

## Notes

- Coordinate system: Y-axis is inverted (640 - Y for screen coordinates)
- Frame rate: 50 FPS (0.02s per frame)
- Ground level: Y ≈ 360 in game coordinates
- Visualizations are shown after game 9 (`games_played >= 9`)
