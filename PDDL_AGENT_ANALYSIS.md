# PDDL Agent - Comprehensive Analysis

## Overview

The **PDDLAgent** is a planning-based AI agent that uses **PDDL (Planning Domain Definition Language)** to solve Science Birds (Angry Birds-style) levels. It combines symbolic planning with machine learning to adapt its physics model based on observed trajectories.

## Architecture

### Core Components

1. **PDDL Planning System**
   - Uses ENHSP (Expressive Numeric Heuristic Search Planner) solver
   - Domain file: `base_domain.pddl` (defines actions, processes, predicates)
   - Problem file: `problem.pddl` (generated dynamically for each level)
   - Solution file: `solution.pddl` (contains planned actions)

2. **World Model** (`world_model/world_model.py`)
   - Physics simulation parameters:
     - `gravity`: Default 87.2 (learned/updated)
     - `velocity`: Default 175.9259 (learned/updated)
   - Uses Taylor series approximations for sin/cos calculations
   - Maintains knowledge base (KB) for learned collision effects

3. **Trajectory Analysis**
   - Extracts real trajectories from batch ground truth
   - Segments trajectories by events (collisions, hits, platform collisions)
   - Learns physics parameters from observed flight paths

4. **Event Learning** (`events/learn_events.py`)
   - Learns collision effects using linear regression
   - Updates knowledge base with learned models
   - Injects learned effects into PDDL domain file

## Workflow

### Main Solve Loop (`solve()` method)

```
1. Get current game state (vision/ground truth)
2. Get action to perform (PDDL planning)
3. Execute shot and record batch ground truth
4. Analyze observed trajectory
5. Learn from trajectory:
   - Learn physics parameters (gravity, velocity)
   - Learn collision effects
6. Update world model
7. Calculate metrics (RMSE, wins)
```

### Action Planning (`get_action_to_perform()`)

1. **Extract Game Objects**:
   - Birds (position, type, mass, radius, velocity)
   - Pigs (position, mass, radius, life)
   - Blocks (wood/ice/stone/TNT - position, life, mass, stability)
   - Platforms (position, dimensions)

2. **Generate PDDL Problem**:
   - Convert game objects to PDDL predicates
   - Set initial state (positions, velocities, angles)
   - Set goal (all pigs dead: `(pig_dead ?p)`)

3. **Run PDDL Planner**:
   - Calls ENHSP solver: `java -jar enhsp-20.jar`
   - Uses SAT planner mode
   - Timeout: 200 seconds

4. **Parse Solution**:
   - Extracts shooting actions (angle adjustments)
   - Returns list of `(action, angle)` tuples

### Learning Process (`learn_process()`)

1. **Polynomial Fitting**:
   - Fits polynomials to observed trajectory (x and y separately)
   - Determines optimal polynomial degree using residual analysis

2. **Parameter Extraction**:
   - Extracts initial velocity from polynomial derivatives
   - Extracts gravity from second derivative of y-polynomial

3. **World Model Update**:
   - Creates new WorldModel with learned parameters
   - Updates agent's world model for next planning cycle

### Event Detection & Learning

**Event Types Detected**:
- `ground_collision`: Bird hits the ground
- `hit`: Bird collides with pig
- `platform_collision`: Bird hits platform

**Event Learning Process**:
1. Segment trajectory by detected events
2. For each collision event:
   - Extract pre-event state (x, y, v_x, v_y)
   - Extract post-event state
   - Train linear regression model: `post_state = f(pre_state)`
3. Store learned models in knowledge base
4. Inject learned effects into PDDL domain file

**Knowledge Base Structure**:
```python
kb = {
    "collision": {
        "states": [pre_event_states...],
        "variables": {
            "v_x": {"value": [...], "model": LinearRegression()},
            "v_y": {"value": [...], "model": LinearRegression()},
            "y": {"value": [...], "model": LinearRegression()}
        }
    }
}
```

## Key Files

### Core Agent
- `pddl_agent.py` - Main agent class
- `pddl_parser.py` - PDDL file generation and parsing
- `pddl_objects.py` - Game object extraction and conversion

### World Model
- `world_model/world_model.py` - Physics model with parameters
- `world_model/params.py` - Parameter definitions (gravity, velocity, etc.)
- `world_model/process.py` - Process definitions (flight)

### Learning & Analysis
- `trajectory_parser.py` - Extract and construct trajectories
- `segments.py` - Event detection and trajectory segmentation
- `events/learn_events.py` - Collision effect learning
- `events/event_conditions.py` - Event detection conditions
- `optimizer.py` - Parameter optimization utilities
- `metrics.py` - RMSE calculation for trajectory comparison

### PDDL Files
- `base_domain.pddl` - PDDL domain definition (actions, processes, predicates)
- `base_domain_modified.pddl` - Domain with learned effects injected
- `problem.pddl` - Generated problem file for each level
- `solution.pddl` - Planner output

## Configuration

### Initialization Parameters
```python
PDDLAgent(
    agent_ind,           # Agent identifier
    agent_configs,       # Connection configs
    min_deg=-4,          # Minimum shooting angle
    max_deg=78,          # Maximum shooting angle
    deg_step=1,          # Angle increment step
    learn=True           # Enable learning mode
)
```

### World Model Defaults
- `gravity`: 90 (updated through learning)
- `velocity`: 200 (updated through learning)
- `sim_speed`: 20x (simulation speed multiplier)

## Learning Mechanism

### Physics Parameter Learning
1. Observe bird trajectory after shot
2. Fit polynomial to trajectory (x and y components)
3. Extract:
   - Initial velocity from first derivative
   - Gravity from second derivative of y-component
4. Update world model parameters

### Collision Effect Learning
1. Detect collision events in trajectory
2. For each collision:
   - Record pre-collision state (x, y, v_x, v_y)
   - Record post-collision state
3. Train linear regression: `post_var = f(pre_state)`
4. Store model in knowledge base
5. Inject learned equations into PDDL domain

**Example Learned Effect**:
```
v_x_after = bias + c1*x + c2*y + c3*v_x + c4*v_y
```

## Metrics & Evaluation

### Tracked Metrics
- `rmse`: Root Mean Squared Error (observed vs estimated trajectory)
- `suggested_rmse`: RMSE with learned parameters
- `wins`: List of win/loss outcomes
- `aggravate_score`: Cumulative score across levels

### Visualization (commented out)
- `visualize_compare()`: Compare trajectories
- `visualize_rmse()`: Plot RMSE over time
- `visualize_rmse_vs_suggsted()`: Compare old vs new RMSE
- `visuallize_wins_percentage()`: Win rate visualization

## Strengths

1. **Adaptive Physics Model**: Learns gravity and velocity from observations
2. **Event-Based Learning**: Learns collision effects automatically
3. **Symbolic Planning**: Uses PDDL for structured action planning
4. **Continuous Improvement**: Updates world model after each shot

## Limitations

1. **Planning Timeout**: 200-second timeout may be insufficient for complex levels
2. **Fallback Strategy**: Falls back to 45-degree shot if planning fails
3. **Limited Event Types**: Only learns ground collisions currently
4. **Knowledge Base Size**: Limited to 3 entries (`kb_max_size = 3`)

## Dependencies

- **ENHSP**: Java-based PDDL planner (`enhsp-20.jar`)
- **NumPy**: Numerical computations
- **Scikit-learn**: Linear regression for learning
- **Ruptures**: Changepoint detection (PELT algorithm)
- **Scipy**: Polynomial fitting

## Current Status

- **Active Learning**: Enabled (`self.learn = True`)
- **Default Agent**: Currently set in `main.py` (line 36)
- **Visualization**: Disabled (`self.visualize = False`)
- **Simulation Speed**: 20x normal speed

## Future Improvements

1. Learn more event types (block collisions, TNT explosions)
2. Increase knowledge base capacity
3. Add parameter sensitivity analysis
4. Implement grid search for parameter optimization
5. Add visualization tools for debugging
6. Improve fallback strategies when planning fails

