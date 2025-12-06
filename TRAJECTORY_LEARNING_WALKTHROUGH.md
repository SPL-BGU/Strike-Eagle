# Trajectory Learning - Code Walkthrough

## Complete Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ 1. SHOOT BIRD                                               │
│    - Agent plans shot using PDDL                            │
│    - Executes: shoot_and_record_ground_truth()              │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. GET BATCH GROUND TRUTH                                    │
│    batch_gt = [frame0, frame1, frame2, ..., frameN]        │
│    Each frame contains all game objects at that moment      │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. EXTRACT TRAJECTORY                                        │
│    extract_real_trajectory(batch_gt, ...)                   │
│    → Parses each frame                                       │
│    → Finds bird position in each frame                       │
│    → Returns: [(x₀,y₀), (x₁,y₁), ..., (xₙ,yₙ)]              │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. DETECT EVENTS                                             │
│    getSegmentsEvents(trajectory)                             │
│    → Detects: ground_collision, hit, platform_collision    │
│    → Returns: event_indexes = [collision_frame_1, ...]       │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. SEGMENT TRAJECTORY                                        │
│    parts = np.split(trajectory, event_indexes)               │
│    → parts[0] = trajectory BEFORE first collision            │
│    → parts[1] = trajectory AFTER first collision             │
│    → Only use parts[0] for learning!                        │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. LEARN PHYSICS (learn_process)                            │
│    ┌──────────────────────────────────────────────────┐   │
│    │ 6a. Create time axis                               │   │
│    │     t = [0, 1/50, 2/50, 3/50, ...]                │   │
│    │     (assuming 50 fps)                              │   │
│    └──────────────────────────────────────────────────┘   │
│    ┌──────────────────────────────────────────────────┐   │
│    │ 6b. Fit polynomial to X positions                  │   │
│    │     poly_x = fit_polynomial(t, x_positions)        │   │
│    │     → x(t) = a₀ + a₁·t + a₂·t² + ...              │   │
│    └──────────────────────────────────────────────────┘   │
│    ┌──────────────────────────────────────────────────┐   │
│    │ 6c. Fit polynomial to Y positions                  │   │
│    │     poly_y = fit_polynomial(t, y_positions)       │   │
│    │     → y(t) = b₀ + b₁·t + b₂·t² + ...              │   │
│    └──────────────────────────────────────────────────┘   │
│    ┌──────────────────────────────────────────────────┐   │
│    │ 6d. Extract velocity from derivatives              │   │
│    │     v₀ₓ = poly_x.deriv().coef[0]  (dx/dt at t=0)│   │
│    │     v₀ᵧ = poly_y.coef[1]           (dy/dt at t=0)│   │
│    │     v = √(v₀ₓ² + v₀ᵧ²)                           │   │
│    └──────────────────────────────────────────────────┘   │
│    ┌──────────────────────────────────────────────────┐   │
│    │ 6e. Extract gravity from second derivative        │   │
│    │     g = |poly_y.deriv(2)(0)|  (d²y/dt² at t=0)    │   │
│    │     (Note: d²y/dt² = -g, so g = |d²y/dt²|)       │   │
│    └──────────────────────────────────────────────────┘   │
│    ┌──────────────────────────────────────────────────┐   │
│    │ 6f. Create new WorldModel                          │   │
│    │     return WorldModel({gravity: g, velocity: v}) │   │
│    └──────────────────────────────────────────────────┘   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 7. UPDATE WORLD MODEL                                        │
│    self.world_model = new_world_model                       │
│    → Next shot will use these learned parameters             │
└─────────────────────────────────────────────────────────────┘
```

## Detailed Code Flow

### Step 1: Shooting and Data Collection

```python
# pddl_agent.py:98
batch_gt = self.ar.shoot_and_record_ground_truth(
    release_point.X, 
    release_point.Y, 
    0,      # release time
    0,      # tap time
    1,      # ground truth frequency (every frame)
    0       # batch option (record all)
)
```

**What happens**: Game records ground truth data every frame during the shot.

### Step 2: Extract Trajectory

```python
# pddl_agent.py:105
groundtruth_trajectories, groundtruth_objects = extract_real_trajectory(
    batch_gt, angle, self.model, self.target_class
)
```

**Inside `extract_real_trajectory()`**:
```python
# trajectory_parser.py:39-45
def extract_real_trajectory(raw_trajectory, ...):
    # Parse each frame in batch_gt
    combined_trajectory = [
        GroundTruthReader(shoot, model, target_class).allObj 
        for shoot in raw_trajectory
    ]
    
    # Extract bird position from each frame
    for temporal_state in combined_trajectory:
        for object_type, object_list in temporal_state.items():
            if object_type == "redBird":
                entity_trajectories["redBird_0"].append(
                    [entity.X, 640 - entity.Y]  # Convert coordinates
                )
    
    return entity_trajectories, entity_dimensions
```

**Result**: 
```python
groundtruth_trajectories = {
    "redBird_0": [
        [100, 300],  # Frame 0
        [105, 295],  # Frame 1
        [110, 288],  # Frame 2
        ...
    ]
}
```

### Step 3: Event Detection

```python
# pddl_agent.py:110
event_indexes_by_event, objects_features = getSegmentsEvents(
    groundtruth_trajectories, 
    groundtruth_objects
)
```

**Inside `getSegmentsEvents()`**:
```python
# segments.py:58-80
def getSegmentsEvents(groundtruth_trajectories, groundtruth_objects):
    # Calculate features for each frame
    objects_features = {}
    for object, traj in groundtruth_trajectories.items():
        objects_features[object] = calculate_features(np.stack(traj))
        # Features: x, y, v_x, v_y, a_x, a_y
    
    # Check for events at each frame
    event_indexes = check_events(
        objects_features, 
        groundtruth_objects,
        [
            {"name": "ground_collision", "func": is_ground_collision},
            {"name": "hit", "func": is_hit},
            {"name": "platform_collision", "func": is_platform_collision}
        ]
    )
    
    return event_indexes, objects_features
```

**Event Detection Example**:
```python
# segments.py:4-9 (is_ground_collision)
def is_ground_collision(frames, groundtruth_objects, i):
    epsilon = 3
    if not "redBird_0" in frames[i]:
        return False
    # Check if bird crossed ground threshold
    return (i > 0 and 
            frames[i-1]["redBird_0"]['y'] <= epsilon and 
            frames[i]["redBird_0"]['y'] > epsilon)
```

**Result**:
```python
event_indexes_by_event = {
    "ground_collision": [45, 78],  # Collisions at frames 45 and 78
    "hit": [23],                    # Hit at frame 23
    "platform_collision": []
}
```

### Step 4: Segment Trajectory

```python
# pddl_agent.py:116-118
event_indexes = sorted([val for values in event_indexes_by_event.values() 
                        for val in values])
parts = np.split(bird_observed_trajectory, event_indexes)
```

**Example**:
```python
# If trajectory has 100 points and collision at frame 45:
parts[0] = trajectory[0:45]    # First segment (before collision)
parts[1] = trajectory[45:100]  # Second segment (after collision)

# Only use parts[0] for learning
bird_observed_trajectory = parts[0]
```

### Step 5: Learn Physics Parameters

```python
# pddl_agent.py:133
new_world_model = self.learn_process(bird_observed_trajectory)
```

**Inside `learn_process()`**:

```python
# pddl_agent.py:217-240
def learn_process(self, observed_trajectory: np.ndarray):
    # Step 5a: Create time axis
    # Assuming 50 fps: each frame = 1/50 seconds
    function_range = np.array(range(len(observed_trajectory))) / 50
    # Result: [0, 0.02, 0.04, 0.06, ...]
    
    # Step 5b: Fit polynomial to X positions
    rank_x, poly_x = get_poly_rank(function_range, observed_trajectory[:, 0])
    # poly_x is a Polynomial object: x(t) = a₀ + a₁·t + a₂·t² + ...
    
    # Step 5c: Fit polynomial to Y positions  
    rank_y, poly_y = get_poly_rank(function_range, observed_trajectory[:, 1])
    # poly_y is a Polynomial object: y(t) = b₀ + b₁·t + b₂·t² + ...
    
    # Step 5d: Extract initial velocity X
    v0_x = poly_x.deriv().coef[0]
    # poly_x.deriv() = dx/dt = a₁ + 2·a₂·t + ...
    # At t=0: v₀ₓ = a₁
    
    # Step 5e: Extract initial velocity Y
    v0_y = poly_y.coef[1]
    # poly_y = b₀ + b₁·t + b₂·t² + ...
    # At t=0: v₀ᵧ = b₁
    
    # Step 5f: Extract gravity
    gravity = abs(poly_y.deriv(2)(0))
    # poly_y.deriv(2) = d²y/dt² = 2·b₂ + ...
    # At t=0: d²y/dt² = 2·b₂ = -g
    # So: g = |2·b₂|
    
    # Step 5g: Calculate total velocity
    velocity = math.sqrt(v0_x ** 2 + v0_y ** 2)
    
    # Step 5h: Create new world model
    return WorldModel({
        Params.gravity: gravity,
        Params.velocity: velocity
    })
```

### Step 6: Polynomial Fitting Details

**Inside `get_poly_rank()`**:

```python
# optimizer.py:59-70
def get_poly_rank(x, y, max_rank=5, threshold=1):
    # Try different polynomial degrees
    resids_coeff = []
    for degree in range(0, max_rank):  # 0, 1, 2, 3, 4
        resid, poly = get_resid(x, y, degree)
        resids_coeff.append((resid, poly))
    
    # Extract residuals
    resids = [r[0] for r in resids_coeff]
    polys = [r[1] for r in resids_coeff]
    
    # Calculate improvement from each degree
    resids_diff = -np.diff(resids)
    # resids_diff[i] = improvement from degree i to i+1
    
    # Find where improvement becomes small
    condition = resids_diff < threshold
    # Stop when adding more terms doesn't help much
    
    rank = np.argmax(condition) if np.any(condition) else 3
    return rank, polys[rank]
```

**Example**:
```python
# Try fitting trajectory with different degrees:
degree 0: residual = 1000  (constant: y = c)
degree 1: residual = 500   (linear: y = a + b·t)    improvement = 500
degree 2: residual = 50   (quadratic: y = a + b·t + c·t²)  improvement = 450
degree 3: residual = 48   (cubic)                  improvement = 2
degree 4: residual = 47.5 (quartic)                improvement = 0.5

# If threshold = 1:
# resids_diff = [500, 450, 2, 0.5]
# condition = [False, False, False, True]  # 0.5 < 1
# rank = 3 (first True)

# Use degree 3 polynomial
```

### Step 7: Update World Model

```python
# pddl_agent.py:153-157
print(f"Old values- {self.world_model.hyperparams_values}")
print(f"New values- gravity: {new_world_model.hyperparams_values}")

# Update world model
self.world_model = new_world_model
self.world_model.kb = self.kb  # Keep knowledge base
```

**Example Output**:
```
Old values- {Params.gravity: 90, Params.velocity: 200}
New values- gravity: {Params.gravity: 87.2, Params.velocity: 195.5}
```

## Concrete Example

### Input: Observed Trajectory
```python
observed_trajectory = np.array([
    [100, 300],  # t=0.00s
    [105, 295],  # t=0.02s
    [110, 288],  # t=0.04s
    [115, 279],  # t=0.06s
    [120, 268],  # t=0.08s
    ...
])
```

### Step 1: Create Time Axis
```python
function_range = [0, 0.02, 0.04, 0.06, 0.08, ...]
```

### Step 2: Fit Polynomials
```python
# X-component
poly_x = Polynomial.fit(function_range, [100, 105, 110, 115, 120, ...])
# Result: x(t) ≈ 100 + 250·t + 0·t²

# Y-component  
poly_y = Polynomial.fit(function_range, [300, 295, 288, 279, 268, ...])
# Result: y(t) ≈ 300 + 200·t - 87·t²
```

### Step 3: Extract Parameters
```python
# Velocity X
v0_x = poly_x.deriv().coef[0]  # = 250

# Velocity Y
v0_y = poly_y.coef[1]  # = 200

# Gravity
gravity = abs(poly_y.deriv(2)(0))  # = |2 × (-87/2)| = 87

# Total velocity
velocity = sqrt(250² + 200²) = 320.16
```

### Step 4: Update Model
```python
new_world_model = WorldModel({
    Params.gravity: 87,
    Params.velocity: 320.16
})
```

## Why This Works

1. **Physics is polynomial**: Projectile motion equations are inherently polynomial
2. **Derivatives = physics**: 
   - First derivative = velocity
   - Second derivative = acceleration (gravity)
3. **First segment is clean**: Before collisions, only gravity affects motion
4. **Adaptive fitting**: Polynomial degree selection prevents overfitting

## Validation

After learning, the agent validates by comparing:

```python
# pddl_agent.py:138-142
estimated_trajectory = construct_trajectory(
    bird_observed_trajectory[0], 
    angle, 
    self.world_model,  # OLD parameters
    limit
)

suggested_trajectory = construct_trajectory(
    bird_observed_trajectory[0], 
    angle, 
    new_world_model,  # NEW learned parameters
    limit
)

# Calculate errors
old_rmse = calculate_rmse(observed, estimated)      # Should be higher
new_rmse = calculate_rmse(observed, suggested)    # Should be lower
```

The new parameters should produce a trajectory that **better matches** the observed one!

