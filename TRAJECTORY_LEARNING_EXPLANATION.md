# How PDDL Agent Learns Trajectories

## Overview

The PDDL agent learns physics parameters (gravity and velocity) by analyzing the **observed trajectory** of the bird after shooting. It uses **polynomial fitting** to extract physics parameters from the actual flight path.

## Step-by-Step Learning Process

### Step 1: Collect Observed Trajectory

After shooting, the agent receives **batch ground truth** - a sequence of game states showing the bird's actual position over time:

```python
# In solve() method (line 98)
batch_gt = self.ar.shoot_and_record_ground_truth(release_point.X, release_point.Y, 0, 0, 1, 0)
```

### Step 2: Extract Real Trajectory from Ground Truth

The batch ground truth contains multiple frames. The agent extracts the bird's position from each frame:

```python
# Line 105
groundtruth_trajectories, groundtruth_objects = extract_real_trajectory(
    batch_gt, angle, self.model, self.target_class
)
```

**Result**: A numpy array of `(x, y)` positions over time:
```python
bird_observed_trajectory = [
    [x0, y0],  # Frame 0
    [x1, y1],  # Frame 1
    [x2, y2],  # Frame 2
    ...
]
```

### Step 3: Segment Trajectory by Events

The trajectory is split at collision points (ground hits, pig hits, platform collisions):

```python
# Line 110-118
event_indexes_by_event, objects_features = getSegmentsEvents(...)
event_indexes = sorted([val for values in event_indexes_by_event.values() for val in values])
parts = np.split(bird_observed_trajectory, event_indexes)
```

**Why?** The agent only learns from the **first flight segment** (before any collisions), because:
- Physics is simpler (just gravity and initial velocity)
- No collision effects to complicate the model
- Matches the theoretical projectile motion equation

```python
# Line 131 - Only use first segment
bird_observed_trajectory = parts[0]  # First segment before any collisions
```

### Step 4: Learn Physics Parameters

The core learning happens in `learn_process()`:

```python
def learn_process(self, observed_trajectory: np.ndarray):
    # Create time axis (normalized by frame rate)
    function_range = np.array(range(len(observed_trajectory))) / 50
    
    # Fit polynomials to x and y components separately
    rank_x, poly_x = get_poly_rank(function_range, observed_trajectory[:, 0])
    rank_y, poly_y = get_poly_rank(function_range, observed_trajectory[:, 1])
    
    # Extract initial velocities from polynomial derivatives
    v0_x = poly_x.deriv().coef[0]  # First derivative at t=0
    v0_y = poly_y.coef[1]           # First derivative at t=0
    
    # Extract gravity from second derivative of y-component
    gravity = abs(poly_y.deriv(2)(0))  # Second derivative at t=0
    
    # Calculate total initial velocity magnitude
    velocity = math.sqrt(v0_x ** 2 + v0_y ** 2)
    
    # Create new world model with learned parameters
    return WorldModel({
        Params.gravity: gravity,
        Params.velocity: velocity
    })
```

## The Mathematics Behind It

### Projectile Motion Equations

The theoretical trajectory of a projectile follows:

```
x(t) = x₀ + v₀ₓ·t
y(t) = y₀ + v₀ᵧ·t - ½·g·t²
```

Where:
- `v₀ₓ` = initial velocity in x-direction
- `v₀ᵧ` = initial velocity in y-direction  
- `g` = gravity acceleration
- `t` = time

### Polynomial Fitting

The agent fits polynomials to the observed trajectory:

**For X-component:**
```
x(t) ≈ a₀ + a₁·t + a₂·t² + ...
```

**For Y-component:**
```
y(t) ≈ b₀ + b₁·t + b₂·t² + ...
```

### Parameter Extraction

**Initial Velocity X:**
```python
v0_x = poly_x.deriv().coef[0]  # Coefficient of t in x(t)
```

**Initial Velocity Y:**
```python
v0_y = poly_y.coef[1]  # Coefficient of t in y(t)
```

**Gravity:**
```python
gravity = abs(poly_y.deriv(2)(0))  # Second derivative of y(t) at t=0
# This gives: d²y/dt² = -g, so g = |d²y/dt²|
```

**Total Velocity:**
```python
velocity = sqrt(v0_x² + v0_y²)
```

## Polynomial Degree Selection

The agent uses `get_poly_rank()` to find the optimal polynomial degree:

```python
def get_poly_rank(x, y, max_rank=5, threshold=1):
    # Try degrees 0 to max_rank
    resids_coeff = [get_resid(x, y, degree) for degree in range(0, max_rank)]
    
    # Calculate residual differences
    resids_diff = -np.diff(resids)
    
    # Find where improvement becomes small (< threshold)
    condition = resids_diff < threshold
    rank = np.argmax(condition) if np.any(condition) else 3
    
    return rank, polys[rank]
```

**Logic**: 
- Fit polynomials of increasing degree (0, 1, 2, 3, 4, 5)
- Calculate residual (error) for each
- Stop when adding more terms doesn't significantly reduce error
- This prevents overfitting

## Visual Example

```
Observed Trajectory (actual bird path):
    *
   * *
  *   *
 *     *
*       *

Fitted Polynomial:
x(t) = 100 + 150·t + 0·t² + ...
y(t) = 300 + 200·t - 87·t² + ...

Extracted Parameters:
- v₀ₓ = 150 (from x polynomial)
- v₀ᵧ = 200 (from y polynomial)
- g = 87 (from y polynomial second derivative)
- v = √(150² + 200²) = 250
```

## Why This Works

1. **Projectile motion is polynomial**: The physics equations are inherently polynomial
2. **First segment is clean**: Before collisions, only gravity affects the bird
3. **Derivatives give physics**: Mathematical derivatives directly correspond to velocity and acceleration
4. **Adaptive fitting**: Polynomial degree selection prevents overfitting to noise

## Learning Loop

```
1. Shoot bird at angle θ
2. Observe actual trajectory: [(x₀,y₀), (x₁,y₁), ..., (xₙ,yₙ)]
3. Fit polynomials: x(t) and y(t)
4. Extract: gravity, velocity
5. Update world model
6. Use updated model for next shot planning
```

## Comparison with Estimated Trajectory

After learning, the agent compares:

```python
# Trajectory using OLD parameters
estimated_trajectory = construct_trajectory(..., self.world_model, ...)

# Trajectory using NEW learned parameters  
suggested_trajectory = construct_trajectory(..., new_world_model, ...)

# Calculate error
old_rmse = calculate_rmse(observed, estimated)
new_rmse = calculate_rmse(observed, suggested)
```

The new parameters should have **lower RMSE**, meaning they better match reality.

## Key Code Locations

- **Learning entry point**: `pddl_agent.py:133` - `learn_process()`
- **Polynomial fitting**: `optimizer.py:59` - `get_poly_rank()`
- **Trajectory extraction**: `trajectory_parser.py:39` - `extract_real_trajectory()`
- **Trajectory construction**: `trajectory_parser.py:48` - `construct_trajectory()`
- **Event segmentation**: `segments.py:58` - `getSegmentsEvents()`

## Limitations

1. **Only learns from first segment**: Ignores trajectory after collisions
2. **Assumes constant gravity**: Doesn't account for gravity variations
3. **No air resistance**: Assumes ideal projectile motion
4. **Polynomial degree limit**: Max degree 5 may not capture all dynamics
5. **Time normalization**: Uses fixed frame rate (50 fps) assumption

## Future Improvements

1. Learn from multiple segments (with collision corrections)
2. Account for air resistance
3. Learn gravity variations
4. Use more sophisticated curve fitting (splines, neural networks)
5. Learn bird-specific parameters (different birds have different physics)

