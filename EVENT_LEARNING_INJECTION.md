# Event Learning and PDDL Domain Injection

## Overview

The PDDL agent learns collision effects by observing real trajectories and automatically injects learned physics equations into the PDDL domain file. This allows the planner to use learned physics models instead of hardcoded values.

## Event Learning Process

### Step 1: Event Detection

After executing a shot, the agent segments the trajectory by collision events:

```python
# In solve() method (line 121)
event_indexes_by_event, objects_features = getSegmentsEvents(
    groundtruth_trajectories, groundtruth_objects
)

collisions = event_indexes_by_event["ground_collision"]
```

**Event Types Detected:**
- `ground_collision`: Bird hits the ground
- `hit`: Bird collides with pig
- `platform_collision`: Bird hits platform

### Step 2: Learning from Collisions

For each detected collision, the agent learns how the collision affects the bird's state:

```python
# Lines 137-139 in pddl_agent.py
for collision_index in collisions:
    update_model_effects(
        "collision", 
        self.kb, 
        bird_observed_features[collision_index],      # Pre-collision state
        bird_observed_features[collision_index + 1]   # Post-collision state
    )
```

**What Gets Learned:**
- Pre-collision state: `{x, y, v_x, v_y}` (position and velocity before collision)
- Post-collision state: `{x, y, v_x, v_y}` (position and velocity after collision)
- Linear regression models: `post_variable = f(pre_state)`

### Step 3: Model Training

For each affected variable (`v_x`, `v_y`, `y`), a linear regression model is trained:

```python
# In learn_events.py:update_model_effects()
# Features: [x_bird, y_bird, vx_bird, vy_bird]
# Target: post-collision value of the variable

states = make_feature_vector(kb[event_name]["states"])  # Pre-collision states
y = np.array(variable["value"])                         # Post-collision values
variable["model"] = polyfit_var(states, y)              # Train LinearRegression
```

**Model Format:**
```
post_variable = intercept + coef_x * x_bird + coef_y * y_bird + 
                coef_vx * vx_bird + coef_vy * vy_bird
```

### Step 4: Knowledge Base Storage

Learned models are stored in the knowledge base:

```python
kb = {
    "collision": {
        "states": [pre_event_state_1, pre_event_state_2, ...],
        "variables": {
            "v_x": {
                "value": [post_vx_1, post_vx_2, ...],
                "model": LinearRegression()  # Trained model
            },
            "v_y": {
                "value": [post_vy_1, post_vy_2, ...],
                "model": LinearRegression()  # Trained model
            },
            "y": {
                "value": [post_y_1, post_y_2, ...],
                "model": LinearRegression()  # Trained model
            }
        }
    }
}
```

## PDDL Domain Injection

### Step 1: Placeholder Detection

The base PDDL domain file (`base_domain.pddl`) contains placeholders that need to be replaced:

```pddl
(:event collision_ground
    :parameters (?b - bird)
    :precondition (and
        (= (active_bird) (bird_id ?b))
        (bird_released ?b)
        (<= (y_bird ?b) 0)
        (> (vx_bird ?b) 0)
    )
    :effect (and
        (assign (y_bird ?b) {SE-collision-y})
        (assign (vy_bird ?b) {SE-collision-v_y})
        (assign (vx_bird ?b) {SE-collision-v_x})
        (assign (bounce_count ?b) (+ (bounce_count ?b) 1))
    )
)
```

**Placeholders:**
- `{SE-collision-y}` → Post-collision y position
- `{SE-collision-v_y}` → Post-collision vertical velocity
- `{SE-collision-v_x}` → Post-collision horizontal velocity

### Step 2: Equation Generation

When a model is available, the injection function generates a PDDL-compatible equation:

```python
# In pddl_parser.py:inject_domain_file()

for variable_name, variable_data in world_model.kb["collision"]["variables"].items():
    placeholder = "{{SE-collision-{}}}".format(variable_name)
    
    if variable_data["model"] == None:
        # No model available - inject 0.0
        new_content = new_content.replace(placeholder, "0.0")
        continue
    
    # Extract model coefficients
    coefs = variable_data["model"].coef_      # [coef_x, coef_y, coef_vx, coef_vy]
    bias = variable_data["model"].intercept_  # Intercept term
    vars = ['x_bird', 'y_bird', 'vx_bird', 'vy_bird']
    
    # Create PDDL terms
    terms = [
        f"(* {c:.4f} ({v} ?b))"
        for c, v in zip(coefs, vars)
        if abs(c) >= threshold  # Skip negligible coefficients
    ]
    
    # Nest terms: (+ term1 (+ term2 (+ term3 term4)))
    nested_terms = nest_terms(terms)
    
    # Add bias if significant
    if abs(bias) > threshold:
        equation = f"(+ {bias:.4f} {nested_terms})"
    else:
        equation = nested_terms
    
    # Replace placeholder
    new_content = new_content.replace(placeholder, equation)
```

### Step 3: Fallback for Missing Models

If no models are available (no collisions observed yet), placeholders are replaced with `0.0`:

```python
# Replace any remaining placeholders (not in KB) with 0 as fallback
remaining_placeholders = re.findall(r'\{\{SE-collision-[^}]+\}\}', new_content)
for placeholder in remaining_placeholders:
    new_content = new_content.replace(placeholder, "0.0")
```

### Step 4: Save Modified Domain

The modified domain is saved as `base_domain_modified.pddl`:

```python
output_path = os.path.join(base_dir, f"{base_name}_modified.pddl")
with open(output_path, "w") as file:
    file.write(new_content)
```

## Example Injections

### Example 1: With Learned Models

**Before Injection (base_domain.pddl):**
```pddl
:effect (and
    (assign (y_bird ?b) {SE-collision-y})
    (assign (vy_bird ?b) {SE-collision-v_y})
    (assign (vx_bird ?b) {SE-collision-v_x})
)
```

**After Injection (base_domain_modified.pddl):**
```pddl
:effect (and
    (assign (y_bird ?b) (+ 4.0000 (* 20.0000 (vy_bird ?b))))
    (assign (vy_bird ?b) (+ -0.1350 (+ (* 0.0003 (x_bird ?b)) (+ (* -0.0552 (vx_bird ?b)) (* 1.2671 (vy_bird ?b))))))
    (assign (vx_bird ?b) (+ 0.0179 (+ (* -0.0000 (x_bird ?b)) (+ (* 0.5688 (vx_bird ?b)) (* -0.5447 (vy_bird ?b))))))
)
```

### Example 2: No Models Available

**After Injection (when no collisions observed):**
```pddl
:effect (and
    (assign (y_bird ?b) 0.0)
    (assign (vy_bird ?b) 0.0)
    (assign (vx_bird ?b) 0.0)
)
```

## Code Flow

```
1. solve() executes shot
   ↓
2. Extract trajectory and detect events
   ↓
3. For each collision:
   update_model_effects() → Train LinearRegression models
   ↓
4. Store models in self.kb
   ↓
5. Assign KB to world model:
   self.world_model.kb = self.kb
   ↓
6. get_action_to_perform() called with world_model
   ↓
7. Check if KB exists:
   if agent_world_model.kb != None:
       inject_domain_file() → Generate equations and replace placeholders
   ↓
8. Use modified domain file for PDDL planning:
   domain_path = 'base_domain_modified.pddl'
```

## Key Functions

### `update_model_effects()` (learn_events.py)
- **Purpose**: Train linear regression models from collision observations
- **Input**: Pre-collision state, post-collision state
- **Output**: Updates KB with trained models
- **Algorithm**: Uses `PolynomialFeatures(degree=1)` + `LinearRegression`

### `inject_domain_file()` (pddl_parser.py)
- **Purpose**: Inject learned equations into PDDL domain file
- **Input**: Path to base domain, WorldModel with KB
- **Output**: Creates `base_domain_modified.pddl` with injected equations
- **Handles**: Missing models (injects `0.0`), coefficient thresholding, PDDL syntax generation

### `nest_terms()` (pddl_parser.py)
- **Purpose**: Convert list of PDDL terms into nested addition expression
- **Input**: List of terms like `["(* 1.5 (x_bird ?b))", "(* 2.0 (y_bird ?b))"]`
- **Output**: Nested PDDL expression: `(+ (* 1.5 (x_bird ?b)) (* 2.0 (y_bird ?b)))`

## Configuration

### Knowledge Base Initialization

```python
self.kb = {
    "collision": {
        "states": [],
        "variables": {
            "v_x": {"value": [], "model": None},
            "v_y": {"value": [], "model": None},
            "y": {"value": [], "model": None}
        }
    }
}
self.kb_max_size = 3  # Maximum number of collision states stored
```

### Injection Thresholds

```python
threshold = 1e-8  # Coefficients below this are considered zero and skipped
```

## Limitations

1. **Limited Event Types**: Currently only learns `ground_collision` effects
2. **Knowledge Base Size**: Limited to 3 stored states (`kb_max_size = 3`)
3. **Linear Models Only**: Uses linear regression (degree 1), may not capture non-linear effects
4. **No Model Validation**: Doesn't check model quality before injection
5. **Single Collision Type**: Only one collision type ("collision") is supported

## When Injection Happens

- **Enabled**: Always (if `kb != None`, which is always true after initialization)
- **Triggered**: Every time `get_action_to_perform()` is called
- **Result**: Overwrites `base_domain_modified.pddl` with latest learned models

## Error Handling

### Missing Models
- If `model == None`: Placeholder replaced with `0.0`
- This ensures valid PDDL syntax even with no learning data

### Missing Placeholders
- If placeholder not found in domain file: No replacement (original placeholder remains)
- Regex fallback catches any remaining placeholders after main loop

### Invalid Models
- No explicit validation of model coefficients
- If model predicts invalid values (NaN, Inf), they will be injected (may cause planner errors)

## Regularization in Event Learning

The event learning system uses regression to learn how collisions affect the bird's state. Below are the 4 versions of the loss functions with different regularization strategies:

### 1. Without Regularization (Ordinary Least Squares)

**Loss Function:**

$$\mathcal{L}_{OLS} = \sum_{i=1}^{n} \left( y_i - \hat{y}_i \right)^2 = \sum_{i=1}^{n} \left( y_i - \left( \beta_0 + \sum_{j=1}^{p} \beta_j x_{ij} \right) \right)^2$$

**Expanded Form for Collision Learning:**

$$\mathcal{L}_{OLS} = \sum_{i=1}^{n} \left( v_{y,\text{after}}^{(i)} - \left( \beta_0 + \beta_1 x^{(i)} + \beta_2 y^{(i)} + \beta_3 v_x^{(i)} + \beta_4 v_y^{(i)} \right) \right)^2$$

**Properties:**
- No penalty on coefficient magnitude
- Prone to overfitting with small samples
- All features get non-zero coefficients
- No feature selection

---

### 2. With L1 Regularization (Lasso)

**Loss Function:**

$$\mathcal{L}_{L1} = \sum_{i=1}^{n} \left( y_i - \hat{y}_i \right)^2 + \alpha \sum_{j=1}^{p} |\beta_j|$$

**Expanded Form:**

$$\mathcal{L}_{L1} = \sum_{i=1}^{n} \left( v_{y,\text{after}}^{(i)} - \left( \beta_0 + \beta_1 x^{(i)} + \beta_2 y^{(i)} + \beta_3 v_x^{(i)} + \beta_4 v_y^{(i)} \right) \right)^2 + \alpha \left( |\beta_1| + |\beta_2| + |\beta_3| + |\beta_4| \right)$$

**Properties:**
- L1 penalty drives irrelevant coefficients to **exactly zero**
- Automatic **feature selection** (sparse solutions)
- Useful when only a few features matter (e.g., only $v_y$ affects $v_{y,\text{after}}$)
- Results in simpler, more interpretable equations

---

### 3. With L2 Regularization (Ridge)

**Loss Function:**

$$\mathcal{L}_{L2} = \sum_{i=1}^{n} \left( y_i - \hat{y}_i \right)^2 + \alpha \sum_{j=1}^{p} \beta_j^2$$

**Expanded Form:**

$$\mathcal{L}_{L2} = \sum_{i=1}^{n} \left( v_{y,\text{after}}^{(i)} - \left( \beta_0 + \beta_1 x^{(i)} + \beta_2 y^{(i)} + \beta_3 v_x^{(i)} + \beta_4 v_y^{(i)} \right) \right)^2 + \alpha \left( \beta_1^2 + \beta_2^2 + \beta_3^2 + \beta_4^2 \right)$$

**Properties:**
- L2 penalty **shrinks** all coefficients toward zero
- Coefficients remain non-zero but small
- Prevents large coefficient values (stabilizes training)
- Better when all features contribute somewhat

---

### 4. With Both L1 + L2 Regularization (ElasticNet)

**Loss Function:**

$$\mathcal{L}_{ElasticNet} = \sum_{i=1}^{n} \left( y_i - \hat{y}_i \right)^2 + \alpha \left[ \rho \sum_{j=1}^{p} |\beta_j| + \frac{(1-\rho)}{2} \sum_{j=1}^{p} \beta_j^2 \right]$$

**Expanded Form:**

$$\mathcal{L}_{ElasticNet} = \sum_{i=1}^{n} \left( v_{y,\text{after}}^{(i)} - \left( \beta_0 + \beta_1 x^{(i)} + \beta_2 y^{(i)} + \beta_3 v_x^{(i)} + \beta_4 v_y^{(i)} \right) \right)^2 + \alpha \cdot \rho \left( |\beta_1| + |\beta_2| + |\beta_3| + |\beta_4| \right) + \frac{\alpha (1-\rho)}{2} \left( \beta_1^2 + \beta_2^2 + \beta_3^2 + \beta_4^2 \right)$$

**Properties:**
- Combines benefits of L1 (sparsity) and L2 (stability)
- $\rho$ controls the mix: $\rho=1$ is Lasso, $\rho=0$ is Ridge
- **Currently used in the codebase** with `l1_ratio=0.5` (balanced)
- Best for small sample sizes with correlated features

---

### Parameter Definitions

| Symbol | Description |
|--------|-------------|
| $y_i$ | Actual post-collision value (e.g., $v_{y,\text{after}}$) |
| $\hat{y}_i$ | Predicted post-collision value |
| $\beta_0$ | Intercept (bias) term |
| $\beta_j$ | Coefficient for feature $j$ |
| $x^{(i)}$ | Bird x-position before collision |
| $y^{(i)}$ | Bird y-position before collision |
| $v_x^{(i)}$ | Bird horizontal velocity before collision |
| $v_y^{(i)}$ | Bird vertical velocity before collision |
| $\alpha$ | Overall regularization strength |
| $\rho$ | L1 ratio (mix between L1 and L2) |
| $n$ | Number of training samples (collisions observed) |
| $p$ | Number of features (4: x, y, v_x, v_y) |

---

### Code Implementation

The `EventModelManager` class in `learn_events.py` implements ElasticNet:

```python
# In _train_general_model() at line 386
model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)
model.fit(X_poly, y)
```

**Default Parameters:**
- `alpha = 1.0` - Regularization strength
- `l1_ratio = 0.5` - Balanced between L1 and L2

---

### Why ElasticNet for Collision Learning?

1. **Sparse Features**: Position (x, y) typically has zero or near-zero effect on velocity change
2. **Small Samples**: Only a few collisions are observed, regularization prevents overfitting
3. **Correlated Features**: v_x and v_y are often correlated in collision trajectories
4. **Interpretable Results**: L1 component zeros out irrelevant features, leaving clean physics equations

**Expected Learned Equation (with ElasticNet):**
```
v_y_after = β₀ + β₄ * v_y_before
```
Where β₁, β₂, β₃ ≈ 0 (position and v_x don't affect v_y bounce)

---

## Future Improvements

1. **More Event Types**: Learn effects for pig hits, block collisions, TNT explosions
2. **Model Validation**: Check model quality (R², residual analysis) before injection
3. **Non-linear Models**: Use polynomial regression or neural networks for complex effects
4. **Multiple Collision Types**: Separate models for different collision types
5. **Model Persistence**: Save/load learned models across sessions
6. **Incremental Learning**: Update models incrementally instead of retraining from scratch
7. **Confidence Thresholds**: Only inject models with sufficient confidence

## Related Files

- **Main Agent**: `agents/pddl/pddl_agent.py` (lines 133-139, 219-223)
- **Event Learning**: `agents/pddl/pddl_files/events/learn_events.py`
- **Injection Logic**: `agents/pddl/pddl_files/pddl_parser.py` (lines 67-114)
- **Domain Files**: 
  - `agents/pddl/pddl_files/base_domain.pddl` (original with placeholders)
  - `agents/pddl/pddl_files/base_domain_modified.pddl` (generated with injections)

## See Also

- [PDDL_AGENT_ANALYSIS.md](PDDL_AGENT_ANALYSIS.md) - Overall agent architecture
- [TRAJECTORY_LEARNING_EXPLANATION.md](TRAJECTORY_LEARNING_EXPLANATION.md) - Physics parameter learning



