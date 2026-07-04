# Collision Event Learning: Problem Analysis and CART Solution

## Executive Summary

Analysis of 24 collision samples revealed that ground collision physics varies by **impact angle**, with cluster-based models showing **+70% improvement** for v_x prediction over single models.

---

## Problem Statement

### The Challenge

When a bird collides with the ground in Angry Birds, the post-collision velocities (v_x, v_y) depend on pre-collision state. The question: **Is one collision model sufficient, or do different impact angles require different models?**

### Current Approach

Single linear model for all collisions:
```
v_x_after = f(x, y, v_x_before, v_y_before)
v_y_after = g(x, y, v_x_before, v_y_before)
```

---

## Experimental Results

### Data Collection

- **Method**: Cluster analysis mode cycling through angles 20°-75° (5° steps)
- **Samples collected**: 24 collision events
- **Features**: Pre-collision state (x, y, v_x, v_y)
- **Target**: Post-collision velocities (v_x_after, v_y_after)

### Cluster Analysis Results

| Samples | Variable | Single Model RMSE | Cluster Model RMSE | Improvement |
|---------|----------|-------------------|--------------------|--------------| 
| 12 | v_y | 9.50 | 10.64 | **-12.0%** (worse) |
| 12 | v_x | 31.80 | 34.03 | **-7.0%** (worse) |
| 24 | v_y | 6.48 | 4.69 | **+27.5%** (better) |
| 24 | v_x | 22.03 | 6.48 | **+70.6%** (better) |

### Cluster Statistics (24 samples)

| Variable | Cluster | Impact Angle Range | Ratio (Mean ± Std) |
|----------|---------|-------------------|---------------------|
| v_y | Shallow (26°-48°) | 0.431 ± 0.063 | **-0.287 ± 0.072** |
| v_y | Steep (52°-75°) | 0.662 ± 0.077 | **-0.284 ± 0.034** |
| v_x | Shallow (26°-48°) | 0.431 ± 0.063 | **0.646 ± 0.130** |
| v_x | Steep (52°-75°) | 0.662 ± 0.077 | **0.313 ± 0.290** |

### Key Findings

1. **v_y restitution is angle-independent**: Both clusters have ~-0.28 ratio
2. **v_x friction is strongly angle-dependent**: 0.65 (shallow) vs 0.31 (steep)
3. **Minimum samples needed**: ~12 per cluster before improvement appears
4. **Silhouette score**: 0.58 (good cluster separation)

---

## Solution: CART (Classification and Regression Trees)

### Why CART?

| Advantage | Explanation |
|-----------|-------------|
| **Automatic clustering** | Finds optimal threshold from data |
| **No configuration** | Uses cross-validation for tree depth |
| **PDDL-compatible** | Tree structure maps to conditional effects |
| **Single algorithm** | Learns splits AND predictions together |

### How CART Works

```
Input: Pre-collision state [x, y, v_x, v_y]
Target: Post-collision ratio (v_x_after/v_x_before)

CART Algorithm:
1. Try ALL possible splits on ALL features
2. Choose split with maximum impurity (MSE) reduction
3. Recurse until stopping criterion met
4. Each leaf predicts mean of its samples
```

### Expected CART Output for Collision Data

```
                impact_factor < 0.52?
                /                    \
             YES                      NO
              |                        |
       SHALLOW IMPACT            STEEP IMPACT
       v_x_ratio = 0.65          v_x_ratio = 0.31
       v_y_ratio = -0.29         v_y_ratio = -0.28
```

Where `impact_factor = |v_y| / (|v_x| + |v_y|)`

### CART vs Current K-Means Approach

| Aspect | K-Means + LOO-CV | CART |
|--------|------------------|------|
| Threshold | Manual (0.5) | **Learned from data** |
| # Clusters | Manual (k=2) | **Automatic (tree depth)** |
| Configuration | Silhouette threshold | **None (CV pruning)** |
| Output | Cluster labels | **Decision rules** |
| PDDL translation | Complex | **Direct mapping** |

---

## PDDL Integration

### Current Single-Model Injection

```pddl
(:event collision_ground
    :effect (and
        (assign (vx_bird ?b) (+ bias (* c1 (vx_bird ?b)) (* c2 (vy_bird ?b))))
        (assign (vy_bird ?b) (* -0.28 (vy_bird ?b)))
    )
)
```

### CART-Based Conditional Injection

```pddl
(:event collision_ground
    :effect (and
        ; Shallow impact: impact_factor < 0.52
        (when (< (/ (abs (vy_bird ?b)) 
                    (+ (abs (vx_bird ?b)) (abs (vy_bird ?b)))) 
                 0.52)
            (and
                (assign (vx_bird ?b) (* 0.65 (vx_bird ?b)))
                (assign (vy_bird ?b) (* -0.29 (vy_bird ?b)))
            )
        )
        ; Steep impact: impact_factor >= 0.52
        (when (>= (/ (abs (vy_bird ?b)) 
                     (+ (abs (vx_bird ?b)) (abs (vy_bird ?b)))) 
                  0.52)
            (and
                (assign (vx_bird ?b) (* 0.31 (vx_bird ?b)))
                (assign (vy_bird ?b) (* -0.28 (vy_bird ?b)))
            )
        )
    )
)
```

---

## Decision Function: When to Use Clusters

### Configuration-Free Approach (LOO-CV)

```python
def should_use_clusters(single_rmse, cluster_rmse):
    """No configuration needed - just compare."""
    return cluster_rmse < single_rmse
```

This naturally handles overfitting:
- **Few samples** → cluster models overfit → higher RMSE → use single model
- **Many samples** → cluster models generalize → lower RMSE → use clusters

---

## Implementation Recommendations

### Phase 1: Data Collection
- Continue cluster_analysis_mode until 24+ collision samples
- Ensure angle diversity (20°-80° range)

### Phase 2: Model Selection
- Implement CART-based collision model
- Use cross-validation to determine optimal tree depth
- Compare LOO-CV RMSE: CART vs single linear model

### Phase 3: PDDL Integration
- Modify `inject_domain_file()` to generate conditional effects
- Tree leaves → PDDL `when` conditions
- Validate with ENHSP planner

---

## Appendix: Physics Interpretation

### Why v_x is Angle-Dependent

| Impact Type | Physics | v_x Retention |
|-------------|---------|---------------|
| **Shallow** (20°-50°) | Bird grazes ground, maintains horizontal momentum | ~65% |
| **Steep** (50°-80°) | Bird digs into ground, loses horizontal momentum | ~31% |

### Why v_y is Angle-Independent

The coefficient of restitution (bounciness) is a material property:
- Ground material is constant
- Bird material is constant
- Therefore: v_y_ratio ≈ -0.28 regardless of angle

---

## References

- Silhouette Score: Rousseeuw, P.J. (1987). "Silhouettes: a Graphical Aid"
- CART: Breiman, L. et al. (1984). "Classification and Regression Trees"
- PDDL+: Fox, M. & Long, D. (2006). "Modelling Mixed Discrete-Continuous Domains"
