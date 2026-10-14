# PDDL domain catalog (`angry_birds_scaled`)

Source domain: [`pddl_files/base_domain_modified.pddl`](pddl_files/base_domain_modified.pddl).

Object instances are named `bird_*`, `pig_*`, `block_*`, `platform_*`, and optionally `external_agent_*` in the problem file. Problem init is built in [`pddl_files/pddl_parser.py`](pddl_files/pddl_parser.py) from vision via [`pddl_files/pddl_objects.py`](pddl_files/pddl_objects.py).

---

## Domain types

| Type | Description |
|------|-------------|
| `bird` | Slingshot projectile |
| `pig` | Target |
| `block` | Destructible structure (wood, stone, TNT, …) |
| `platform` | Ground / hill / ramp (sliding surface) |
| `external_agent` | Moving level hazards (magician, wizard, butterfly, worm) |

---

## Processes (continuous dynamics)

| Name | Parameters | Role |
|------|------------|------|
| `increasing_angle` | `?b` bird | While aiming: decreases `(angle)`, updates `(angle_rad)`, `(cosine)`, `(sinus)` |
| `decreasing_force` | `?b` bird | While aiming: decreases `(v_bird_multiplier)` between `(min_force)` and `(max_force)` |
| `flying` | `?b` bird | After release: ballistic motion with `(gravity)` until `y_bird > 0` and not platform-sliding |
| `platform_sliding` | `?b` bird, `?pl` platform | Bird rolls along platform top line; friction and `(platform_slide_accel)` |
| `block_platform_rolling` | `?bl` block, `?pl` platform | Block rolls on platform while `(block_slide_active)` |
| `agent_movement` | `?ea` external_agent | Moves agent by `(vx_agent)`, `(vy_agent)`; advances `(timing_agent)` after shot is adjusted |

---

## Events (discrete transitions)

| Name | Parameters | Role |
|------|------------|------|
| `collision_ground` | `?b` bird | Bird hits ground (`y_bird ≤ 0`); bounce / damp velocities |
| `explode_bird` | `?b` bird | End of bird flight (bounces ≥ 3 or `x_bird ≥ 800`); advance `(active_bird)` |
| `collision_pig_kill` | `?b` bird, `?p` pig | Direct hit kill when bird speed ≥ pig life; elastic-style velocity update |
| `collision_block_bounce_off` | `?b` bird, `?bl` block | AABB overlap, block too strong to break; bird stops, block pushed (learned ratio) |
| `collision_block` | `?b` bird, `?bl` block | AABB overlap, bird breaks through; block gets velocity, bird shot ends |
| `remove_unsupported_block` | `?bl_bottom`, `?bl_top` block | Top block falls when support below is gone |
| `explode_block` | `?bl_tnt`, `?bl_near` block | TNT chain: nearby blocks destroyed when TNT `(block_life ≤ 0)` |
| `explode_pig` | `?bl_tnt` block, `?p` pig | TNT blast kills pig within ~50 px |
| `remove_unsupported_pig` | `?bl_bottom` block, `?p` pig | Pig falls / dies when support block is invalid |
| `collision_block_pig_kill` | `?bl` block, `?p` pig | Rolling block (`v_block > 10`) kills pig on overlap |
| `collision_platform` | `?b` bird, `?pl` platform | Bird lands on platform; start `(platform_slide_active)` / `(bird_sliding)` |
| `platform_slide_handoff` | `?b` bird, `?from`, `?to` platform | Slide continues on chained platform `(platform_chain)` |
| `platform_slide_stop` | `?b` bird, `?pl` platform | Friction stops bird on platform (`vx_bird ≤ 0`) |
| `platform_slide_end` | `?b` bird, `?pl` platform | Bird leaves platform span; clear slide flags |
| `collision_block_platform` | `?bl` block, `?pl` platform | Moving block meets platform; start block slide |
| `block_platform_roll_handoff` | `?bl` block, `?from`, `?to` platform | Block roll handoff along `(platform_chain)` |
| `block_platform_roll_end` | `?bl` block, `?pl` platform | Block stops rolling or leaves platform |
| `explode_block_from_bird` | `?b` bird, `?bl_near` block | Black bird or post-bounce blast radius (~50 px) destroys block |
| `explode_pig_from_bird` | `?b` bird, `?p` pig | Black bird / end-of-shot / sliding blast kills pig (~50 px) |
| `explode_pig_from_egg` | `?b` bird, `?p` pig | White bird egg (`bird_tapped`) kills pig (~40 px) |
| `agent_1_timed_change_direction` | `?ea` external_agent | Wizard (`agent_type = 1`): reverse velocity every 1000 timing units |
| `agent_x_border_change_direction` | `?ea` external_agent | Reflect `(vx_agent)` at x borders |
| `agent_y_border_change_direction` | `?ea` external_agent | Reflect `(vy_agent)` at y borders |
| `collision_agent` | `?b` bird, `?ea` external_agent | Bird hits agent; bird stops |

---

## Instantaneous actions (planner / player)

| Name | Parameters | Role |
|------|------------|------|
| `set_force` | `?b` bird | Lock force while aiming (`force_locked`) |
| `pa-twang` | `?b` bird | Release: set `(vx_bird)`, `(vy_bird)` from angle and force; `(bird_released)` |
| `yellow_bird_action` | `?b` bird | Yellow bird tap: double speed |
| `white_bird_action` | `?b` bird | White bird tap: drop egg / shrink radius |
| `black_bird_action` | `?b` bird | Black bird tap: trigger explosion (`bounce_count → 3`) |
| `switch_bird` | `?b1` bird | Next bird when current bird stopped after release |

---

## Global predicates (not tied to one object)

| Predicate | Meaning |
|-----------|---------|
| `(angle_adjusted)` | Slingshot angle pass finished (released or reset) |
| `(pig_killed)` | At least one pig kill this level (score / bookkeeping) |
| `(force_locked)` | Force multiplier frozen until `pa-twang` |

---

## Global numeric fluents (scenario / slingshot)

| Fluent | Typical init / use |
|--------|---------------------|
| `(mod)` | Simulation mode flag (0 = integrate, 2 = collision handled) |
| `(cosine)`, `(sinus)` | Aim trig (updated by `increasing_angle`) |
| `(gravity)` | From world model |
| `(angle)`, `(angle_rad)`, `(angle_rate)`, `(angle_bias)` | Slingshot dial |
| `(min_angle)`, `(max_angle)` | Aim limits |
| `(active_bird)` | Index of bird currently in play |
| `(ground_y_damper)`, `(ground_x_damper)` | Ground interaction tuning |
| `(base_life_*_multiplier)`, `(base_mass_*_multiplier)` | Material tables |
| `(meta_wood_multiplier)`, `(meta_stone_multiplier)`, `(meta_ice_multiplier)`, `(meta_platform_size)` | Meta-level scaling |
| `(v_bird_multiplier)`, `(min_force)`, `(max_force)`, `(force_rate)` | Force search while aiming |
| `(fall_damage)`, `(explosion_damage)` | Damage constants (domain) |
| `(points_score)` | Score accumulator |
| `(total-time)` | Plan metric when minimizing time |

---

## Object attributes by type

### Bird (`?b`)

**Numeric fluents (per bird)**

| Attribute | PDDL fluent | Set in problem (vision) | Notes |
|-----------|-------------|-------------------------|--------|
| x | `(x_bird ?b)` | yes (`get_birds`) | Sling reference point; `pa-twang` applies kick |
| y | `(y_bird ?b)` | yes | PDDL y = `640 - screen_y` |
| speed magnitude | `(v_bird ?b)` | yes | From world model velocity hyperparameter |
| vx | `(vx_bird ?b)` | init 0 | Set at `pa-twang` |
| vy | `(vy_bird ?b)` | init 0 | Set at `pa-twang` |
| mass | `(m_bird ?b)` | yes | `width × height` (approx.) |
| id | `(bird_id ?b)` | yes | 0, 1, … |
| bounce / shot phase | `(bounce_count ?b)` | yes (0) | Also used to end flight |
| type | `(bird_type ?b)` | yes | RED=0, YELLOW=1, BLACK=2, WHITE=3, BLUE=4 |
| radius | `(bird_radius ?b)` | yes | `max(w,h)/2` |

**Numeric fluents (bird–block pairs)**

| Attribute | PDDL fluent | Set in problem |
|-----------|-------------|----------------|
| damage factor vs block | `(bird_block_damage ?b ?bl)` | yes (default or shelter override) |

**Predicates**

| Predicate | When true |
|-----------|-----------|
| `(bird_released ?b)` | After `pa-twang` |
| `(bird_tapped ?b)` | Special ability used |
| `(platform_slide_active ?b)` | Rolling on a platform |
| `(bird_sliding ?b ?pl)` | On platform `?pl` |

---

### Pig (`?p`)

**Numeric fluents**

| Attribute | PDDL fluent | Set in problem (`get_pigs`) |
|-----------|-------------|-----------------------------|
| x | `(x_pig ?p)` | yes (centre) |
| y | `(y_pig ?p)` | yes |
| radius | `(pig_radius ?p)` | yes (`min(w,h)/2`) |
| mass | `(m_pig ?p)` | yes |
| life / hit threshold | `(pig_life ?p)` | yes (typically 1) |

**Predicates**

| Predicate | When true |
|-----------|-----------|
| `(pig_dead ?p)` | Pig eliminated (goal: `(pig_dead pig_0)` etc.) |

---

### Block (`?bl`)

**Numeric fluents**

| Attribute | PDDL fluent | Set in problem (`get_blocks`) |
|-----------|-------------|----------------------------------|
| x | `(x_block ?bl)` | yes (AABB centre) |
| y | `(y_block ?bl)` | yes |
| width | `(block_width ?bl)` | yes |
| height | `(block_height ?bl)` | yes |
| life | `(block_life ?bl)` | yes (material × multiplier) |
| mass | `(block_mass ?bl)` | yes |
| stability | `(block_stability ?bl)` | yes (default 1) |
| speed | `(v_block ?bl)` | init 0 |
| vx | `(vx_block ?bl)` | init 0 |
| vy | `(vy_block ?bl)` | init 0 |

**Predicates**

| Predicate | When true |
|-----------|-----------|
| `(block_explosive ?bl)` | TNT / explosive block |
| `(block_slide_active ?bl)` | Rolling on platform |
| `(block_sliding ?bl ?pl)` | On platform `?pl` |

**Problem data only (not written to PDDL init)**

| Key | Purpose |
|-----|---------|
| `_block_angle` | Vision rotation (deg) |
| `_block_type` | wood / stone / TNT / … |
| `_sheltering_block` | Higher `bird_block_damage` for shelter tuning |

---

### Platform (`?pl`)

**Numeric fluents** (geometry filled / derived in `apply_enhsp_platform_geometry`)

| Attribute | PDDL fluent | Source |
|-----------|-------------|--------|
| x (centre) | `(x_platform ?pl)` | `get_platforms` |
| y (centre) | `(y_platform ?pl)` | `get_platforms` |
| width | `(platform_width ?pl)` | vision / merged fragments |
| height | `(platform_height ?pl)` | MBR or thin ramp thickness |
| top y at left edge | `(platform_top_y_left ?pl)` | derived |
| top y at right edge | `(platform_top_y_right ?pl)` | derived |
| top y min / max | `(platform_top_y_min ?pl)`, `(platform_top_y_max ?pl)` | derived |
| left x | `(platform_x_left ?pl)` | derived |
| slope dy/dx | `(platform_slope ?pl)` | derived |
| line intercept | `(platform_line_c ?pl)` | `y = c + slope × x` |
| roll acceleration | `(platform_slide_accel ?pl)` | derived from slope |

**Predicates**

| Predicate | When true |
|-----------|-----------|
| `(platform_chain ?from ?to)` | Touching platforms (handoff) |
| `(platform_has_next ?pl)` | This platform has a chained successor |

**Problem data only**

| Key | Purpose |
|-----|---------|
| `_platform_sloped` | Sloped top from vertices |
| `_platform_surface_x0/y0/x1/y1` | Top segment endpoints |
| `_platform_surface_angle_deg` | Debug / slope detection |
| `_platform_chain_next` | Becomes `(platform_chain …)` in init |

---

### External agent (`?ea`)

**Numeric fluents** (declared in domain; populated when agents appear in problem data)

| Attribute | PDDL fluent |
|-----------|-------------|
| x | `(x_agent ?ea)` |
| y | `(y_agent ?ea)` |
| width | `(agent_width ?ea)` |
| height | `(agent_height ?ea)` |
| vx | `(vx_agent ?ea)` |
| vy | `(vy_agent ?ea)` |
| internal clock | `(timing_agent ?ea)` |
| x min / max border | `(x_min_border ?ea)`, `(x_max_border ?ea)` |
| y min / max border | `(y_min_border ?ea)`, `(y_max_border ?ea)` |
| type | `(agent_type ?ea)` — MAGICIAN=0, WIZARD=1, BUTTERFLY=2, WORM=3 |

**Predicates**

| Predicate | When true |
|-----------|-----------|
| `(agent_dead ?ea)` | Agent no longer active |

---

## Counts (current domain file)

| Kind | Count |
|------|------:|
| Processes | 6 |
| Events | 27 |
| Actions | 6 |
