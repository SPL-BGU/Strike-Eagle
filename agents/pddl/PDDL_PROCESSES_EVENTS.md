# PDDL domain: processes and events

Source: [`pddl_files/base_domain_modified.pddl`](pddl_files/base_domain_modified.pddl) (`angry_birds_scaled`).

CSV export: [`PDDL_PROCESSES_EVENTS.csv`](PDDL_PROCESSES_EVENTS.csv) (same rows; entity columns `*_in` / `*_out`; multi-value cells use `;` separators).

**Notation**

- **Inputs** — fluents and predicates read in `:precondition` (including inequality bounds).
- **Outputs** — fluent updates and predicate adds/deletes in `:effect`.
- Entity columns list only symbols that appear for that type in preconditions or effects.
- `#t` — continuous time (processes only).

## Entity fluent reference

| Entity | Fluents | Predicates |
|--------|---------|------------|
| **bird** `?b` | `x_bird`, `y_bird`, `v_bird`, `vx_bird`, `vy_bird`, `m_bird`, `bird_id`, `bird_type`, `bird_radius`, `bounce_count`, `bird_block_damage(?b,?bl)` | `bird_released`, `bird_tapped`, `platform_slide_active`, `bird_sliding(?b,?pl)` |
| **pig** `?p` | `x_pig`, `y_pig`, `pig_radius`, `m_pig`, `pig_life` | `pig_dead` |
| **block** `?bl` | `x_block`, `y_block`, `block_width`, `block_height`, `block_life`, `block_mass`, `block_stability`, `v_block`, `vx_block`, `vy_block` | `block_explosive`, `block_slide_active`, `block_sliding(?bl,?pl)` |
| **platform** `?pl` | `x_platform`, `y_platform`, `platform_width`, `platform_height`, `platform_top_y_*`, `platform_x_left`, `platform_slope`, `platform_slide_accel`, `platform_line_c` | `platform_chain(?from,?to)`, `platform_has_next` |
| **external_agent** `?ea` | `x_agent`, `y_agent`, `agent_width`, `agent_height`, `vx_agent`, `vy_agent`, `timing_agent`, `agent_type`, `x_min_border`, `x_max_border`, `y_min_border`, `y_max_border` | `agent_dead` |
| **global** | `active_bird`, `angle`, `angle_rad`, `angle_bias`, `angle_rate`, `min_angle`, `max_angle`, `cosine`, `sinus`, `gravity`, `v_bird_multiplier`, `min_force`, `max_force`, `force_rate`, `mod`, `points_score`, `fall_damage`, `explosion_damage`, meta/base multipliers, … | `angle_adjusted`, `force_locked`, `pig_killed` |

---

## Summary table

| Name | Kind | Parameters |
|------|------|------------|
| `increasing_angle` | process | `?b` bird |
| `decreasing_force` | process | `?b` bird |
| `flying` | process | `?b` bird |
| `platform_sliding` | process | `?b` bird, `?pl` platform |
| `block_platform_rolling` | process | `?bl` block, `?pl` platform |
| `agent_movement` | process | `?ea` external_agent |
| `collision_ground` | event | `?b` bird |
| `explode_bird` | event | `?b` bird |
| `collision_pig_kill` | event | `?b` bird, `?p` pig |
| `collision_block_bounce_off` | event | `?b` bird, `?bl` block |
| `collision_block` | event | `?b` bird, `?bl` block |
| `remove_unsupported_block` | event | `?bl_bottom`, `?bl_top` block |
| `explode_block` | event | `?bl_tnt`, `?bl_near` block |
| `explode_pig` | event | `?bl_tnt` block, `?p` pig |
| `remove_unsupported_pig` | event | `?bl_bottom` block, `?p` pig |
| `collision_block_pig_kill` | event | `?bl` block, `?p` pig |
| `collision_platform` | event | `?b` bird, `?pl` platform |
| `platform_slide_handoff` | event | `?b` bird, `?from` `?to` platform |
| `platform_slide_stop` | event | `?b` bird, `?pl` platform |
| `platform_slide_end` | event | `?b` bird, `?pl` platform |
| `collision_block_platform` | event | `?bl` block, `?pl` platform |
| `block_platform_roll_handoff` | event | `?bl` block, `?from` `?to` platform |
| `block_platform_roll_end` | event | `?bl` block, `?pl` platform |
| `explode_block_from_bird` | event | `?b` bird, `?bl_near` block |
| `explode_pig_from_bird` | event | `?b` bird, `?p` pig |
| `explode_pig_from_egg` | event | `?b` bird, `?p` pig |
| `agent_1_timed_change_direction` | event | `?ea` external_agent |
| `agent_x_border_change_direction` | event | `?ea` external_agent |
| `agent_y_border_change_direction` | event | `?ea` external_agent |
| `collision_agent` | event | `?b` bird, `?ea` external_agent |

---

## Processes (continuous)

### `increasing_angle`

| | |
|---|---|
| **bird** | **In:** `bird_id`; **pred:** not `bird_released`, not `angle_adjusted` |
| **global** | **In:** `active_bird`, `angle`, `min_angle`, `max_angle`, `angle_rate`, `angle_bias`, `angle_rad` · **Out:** `angle` ↓, `angle_rad`, `cosine`, `sinus` |

Aim angle decreases while the slingshot is active and angle is not locked.

---

### `decreasing_force`

| | |
|---|---|
| **bird** | **In:** `bird_id`; **pred:** not `bird_released`, not `force_locked` |
| **global** | **In:** `active_bird`, `v_bird_multiplier`, `min_force`, `max_force`, `force_rate` · **Out:** `v_bird_multiplier` ↓ |

Pull strength ramps down until `set_force` locks it.

---

### `flying`

| | |
|---|---|
| **bird** | **In:** `bird_id`, `x_bird`, `y_bird`, `vx_bird`, `vy_bird`; **pred:** `bird_released`, not `platform_slide_active`, `y_bird` > 0 |
| **global** | **In:** `active_bird`, `gravity` · **Out:** `x_bird` ↑, `y_bird` ↑, `vy_bird` ↓, `mod` ← 0 |

Ballistic motion between platform slide and ground contact.

---

### `platform_sliding`

| | |
|---|---|
| **bird** | **In:** `bird_id`, `x_bird`, `y_bird`, `vx_bird`, `vy_bird`, `bird_radius`; **pred:** `bird_released`, `platform_slide_active`, `bird_sliding(?b,?pl)`, `vx_bird` > 0 |
| **platform** | **In:** `x_platform`, `platform_width`, `platform_line_c`, `platform_slope`, `platform_slide_accel` |
| **global** | **In:** `active_bird`, `gravity` · **Out:** `y_bird`, `vy_bird` ← 0, `x_bird` ↑, `vx_bird` ↑/↓ (gravity + friction), `mod` ← 0 |

Bird rolls along platform top edge while slide mode is active.

---

### `block_platform_rolling`

| | |
|---|---|
| **block** | **In:** `x_block`, `y_block`, `block_width`, `block_height`, `vx_block`, `vy_block`, `v_block`; **pred:** `block_slide_active`, `block_sliding(?bl,?pl)`, `vx_block` > 0, `v_block` > 0 |
| **platform** | **In:** `x_platform`, `platform_width`, `platform_line_c`, `platform_slope`, `platform_slide_accel` |
| **global** | **In:** `gravity` · **Out:** `y_block`, `vy_block` ← 0, `x_block` ↑, `vx_block`, `v_block`, `mod` ← 0 |

Rolling block kinematics on a platform.

---

### `agent_movement`

| | |
|---|---|
| **external_agent** | **In:** `x_agent`, `y_agent`, `vx_agent`, `vy_agent`, `timing_agent`; **pred:** not `agent_dead` |
| **global** | **In:** **pred:** `angle_adjusted` · **Out:** `x_agent` ↑, `y_agent` ↑, `timing_agent` ↑ |

Agents move after the bird has been launched (`angle_adjusted`).

---

## Events (instantaneous)

### `collision_ground`

| | |
|---|---|
| **bird** | **In:** `bird_id`, `x_bird`, `y_bird`, `vx_bird`, `vy_bird`; **pred:** `bird_released`, `y_bird` ≤ 0, `vx_bird` > 0 · **Out:** `x_bird`, `y_bird`, `vx_bird`, `vy_bird` |
| **global** | **In:** `active_bird` · **Out:** `mod` ← 2 |

Ground bounce / damping model.

---

### `explode_bird`

| | |
|---|---|
| **bird** | **In:** `bird_id`, `bounce_count`, `x_bird`; **pred:** `bird_released`, `angle_adjusted`, (`bounce_count` ≥ 3 or `x_bird` ≥ 800) |
| **global** | **In:** `active_bird`, `min_angle` · **Out:** `angle` ← `min_angle`, `active_bird` ↑, **pred:** delete `angle_adjusted` |

Ends the current bird’s shot and advances to the next bird.

---

### `collision_pig_kill`

| | |
|---|---|
| **bird** | **In:** `bird_id`, `v_bird`, `x_bird`, `y_bird`, `vx_bird`, `vy_bird`, `bird_radius`, `m_bird` · **Out:** `vx_bird`, `vy_bird` (elastic collision) |
| **pig** | **In:** `x_pig`, `y_pig`, `pig_radius`, `m_pig`, `pig_life`; **pred:** not `pig_dead`, overlap + `pig_life` ≤ `v_bird` · **Out:** **pred:** `pig_dead` |
| **global** | **In:** `active_bird` · **Out:** **pred:** `pig_killed`; `points_score` ↑ 5000 |

Direct bird–pig hit with enough speed to meet life threshold.

---

### `collision_block_bounce_off`

| | |
|---|---|
| **bird** | **In:** `bird_id`, `v_bird`, `x_bird`, `y_bird`, `bird_radius`, `vx_bird`, `vy_bird`, `bird_block_damage(?b,?bl)` · **Out:** `v_bird`, `vx_bird`, `vy_bird` ← 0, `bounce_count` ← 3 |
| **block** | **In:** `x_block`, `y_block`, `block_width`, `block_height`, `block_life`, `block_stability` (AABB overlap; life > 0; `v_bird` ≥ stability; damage check fails) · **Out:** `vx_block`, `vy_block`, `v_block` (scaled from bird velocity) |
| **global** | **In:** `active_bird` | |

Bird bounces off; block gets negligible push (coefficients 0 in domain).

---

### `collision_block`

| | |
|---|---|
| **bird** | **In:** `bird_id`, `v_bird`, `x_bird`, `y_bird`, `bird_radius`, `vx_bird`, `vy_bird`, `bird_block_damage(?b,?bl)` · **Out:** `bounce_count` ← 3 |
| **block** | **In:** same geometry/life as bounce-off; damage check succeeds (`block_life`² ≤ kinetic × damage²) · **Out:** `vx_block`, `vy_block`, `v_block` |
| **global** | **In:** `active_bird` | |

Bird breaks through (stops counting as continuing flight via `bounce_count`).

---

### `remove_unsupported_block`

| | |
|---|---|
| **block** `?bl_top` | **In:** `x_block`, `y_block`, `block_width`, `block_height`, `block_life`, `block_stability` · **Out:** `block_life` ↓ 100, `y_block`, `block_stability` ← 0 |
| **block** `?bl_bottom` | **In:** `x_block`, `y_block`, `block_life`, `block_stability` (support gone: life ≤ 0 or stability ≤ 0); stacked geometry |
| **global** | **Out:** `points_score` ↑ 500 |

Top block falls when support block is destroyed/unstable.

---

### `explode_block`

| | |
|---|---|
| **block** `?bl_tnt` | **In:** `x_block`, `y_block`; **pred:** `block_explosive`, `block_life` ≤ 0 |
| **block** `?bl_near` | **In:** `x_block`, `y_block`, `block_life`, `block_stability` (within ±100) · **Out:** `block_life` ← 0, `block_stability` ← 0 |
| **global** | **Out:** `points_score` ↑ 1000 |

TNT chain reaction on nearby blocks.

---

### `explode_pig`

| | |
|---|---|
| **block** `?bl_tnt` | **In:** `x_block`, `y_block`; **pred:** `block_explosive`, `block_life` ≤ 0 |
| **pig** | **In:** `x_pig`, `y_pig`; **pred:** not `pig_dead` (within ±50 of TNT) · **Out:** **pred:** `pig_dead` |
| **global** | **Out:** **pred:** `pig_killed`; `points_score` ↑ 5000 |

Pig killed by TNT blast radius.

---

### `remove_unsupported_pig`

| | |
|---|---|
| **block** `?bl_bottom` | **In:** `x_block`, `y_block`, `block_width`, `block_height`, `block_life`, `block_stability` (support failed) |
| **pig** | **In:** `x_pig`, `y_pig`, `pig_radius`; **pred:** not `pig_dead` (pig above block footprint) · **Out:** **pred:** `pig_dead` |
| **global** | **Out:** **pred:** `pig_killed`; `points_score` ↑ 5000 |

Pig dies when its supporting block is gone.

---

### `collision_block_pig_kill`

| | |
|---|---|
| **block** | **In:** `x_block`, `y_block`, `block_width`, `block_height`, `block_life`, `v_block` (`v_block` > 10, overlap with pig) |
| **pig** | **In:** `x_pig`, `y_pig`, `pig_radius`; **pred:** not `pig_dead` · **Out:** **pred:** `pig_dead` |
| **global** | **Out:** **pred:** `pig_killed`; `points_score` ↑ 5000 |

Rolling/falling block crushes pig.

---

### `collision_platform`

| | |
|---|---|
| **bird** | **In:** `bird_id`, `x_bird`, `y_bird`, `v_bird`, `vx_bird`, `bird_radius`; **pred:** `bird_released`, not `platform_slide_active`, `v_bird` > 0, approach from above · **Out:** `y_bird`, `vx_bird`, `v_bird`, `vy_bird` ← 0, **pred:** `platform_slide_active`, `bird_sliding(?b,?pl)` |
| **platform** | **In:** `x_platform`, `platform_width`, `platform_height`, `platform_line_c`, `platform_slope` |
| **global** | **In:** `active_bird` · **Out:** `mod` ← 2 |

First landing on a platform; enters slide mode.

---

### `platform_slide_handoff`

| | |
|---|---|
| **bird** | **In:** `x_bird`, `bird_radius`, `vx_bird`; **pred:** `platform_slide_active`, `bird_sliding(?b,?from)`, `vx_bird` > 0 · **Out:** `y_bird`, `vy_bird` ← 0, **pred:** delete `bird_sliding(?b,?from)`, add `bird_sliding(?b,?to)` |
| **platform** | **In:** `x_platform`, `platform_width`, `platform_line_c`, `platform_slope`; **pred:** `platform_chain(?from,?to)` |

Bird slide transfers to the next chained platform.

---

### `platform_slide_stop`

| | |
|---|---|
| **bird** | **In:** `vx_bird`, `bounce_count`; **pred:** `platform_slide_active`, `bird_sliding(?b,?pl)`, `vx_bird` ≤ 0, `bounce_count` < 3 · **Out:** `vx_bird`, `v_bird` ← 0, `bounce_count` ← 3 |

Friction stops the bird on the platform (shot ends).

---

### `platform_slide_end`

| | |
|---|---|
| **bird** | **In:** `x_bird`, `bird_radius`; **pred:** `platform_slide_active`, `bird_sliding(?b,?pl)` (left platform edge or no `platform_has_next`) · **Out:** **pred:** delete `platform_slide_active`, delete `bird_sliding(?b,?pl)` |
| **platform** | **In:** `x_platform`, `platform_width`; **pred:** `platform_has_next` |

Bird leaves platform surface; returns to free flight or fall.

---

### `collision_block_platform`

| | |
|---|---|
| **block** | **In:** `x_block`, `y_block`, `block_width`, `block_height`, `block_life`, `v_block`, `vx_block`, `vy_block`; **pred:** not `block_slide_active`, `block_life` > 0, `v_block` > 0 · **Out:** `y_block`, `vy_block` ← 0, `v_block`, **pred:** `block_slide_active`, `block_sliding(?bl,?pl)` |
| **platform** | **In:** `x_platform`, `platform_width`, `platform_top_y_min`, `platform_top_y_max`, `platform_height`, `platform_line_c`, `platform_slope` |
| **global** | **Out:** `mod` ← 2 |

Moving block lands and starts rolling on platform.

---

### `block_platform_roll_handoff`

| | |
|---|---|
| **block** | **In:** `x_block`, `block_width`, `block_height`, `vx_block`; **pred:** `block_slide_active`, `block_sliding(?bl,?from)`, `vx_block` > 0 · **Out:** `y_block`, `vy_block` ← 0, **pred:** delete `block_sliding(?bl,?from)`, add `block_sliding(?bl,?to)` |
| **platform** | **In:** `x_platform`, `platform_width`, `platform_line_c`, `platform_slope`; **pred:** `platform_chain(?from,?to)` |

Block roll moves to next chained platform.

---

### `block_platform_roll_end`

| | |
|---|---|
| **block** | **In:** `vx_block`, `v_block`, `x_block`, `block_width`; **pred:** `block_slide_active`, `block_sliding(?bl,?pl)` (stopped or off edge) · **Out:** **pred:** delete `block_slide_active`, delete `block_sliding(?bl,?pl)` |
| **platform** | **In:** `x_platform`, `platform_width`; **pred:** `platform_has_next` |

Block stops sliding or leaves platform.

---

### `explode_block_from_bird`

| | |
|---|---|
| **bird** | **In:** `bird_id`, `bird_type`, `bounce_count`, `x_bird`, `y_bird` (`bird_type` = 2 BLACK or `bounce_count` > 0) |
| **block** | **In:** `x_block`, `y_block`, `block_life`, `block_stability` (within ±50) · **Out:** `block_life` ← 0, `block_stability` ← 0 |
| **global** | **In:** `active_bird` | |

Black bird / post-tap explosion damages nearby blocks.

---

### `explode_pig_from_bird`

| | |
|---|---|
| **bird** | **In:** `bird_id`, `bird_type`, `bounce_count`, `x_bird`, `y_bird`; **pred:** `platform_slide_active` optional (`bird_type` = 2, or `bounce_count` = 3, or sliding) |
| **pig** | **In:** `x_pig`, `y_pig`; **pred:** not `pig_dead` (within ±50) · **Out:** **pred:** `pig_dead` |
| **global** | **In:** `active_bird` · **Out:** **pred:** `pig_killed`; `points_score` ↑ 5000 |

Bird explosion radius kills pig.

---

### `explode_pig_from_egg`

| | |
|---|---|
| **bird** | **In:** `bird_id`, `bird_type`, `bounce_count`, `x_bird`, `y_bird`; **pred:** `bird_type` = 3 (WHITE), `bounce_count` = 3, `bird_tapped` |
| **pig** | **In:** `x_pig`, `y_pig`; **pred:** not `pig_dead` (within ±40) · **Out:** **pred:** `pig_dead` |
| **global** | **In:** `active_bird` · **Out:** **pred:** `pig_killed`; `points_score` ↑ 5000 |

White bird egg splash kill (smaller radius than black explosion).

---

### `agent_1_timed_change_direction`

| | |
|---|---|
| **external_agent** | **In:** `agent_type`, `timing_agent`, `vx_agent`, `vy_agent`; **pred:** not `agent_dead`, `agent_type` = 1 (WIZARD), `timing_agent` ≥ 1000 · **Out:** `vx_agent`, `vy_agent` (flip sign) |
| **global** | **In:** **pred:** `angle_adjusted` | |

Wizard reverses velocity on a timer.

---

### `agent_x_border_change_direction`

| | |
|---|---|
| **external_agent** | **In:** `x_agent`, `agent_width`, `vx_agent`, `x_min_border`, `x_max_border`; **pred:** not `agent_dead`, hit x border · **Out:** `vx_agent` (flip) |
| **global** | **In:** **pred:** `angle_adjusted` | |

---

### `agent_y_border_change_direction`

| | |
|---|---|
| **external_agent** | **In:** `y_agent`, `agent_height`, `vy_agent`, `y_min_border`, `y_max_border`; **pred:** not `agent_dead`, hit y border · **Out:** `vy_agent` (flip) |
| **global** | **In:** **pred:** `angle_adjusted` | |

---

### `collision_agent`

| | |
|---|---|
| **bird** | **In:** `bird_id`, `v_bird`, `x_bird`, `y_bird`, `bird_radius` · **Out:** `v_bird`, `vx_bird`, `vy_bird` ← 0, `bounce_count` ← 3 |
| **external_agent** | **In:** `x_agent`, `y_agent`, `agent_width`, `agent_height` |
| **global** | **In:** `active_bird` | |

Bird stopped by external agent hitbox.

---

## Related discrete actions (not processes/events)

These are `:action` definitions in the same domain file (planner-chosen, not triggered continuously):

| Action | Parameters | Role |
|--------|------------|------|
| `set_force` | `?b` | Lock pull strength (`force_locked`) |
| `pa-twang` | `?b` | Release bird with `vx_bird`/`vy_bird` from angle and force |
| `yellow_bird_action` | `?b` | Speed boost (type 1) |
| `white_bird_action` | `?b` | Lay egg / shrink radius (type 3) |
| `black_bird_action` | `?b` | Trigger explosion end state (type 2) |
| `switch_bird` | `?b1` | Advance `active_bird` when bird at rest |
