# PDDL domain: processes and events

Source: [`pddl_files/base_domain_modified.pddl`](pddl_files/base_domain_modified.pddl) (`angry_birds_scaled`).

CSV export: [`PDDL_PROCESSES_EVENTS.csv`](PDDL_PROCESSES_EVENTS.csv) (same rows plus the discrete actions as `kind=action`; columns `name, kind, parameters, precondition, learnt, effect, inputs, description`; multi-value cells use `;` separators).

**Notation**

- **Parameters** — the typed `:parameters` list.
- **Precondition** — the `:precondition` conjunction, one condition per line (`or` written inline). Fluent arguments are dropped (`x_bird` = `(x_bird ?b)`) unless a second object of the same type is involved.
- **Effect** — the `:effect` list. `+=` / `-=` are `increase` / `decrease`, `←` is `assign`, `+pred` / `−pred` are predicate add / delete.
- **Inputs** — every fluent and predicate the definition reads, in the precondition or on the right-hand side of an effect, grouped by entity.
- `#t` — continuous time (processes only); the domain time unit is 0.01 s, so most rates use `#t/100`.
- `r` = `bird_radius`, `w`/`h` = `block_width`/`block_height`, `pw` = `platform_width`, `line(pl, x)` = `platform_line_c + platform_slope · x` (platform top edge at `x`).
- **Learnt** — present only on definitions whose body is overridden from learned data (see next section). Values marked *(injected)* come from those placeholders; the numbers shown are from the current `base_domain_modified.pddl` and change from run to run.

## Learnt / overridden definitions

The template is [`pddl_files/base_domain.pddl`](pddl_files/base_domain.pddl). Before every ENHSP call, `inject_domain_file` (`pddl_parser.py`) replaces its `{SE-…}` placeholders, using the world model's `kb`, and writes `base_domain_modified.pddl`. Every other process and event is hand-written and fixed.

| Placeholder | Overrides | Source (`kb` key) | Selection / fallback |
|---|---|---|---|
| `{SE-collision-ground-effect}` | **whole effect** of `collision_ground` | `collision` (ground-contact samples from GT trajectories, `learn_collision_effects`) | Tried in order: affine **LR** (4-feature PDDL LR, used if it passes the R² gates and has no extended features) → **M5** model tree (`lineartree`) → **placeholder**. The placeholder is a roll: `y ← 0.35r`, `vx ×0.35`, `vy ×0.5`, `v_bird ← 0.35·vx`, `x` advance, `mod ← 2`. With `enhsp_ground_placeholder`, the LR/M5 step is skipped and the placeholder is always used. |
| `{SE-collision-platform-effect}` | **whole effect** of `collision_platform` | `platform_collision` (`learn_platform_effects`) | All options snap `y` to the stand line, set `vy ← 0` and start the slide. Tried in order: the LR `v_x` model (same formula for `vx_bird` and `v_bird`) → the learned `v_x` ratio model (n ≥ 5, R² ≥ 0) → heuristic `vx ×0.35`. M5 is never used here. A hard-stop variant (`planning_pessimistic_platform`) exists, but every call site passes `False`. |
| `{SE-platform-friction}` | the `decrease (vx_…)` friction term in `platform_sliding` and `block_platform_rolling` | `_platform_friction.samples` | Median of the samples; default `10.0` px/s² when there are none. |
| `{SE-block-push-ratio}` | `vx_block`, `vy_block`, `v_block` assignments in `collision_block` and `collision_block_bounce_off` | `_block_push.samples` (`learn_block_impulse_effects`) | Median of the samples; default `0.18` when there are none. `_block_push` is **not** persisted by `world_model_persistence`, so it starts again from the default on every run. |
| `{SE-block-platform-roll-effect}` | **whole effect** of `collision_block_platform` | — | Generated, but not learnt: always the fixed roll entry (snap to the surface, `vy ← 0`, `v_block ← vx_block`, start rolling). |

**Not overridden in the domain:** `flying` is hand-written ballistics. `inject_learned_transitions` exists but is never called. Learned `v_bird` and `gravity` from the world model reach the planner only as problem `:init` values, so the domain bodies don't change. `angle_bias` is a constant 0. The angle-residual correction is applied at execution time, outside the planner.

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

| Name | Kind | Parameters | Learnt / overridden |
|------|------|------------|---------------------|
| `increasing_angle` | process | `?b - bird` | — |
| `decreasing_force` | process | `?b - bird` | — |
| `flying` | process | `?b - bird` | — (learned `gravity` via problem init) |
| `platform_sliding` | process | `?b - bird`, `?pl - platform` | **friction** `{SE-platform-friction}` |
| `block_platform_rolling` | process | `?bl - block`, `?pl - platform` | **friction** `{SE-platform-friction}` |
| `agent_movement` | process | `?ea - external_agent` | — |
| `collision_ground` | event | `?b - bird` | **whole effect** `{SE-collision-ground-effect}` |
| `explode_bird` | event | `?b - bird` | — |
| `collision_pig_kill` | event | `?b - bird`, `?p - pig` | — |
| `collision_block_bounce_off` | event | `?b - bird`, `?bl - block` | **push ratio** `{SE-block-push-ratio}` |
| `collision_block` | event | `?b - bird`, `?bl - block` | **push ratio** `{SE-block-push-ratio}` |
| `remove_unsupported_block` | event | `?bl_bottom - block`, `?bl_top - block` | — |
| `explode_block` | event | `?bl_tnt - block`, `?bl_near - block` | — |
| `explode_pig` | event | `?bl_tnt - block`, `?p - pig` | — |
| `remove_unsupported_pig` | event | `?bl_bottom - block`, `?p - pig` | — |
| `collision_block_pig_kill` | event | `?bl - block`, `?p - pig` | — |
| `collision_platform` | event | `?b - bird`, `?pl - platform` | **whole effect** `{SE-collision-platform-effect}` |
| `platform_slide_handoff` | event | `?b - bird`, `?from - platform`, `?to - platform` | — |
| `platform_slide_stop` | event | `?b - bird`, `?pl - platform` | — |
| `platform_slide_end` | event | `?b - bird`, `?pl - platform` | — |
| `collision_block_platform` | event | `?bl - block`, `?pl - platform` | generated effect `{SE-block-platform-roll-effect}` (fixed, not learnt) |
| `block_platform_roll_handoff` | event | `?bl - block`, `?from - platform`, `?to - platform` | — |
| `block_platform_roll_end` | event | `?bl - block`, `?pl - platform` | — |
| `explode_block_from_bird` | event | `?b - bird`, `?bl_near - block` | — |
| `explode_pig_from_bird` † | event | `?b - bird`, `?p - pig` | — |
| `explode_pig_from_egg` † | event | `?b - bird`, `?p - pig` | — |
| `agent_1_timed_change_direction` † | event | `?ea - external_agent` | — |
| `agent_x_border_change_direction` † | event | `?ea - external_agent` | — |
| `agent_y_border_change_direction` † | event | `?ea - external_agent` | — |
| `collision_agent` † | event | `?b - bird`, `?ea - external_agent` | — |

† These come after the stray `)` that closes `explode_block_from_bird` (line ~695), which also closes the `define`. As written they are outside the domain, and so is `agent_movement` and `switch_bird`.

---

## Processes (continuous)

### `increasing_angle`

| | |
|---|---|
| **Parameters** | `?b - bird` |
| **Precondition** | `active_bird = bird_id`<br>not `bird_released`<br>not `angle_adjusted`<br>`min_angle ≤ angle ≤ max_angle` |
| **Effect** | `angle -= #t · angle_rate`<br>`angle_rad ← (angle − #t·angle_rate − angle_bias) · π/180`<br>`cosine ← 1 − a²/2 + a⁴/24 − a⁶/720` (a = `angle_rad`)<br>`sinus ← a − a³/6 + a⁵/120` |
| **Inputs** | bird: `bird_id`, `bird_released` · global: `active_bird`, `angle`, `min_angle`, `max_angle`, `angle_rate`, `angle_bias`, `angle_rad`, `angle_adjusted` |

The aim angle *decreases* while the bird is in the sling and the angle is not locked. The trig uses the post-step angle.

---

### `decreasing_force`

| | |
|---|---|
| **Parameters** | `?b - bird` |
| **Precondition** | `active_bird = bird_id`<br>not `bird_released`<br>not `force_locked`<br>`min_force ≤ v_bird_multiplier ≤ max_force` |
| **Effect** | `v_bird_multiplier -= #t · force_rate` |
| **Inputs** | bird: `bird_id`, `bird_released` · global: `active_bird`, `v_bird_multiplier`, `min_force`, `max_force`, `force_rate`, `force_locked` |

Pull strength ramps down until `set_force` locks it.

---

### `flying`

| | |
|---|---|
| **Parameters** | `?b - bird` |
| **Precondition** | `active_bird = bird_id`<br>`bird_released`<br>not `platform_slide_active`<br>`y_bird > 0` |
| **Effect** | `y_bird += #t/100 · vy_bird`<br>`vy_bird -= #t/100 · gravity`<br>`x_bird += #t/100 · vx_bird`<br>`mod ← 0` |
| **Inputs** | bird: `bird_id`, `x_bird`, `y_bird`, `vx_bird`, `vy_bird`, `bird_released`, `platform_slide_active` · global: `active_bird`, `gravity` |

Ballistic flight whenever the bird is released and not sliding.

---

### `platform_sliding`

| | |
|---|---|
| **Parameters** | `?b - bird`, `?pl - platform` |
| **Precondition** | `active_bird = bird_id`<br>`bird_released`<br>`platform_slide_active(?b)`<br>`bird_sliding(?b, ?pl)`<br>`vx_bird > 0`<br>`x_bird − 0.35r ≤ x_platform + pw/2`<br>`x_bird + 0.35r ≥ x_platform − pw/2` |
| **Learnt** | `{SE-platform-friction}` — the friction term (median of `kb._platform_friction`, default 10.0) |
| **Effect** | `y_bird ← line(pl, x_bird) + 0.35r`<br>`vy_bird ← 0`<br>`x_bird += #t/100 · vx_bird`<br>`vx_bird += #t/100 · gravity · platform_slide_accel`<br>`vx_bird -= #t/100 · 10.0` *(injected friction)*<br>`mod ← 0` |
| **Inputs** | bird: `bird_id`, `x_bird`, `vx_bird`, `bird_radius`, `bird_released`, `platform_slide_active`, `bird_sliding` · platform: `x_platform`, `platform_width`, `platform_line_c`, `platform_slope`, `platform_slide_accel` · global: `active_bird`, `gravity` |

The bird rides the platform's top edge. Slope acceleration and friction act on `vx_bird`.

---

### `block_platform_rolling`

| | |
|---|---|
| **Parameters** | `?bl - block`, `?pl - platform` |
| **Precondition** | `block_slide_active(?bl)`<br>`block_sliding(?bl, ?pl)`<br>`vx_block > 0`<br>`v_block > 0`<br>`x_block − w/2 ≤ x_platform + pw/2`<br>`x_block + w/2 ≥ x_platform − pw/2` |
| **Learnt** | `{SE-platform-friction}` — the friction term (median of `kb._platform_friction`, default 10.0) |
| **Effect** | `y_block ← line(pl, x_block) + h/2`<br>`vy_block ← 0`<br>`x_block += #t/100 · vx_block`<br>`vx_block += #t/100 · gravity · platform_slide_accel`<br>`vx_block -= #t/100 · 10.0` *(injected friction)*<br>`v_block ← vx_block`<br>`mod ← 0` |
| **Inputs** | block: `x_block`, `block_width`, `block_height`, `vx_block`, `v_block`, `block_slide_active`, `block_sliding` · platform: `x_platform`, `platform_width`, `platform_line_c`, `platform_slope`, `platform_slide_accel` · global: `gravity` |

A pushed block rolls along a platform top.

---

### `agent_movement` †

| | |
|---|---|
| **Parameters** | `?ea - external_agent` |
| **Precondition** | `angle_adjusted`<br>not `agent_dead(?ea)` |
| **Effect** | `x_agent += #t · vx_agent`<br>`y_agent += #t · vy_agent`<br>`timing_agent += #t · 200` |
| **Inputs** | external_agent: `x_agent`, `y_agent`, `vx_agent`, `vy_agent`, `timing_agent`, `agent_dead` · global: `angle_adjusted` |

External agents move once the bird has been launched.

---

## Events (instantaneous)

### `collision_ground`

| | |
|---|---|
| **Parameters** | `?b - bird` |
| **Precondition** | `active_bird = bird_id`<br>`bird_released`<br>`y_bird ≤ 0`<br>`vx_bird > 0` |
| **Learnt** | `{SE-collision-ground-effect}` — **entire effect** replaced: LR → M5 → roll placeholder, from `kb.collision`. Shown here: the current LR. |
| **Effect** *(injected)* | `x_bird ← 1.0·x − 0.4737·y + 0.0189·vx − 0.0158·vy`<br>`y_bird ← 15.0 − 3.1053·y − 0.0158·vx + 0.0632·vy`<br>`vx_bird ← 1.0·vx`<br>`vy_bird ← −200.0 + 37.7193·y + 0.1579·vx − 1.6316·vy`<br>`mod ← 2` |
| **Inputs** | bird: `bird_id`, `x_bird`, `y_bird`, `vx_bird`, `vy_bird`, `bird_released` · global: `active_bird` |

Learned ground bounce. It fires at `y ≤ 0`, not at the game floor of 360 (known bug #3).

---

### `explode_bird`

| | |
|---|---|
| **Parameters** | `?b - bird` |
| **Precondition** | `active_bird = bird_id`<br>`bird_released`<br>`bounce_count ≥ 3` **or** `x_bird ≥ 800`<br>`angle_adjusted` |
| **Effect** | `angle ← min_angle`<br>`active_bird ← active_bird + 1`<br>−`angle_adjusted` |
| **Inputs** | bird: `bird_id`, `bounce_count`, `x_bird`, `bird_released` · global: `active_bird`, `min_angle`, `angle_adjusted` |

Ends the current bird's shot and moves on to the next bird.

---

### `collision_pig_kill`

| | |
|---|---|
| **Parameters** | `?b - bird`, `?p - pig` |
| **Precondition** | `active_bird = bird_id`<br>`v_bird > 0`<br>`(x_bird − x_pig)² + (y_bird − y_pig)² ≤ (r + pig_radius − 4.0)²`<br>not `pig_dead(?p)`<br>`pig_life ≤ v_bird` |
| **Effect** | `vx_bird`, `vy_bird` ← 2D elastic-collision response, scaled by `2·m_pig/(m_bird + m_pig)`<br>+`pig_dead(?p)`<br>+`pig_killed`<br>`points_score += 5000` |
| **Inputs** | bird: `bird_id`, `v_bird`, `x_bird`, `y_bird`, `vx_bird`, `vy_bird`, `bird_radius`, `m_bird` · pig: `x_pig`, `y_pig`, `pig_radius`, `m_pig`, `pig_life`, `pig_dead` · global: `active_bird`, `points_score` |

Direct bird–pig hit. The overlap test uses a 4.0 epsilon; the Python sim uses 2.0.

---

### `collision_block_bounce_off`

| | |
|---|---|
| **Parameters** | `?b - bird`, `?bl - block` |
| **Precondition** | `active_bird = bird_id`<br>`v_bird > 0`<br>AABB overlap: `|x_bird − x_block| ≤ r + w/2`, `|y_bird − y_block| ≤ r + h/2`<br>`block_life > 0`<br>`v_bird ≥ block_stability`<br>`block_life² > (vx_bird² + vy_bird²) · bird_block_damage²` |
| **Learnt** | `{SE-block-push-ratio}` — the 0.18 factor (median of `kb._block_push`, default 0.18, not persisted) |
| **Effect** | `vx_block ← 0.18 · vx_bird` *(injected push ratio)*<br>`vy_block ← 0.18 · vy_bird`<br>`v_block ← 0.18 · v_bird`<br>`v_bird ← 0`, `vx_bird ← 0`, `vy_bird ← 0`<br>`bounce_count ← 3` |
| **Inputs** | bird: `bird_id`, `v_bird`, `x_bird`, `y_bird`, `vx_bird`, `vy_bird`, `bird_radius`, `bird_block_damage(?b,?bl)` · block: `x_block`, `y_block`, `block_width`, `block_height`, `block_life`, `block_stability` · global: `active_bird` |

The bird can't break the block: it stops dead and pushes the block.

---

### `collision_block`

| | |
|---|---|
| **Parameters** | `?b - bird`, `?bl - block` |
| **Precondition** | `active_bird = bird_id`<br>`v_bird > 0`<br>AABB overlap (as above)<br>`block_life > 0`<br>`block_life² ≤ (vx_bird² + vy_bird²) · bird_block_damage²` |
| **Learnt** | `{SE-block-push-ratio}` — the 0.18 factor (median of `kb._block_push`, default 0.18, not persisted) |
| **Effect** | `vx_block ← 0.18 · vx_bird` *(injected push ratio)*<br>`vy_block ← 0.18 · vy_bird`<br>`v_block ← 0.18 · v_bird`<br>`bounce_count ← 3` |
| **Inputs** | bird: `bird_id`, `v_bird`, `x_bird`, `y_bird`, `vx_bird`, `vy_bird`, `bird_radius`, `bird_block_damage(?b,?bl)` · block: `x_block`, `y_block`, `block_width`, `block_height`, `block_life` · global: `active_bird` |

The bird breaks through. `bounce_count ← 3` ends its shot, via `explode_bird`. Block life is *not* reduced.

---

### `remove_unsupported_block`

| | |
|---|---|
| **Parameters** | `?bl_bottom - block`, `?bl_top - block` |
| **Precondition** | `block_life(top) > 0`<br>`block_stability(top) > 0`<br>`block_life(bottom) ≤ 0` **or** `block_stability(bottom) ≤ 0`<br>`|x_block(bottom) − x_block(top)| ≤ w(top)/2`<br>`y_block(bottom) ≤ y_block(top) − h(top)/2` |
| **Effect** | `block_life(top) ← block_life(top) − 100`<br>`y_block(top) ← h(top)/2`<br>`block_stability(top) ← 0`<br>`points_score += 500` |
| **Inputs** | block (both): `x_block`, `y_block`, `block_life`, `block_stability`; top only: `block_width`, `block_height` · global: `points_score` |

The top block falls when its support is destroyed or unstable.

---

### `explode_block`

| | |
|---|---|
| **Parameters** | `?bl_tnt - block`, `?bl_near - block` |
| **Precondition** | `block_life(near) > 0`<br>`block_stability(near) > 0`<br>`|x_block(tnt) − x_block(near)| ≤ 100`<br>`|y_block(tnt) − y_block(near)| ≤ 100`<br>`block_explosive(tnt)`<br>`block_life(tnt) ≤ 0` |
| **Effect** | `block_life(near) ← 0`<br>`block_stability(near) ← 0`<br>`points_score += 1000` |
| **Inputs** | block: `x_block`, `y_block`, `block_life` (both), `block_stability` (near), `block_explosive` (tnt) · global: `points_score` |

TNT chain reaction. The radius is 100, widened from 50 to model chains. It never fires in practice because `block_explosive` is never set (bug #11).

---

### `explode_pig`

| | |
|---|---|
| **Parameters** | `?bl_tnt - block`, `?p - pig` |
| **Precondition** | not `pig_dead(?p)`<br>`block_explosive(tnt)`<br>`block_life(tnt) ≤ 0`<br>`|x_block − x_pig| ≤ 50`<br>`|y_block − y_pig| ≤ 50` |
| **Effect** | +`pig_dead(?p)`<br>+`pig_killed`<br>`points_score += 5000` |
| **Inputs** | block: `x_block`, `y_block`, `block_life`, `block_explosive` · pig: `x_pig`, `y_pig`, `pig_dead` · global: `points_score` |

A pig dies inside the TNT blast radius.

---

### `remove_unsupported_pig`

| | |
|---|---|
| **Parameters** | `?bl_bottom - block`, `?p - pig` |
| **Precondition** | not `pig_dead(?p)`<br>`block_life < 0` **or** `block_stability ≤ 0`<br>`|x_pig − x_block| ≤ w/2`<br>`y_block + h/2 ≤ y_pig ≤ y_block + h + pig_radius` |
| **Effect** | +`pig_dead(?p)`<br>+`pig_killed`<br>`points_score += 5000` |
| **Inputs** | block: `x_block`, `y_block`, `block_width`, `block_height`, `block_life`, `block_stability` · pig: `x_pig`, `y_pig`, `pig_radius`, `pig_dead` · global: `points_score` |

A pig dies when the block under it fails. Note the strict `< 0` on life here, versus `≤ 0` in `remove_unsupported_block`.

---

### `collision_block_pig_kill`

| | |
|---|---|
| **Parameters** | `?bl - block`, `?p - pig` |
| **Precondition** | not `pig_dead(?p)`<br>`block_life > 0`<br>`v_block > 10`<br>AABB overlap: `|x_block − x_pig| ≤ w/2 + pig_radius`, `|y_block − y_pig| ≤ h/2 + pig_radius` |
| **Effect** | +`pig_dead(?p)`<br>+`pig_killed`<br>`points_score += 5000` |
| **Inputs** | block: `x_block`, `y_block`, `block_width`, `block_height`, `block_life`, `v_block` · pig: `x_pig`, `y_pig`, `pig_radius`, `pig_dead` · global: `points_score` |

A moving block (pushed or rolling) crushes a pig.

---

### `collision_platform`

| | |
|---|---|
| **Parameters** | `?b - bird`, `?pl - platform` |
| **Precondition** | `active_bird = bird_id`<br>`bird_released`<br>not `platform_slide_active(?b)`<br>`v_bird > 0`<br>`|x_bird − x_platform| ≤ 1.1r + pw/2`<br>`y_bird + r ≥ line(pl, x_bird) − platform_height`<br>`y_bird − r ≤ line(pl, x_bird)`<br>`y_bird > line(pl, x_bird) + 0.35r` (approach from above) |
| **Learnt** | `{SE-collision-platform-effect}` — **entire effect** replaced. `vx_bird` and `v_bird` come from: LR `v_x` → learned ratio → heuristic ×0.35, from `kb.platform_collision`. The snap, `vy ← 0` and slide-start lines are always present. Shown here: the current learned value, 1.0·vx. |
| **Effect** *(injected)* | `y_bird ← line(pl, x_bird) + 0.35r`<br>`vx_bird ← 1.0 · vx_bird`<br>`v_bird ← 1.0 · vx_bird`<br>`vy_bird ← 0`<br>+`platform_slide_active(?b)`<br>+`bird_sliding(?b, ?pl)`<br>`mod ← 2` |
| **Inputs** | bird: `bird_id`, `x_bird`, `y_bird`, `v_bird`, `vx_bird`, `bird_radius`, `bird_released`, `platform_slide_active` · platform: `x_platform`, `platform_width`, `platform_height`, `platform_line_c`, `platform_slope` · global: `active_bird` |

First landing on a platform; the bird switches to slide mode.

---

### `platform_slide_handoff`

| | |
|---|---|
| **Parameters** | `?b - bird`, `?from - platform`, `?to - platform` |
| **Precondition** | `platform_slide_active(?b)`<br>`bird_sliding(?b, ?from)`<br>`platform_chain(?from, ?to)`<br>`vx_bird > 0`<br>`x_bird − 0.35r > x_platform(from) + pw(from)/2` |
| **Effect** | −`bird_sliding(?b, ?from)`<br>+`bird_sliding(?b, ?to)`<br>`y_bird ← line(to, x_bird) + 0.35r`<br>`vy_bird ← 0` |
| **Inputs** | bird: `x_bird`, `vx_bird`, `bird_radius`, `platform_slide_active`, `bird_sliding` · platform: `x_platform`, `platform_width` (from); `platform_line_c`, `platform_slope` (to); `platform_chain` |

The slide carries over onto the next platform in the chain.

---

### `platform_slide_stop`

| | |
|---|---|
| **Parameters** | `?b - bird`, `?pl - platform` |
| **Precondition** | `platform_slide_active(?b)`<br>`bird_sliding(?b, ?pl)`<br>`vx_bird ≤ 0`<br>`bounce_count < 3` |
| **Effect** | `vx_bird ← 0`<br>`v_bird ← 0`<br>`bounce_count ← 3` |
| **Inputs** | bird: `vx_bird`, `bounce_count`, `platform_slide_active`, `bird_sliding` |

Friction stops the bird on the platform, which ends the shot. The slide flags stay set.

---

### `platform_slide_end`

| | |
|---|---|
| **Parameters** | `?b - bird`, `?pl - platform` |
| **Precondition** | `platform_slide_active(?b)`<br>`bird_sliding(?b, ?pl)`<br>( not `platform_has_next(?pl)` **and** `x_bird − 0.35r > x_platform + pw/2` ) **or** `x_bird + 0.35r < x_platform − pw/2` |
| **Effect** | −`platform_slide_active(?b)`<br>−`bird_sliding(?b, ?pl)` |
| **Inputs** | bird: `x_bird`, `bird_radius`, `platform_slide_active`, `bird_sliding` · platform: `x_platform`, `platform_width`, `platform_has_next` |

The bird slides off the platform edge and `flying` takes over again.

---

### `collision_block_platform`

| | |
|---|---|
| **Parameters** | `?bl - block`, `?pl - platform` |
| **Precondition** | not `block_slide_active(?bl)`<br>`block_life > 0`<br>`v_block > 0`<br>`|x_block − x_platform| ≤ w/2 + pw/2`<br>`platform_top_y_min − platform_height/2 ≤ y_block − h/2 ≤ platform_top_y_max + h/2` |
| **Learnt** | Generated, not learnt: `{SE-block-platform-roll-effect}` is always replaced by this fixed body |
| **Effect** | `y_block ← line(pl, x_block) + h/2`<br>`vy_block ← 0`<br>`v_block ← vx_block`<br>+`block_slide_active(?bl)`<br>+`block_sliding(?bl, ?pl)`<br>`mod ← 2` |
| **Inputs** | block: `x_block`, `y_block`, `block_width`, `block_height`, `block_life`, `v_block`, `vx_block`, `block_slide_active` · platform: `x_platform`, `platform_width`, `platform_height`, `platform_top_y_min`, `platform_top_y_max`, `platform_line_c`, `platform_slope` |

A moving block lands on a platform and starts rolling.

---

### `block_platform_roll_handoff`

| | |
|---|---|
| **Parameters** | `?bl - block`, `?from - platform`, `?to - platform` |
| **Precondition** | `block_slide_active(?bl)`<br>`block_sliding(?bl, ?from)`<br>`platform_chain(?from, ?to)`<br>`vx_block > 0`<br>`x_block − w/2 > x_platform(from) + pw(from)/2` |
| **Effect** | −`block_sliding(?bl, ?from)`<br>+`block_sliding(?bl, ?to)`<br>`y_block ← line(to, x_block) + h/2`<br>`vy_block ← 0` |
| **Inputs** | block: `x_block`, `block_width`, `block_height`, `vx_block`, `block_slide_active`, `block_sliding` · platform: `x_platform`, `platform_width` (from); `platform_line_c`, `platform_slope` (to); `platform_chain` |

The block's roll carries over onto the next platform in the chain.

---

### `block_platform_roll_end`

| | |
|---|---|
| **Parameters** | `?bl - block`, `?pl - platform` |
| **Precondition** | `block_slide_active(?bl)`<br>`block_sliding(?bl, ?pl)`<br>`vx_block ≤ 0` **or** `v_block ≤ 0` **or** ( not `platform_has_next(?pl)` **and** `x_block − w/2 > x_platform + pw/2` ) **or** `x_block + w/2 < x_platform − pw/2` |
| **Effect** | −`block_slide_active(?bl)`<br>−`block_sliding(?bl, ?pl)` |
| **Inputs** | block: `x_block`, `block_width`, `vx_block`, `v_block`, `block_slide_active`, `block_sliding` · platform: `x_platform`, `platform_width`, `platform_has_next` |

The block stops, or rolls off the platform.

---

### `explode_block_from_bird`

| | |
|---|---|
| **Parameters** | `?b - bird`, `?bl_near - block` |
| **Precondition** | `active_bird = bird_id`<br>`bird_type = 2` (BLACK)<br>`bounce_count > 0`<br>`block_stability > 0`<br>`block_life > 0`<br>`|x_bird − x_block| ≤ 50`<br>`|y_bird − y_block| ≤ 50` |
| **Effect** | `block_life ← 0`<br>`block_stability ← 0` |
| **Inputs** | bird: `bird_id`, `bird_type`, `bounce_count`, `x_bird`, `y_bird` · block: `x_block`, `y_block`, `block_life`, `block_stability` · global: `active_bird` |

A black bird's explosion destroys nearby blocks. The `or` is commented out, so the bird must be BLACK **and** have bounced. The extra `)` after this event closes the domain early.

---

### `explode_pig_from_bird` †

| | |
|---|---|
| **Parameters** | `?b - bird`, `?p - pig` |
| **Precondition** | `active_bird = bird_id`<br>`bird_type = 2` **or** `bounce_count = 3` **or** `platform_slide_active(?b)`<br>not `pig_dead(?p)`<br>`|x_bird − x_pig| ≤ 50`<br>`|y_bird − y_pig| ≤ 50` |
| **Effect** | +`pig_dead(?p)`<br>+`pig_killed`<br>`points_score += 5000` |
| **Inputs** | bird: `bird_id`, `bird_type`, `bounce_count`, `x_bird`, `y_bird`, `platform_slide_active` · pig: `x_pig`, `y_pig`, `pig_dead` · global: `active_bird`, `points_score` |

Proximity kill: any pig within ±50 of a stopped or sliding bird.

---

### `explode_pig_from_egg` †

| | |
|---|---|
| **Parameters** | `?b - bird`, `?p - pig` |
| **Precondition** | `active_bird = bird_id`<br>`bird_type = 3` (WHITE)<br>`bounce_count = 3`<br>not `pig_dead(?p)`<br>`|x_bird − x_pig| ≤ 40`<br>`|y_bird − y_pig| ≤ 40`<br>`bird_tapped(?b)` |
| **Effect** | +`pig_dead(?p)`<br>+`pig_killed`<br>`points_score += 5000` |
| **Inputs** | bird: `bird_id`, `bird_type`, `bounce_count`, `x_bird`, `y_bird`, `bird_tapped` · pig: `x_pig`, `y_pig`, `pig_dead` · global: `active_bird`, `points_score` |

A white bird's egg kills a pig within ±40.

---

### `agent_1_timed_change_direction` †

| | |
|---|---|
| **Parameters** | `?ea - external_agent` |
| **Precondition** | `angle_adjusted`<br>not `agent_dead(?ea)`<br>`agent_type = 1` (WIZARD)<br>`timing_agent ≥ 1000` |
| **Effect** | `vx_agent ← −vx_agent`<br>`vy_agent ← −vy_agent` |
| **Inputs** | external_agent: `agent_type`, `timing_agent`, `vx_agent`, `vy_agent`, `agent_dead` · global: `angle_adjusted` |

The wizard reverses direction on a timer. `timing_agent` is never reset, so the event re-fires on every step after the threshold.

---

### `agent_x_border_change_direction` †

| | |
|---|---|
| **Parameters** | `?ea - external_agent` |
| **Precondition** | `angle_adjusted`<br>not `agent_dead(?ea)`<br>( `vx_agent > 0` **and** `x_agent + agent_width/2 ≥ x_max_border` ) **or** ( `vx_agent < 0` **and** `x_agent − agent_width/2 ≤ x_min_border` ) |
| **Effect** | `vx_agent ← −vx_agent` |
| **Inputs** | external_agent: `x_agent`, `agent_width`, `vx_agent`, `x_min_border`, `x_max_border`, `agent_dead` · global: `angle_adjusted` |

---

### `agent_y_border_change_direction` †

| | |
|---|---|
| **Parameters** | `?ea - external_agent` |
| **Precondition** | `angle_adjusted`<br>not `agent_dead(?ea)`<br>( `vy_agent > 0` **and** `y_agent + agent_height/2 ≥ y_max_border` ) **or** ( `vy_agent < 0` **and** `y_agent − agent_height/2 ≤ y_min_border` ) |
| **Effect** | `vy_agent ← −vy_agent` |
| **Inputs** | external_agent: `y_agent`, `agent_height`, `vy_agent`, `y_min_border`, `y_max_border`, `agent_dead` · global: `angle_adjusted` |

---

### `collision_agent` †

| | |
|---|---|
| **Parameters** | `?b - bird`, `?ea - external_agent` |
| **Precondition** | `active_bird = bird_id`<br>`v_bird > 0`<br>`|x_bird − x_agent| ≤ r + agent_width/1.75`<br>`|y_bird − y_agent| ≤ r + agent_height/1.75` |
| **Effect** | `v_bird ← 0`, `vx_bird ← 0`, `vy_bird ← 0`<br>`bounce_count ← 3` |
| **Inputs** | bird: `bird_id`, `v_bird`, `x_bird`, `y_bird`, `bird_radius` · external_agent: `x_agent`, `y_agent`, `agent_width`, `agent_height` · global: `active_bird` |

An external agent's hitbox stops the bird.

---

## Related discrete actions (not processes/events)

These are `:action` definitions in the same domain file. The planner chooses them; they do not fire on their own.

| Action | Parameters | Precondition | Effect | Inputs |
|--------|------------|--------------|--------|--------|
| `set_force` | `?b - bird` | `active_bird = bird_id`; not `bird_released`; not `force_locked` | +`force_locked` | `bird_id`, `bird_released`, `active_bird`, `force_locked` |
| `pa-twang` | `?b - bird` | `active_bird = bird_id`; not `bird_released`; not `angle_adjusted`; `force_locked` | `vx_bird ← v_bird·v_bird_multiplier·cosine`; `vy_bird ← v_bird·v_bird_multiplier·sinus`; `x_bird -= 16·cosine`; `y_bird -= 12·sinus`; +`bird_released`; +`angle_adjusted` | `bird_id`, `v_bird`, `x_bird`, `y_bird`, `bird_released`, `active_bird`, `v_bird_multiplier`, `cosine`, `sinus`, `angle_adjusted`, `force_locked` |
| `yellow_bird_action` | `?b - bird` | `active_bird = bird_id`; `bird_released`; not `bird_tapped`; `bounce_count = 0`; `x_bird < 800`; `bird_type = 1` | `vx_bird`, `vy_bird`, `v_bird` ×2; +`bird_tapped` | `bird_id`, `bird_released`, `bird_tapped`, `bounce_count`, `x_bird`, `bird_type`, `vx_bird`, `vy_bird`, `v_bird`, `active_bird` |
| `white_bird_action` | `?b - bird` | same as yellow, with `bird_type = 3` | `vx_bird ← 0`; `vy_bird ← 0`; `bird_radius ← bird_radius/1.5` and `bird_radius ← bird_radius/2` (two conflicting assigns to the same fluent); +`bird_tapped` | `bird_id`, `bird_released`, `bird_tapped`, `bounce_count`, `x_bird`, `bird_type`, `bird_radius`, `active_bird` |
| `black_bird_action` | `?b - bird` | `active_bird = bird_id`; `bird_released`; not `bird_tapped`; `bounce_count < 3`; `x_bird < 800`; `bird_type = 2` | +`bird_tapped`; `bounce_count ← 3` | `bird_id`, `bird_released`, `bird_tapped`, `bounce_count`, `x_bird`, `bird_type`, `active_bird` |
| `switch_bird` † | `?b1 - bird` | `bird_released`; `active_bird = bird_id`; `vx_bird = 0` | `active_bird += 1`; −`angle_adjusted`; `angle ← 0` | `bird_released`, `bird_id`, `vx_bird`, `active_bird` |
