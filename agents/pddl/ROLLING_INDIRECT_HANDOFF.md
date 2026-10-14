# Rolling indirect kill — handoff context (Sep 2026)

Use this doc when continuing work in a new chat. Scenario: **Phy-Q rolling t05**, **block-aim was FF**; shifted to **ENHSP + new PDDL block roll / block→pig kill**.

---

## Goal arc

1. **Local learning (Python):** bird→block impulse LR, block→pig GT contacts, indirect-win conditions from logs.
2. **PDDL (ENHSP):** model indirect kill — block rolls on platform, hits pig — so planner can find roll→kill plans (not only ballistic FF).
3. **Runtime:** block-aim FF **disabled by default** so ENHSP runs.

---

## Code changes (by area)

### Block impulse LR (`agents/pddl/pddl_files/events/learn_events.py`)

- **Primary variant only:** `baseline_rel_geom_mass_size` (8-variant grid commented out).
- **`[BLOCK IMPULSE LR COMPARE]`** disabled (no multi-variant training/log).
- **`train_block_impulse_lr_all_variants`** trains primary only.
- KB: `lr_models` = primary; `lr_models_by_variant` still one entry.

### Block→pig GT analysis (`agents/pddl/pddl_files/events/block_pig_contact.py`)

- Unpadded pig track from `groundtruth_trajectories` (segment padding was hiding pig death).
- On **game WIN:** `analyze_block_pig_shot(..., game_won=True)` + **`[BLOCK-PIG WIN]`** logs (called from `pddl_agent` after outcome).
- Kill frame heuristics: bird hit → pig vanished from GT → impulse peak → closest block–pig contact.
- **`[BLOCK-PIG]` labels** still weak on losses (`hard_contact_no_kill`, `pig_still_active`) because `damage_est` thresholds don’t match Unity-scale numbers and pig settle detector often misses SB death frame.

### PDDL domain (`agents/pddl/pddl_files/base_domain.pddl`)

**Injected into `base_domain_modified.pddl` on every ENHSP plan via `inject_domain_file()`.**

| Addition | Purpose |
|----------|---------|
| `(v_block ?bl) (vx_block ?bl) (vy_block ?bl)` | Block speed for roll + kill |
| `(block_slide_active ?bl) (block_sliding ?bl ?pl)` | Durative roll on platform |
| **`collision_block`** effect | Copies bird speed → `vx_block` / `vy_block` / `v_block` on bird–block hit |
| **`collision_block_platform`** | Block AABB ∩ platform → `{SE-block-platform-roll-effect}` |
| **`block_platform_rolling`** process | Same idea as `platform_sliding` for bird; updates `x_block`, `v_block` |
| **`block_platform_roll_end`** | Stops block slide |
| **`collision_block_pig_kill`** | Pig not dead, **`(> (v_block ?bl) 50)`**, block–pig AABB overlap (pig uses `pig_radius` as half-extent) → `(pig_dead ?p) (pig_killed)` |

**Not in PDDL yet:** block-impulse LR post-state on `collision_block`; forward sim block roll chain.

### Injection (`agents/pddl/pddl_files/pddl_parser.py`)

- Sentinel **`{SE-block-platform-roll-effect}`** → `_inject_block_platform_roll_sentinel`.
- **Bootstrap:** reuses **`platform_collision` KB** (same as bird platform slide): rolling LR → PDDL v_x ratio → heuristic **`PLATFORM_DEFAULT_SPEED_RATIO` (0.5)**.
- Log label: `block_platform_roll=block_roll_sim_heuristic` | `sim_ratio_vx` | `sim_aligned`.
- **`generate_pddl`:** inits `(= (v_block …) 0)`, `vx_block`, `vy_block` per `block_*`.

### Agent / CLI

- **`block_aim_override` default `False`** in `main.py` and `PDDLAgent` → ENHSP path by default.
- Re-enable FF: **`--block-aim-override`**.
- ENHSP inject log includes **`block_platform_roll=…`**.

### Scripts / tmp

- `scripts/analyze_block_pig_kills.py` — parse `[BLOCK-PIG]` from logs.
- `_tmp_indirect_kill_analysis.py` — win vs loss shot comparison (optional).

---

## Log findings (empirical indirect kill)

### What “indirect win” means in logs

- **`[SIM/REAL] MISMATCH: sim predicted miss but level WON (indirect/collapse path)`** — game won via block/collapse; forward sim (bird-only) predicted miss.
- **`hit: []`** — no direct bird→pig GT event.

### Win chain (typical, e.g. `rolling_t05_00078`, `00046`)

1. Bird hits **on-platform wood block** (GT bird–block ~f104–127).
2. **Platform collision** frames — block/bird slide on hill.
3. Many **block→pig** swept contacts; wins tend **closer separation** (~8–10 px on `00078`) vs losses often stuck ~15–18 px.
4. **`[BLOCK-PIG WIN]`** (when game WIN): chain lines `bird_block`, `platform`, `first_block_pig`, `kill_contact`, derived **conditions** string.

### Reference runs

| Log | Notes |
|-----|--------|
| `log/run_20260921_135932.log` | Full train pass; **1/19 win** (`00046` only); `00078` abandoned; **`[BLOCK-PIG WIN]`** on `00046` (kill @ pig vanished ~f187, kill contact f173 sep 16.6). |
| `log/run_20260921_130936.log` | Earlier; 2 wins `00078`, `00046`; no `[BLOCK-PIG WIN]` (before WIN refine). |
| Checkpoint | `saved_models/world_model_20260921_135932.json` — platform n≈111, block impulse n≈43 at end of that run. |

### PDDL vs GT gap

- ENHSP domain now has **static block motion + kill event**; **does not** yet guarantee same physics as game.
- **`collision_block_pig_kill`** threshold **`v_block > 50`** and AABB overlap — calibrated directionally from logs, not fully fit.

---

## First ENHSP trial checklist

**No manual edit of `base_domain_modified.pddl` required** — first plan run regenerates from `base_domain.pddl`.

1. **FF off:** log shows `block_aim_override=False`, not `Block-aim FF ON`.
2. **Java** on PATH; `agents/pddl/pddl_files/enhsp-20.jar` present.
3. **Optional:** `--world-model-load saved_models/world_model_20260921_135932.json` for warmer platform (→ block roll) inject.
4. Confirm inject line: **`block_platform_roll=…`** and modified domain **~750+ lines** (not stale ~674 without block events).

**Defaults (main):** `planner_only=True`, `platform_aim_override=False`, `block_aim_override=False`.

---

## Open / next steps

1. **Forward sim:** block on platform + roll + `collision_block_pig_kill` so sim gate matches ENHSP indirect plans.
2. **Inject block impulse LR** into `collision_block` (post bird–block velocities) instead of copying `v_bird`.
3. **Tune `[BLOCK-PIG]`** — pig death frame from GT (WIN + last pig frame), fix `damage_est` / `light_contact` thresholds.
4. **ENHSP solvability** — rolling levels may need Pass B (shelter/slide); expect slow/unsolvable until layout + physics align.
5. **Optional:** dedicated `block_platform_collision` KB instead of bootstrapping from bird platform KB.

---

## Key paths

```
agents/pddl/pddl_files/base_domain.pddl          # source domain (+ sentinels)
agents/pddl/pddl_files/base_domain_modified.pddl  # generated each ENHSP inject
agents/pddl/pddl_files/events/learn_events.py   # block impulse LR
agents/pddl/pddl_files/events/block_pig_contact.py
agents/pddl/pddl_files/pddl_parser.py           # inject_domain_file, block roll inject
agents/pddl/pddl_agent.py                       # solve, ENHSP, BLOCK-PIG WIN
main.py                                         # CLI defaults (FF off)
PHYQ_SCENARIO_CAPABILITIES.md                   # rolling = IP + RO + BI
```

---

## One-line summary

**Python learns bird→block impulse and logs indirect win chains; PDDL now has block platform roll (injected like bird slide) and block→pig kill at `v_block>50` + AABB; FF is off so ENHSP can plan — sim and impulse injection still lag the game.**
