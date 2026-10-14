# Calibration — Mode B: Actuator (release-point) resolution floor

**Status**: Open. Not blocking wins today but caps how tight our calibration
can ever get. Fix before any further work on angle-calibration accuracy.

**Related code**:
- `src/trajectory_planner/trajectory_planner.py` — `SimpleTrajectoryPlanner.find_release_point_partial_power`
- `agents/pddl/pddl_files/pddl_parser.py` — `pddl_shot_to_release_point`
- `agents/pddl/pddl_files/bambirds_shot_helper.py` — BamBirds angle correction (Mode A fix)

---

## Two calibration failure modes

We identified two independent sources of "PDDL asked for angle X, game
produced angle Y ≠ X" during the Aug 2026 calibration work.

### Mode A — Angle-dependent sling deficit (FIXED)

The game's sling always launches slightly **flatter** than the pull angle.
Deficit magnitude depends on angle (larger at low angles, ~0 near vertical).
Root cause: gravity acts on the bird during the snap-back animation,
tilting the release-velocity vector downward. Bigger effect on flatter
shots because horizontal snap-back travel is longer.

**Fix**: BamBirds' `actual_to_launch(θ)` polynomial, ported verbatim in
`bambirds_shot_helper.py`. When PDDL wants the bird to fly at θ, we ask
the game to pull at `actual_to_launch(θ) = θ + fChangeAngle(θ)` so that
the game's flatter fire produces the θ we actually want.

After the fix (run `run_20260822_190314.log`):
- Systematic negative bias eliminated (was −2.6° mean, now −1.3° mean)
- Sign of Δθ now roughly symmetric (was 100 % negative, now 58 % negative)
- Win rate jumped from ~75 % to 6/6 on training templates

### Mode B — Actuator resolution (OPEN — this doc)

Even with the perfect angle correction, we cannot express fine-grained
angles to the game because the release point is an integer pixel and
the pullback distance is tiny.

---

## The problem in detail

`SimpleTrajectoryPlanner.find_release_point_partial_power` computes the
release coordinate as:

```python
mag = sling.height * 5 * v_portion
release = Point2D(
    round(ref.X - mag * cos(theta)),
    round(ref.Y + mag * sin(theta)),
)
```

For ScienceBirds 6.6, `sling.height` from the vision module is
consistently **3–4 px**. So for a full-force shot:

```
mag = 3.5 * 5 * 1.0 ≈ 18 px
```

An 18 px pullback swept over the full angular range 0–90° gives us:

```
angular resolution = 18 px / 90° ≈ 0.2 px per degree
```

Because release coordinates are integers, the smallest angular change we
can express is `1 px / 0.2 (px/deg) ≈ 5°`. Any request finer than that
snaps to the nearest reachable release pixel.

## Evidence

From `run_20260822_190314.log`, three shots requesting slightly different
angles all landed on the same release pixel and produced the same
outcome:

| flight_target | BamBirds request | release (rounded) | actual θ |
|---|---|---|---|
| 55.00° | 56.84° | **(87, 268)** | **58.81°** |
| 55.50° | 57.31° | **(87, 268)** | **58.81°** |
| 56.00° | 57.79° | **(87, 268)** | **58.81°** |

Same in the other direction:

| flight_target | BamBirds request | release (rounded) | actual θ |
|---|---|---|---|
| 67.5° | 68.71° | (90, 270) | 61.75° |
| 68.5° | 69.66° | **(91, 267)** | **62.72°** (1 px shift = 1° change) |

Across all 24 shots in the run, the game only ever produced **~13 distinct
launch angles** — 43.2, 46.9, 50.6, 50.9, 58.8, 59.4, 61.75, 61.85, 62.7,
68.4, 73.2, 73.9, 80.15° — one per reachable release pixel.

BamBirds' typical corrections are 1–2°, which is less than one release
pixel's worth of movement. So the correction *is* computed but *is not*
transmitted to the game. Only very large corrections (>4°, roughly at
vertical shots) actually shift the release pixel.

## Why this doesn't block wins today

- ENHSP has retry logic (`planner_diversified`) that samples alternative
  angles. Given ~13 reachable actual angles across 0–90°, at least one
  will usually fall close to an ENHSP plan.
- Levels have some tolerance — a shot that arrives 5° off can still kill
  the pig via ground roll, block cascade, or lucky obstacle bounce.
- The observed win rate is 6/6 on training templates, mostly on
  attempts 2–5.

## Why it *does* matter

- **Attempt-1 win rate is capped**: since ENHSP plans for a specific angle
  and the game snaps to the nearest bin, the first shot often lands 2–3°
  off — sometimes enough to hit an obstacle instead of the pig.
- **BamBirds calibration is effectively wasted**: we ported a
  high-fidelity angle-correction polynomial that can only deliver ~5°
  precision to the game.
- **Any future calibration work (per-angle bias learning, per-level
  scaling, force-dependent lookup tables) will hit the same 5° ceiling**
  until Mode B is addressed.
- **Templates with tight platform-clearance geometry** (thin gaps,
  overhangs) may become unwinnable at any angle because the exact
  clearance angle isn't in the reachable set.

## Root cause diagnosis

Two independent factors combine to produce the small pullback:

1. **Vision reports `sling.height ≈ 3–4 px`**. Real slingshots in the
   game screenshot are much taller (~40–60 px based on visual
   inspection). Whatever CV detector we use for the sling bounding box
   is probably returning only the visible-nub top portion, not the full
   sling body.

2. **`SimpleTrajectoryPlanner` was tuned for classic AngryBirds** where
   `sling.height * 5` gave a reasonable ~200 px pullback. That formula
   makes no sense when the input `sling.height` is 3–4 px — but it was
   never re-tuned for our vision output.

## Fix candidates

### Option 1 — Enlarge the virtual pullback in `SimpleTrajectoryPlanner`

Replace:

```python
mag = sling.height * 5 * v_portion
```

with something like:

```python
mag = max(sling.height * 5, 200) * v_portion   # or a fixed 1000 like BamBirds
```

**Pros**: One-line change. Immediate ~10× better angular resolution.
**Risk**: SB6.6 may use pull distance to determine launch velocity. If it
does *not* clamp at max_pull, a 10× pullback gives 10× velocity. Would
break partial-force shots and change every calibrated constant.

**Test to de-risk (cheap)**: fire two shots at the same angle, one with
`mag=20` and one with `mag=200`. Measure the launch speed from GT
trajectory. If identical, SB6.6 clamps and the enlargement is safe. If
different, we know the game reads pull distance and this option is out.

BamBirds itself uses `mag = 1000` in `ShotHelper.angleToReleasePoint`,
confirming that at least the classic AB engine clamps.

### Option 2 — Fix vision to report the real sling height

Track down which module produces the `sling.height` value and correct
its bounding-box logic for the SB6.6 sling sprite. If it returns
~40 px instead of ~3.5 px, `mag = 40 * 5 = 200` naturally and Option 1
is unnecessary.

**Pros**: Root-cause fix. Also fixes any other consumer of `sling.height`
that might be silently miscalibrated.
**Risk**: Bigger surgery. May cascade into other vision-dependent code
(e.g., `get_reference_point` uses `sling.width`, not height — but if the
detector under-reports one dimension, it may under-report others too).

### Option 3 — Sub-pixel release protocol

If `ActionRobot.cshoot` / `shoot_and_record_ground_truth` accepts floats
over the SB TCP protocol, pass fractional release coordinates and let
the game interpolate.

**Pros**: True sub-pixel precision without changing pullback.
**Risk**: SB protocol probably wants ints — need to check the wire
format. If it accepts only ints, this option is closed.

## Recommended sequence

**Implementation status (2026-08-22 late)**: Mode B knob is live in
`pddl_parser.py` as `MODE_B_MAG_MULTIPLIER` (default **4.0**) and
`MODE_B_MIN_MAG_AT_FULL_FORCE` (default **72.0** px at force=1.0).
The PDDL agent uses `pddl_release_point_with_mag(sling, θ, mag_px)`.
Coincident-contact pig kills at platform touch use full `bird_r + pig_r`
radius in `simulate_pddl_shot_plan` (see `_bird_kills_pig(..., coincident=True)`).

Every shot logs `[MODE B DIAG]` with `sling.height`, `v_portion`, the
raw `base_mag`, the `scaled_mag` actually sent, and the resulting
`angular_res` in px/°. Compare against `[SHOT DIAG POST]` `|v|` and
`Δθ` to see the effect.

**Calibration protocol**:

1. Run baseline at `MODE_B_MAG_MULTIPLIER = 1.0`, note mean `|v|` and
   `Δθ` distribution (already have this from `run_20260822_190314.log`).
2. Bump to **2.0**, re-run one template. Check that partial-force shots
   (`force < 1.0`) still show `|v|` roughly halved vs `force=1.0` shots
   at the same angle. If yes → safe, continue.
3. Bump to **4.0**, then **8.0**. Watch for the point where `|v|` at
   `force=1.0` stops growing (velocity clamp — that's our max_pull).
4. Set `MODE_B_MAG_MULTIPLIER` to the largest value that still shows
   proportional velocity response for the partial-force range we
   actually use (typically 0.6–1.0 → so we need proportionality above
   `0.6 × MULT` px pullback).
5. If step 4 fails (i.e., no value gives both good resolution *and* good
   partial force), fall back to Option 2 (vision fix for `sling.height`).
6. After Mode B is dialed in, revisit BamBirds coefficients — they were
   tuned on classic AB, and now that our corrections can actually reach
   the game we can measure the residual bias and refit the polynomial
   to our SB6.6 data (see `recalculate_scaling_factor` in
   `bambirds_shot_helper.py`).

## Historical context

- 2026-08-22, `run_20260822_182936.log`: first diagnostic run showing
  ~118 px position mismatch — later confirmed to be a phantom from a
  coordinate-system flip bug (Y-up PDDL vs Y-down screen).
- 2026-08-22, `run_20260822_183646.log`: coordinate bug fixed;
  systematic −2.6° angle bias exposed.
- 2026-08-22, `run_20260822_185323.log`: BamBirds correction added but
  showed almost no effect — first evidence of Mode B.
- 2026-08-22, `run_20260822_190314.log`: with `ANGLE_BIAS_DEGREES = 6`
  still in place (Mode A applied), win rate jumped to 6/6; residual ±3°
  variance around zero attributed to Mode B pixel snapping.

## References

- BamBirds `ShotHelper.java`: <https://github.com/dwolter/BamBirds/blob/master/common/src/main/java/de/uniba/sme/bambirds/common/utils/ShotHelper.java>
- BamBirds `Settings.java` (RedBird coefficients): <https://github.com/dwolter/BamBirds/blob/master/common/src/main/java/de/uniba/sme/bambirds/common/utils/Settings.java>
- AIBIRDS PDDL+ paper (same architecture as ours): <https://ojs.aaai.org/index.php/ICAPS/article/download/27232/27005/31301>
