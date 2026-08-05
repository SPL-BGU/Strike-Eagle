from string import Template

import math
import numpy as np
import os

from agents.pddl.pddl_files.world_model.params import Params
from agents.pddl.pddl_files.world_model.world_model import WorldModel
from agents.pddl.pddl_files.events.learn_events import m5_collision_leaves_for_pddl, make_feature_vector

# Print [COLLISION-INJECT] lines during domain injection.
COLLISION_INJECT_DEBUG = True

# Force search constants (joint angle × force PDDL planning)
FORCE_MIN = 0.2
FORCE_MAX = 1.0
FORCE_RATE = 0.1
MAG_MULT = 0.5          # keeps force→velocity in linear regime (force_velocity_curve_05.png)
FORCE_V_SCALE = MAG_MULT / 5.0   # = 0.1; converts force fraction to find_release_point_partial_power v_portion

# Shrink effective pig hit radius so PDDL requires a deeper hit (avoids top-graze false kills).
PIG_HIT_EPSILON = 5.5


def pddl_force_to_v_portion(force: float) -> float:
    """
    Map PDDL ``v_bird_multiplier`` (0.2–1.0) to ``SimpleTrajectoryPlanner`` v_portion.

    ``find_release_point_partial_power`` uses ``mag = sling.height * 5 * v_portion``.
    Force-learning and normal execution both target ``mag = sling.height * MAG_MULT * force``.
    Therefore ``v_portion = MAG_MULT * force / 5 = force * FORCE_V_SCALE``.

    Note: ``v_portion=1.0`` in the trajectory planner API is *full* screen pullback
    (``mag = height * 5``). PDDL force 1.0 maps to ``v_portion=0.1`` — not full pull —
    by design so shots stay in the linear force→velocity regime.
    """
    force = max(FORCE_MIN, min(FORCE_MAX, float(force)))
    return force * FORCE_V_SCALE


def pullback_pixels(sling_height: float, force: float) -> float:
    """Slingshot pullback magnitude (px) for a PDDL force fraction."""
    force = max(FORCE_MIN, min(FORCE_MAX, float(force)))
    return float(sling_height) * MAG_MULT * force


def launch_speed_at_force(v_bird: float, force: float, force_lr_model=None) -> float:
    """
    Expected launch speed |v| for a PDDL force fraction.

    Default (no learned model): ``v_bird * force``, matching base_domain ``pa-twang``:
      ``vx = v_bird * v_bird_multiplier * cos(angle_rad)``.

    When ``force_lr_model`` is fitted from observed shots, uses affine ``a*force + b``
    which better matches the game's non-zero intercept curve.
    """
    force = max(FORCE_MIN, min(FORCE_MAX, float(force)))
    if force_lr_model is not None:
        try:
            return float(force_lr_model.predict(np.array([[force]], dtype=float))[0])
        except Exception:
            pass
    return float(v_bird) * force


def _collision_inject_log(msg: str) -> None:
    if COLLISION_INJECT_DEBUG:
        print(f"[COLLISION-INJECT] {msg}")


_COLLISION_VAR_ORDER = ("y", "v_x", "v_y")
_COLLISION_PDDL_FLUENT = {"y": "y_bird", "v_x": "vx_bird", "v_y": "vy_bird"}
_PDDL_AFFINE_VARS = ("x_bird", "y_bird", "vx_bird", "vy_bird")

PLATFORM_COLLISION_MIN_SAMPLES = 5
PLATFORM_MAX_POST_SPEED = 220.0
# Observed platform slides (Template 4): post_speed varies; use conservative 50% for
# cold-start planning so ENHSP does not assume aggressive shallow slides reach the pig.
PLATFORM_DEFAULT_SPEED_RATIO = 0.50
PLATFORM_DEFAULT_VY_RATIO = 0.50
PLATFORM_SURFACE_RADIUS_FACTOR = 0.35
PLATFORM_SWEEP_SUBSTEPS = 5
PLATFORM_MARGIN_MULT = 1.1
# Measured launch angle often deviates from PDDL flight angle; validate plans at ±slack.
ANGLE_EXECUTION_SLACK_DEG = 4.0
# Platform/gap levels: wider slack band for nonlinear slingshot + overshoot-up misses.
PLATFORM_EXECUTION_SLACK_DEG = 6.0
# When plan clears platforms (gap/over-flight), nudge dial down so game overshoot-up misses top shelf.
GAP_CLEARANCE_ANGLE_NUDGE_DEG = 2.0
# Bootstrap: platform is non-changeable — terminate flight without mutating vx/vy/y/v_bird.
PLATFORM_COLLISION_BOOTSTRAP = (
    "(assign (bounce_count ?b) 3)\n            "
    "(assign (mod) 2)"
)
# Legacy full zero-out (used when explicit stop semantics are required elsewhere).
PLATFORM_COLLISION_HARD_STOP = (
    "(assign (v_bird ?b) 0)\n            "
    "(assign (vx_bird ?b) 0)\n            "
    "(assign (vy_bird ?b) 0)\n            "
    "(assign (bounce_count ?b) 3)\n            "
    "(assign (mod) 2)"
)


def _build_platform_bootstrap_effect() -> str:
    """PDDL effect when platform KB is cold — end flight, no bird state mutation."""
    return PLATFORM_COLLISION_BOOTSTRAP


def _build_platform_slide_placeholder() -> str:
    """
    PDDL effect when platform KB has samples but M5 is not ready yet.
    Bird lands on platform top, retains ~50% horizontal/vertical speed components.
    v_bird is left unchanged (no sqrt in ENHSP); vx/vy drive continued flight.
    """
    ratio = PLATFORM_DEFAULT_SPEED_RATIO
    vy_ratio = PLATFORM_DEFAULT_VY_RATIO
    surf = PLATFORM_SURFACE_RADIUS_FACTOR
    return (
        f"(assign (y_bird ?b) (+ (+ (y_platform ?pl) (/ (platform_height ?pl) 2)) "
        f"(* (bird_radius ?b) {surf})))\n            "
        f"(assign (vx_bird ?b) (* (vx_bird ?b) {ratio}))\n            "
        f"(assign (vy_bird ?b) (* (vy_bird ?b) {vy_ratio}))\n            "
        f"(assign (bounce_count ?b) (+ (bounce_count ?b) 1))\n            "
        f"(assign (mod) 2)"
    )
# segments.calculate_features uses y_pddl - 360 (same frame as ground collision learning)
_LEARNING_Y_OFFSET = 360.0


def format_affine_assign_rhs(coefs, intercept, bird_var="?b", threshold=1e-8, precision=4):
    """
    Build nested (+ … (* c (fluent ?b)) …) PDDL numeric expression for affine bird fluents.
    coefs: length-4 iterable matching x, y, vx, vy.
    """
    c = np.asarray(coefs, dtype=float).reshape(-1)
    if c.size < 4:
        pad = np.zeros(4, dtype=float)
        pad[: c.size] = c
        c = pad
    else:
        c = c[:4]
    bias = float(intercept)
    fmt = f"{{:.{precision}f}}"

    def nest_terms(term_list):
        if not term_list:
            return "0.0"
        if len(term_list) == 1:
            return term_list[0]
        return f"(+ {term_list[0]} {nest_terms(term_list[1:])})"

    terms = [
        f"(* {fmt.format(float(co))} ({v} {bird_var}))"
        for co, v in zip(c, _PDDL_AFFINE_VARS)
        if abs(float(co)) >= threshold
    ]
    nested_terms = nest_terms(terms[:4])
    if abs(bias) > threshold:
        return f"(+ {fmt.format(bias)} {nested_terms})"
    return nested_terms if nested_terms else "0.0"


def _build_collision_ground_effect_m5(collision_vars):
    """Backward-compatible alias for ground bounce M5 injection."""
    return _build_collision_effect_m5(collision_vars, bounce_mode="ground")


def _build_collision_effect_m5(collision_vars, bounce_mode="ground"):
    """
    Build M5 piecewise collision effect for PDDL injection.

    bounce_mode:
      - 'ground': increment bounce_count (bird may continue after bounce)
      - 'platform': increment bounce_count + update v_bird magnitude (slide, not hard stop)

    Returns effect body string, or None if M5 models not available yet.
    """
    per_var_leaves = {}
    for vn in _COLLISION_VAR_ORDER:
        vd = collision_vars.get(vn, {})
        mc = vd.get("model_comparison")
        m5 = mc.get("m5_model") if isinstance(mc, dict) else None

        leaves = m5_collision_leaves_for_pddl(m5, debug=COLLISION_INJECT_DEBUG, var_label=vn)
        if not leaves:
            _collision_inject_log(f"M5 not ready for '{vn}' ({bounce_mode})")
            return None

        _collision_inject_log(f"M5 '{bounce_mode}/{vn}': {len(leaves)} leaf(s)")
        per_var_leaves[vn] = leaves

    lines = []
    for vn in _COLLISION_VAR_ORDER:
        pb = _COLLISION_PDDL_FLUENT[vn]
        for path, coef, intercept in per_var_leaves[vn]:
            rhs = format_affine_assign_rhs(coef, intercept)
            if not path:
                lines.append(f"(assign ({pb} ?b) {rhs})")
            else:
                cond = path[0] if len(path) == 1 else "(and " + " ".join(path) + ")"
                lines.append(f"(when {cond} (assign ({pb} ?b) {rhs}))")

    if bounce_mode == "platform":
        # Do not assign v_bird here — ENHSP numeric fluents have no sqrt(); prior v_bird
        # stays > 0 so the bird can keep flying on vx/vy (same as ground M5).
        lines.append("(assign (bounce_count ?b) (+ (bounce_count ?b) 1))")
        lines.append("(assign (mod) 2)")
    else:
        lines.append("(assign (bounce_count ?b) (+ (bounce_count ?b) 1))")

    body = "\n            ".join(lines)
    _collision_inject_log(f"M5 {bounce_mode} injection: {len(lines)} statements")
    return body


def _learning_y_to_pddl(learning_y: float) -> float:
    return float(learning_y) + _LEARNING_Y_OFFSET


def _pddl_y_to_learning(pddl_y: float) -> float:
    return float(pddl_y) - _LEARNING_Y_OFFSET


def _platform_kb_ready(platform_kb: dict) -> bool:
    if not platform_kb:
        return False
    states = platform_kb.get("states") or []
    if len(states) < PLATFORM_COLLISION_MIN_SAMPLES:
        return False
    return _platform_kb_has_models(platform_kb)


def _platform_kb_has_models(platform_kb: dict) -> bool:
    if not platform_kb:
        return False
    states = platform_kb.get("states") or []
    if len(states) < 1:
        return False
    for vn in _COLLISION_VAR_ORDER:
        model = platform_kb.get("variables", {}).get(vn, {}).get("model")
        if model is None:
            return False
    return True


def _clamp_platform_post_velocity(vx: float, vy: float, pre_vx: float, pre_vy: float):
    """Cap learned post-platform speed to avoid M5 blow-ups with few samples."""
    pre_speed = math.hypot(pre_vx, pre_vy)
    cap = PLATFORM_MAX_POST_SPEED
    if pre_speed > 0:
        cap = min(PLATFORM_MAX_POST_SPEED, max(pre_speed * 1.25, 40.0))
    speed = math.hypot(vx, vy)
    if speed > cap and speed > 0:
        scale = cap / speed
        vx *= scale
        vy *= scale
    return vx, vy


def predict_platform_collision_post_state(platform_kb: dict, x: float, y_pddl: float,
                                          vx: float, vy: float):
    """
    Predict post-platform-contact state using General models from platform_collision KB.
    Inputs/outputs use PDDL coordinates for y and vy.
    """
    if not _platform_kb_has_models(platform_kb):
        return None

    pre_vx, pre_vy = float(vx), float(vy)
    pre = {
        "x": float(x),
        "y": _pddl_y_to_learning(y_pddl),
        "v_x": pre_vx,
        "v_y": pre_vy,
    }
    X = make_feature_vector([pre])
    predicted = {}
    for var in _COLLISION_VAR_ORDER:
        model = platform_kb["variables"][var]["model"]
        if hasattr(model, "poly_features"):
            X_in = model.poly_features.transform(X)
            val = float(model.predict(X_in)[0])
        else:
            val = float(model.predict(X)[0])
        if var == "y":
            val = _learning_y_to_pddl(val)
        predicted[var] = val

    out_vx, out_vy = _clamp_platform_post_velocity(
        predicted["v_x"], predicted["v_y"], pre_vx, pre_vy
    )
    predicted["v_x"] = out_vx
    predicted["v_y"] = out_vy
    return predicted


def _platform_bounds(plat: dict):
    half_w = plat["w"] / 2
    half_h = plat["h"] / 2
    return (
        plat["x"] - half_w,
        plat["x"] + half_w,
        plat["y"] - half_h,
        plat["y"] + half_h,
    )


def _extend_platforms_for_pig(platforms: list, px: float, py: float, pr: float):
    """Extend platform AABB upward when a pig stands on the hill surface above the MBR top."""
    if px is None or py is None or not platforms:
        return platforms
    extended = []
    for plat in platforms:
        p = dict(plat)
        left, right, bottom, top = _platform_bounds(p)
        if left - pr <= px <= right + pr:
            pig_surface = py - pr
            if pig_surface > top:
                extra = pig_surface - top
                p["h"] = p["h"] + extra
                p["y"] = p["y"] + extra / 2
        extended.append(p)
    return extended


def extend_platforms_in_problem_data(problem_data: dict, debug: bool = False) -> dict:
    """
    Extend platform objects in problem_data so PDDL and forward sim share hill geometry.

    Vision hill MBRs are often shorter than the surface the pig stands on; extend upward
    to the pig stand line when the pig is horizontally over the platform.
    """
    pig_key = next((k for k in problem_data if k.startswith("pig_")), None)
    if not pig_key:
        return problem_data

    pig = problem_data[pig_key]
    px = float(pig["x_pig"])
    py = float(pig["y_pig"])
    pr = float(pig.get("pig_radius", 3.5))

    platforms = []
    for key, val in problem_data.items():
        if not key.startswith("platform_"):
            continue
        platforms.append({
            "name": key,
            "x": float(val["x_platform"]),
            "y": float(val["y_platform"]),
            "w": float(val["platform_width"]),
            "h": float(val["platform_height"]),
        })

    if not platforms:
        return problem_data

    extended = _extend_platforms_for_pig(platforms, px, py, pr)
    for plat in extended:
        key = plat["name"]
        old = problem_data[key]
        old_y = float(old["y_platform"])
        old_h = float(old["platform_height"])
        if plat["y"] == old_y and plat["h"] == old_h:
            continue
        _, _, _, old_top = _platform_bounds(
            {"x": plat["x"], "y": old_y, "w": plat["w"], "h": old_h}
        )
        _, _, _, new_top = _platform_bounds(plat)
        if debug:
            print(
                f"[PLATFORM DEBUG] {key}: extended for pig stand line "
                f"(top {old_top:.1f} -> {new_top:.1f}, "
                f"y {old_y:.1f}->{plat['y']:.1f}, h {old_h:.1f}->{plat['h']:.1f})"
            )
        problem_data[key] = {
            **old,
            "x_platform": plat["x"],
            "y_platform": plat["y"],
            "platform_width": plat["w"],
            "platform_height": plat["h"],
        }

    return problem_data


def _bird_overlaps_platform(x: float, y: float, br: float, plat: dict,
                            margin_mult: float = PLATFORM_MARGIN_MULT) -> bool:
    left, right, bottom, top = _platform_bounds(plat)
    margin = br * margin_mult
    return (
        x - margin <= right
        and x + margin >= left
        and y + margin >= bottom
        and y - margin <= top
    )


def _sweep_platform_contact(x0: float, y0: float, x1: float, y1: float, br: float, plat: dict):
    """Return (hit, contact_x, contact_y) along motion segment."""
    substeps = max(2, PLATFORM_SWEEP_SUBSTEPS)
    for i in range(substeps + 1):
        t = i / substeps
        sx = x0 + (x1 - x0) * t
        sy = y0 + (y1 - y0) * t
        if _bird_overlaps_platform(sx, sy, br, plat):
            return True, sx, sy
    return False, x0, y0


def _default_platform_slide_post_state(plat: dict, x: float, y: float,
                                       vx: float, vy: float, br: float):
    """
    Heuristic hill slide when platform KB is cold.
    Matches observed game behavior: ~50% speed retention after platform contact (conservative).
    """
    _, _, _, plat_top = _platform_bounds(plat)
    pre_speed = math.hypot(vx, vy)
    if pre_speed < 5.0:
        return None

    ratio = PLATFORM_DEFAULT_SPEED_RATIO
    out_vx = vx * ratio
    out_vy = vy * PLATFORM_DEFAULT_VY_RATIO
    if abs(out_vx) < 8.0:
        out_vx = math.copysign(max(pre_speed * ratio, 8.0), vx if abs(vx) > 1e-6 else 1.0)
    surface_y = plat_top + br * 0.35
    out_vx, out_vy = _clamp_platform_post_velocity(out_vx, out_vy, vx, vy)
    return {"y": surface_y, "v_x": out_vx, "v_y": out_vy}


def _resolve_platform_post_state(platform_kb: dict, plat: dict, x: float, y: float,
                               vx: float, vy: float, br: float, debug: bool = False):
    predicted = predict_platform_collision_post_state(platform_kb, x, y, vx, vy)
    if predicted is not None:
        if debug:
            post_speed = math.hypot(predicted["v_x"], predicted["v_y"])
            print(
                f"[SIM DEBUG]   Learned post-platform: y={predicted['y']:.1f}, "
                f"vx={predicted['v_x']:.1f}, vy={predicted['v_y']:.1f}, speed={post_speed:.1f}"
            )
        return predicted
    if debug:
        print(
            "[SIM DEBUG]   Bootstrap platform hard stop (non-changeable; no KB slide model)"
        )
    return None


def _apply_platform_contact(platform_kb: dict, plat: dict, x: float, y: float,
                            vx: float, vy: float, br: float, debug: bool = False):
    """
    Apply platform contact response. Returns (new_x, new_y, new_vx, new_vy, slide_continued, hard_stop).
    """
    predicted = _resolve_platform_post_state(platform_kb, plat, x, y, vx, vy, br, debug=debug)
    if predicted is None:
        return x, y, vx, vy, False, True

    post_speed = math.hypot(predicted["v_x"], predicted["v_y"])
    slide = post_speed > 5.0
    return x, predicted["y"], predicted["v_x"], predicted["v_y"], slide, not slide

problem_template = Template("""(define (problem sample_problem)
    (:domain angry_birds_scaled)
    (:objects
        $objects
    )
    (:init
        $initial
    )
    (:goal
        ; Define your goal conditions here
        (and
            $goal
        )
    )
)
""")


# Angle bias correction (in degrees) - used inside PDDL domain physics only.
# Game slingshot execution uses AngleCalibrator (nonlinear pull → measured launch).
ANGLE_BIAS_DEGREES = 6.0
DEG_TO_RAD = 0.01745329252
# Slingshot pull clamp (screen-space aim); separate from ENHSP dial min (may be negative).
GAME_PULL_MIN = 15.0
GAME_PULL_MAX = 85.0
# Shallow ENHSP dial angles are hard to reproduce; platform levels use a higher floor.
PLANNER_MIN_ANGLE_PLATFORM = 20.0


class AngleCalibrator:
    """
    Convert PDDL planner dial angle to game slingshot pull angle.

    PDDL flight uses (dial - ANGLE_BIAS_DEGREES). The slingshot does not obey a fixed
    offset: shallow pulls lose much more angle than steep ones. This class inverts
    (game_pull → measured launch) so measured ≈ PDDL flight angle.
    """

    # Bootstrap (game_pull, measured_θ) at force≈1.0 from Science Birds calibration.
    _BOOTSTRAP = (
        (22.0, 8.5),
        (24.0, 11.0),
        (25.2, 14.4),
        (26.5, 14.4),
        (28.0, 23.4),
        (59.0, 58.8),
        (75.5, 73.9),
        (81.5, 77.9),
    )

    def __init__(self):
        self._samples = []

    def pddl_flight_angle(self, pddl_dial: float) -> float:
        return float(pddl_dial) - ANGLE_BIAS_DEGREES

    def raw_game_pull_for_pddl_dial(self, pddl_dial: float) -> float:
        """Unclamped slingshot pull (degrees) for a PDDL dial target flight angle."""
        return self._invert_measured_to_pull(self.pddl_flight_angle(pddl_dial))

    def is_dial_executable(
        self,
        pddl_dial: float,
        min_pull: float = GAME_PULL_MIN,
        max_pull: float = GAME_PULL_MAX,
    ) -> bool:
        """True when the calibrator does not need to clamp pull for this dial."""
        pull = self.raw_game_pull_for_pddl_dial(pddl_dial)
        return min_pull <= pull <= max_pull

    def executable_dial_bounds(
        self,
        dial_min: float = -20.0,
        dial_max: float = 89.0,
        dial_step: float = 0.5,
        min_pull: float = GAME_PULL_MIN,
        max_pull: float = GAME_PULL_MAX,
    ):
        """
        Scan PDDL dial range; return (min_dial, max_dial) that map to unclamped game pulls.
        ENHSP should not plan outside this range if shots must match PDDL flight angles.
        """
        executable = [
            d for d in np.arange(dial_min, dial_max + 1e-6, dial_step)
            if self.is_dial_executable(d, min_pull=min_pull, max_pull=max_pull)
        ]
        if not executable:
            return float(dial_min), float(dial_max)
        return float(min(executable)), float(max(executable))

    def game_pull_for_pddl_dial(
        self,
        pddl_dial: float,
        min_pull: float = GAME_PULL_MIN,
        max_pull: float = GAME_PULL_MAX,
    ) -> float:
        target = self.pddl_flight_angle(pddl_dial)
        pull = self._invert_measured_to_pull(target)
        return max(min_pull, min(max_pull, pull))

    def record_shot(
        self, pddl_dial: float, game_pull: float, measured_deg: float, force: float = 1.0
    ) -> None:
        """Append a calibration sample (full-force shots only)."""
        if force < 0.95:
            return
        self._samples.append((float(game_pull), float(measured_deg)))

    def _get_curve(self):
        merged = {}
        for pull, meas in self._BOOTSTRAP + tuple(self._samples):
            if pull in merged:
                merged[pull] = 0.5 * (merged[pull] + meas)
            else:
                merged[pull] = meas
        pulls = np.array(sorted(merged.keys()), dtype=float)
        measured = np.array([merged[p] for p in pulls], dtype=float)
        return pulls, measured

    def _invert_measured_to_pull(self, target_measured: float) -> float:
        pulls, measured = self._get_curve()
        target = float(target_measured)
        if len(pulls) < 2:
            return target

        if target <= measured[0]:
            dm = max(measured[1] - measured[0], 1e-6)
            dp = pulls[1] - pulls[0]
            return float(pulls[0] + dp * (target - measured[0]) / dm)

        if target >= measured[-1]:
            dm = max(measured[-1] - measured[-2], 1e-6)
            dp = pulls[-1] - pulls[-2]
            return float(pulls[-1] + dp * (target - measured[-1]) / dm)

        return float(np.interp(target, measured, pulls))


def _initial_angle_trig(start_angle_deg: float):
    """angle_rad / cosine / sinus consistent with base_domain increasing_angle."""
    angle_rad = (start_angle_deg - ANGLE_BIAS_DEGREES) * DEG_TO_RAD
    return angle_rad, float(np.cos(angle_rad)), float(np.sin(angle_rad))


# pa-twang position kick in base_domain.pddl (decrease x_bird / y_bird at release)
PA_TWANG_X_KICK = 16
PA_TWANG_Y_KICK = 12


def pddl_bird_position_after_pa_twang(ref_x: float, ref_y: float, angle_deg: float):
    """
    PDDL bird (x, y) immediately after pa-twang, matching base_domain.pddl:
      (decrease (x_bird ?b) (* 16 (cosine)))
      (decrease (y_bird ?b) (* 12 (sinus)))
    """
    _, cosine, sinus = _initial_angle_trig(angle_deg)
    return ref_x - PA_TWANG_X_KICK * cosine, ref_y - PA_TWANG_Y_KICK * sinus


def pddl_bird_position_before_pa_twang(after_x: float, after_y: float, angle_deg: float):
    """Inverse of pa-twang position kick: ref point given release position and launch angle."""
    _, cosine, sinus = _initial_angle_trig(angle_deg)
    return after_x + PA_TWANG_X_KICK * cosine, after_y + PA_TWANG_Y_KICK * sinus


def generate_pddl(problem_data: dict, init_angle, angel_rate, world_model: WorldModel,
                  min_angle: float = -4, max_angle: float = 85,
                  force_min: float = FORCE_MIN, force_max: float = FORCE_MAX):
    objects = list()
    goals = list()
    # Start at max_angle so increasing_angle can run (requires angle <= max_angle)
    start_angle = float(init_angle) if init_angle is not None else float(max_angle)
    angle_rad, cosine, sinus = _initial_angle_trig(start_angle)
    initial_state = [
        f"(= (angle) {start_angle})",
        f"(= (angle_rad) {angle_rad})",
        f"(= (angle_rate) {angel_rate})",
        f"(= (angle_bias) {ANGLE_BIAS_DEGREES})",  # Bias correction for slingshot
        f"(= (min_angle) {min_angle})",
        f"(= (max_angle) {max_angle})",
        f"(= (cosine) {cosine})",
        f"(= (sinus) {sinus})",
        f"(= (bounce_count) 0)",
        f"(= (gravity) {world_model.hyperparams_values[Params.gravity]})",
        f"(= (active_bird) 0)",
        f"(= (ground_y_damper) 0.1)",
        f"(= (ground_x_damper) 0.5)",
        # Force search fluents — v_bird_multiplier sweeps force_max→force_min at FORCE_RATE.
        # Starting at max means planner prefers full-force shots and only falls back to partial.
        f"(= (v_bird_multiplier) {force_max})",
        f"(= (force_rate) {FORCE_RATE})",
        f"(= (min_force) {force_min})",
        f"(= (max_force) {force_max})",
    ]
    for object, object_data in problem_data.items():
        objects.append(f"{object} - {object.split('_')[0]}")
        if "pig" in object:
            goals.append(f"(pig_dead {object})")
        for pred, val in object_data.items():
            # Skip keys starting with '_' (visualization-only data)
            if pred.startswith('_'):
                continue
            initial_state.append(f"(= ({pred} {object}) {val})")
        # relations
        for other_object in problem_data:
            if 'bird' in object and 'block' in other_object:
                initial_state.append(f'(= (bird_block_damage {object} {other_object}) 0.01)')

    objects_str = "\n".join(objects)
    goals_str = "\n".join(goals)
    initial_state_str = "\n".join(initial_state)
    return objects_str, initial_state_str, goals_str


def write_problem_file(path: str, problem_data: dict, init_angle: float, angel_rate: float, world_model: WorldModel,
                       min_angle: float = -4, max_angle: float = 85,
                       force_min: float = FORCE_MIN, force_max: float = FORCE_MAX):
    extend_platforms_in_problem_data(problem_data)
    objects, initial_state, goals = generate_pddl(
        problem_data, init_angle, angel_rate, world_model, min_angle, max_angle,
        force_min=force_min, force_max=force_max,
    )
    problem = problem_template.substitute({"objects": objects, "initial": initial_state, "goal": goals})
    with open(path, 'w') as file:
        file.write(problem)


def inject_domain_file(path: str, world_model: WorldModel, defer_ground_m5: bool = False):
    """
    Inject learned M5 collision models into the PDDL domain file.

    Ground bounce: kb['collision'] → {SE-collision-ground-effect}
    Platform contact: kb['platform_collision'] → {SE-collision-platform-effect}

    When defer_ground_m5 is True (e.g. hill levels during planning), skip ground M5
    injection to keep ENHSP search tractable.
    """
    _collision_inject_log(f"inject_domain_file input={path!r}")

    ground_placeholder = (
        "(assign (y_bird ?b) 0.0)\n            "
        "(assign (vy_bird ?b) 0.0)\n            "
        "(assign (vx_bird ?b) 0.0)\n            "
        "(assign (bounce_count ?b) (+ (bounce_count ?b) 1))"
    )

    with open(path, "r") as file:
        new_content = file.read()

    ground_sentinel = "{SE-collision-ground-effect}"
    if ground_sentinel in new_content:
        if defer_ground_m5:
            _collision_inject_log("Deferring ground M5 for planning (platform level)")
            new_content = new_content.replace(ground_sentinel, ground_placeholder)
        else:
            collision_vars = world_model.kb["collision"]["variables"]
            m5_body = _build_collision_ground_effect_m5(collision_vars)
            if m5_body is not None:
                new_content = new_content.replace(ground_sentinel, m5_body)
                _collision_inject_log("Injected M5 piecewise ground collision effect")
            else:
                _collision_inject_log("Ground M5 not available yet, using placeholder 0.0")
                new_content = new_content.replace(ground_sentinel, ground_placeholder)

    platform_sentinel = "{SE-collision-platform-effect}"
    if platform_sentinel in new_content:
        platform_kb = world_model.kb.get("platform_collision", {})
        n_samples = len(platform_kb.get("states") or [])
        platform_vars = platform_kb.get("variables") or {}
        if n_samples >= PLATFORM_COLLISION_MIN_SAMPLES and platform_vars:
            m5_body = _build_collision_effect_m5(platform_vars, bounce_mode="platform")
            if m5_body is not None:
                new_content = new_content.replace(platform_sentinel, m5_body)
                _collision_inject_log(
                    f"Injected M5 platform collision effect ({n_samples} samples)"
                )
            else:
                _collision_inject_log(
                    f"Platform M5 not ready ({n_samples} samples), using slide placeholder"
                )
                new_content = new_content.replace(
                    platform_sentinel, _build_platform_slide_placeholder()
                )
        else:
            _collision_inject_log(
                f"Platform learning cold start ({n_samples}/{PLATFORM_COLLISION_MIN_SAMPLES} samples)"
                " — hard stop (bootstrap, no bird state mutation)"
            )
            new_content = new_content.replace(
                platform_sentinel, _build_platform_bootstrap_effect()
            )

    # Save modified file
    base_dir = os.path.dirname(path)
    base_name = os.path.splitext(os.path.basename(path))[0]
    output_path = os.path.join(base_dir, f"{base_name}_modified.pddl")

    with open(output_path, "w") as file:
        file.write(new_content)

    _collision_inject_log(f"Saved to {output_path}")


def inject_learned_transitions(path: str, world_model: WorldModel):
    """
    Inject learned state transition functions into the flying process in the domain file.
    
    Parameters:
    -----------
    path : str
        Path to the base domain PDDL file
    world_model : WorldModel
        WorldModel instance containing learned_transitions attribute
    """
    if not hasattr(world_model, 'learned_transitions') or world_model.learned_transitions is None:
        print("No learned transitions found in world model. Skipping injection.")
        return
    
    with open(path, "r") as file:
        content = file.read()
    new_content = content
    
    learned_transitions = world_model.learned_transitions
    
    # Extract transition equations from learned models
    # We need to convert the learned transition functions to PDDL format
    # The format should match what was previously injected manually
    
    # For now, we'll use placeholders that can be replaced
    # The actual implementation depends on how the transitions are structured
    
    # Check if we have the necessary transitions
    if (learned_transitions.get("x") is not None and 
        learned_transitions.get("y") is not None and
        learned_transitions.get("xdot") is not None and
        learned_transitions.get("ydot") is not None and
        learned_transitions.get("yddot") is not None):
        
        # Get transition strings
        x_transition = learned_transitions["x"].get("string", "")
        y_transition = learned_transitions["y"].get("string", "")
        xdot_transition = learned_transitions["xdot"].get("string", "")
        ydot_transition = learned_transitions["ydot"].get("string", "")
        yddot_transition = learned_transitions["yddot"].get("string", "")
        
        print(f"\nInjecting learned transitions into domain file:")
        print(f"  x: {x_transition}")
        print(f"  y: {y_transition}")
        print(f"  xdot: {xdot_transition}")
        print(f"  ydot: {ydot_transition}")
        print(f"  yddot: {yddot_transition}")
        
        # For now, we'll store the transitions in the world model
        # The actual PDDL injection can be done later when we know the exact format needed
        # This is a placeholder that shows the transitions are available for injection
        
    # Save modified file in same folder
    base_dir = os.path.dirname(path)
    base_name = os.path.splitext(os.path.basename(path))[0]
    output_path = os.path.join(base_dir, f"{base_name}_modified.pddl")
    
    # For now, just copy the content (actual injection logic to be implemented)
    with open(output_path, "w") as file:
        file.write(new_content)
    
    print(f"Learned transitions available for injection. Modified file saved to: {output_path}")
    print("Note: Actual PDDL transition injection logic to be implemented based on format requirements.")


def align_dial_to_planner_grid(
    angle: float,
    start_angle: float,
    step: float,
    dial_min: float,
    dial_max: float,
) -> float:
    """Snap to ENHSP dial grid (start - n*step) and clamp to [dial_min, dial_max]."""
    start_angle = float(start_angle)
    step = float(step)
    dial_min = float(dial_min)
    dial_max = float(dial_max)
    n_lo = math.ceil((start_angle - dial_max) / step - 1e-9)
    n_hi = math.floor((start_angle - dial_min) / step + 1e-9)
    if n_lo > n_hi:
        return dial_min
    n = round((start_angle - float(angle)) / step)
    n = max(n_lo, min(n_hi, n))
    return start_angle - n * step


def action_filter(line):
    return 'pa-twang' in line or 'set_force' in line


def parse_action(line, init_angle, angel_rate, force_min=FORCE_MIN, force_rate=FORCE_RATE):
    """Parse a solution line into a (type, value) action tuple.

    - pa-twang at time n  → ('shoot', angle)   where angle = init_angle - n * angel_rate
    - set_force at time n → ('set_force', force) where force = force_max - n * force_rate
      (force starts at FORCE_MAX and decreases, so planner picks full force first)
    """
    n = float(line.split(':')[0])
    if 'set_force' in line:
        force = FORCE_MAX - n * force_rate
        force = max(force_min, min(FORCE_MAX, force))
        return 'set_force', round(force, 4)
    return 'shoot', init_angle - n * angel_rate


def parse_solution_to_actions(solution_path: str, init_angle, angel_rate,
                               force_min=FORCE_MIN, force_rate=FORCE_RATE):
    """Parse ENHSP solution into a list of (type, value) action tuples.

    Returns a list that may contain both ('set_force', force) and ('shoot', angle).
    """
    with open(solution_path) as solution_file:
        lines = solution_file.readlines()
    relevant = [l for l in lines if action_filter(l)]
    actions = [parse_action(l, init_angle, angel_rate, force_min, force_rate) for l in relevant]
    return actions


# Replan bird ref when planned angle differs from the guess used for pa-twang inverse.
ANGLE_REPLAN_THRESHOLD_DEG = 0.5


def effective_launch_speed(
    v_full: float,
    force: float,
    speed_at_force=None,
    force_lr_model=None,
) -> float:
    """
    Launch speed for a PDDL force fraction (0.2–1.0).

    Mirrors base_domain pa-twang: ``v_launch = v_bird * v_bird_multiplier`` (linear in force).
    Optional ``speed_at_force(v_full, force)`` hook for callers that override the default.
    """
    force = max(FORCE_MIN, min(FORCE_MAX, float(force)))
    if speed_at_force is not None:
        try:
            return float(speed_at_force(v_full, force))
        except Exception:
            pass
    return launch_speed_at_force(v_full, force, force_lr_model=force_lr_model)


def iter_force_grid():
    """Executable force values matching ENHSP decreasing_force grid."""
    force = FORCE_MAX
    while force >= FORCE_MIN - 1e-6:
        yield round(force, 4)
        force -= FORCE_RATE


def simulate_pddl_shot_plan(
    problem_data: dict,
    angle_deg: float,
    gravity: float = None,
    force: float = 1.0,
    dt: float = 0.01,
    max_steps: int = 10000,
    debug: bool = False,
    platform_kb: dict = None,
    speed_at_force=None,
    force_lr_model=None,
):
    """
    Forward-simulate flight after pa-twang using base_domain flying + placeholder ground bounce.

    Returns whether the shot path touches ground (y <= 0) before termination, matching
  collision_ground in the PDDL model when M5 collision is not injected.
    Also checks platform collisions using PDDL domain margins (1.1 bird_radius multipliers).
    Uses swept segment tests and extends platform height to the pig stand line on hills.
    When platform_kb has trained models, applies learned post-contact velocities.
    On bootstrap (no KB), platform contact is a hard stop — no slide state mutation.
    
    Now also returns the full trajectory for visualization.

    The `force` parameter mirrors the PDDL domain's v_bird_multiplier used in pa-twang:
      vx = v_bird * v_bird_multiplier * cosine
      vy = v_bird * v_bird_multiplier * sinus
    Pass the planned force fraction (0.2-1.0). Launch speed is ``v_bird * force`` (linear),
    matching base_domain pa-twang via ``effective_launch_speed``.
    Defaults to 1.0 (full force) for backward compatibility.
    """
    bird_key = next((k for k in problem_data if k.startswith("bird_")), None)
    pig_key = next((k for k in problem_data if k.startswith("pig_")), None)
    if bird_key is None:
        return {"uses_ground_collision": False, "pig_killed_in_sim": False, "ground_touches": 0, 
                "platform_collision": False, "trajectory": []}

    bird = problem_data[bird_key]
    ref_x, ref_y = float(bird["x_bird"]), float(bird["y_bird"])
    v = float(bird.get("v_bird", 180))
    br = float(bird.get("bird_radius", 4.0))  # Default to 4.0 (max of 7x8 bird)

    if gravity is None:
        gravity = 85.0

    launch_x, launch_y = pddl_bird_position_after_pa_twang(ref_x, ref_y, angle_deg)
    x, y = launch_x, launch_y
    _, cosine, sinus = _initial_angle_trig(angle_deg)
    launch_speed = effective_launch_speed(
        v, force, speed_at_force=speed_at_force, force_lr_model=force_lr_model,
    )
    vx = launch_speed * cosine
    vy = launch_speed * sinus

    pig = problem_data.get(pig_key) if pig_key else None
    pr = float(pig.get("pig_radius", 3.5)) if pig else 3.5
    px = float(pig["x_pig"]) if pig else None
    py = float(pig["y_pig"]) if pig else None

    # Collect platform data
    platforms = []
    blocks = []
    for key, val in problem_data.items():
        if key.startswith("platform_"):
            platforms.append({
                "name": key,
                "x": float(val["x_platform"]),
                "y": float(val["y_platform"]),
                "w": float(val["platform_width"]),
                "h": float(val["platform_height"]),
            })
        elif key.startswith("block_"):
            blocks.append({
                "name": key,
                "x": float(val["x_block"]),
                "y": float(val["y_block"]),
                "w": float(val["block_width"]),
                "h": float(val["block_height"]),
            })

    if debug:
        print(f"\n[SIM DEBUG] Simulating shot at angle={angle_deg:.1f}°, force={force:.2f}")
        print(
            f"[SIM DEBUG] Bird: ref=({ref_x:.1f}, {ref_y:.1f}), launch=({launch_x:.1f}, {launch_y:.1f}), "
            f"v_full={v:.1f}, v_launch={launch_speed:.1f}, radius={br:.1f}"
        )
        print(f"[SIM DEBUG] Pig: ({px:.1f}, {py:.1f}), radius={pr:.1f}")
        print(f"[SIM DEBUG] Platforms: {len(platforms)}")
        for p in platforms:
            print(f"[SIM DEBUG]   {p['name']}: center=({p['x']:.1f}, {p['y']:.1f}), size={p['w']:.1f}x{p['h']:.1f}")
            print(f"[SIM DEBUG]     bounds: left={p['x']-p['w']/2:.1f}, right={p['x']+p['w']/2:.1f}, bottom={p['y']-p['h']/2:.1f}, top={p['y']+p['h']/2:.1f}")
        print(f"[SIM DEBUG] Blocks: {len(blocks)}")

    platforms = _extend_platforms_for_pig(platforms, px, py, pr)

    ground_touches = 0
    pig_killed = False
    block_collision = False
    block_hit_name = None
    platform_hit = False
    platform_hit_name = None
    platform_hit_pos = None
    platform_slide_continued = False
    bounce_count = 0
    platforms_responded = set()
    trajectory = [(x, y)]  # Store trajectory for visualization
    step = 0

    for step in range(max_steps):
        if x > 800 or bounce_count >= 3:
            break

        nx = x + vx * dt
        ny = y + vy * dt
        nvy = vy - gravity * dt

        # Platform before pig: swept contact along the integration segment.
        platform_handled = False
        for plat in platforms:
            if plat["name"] in platforms_responded:
                continue
            hit, cx, cy = _sweep_platform_contact(x, y, nx, ny, br, plat)
            if not hit:
                continue

            platform_hit = True
            platform_hit_name = plat["name"]
            platform_hit_pos = (cx, cy)
            platforms_responded.add(plat["name"])
            if debug:
                left, right, bottom, top = _platform_bounds(plat)
                print(
                    f"[SIM DEBUG] Step {step}: PLATFORM HIT '{plat['name']}' "
                    f"at bird pos ({cx:.1f}, {cy:.1f})"
                )
                print(
                    f"[SIM DEBUG]   Platform bounds: [{left:.1f}, {right:.1f}] "
                    f"x [{bottom:.1f}, {top:.1f}]"
                )

            x, y, vx, vy, slide, hard_stop = _apply_platform_contact(
                platform_kb, plat, cx, cy, vx, vy, br, debug=debug
            )
            if hard_stop:
                bounce_count = 3
                platform_handled = True
                break

            if slide:
                platform_slide_continued = True
                bounce_count += 1
            platform_handled = True
            break

        if platform_handled and bounce_count >= 3:
            break

        if not block_collision:
            for blk in blocks:
                blk_left = blk["x"] - blk["w"] / 2
                blk_right = blk["x"] + blk["w"] / 2
                blk_bottom = blk["y"] - blk["h"] / 2
                blk_top = blk["y"] + blk["h"] / 2
                for sx, sy in ((x, y), (nx, ny)):
                    if (sx - br <= blk_right and
                            sx + br >= blk_left and
                            sy + br >= blk_bottom and
                            sy - br <= blk_top):
                        block_collision = True
                        block_hit_name = blk["name"]
                        if debug:
                            print(
                                f"[SIM DEBUG] Step {step}: BLOCK HIT '{blk['name']}' "
                                f"at bird pos ({sx:.1f}, {sy:.1f})"
                            )
                        bounce_count = 3
                        break
                if block_collision:
                    break

        if block_collision:
            break

        if pig is not None and not pig_killed:
            for sx, sy in ((x, y), (nx, ny)):
                dx = sx - px
                dy = sy - py
                effective_r = max(0.0, br + pr - PIG_HIT_EPSILON)
                if effective_r > 0 and (dx * dx + dy * dy) <= effective_r ** 2:
                    pig_killed = True
                    x, y = sx, sy
                    if debug:
                        print(f"[SIM DEBUG] Step {step}: PIG HIT at ({x:.1f}, {y:.1f})")
                    break
            if pig_killed:
                break

        if platform_handled:
            if step % 5 == 0:
                trajectory.append((x, y))
            continue

        x = nx
        y = ny
        vy = nvy

        if step % 5 == 0:
            trajectory.append((x, y))

        if y - br <= 0 and vx > 0:
            ground_touches += 1
            if debug:
                print(f"[SIM DEBUG] Step {step}: GROUND at ({x:.1f}, {y:.1f})")
            y = 0.0
            vy = 0.0
            vx = 0.0
            bounce_count += 1
            trajectory.append((x, y))
            # Placeholder ground effect stops the bird (no M5 bounce in planning sim).
            break

        if y + br <= 0 or (y <= 0 and vy <= 0):
            # Safety: do not integrate below ground when vx is already zero.
            y = max(y, 0.0)
            vy = 0.0
            vx = 0.0
            trajectory.append((x, y))
            break
    trajectory.append((x, y))
    
    if debug:
        print(f"[SIM DEBUG] Simulation ended at step {step}: pos=({x:.1f}, {y:.1f})")
        print(
            f"[SIM DEBUG] Result: pig_killed={pig_killed}, platform_hit={platform_hit}, "
            f"block_collision={block_collision}, ground_touches={ground_touches}"
        )

    return {
        "uses_ground_collision": ground_touches > 0,
        "pig_killed_in_sim": pig_killed,
        "ground_touches": ground_touches,
        "platform_collision": platform_hit,
        "block_collision": block_collision,
        "block_hit_name": block_hit_name,
        "platform_slide_continued": platform_slide_continued,
        "platform_hit_name": platform_hit_name,
        "platform_hit_pos": platform_hit_pos,
        "launch_x": launch_x,
        "launch_y": launch_y,
        "trajectory": trajectory,
        "final_pos": (x, y),
    }


def plan_uses_ground_collision(problem_data: dict, angle_deg: float, gravity: float) -> bool:
    """True when the planned angle's PDDL-style path includes a ground collision."""
    return bool(simulate_pddl_shot_plan(problem_data, angle_deg, gravity=gravity)["uses_ground_collision"])


def ballistic_angle_to_target(
    launch_x: float,
    launch_y: float,
    target_x: float,
    target_y: float,
    velocity: float,
    gravity: float,
    min_angle: float = -20.0,
    max_angle: float = 85.0,
    prefer_low_arc: bool = True,
):
    """
    Solve projectile angle from launch to target (PDDL coords).
    Returns None if no real solution.
    """
    dx = target_x - launch_x
    dy = target_y - launch_y
    if dx <= 0:
        return None
    term = velocity ** 4 - gravity * (gravity * dx ** 2 + 2 * dy * velocity ** 2)
    if term < 0:
        return None
    sqrt_term = float(np.sqrt(term))
    angle_low = float(np.degrees(np.arctan((velocity ** 2 - sqrt_term) / (gravity * dx))))
    angle_high = float(np.degrees(np.arctan((velocity ** 2 + sqrt_term) / (gravity * dx))))
    chosen = angle_low if prefer_low_arc else angle_high
    return max(min_angle, min(max_angle, chosen))
