from string import Template
from typing import Optional

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

# Pig kill when center distance <= bird_radius + pig_radius - PIG_HIT_EPSILON
# (matches ``collision_pig_kill`` in base_domain.pddl). Same rule for mid-air
# and platform-contact steps — no separate coincident full-radius path.
PIG_HIT_EPSILON = 2.0


def _pig_center_distance(sx: float, sy: float, px: float, py: float) -> float:
    return math.hypot(sx - px, sy - py)


def _pig_kill_threshold(br: float, pr: float) -> float:
    """Max center distance for pig kill: ``bird_radius + pig_radius - epsilon``."""
    return max(0.0, br + pr - PIG_HIT_EPSILON)


def _bird_kills_pig(sx: float, sy: float, px: float, py: float,
                    br: float, pr: float) -> bool:
    thresh = _pig_kill_threshold(br, pr)
    if thresh <= 0:
        return False
    return _pig_center_distance(sx, sy, px, py) <= thresh


def _slide_pig_kill_credible(sx: float, sy: float, vx: float, vy: float,
                             px: float, py: float, br: float, pr: float) -> bool:
    """
    Stricter pig-kill check during post-platform slide (not coincident contact).

    run_20260826/00073: sim predicted slide-into-pig kills that the game did not
    reproduce — often the bird was geometrically near the pig but moving away,
    or still too high above the stand line.
    """
    if not _bird_kills_pig(sx, sy, px, py, br, pr):
        return False
    pig_stand = py - pr
    if sy > pig_stand + br + 6.0:
        return False
    if abs(vx) >= 3.0:
        if vx > 0.0 and sx > px + pr:
            return False
        if vx < 0.0 and sx < px - pr:
            return False
    return True


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


_COLLISION_VAR_ORDER = ("y", "v_x", "v_y", "x")
_COLLISION_PDDL_FLUENT = {"y": "y_bird", "v_x": "vx_bird", "v_y": "vy_bird", "x": "x_bird"}
_PDDL_AFFINE_VARS = ("x_bird", "y_bird", "vx_bird", "vy_bird")
ROLLING_LR_MIN_SAMPLES = 3
ROLLING_LR_OUTPUT_VARS = ("x", "y", "v_x", "v_y")
PDDL_PLATFORM_LR_MIN_VX_R2 = 0.35
PDDL_PLATFORM_LR_FALLBACK_MIN_VX_R2 = 0.12
PDDL_SIM_LR_MIN_VX_R2 = 0.35
PDDL_PLATFORM_M5_MIN_SAMPLES = 21
PDDL_PLATFORM_LR_OVERFIT_R2 = 0.99

PLATFORM_COLLISION_MIN_SAMPLES = 5
PLATFORM_MAX_POST_SPEED = 220.0
ROLLING_LR_MAX_V = PLATFORM_MAX_POST_SPEED * 2
ROLLING_LR_MAX_POS = 5000.0
ROLLING_LR_MAX_FEATURE = 1e6
# Observed platform slides (Template 4): post_speed varies; use conservative 50% for
# cold-start planning so ENHSP does not assume aggressive shallow slides reach the pig.
PLATFORM_DEFAULT_SPEED_RATIO = 0.50
PLATFORM_DEFAULT_VY_RATIO = 0.50
PLATFORM_SURFACE_RADIUS_FACTOR = 0.35
# Continuous platform-slide integrator (forward sim): bird stays on surface at vy≈0.
PLATFORM_SLIDE_MIN_SPEED = 3.0
PLATFORM_SLIDE_EDGE_MARGIN = 0.35  # fraction of bird_radius beyond platform x span
PLATFORM_SLIDE_FRICTION_PER_S = 0.0  # game slides at ~constant vx after landing
# One-shot horizontal advance appended to PDDL platform/ground slide effects so
# ENHSP can reach pigs that require multi-frame slide (not modeled as a process).
PLATFORM_PLANNING_SLIDE_SEC = 0.45
# Ground slide: first contact applies LR/heuristic once; no per-step re-fire (run_20260909_175413).
GROUND_SLIDE_MIN_SPEED = 3.0
GROUND_SLIDE_FRICTION_PER_S = 0.0
GROUND_SLIDE_STUCK_EPS = 0.05  # px/step — learned post-state can stall x advance
GROUND_SLIDE_STUCK_MAX_STEPS = 10
PLATFORM_SWEEP_SUBSTEPS = 8
# Match game ``is_platform_collision`` (event_conditions.py): max(bird_dim)/2 + epsilon.
# The old AABB-overlap test with only ``bird_radius`` (4 px) missed grazing ramp
# contacts that the game reports (run_20260823_102616: 44× platform sim=NO game=YES).
PLATFORM_COLLISION_EPSILON = 6.0
# Extra platform AABB inflation in forward sim only (vision vertices are tight vs
# game MBR); does not affect ENHSP problem files.
PLATFORM_SIM_AABB_MARGIN_PX = 2.0
# Bird-radius multiplier for platform-collision detection in the forward sim.
# Was 1.1 (10% safety inflation), but that caused "coincident-contact" false
# negatives: when a pig sits directly on top of a platform, the inflated
# margin makes platform contact fire on the same integration step the bird
# reaches the pig, and — under the pre-fix collision ordering — the platform
# stop terminated the sim before pig-kill was checked, so ENHSP plans that
# ballistically kill the pig were rejected by the sim. Setting to 1.0 matches
# the domain's collision_platform predicate exactly (bird tangent to platform
# surface). Combined with pig-kill being checked BEFORE platform in the loop
# (see simulate_pddl_shot_plan), a same-step pig contact wins.
PLATFORM_MARGIN_MULT = 1.0
# Measured launch angle often deviates from PDDL flight angle; validate plans at ±slack.
ANGLE_EXECUTION_SLACK_DEG = 4.0
# Tighter slack when constraining ENHSP dial search to slingshot-executable angles.
PLANNER_FLIGHT_SLACK_DEG = 2.0
# Extra inflation on block AABBs so ENHSP / forward sim clear rotated obstacles conservatively.
BLOCK_PLANNING_MARGIN_PX = 3.0
# Sheltering blocks (stacked above/over the pig): use tight AABB so ENHSP can thread arcs.
BLOCK_SHELTER_MARGIN_PX = 0.0
BLOCK_SHELTER_X_PAD_PX = 8.0
BLOCK_SHELTER_LIFE_FACTOR = 0.45
BLOCK_SHELTER_DAMAGE = 0.12
DEFAULT_BIRD_BLOCK_DAMAGE = 0.01
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


def _planning_slide_x_advance_effect() -> str:
    sec = PLATFORM_PLANNING_SLIDE_SEC
    return f"(increase (x_bird ?b) (* (vx_bird ?b) {sec}))"


def _append_planning_slide_x_advance(body: str) -> str:
    """Add optimistic post-contact x advance when the effect lacks x_bird mutation."""
    if not body or "x_bird ?b)" in body:
        return body
    return _append_extra_planning_slide_x(body)


def _append_extra_planning_slide_x(
    body: str,
    extra_sec: float = None,
) -> str:
    """Append an extra vx-scaled x bump before (assign (mod) 2)."""
    if not body:
        return body
    sec = PLATFORM_PLANNING_SLIDE_SEC if extra_sec is None else float(extra_sec)
    bump = f"(increase (x_bird ?b) (* (vx_bird ?b) {sec}))"
    mod = "(assign (mod) 2)"
    if mod not in body:
        return f"{body}\n            {bump}\n            {mod}"
    return body.replace(mod, f"{bump}\n            {mod}", 1)


def _build_surface_roll_effect(y_assign_expr: str, *, planning_x_advance: bool = True) -> str:
    """
    Shared PDDL effect for bird rolling/sliding along a surface (platform or ground).
    Retains horizontal speed; v_bird unchanged — vx/vy drive continued motion.

    Does not increment bounce_count: slide contacts can fire every integration step
    while the bird stays on the surface; counting them made explode_bird (>= 3)
    terminate flight after ~3 frames instead of allowing slide-to-pig plans.
    """
    ratio = PLATFORM_DEFAULT_SPEED_RATIO
    vy_ratio = PLATFORM_DEFAULT_VY_RATIO
    lines = [
        f"(assign (y_bird ?b) {y_assign_expr})",
        f"(assign (vx_bird ?b) (* (vx_bird ?b) {ratio}))",
        f"(assign (vy_bird ?b) (* (vy_bird ?b) {vy_ratio}))",
    ]
    if planning_x_advance:
        lines.append(_planning_slide_x_advance_effect())
    lines.append("(assign (mod) 2)")
    return "\n            ".join(lines)


def _build_platform_bootstrap_effect() -> str:
    """PDDL effect when platform KB is cold — end flight, no bird state mutation."""
    return PLATFORM_COLLISION_BOOTSTRAP


def _build_platform_slide_placeholder() -> str:
    """PDDL rolling effect after platform contact (KB has samples but M5 not ready)."""
    surf = PLATFORM_SURFACE_RADIUS_FACTOR
    y_expr = (
        f"(+ (+ (y_platform ?pl) (/ (platform_height ?pl) 2)) "
        f"(* (bird_radius ?b) {surf}))"
    )
    return _build_surface_roll_effect(y_expr)


def _build_ground_roll_placeholder() -> str:
    """PDDL rolling effect after ground contact — same speed retention as platform slide."""
    surf = PLATFORM_SURFACE_RADIUS_FACTOR
    y_expr = f"(* (bird_radius ?b) {surf})"
    return _build_surface_roll_effect(y_expr)
# Terrain surface in PDDL coords (640 - screen_y); matches segments.GROUND_LEVEL and
# event is_ground_collision (relative y = pddl_y - PLAYFIELD_FLOOR_Y).
PLAYFIELD_FLOOR_Y = 360.0
# Center-based slack above floor; matches event_conditions.is_ground_collision epsilon.
GROUND_CONTACT_CENTER_SLACK = 3.0
_LEARNING_Y_OFFSET = PLAYFIELD_FLOOR_Y


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


def _rolling_lr_ready(lr_models: dict, min_samples: int = ROLLING_LR_MIN_SAMPLES) -> bool:
    if not isinstance(lr_models, dict):
        return False
    return all(var in lr_models for var in ROLLING_LR_OUTPUT_VARS)


def _rolling_lr_vx_r2(lr_models: dict) -> Optional[float]:
    entry = (lr_models or {}).get("v_x") or {}
    r2 = entry.get("r2")
    if r2 is None:
        return None
    try:
        return float(r2)
    except (TypeError, ValueError):
        return None


def _rolling_lr_good_for_pddl(
    lr_models: dict,
    min_vx_r2: float = PDDL_PLATFORM_LR_MIN_VX_R2,
    n_samples: int = 0,
) -> bool:
    if not _rolling_lr_ready(lr_models):
        return False
    r2 = _rolling_lr_vx_r2(lr_models)
    if r2 is None or r2 < min_vx_r2:
        return False
    if (
        n_samples > 0
        and n_samples < PDDL_PLATFORM_M5_MIN_SAMPLES
        and r2 >= PDDL_PLATFORM_LR_OVERFIT_R2
    ):
        return False
    return True


def _sanitize_rolling_lr_inputs(x: float, y_pddl: float, vx: float, vy: float):
    if not all(math.isfinite(v) for v in (x, y_pddl, vx, vy)):
        return None
    return (
        float(np.clip(x, -ROLLING_LR_MAX_POS, ROLLING_LR_MAX_POS)),
        float(np.clip(y_pddl, -ROLLING_LR_MAX_POS, ROLLING_LR_MAX_POS)),
        float(np.clip(vx, -ROLLING_LR_MAX_V, ROLLING_LR_MAX_V)),
        float(np.clip(vy, -ROLLING_LR_MAX_V, ROLLING_LR_MAX_V)),
    )


def _rolling_lr_predict_safe(lr_models: dict, pre: dict) -> Optional[dict]:
    from agents.pddl.pddl_files.events.learn_events import make_rolling_lr_feature_vector

    X = make_rolling_lr_feature_vector([pre], lr_models)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    X = np.nan_to_num(
        X, nan=0.0, posinf=ROLLING_LR_MAX_FEATURE, neginf=-ROLLING_LR_MAX_FEATURE,
    )
    X = np.clip(X, -ROLLING_LR_MAX_FEATURE, ROLLING_LR_MAX_FEATURE)
    if not np.all(np.isfinite(X)):
        return None

    predicted = {}
    try:
        for var in ROLLING_LR_OUTPUT_VARS:
            model = lr_models[var]["model"]
            val = float(model.predict(X)[0])
            if not math.isfinite(val):
                return None
            if var == "y":
                val = _learning_y_to_pddl(val)
            predicted[var] = val
    except (ValueError, OverflowError, FloatingPointError):
        return None
    return predicted


def _rolling_lr_models_for_pddl(kb: dict) -> dict:
    """Prefer base 4-feature LR for PDDL; fall back to sim models if base-only."""
    models, _ = _planning_surface_lr_models(kb)
    return models or {}


def _planning_surface_lr_models(kb: dict, n_samples: int = 0):
    """
    4-feature LR models shared by ENHSP inject and forward-sim one-shot contact.

    Primary gate: lr_models_pddl v_x R² ≥ PDDL_PLATFORM_LR_MIN_VX_R2.
    Fallback gate (split): pddl v_x R² ≥ PDDL_PLATFORM_LR_FALLBACK_MIN_VX_R2 and
    extended sim lr_models v_x R² ≥ PDDL_SIM_LR_MIN_VX_R2.
    """
    if not isinstance(kb, dict):
        return None, None
    n_samples = n_samples or len(kb.get("states") or [])
    pddl_models = kb.get("lr_models_pddl") or {}
    if not _rolling_lr_ready(pddl_models):
        sim_models = kb.get("lr_models") or {}
        from agents.pddl.pddl_files.events.learn_events import rolling_lr_uses_extended_features
        if _rolling_lr_ready(sim_models) and not rolling_lr_uses_extended_features(sim_models):
            pddl_models = sim_models
        else:
            return None, None

    if _rolling_lr_good_for_pddl(pddl_models, n_samples=n_samples):
        return pddl_models, "lr_pddl"

    sim_models = kb.get("lr_models") or {}
    pddl_vx_r2 = _rolling_lr_vx_r2(pddl_models)
    sim_vx_r2 = _rolling_lr_vx_r2(sim_models)
    if (
        pddl_vx_r2 is not None
        and pddl_vx_r2 >= PDDL_PLATFORM_LR_FALLBACK_MIN_VX_R2
        and sim_vx_r2 is not None
        and sim_vx_r2 >= PDDL_SIM_LR_MIN_VX_R2
        and not (
            n_samples > 0
            and n_samples < PDDL_PLATFORM_M5_MIN_SAMPLES
            and pddl_vx_r2 >= PDDL_PLATFORM_LR_OVERFIT_R2
        )
    ):
        return pddl_models, "lr_pddl_sim_fallback"

    return None, None


def _m5_kb_ready(kb: dict, m5_min_samples: int = 0) -> bool:
    """True when M5 trees exist and optional sample count is satisfied."""
    if not isinstance(kb, dict):
        return False
    n_samples = len(kb.get("states") or [])
    if m5_min_samples > 0 and n_samples < m5_min_samples:
        return False
    variables = kb.get("variables") or {}
    if not variables:
        return False
    for vn in _COLLISION_VAR_ORDER:
        mc = variables.get(vn, {}).get("model_comparison")
        m5 = mc.get("m5_model") if isinstance(mc, dict) else None
        if not m5_collision_leaves_for_pddl(m5, debug=False, var_label=vn):
            return False
    return True


def _surface_physics_kind(kb: dict, m5_min_samples: int = 0):
    """
    Aligned LR → M5 → placeholder priority for inject and forward sim.
    Returns (kind, meta) where kind is 'lr', 'm5', or 'placeholder'.
    """
    n_samples = len((kb or {}).get("states") or [])
    lr_models, lr_tag = _planning_surface_lr_models(kb, n_samples)
    if lr_models is not None and _build_rolling_lr_effect(lr_models) is not None:
        return "lr", {
            "lr_models": lr_models,
            "lr_tag": lr_tag,
            "n_samples": n_samples,
            "vx_r2": _rolling_lr_vx_r2(lr_models),
        }
    if _m5_kb_ready(kb, m5_min_samples=m5_min_samples):
        return "m5", {"n_samples": n_samples}
    return "placeholder", {"n_samples": n_samples}


def _build_rolling_lr_effect(lr_models: dict) -> Optional[str]:
    """Build PDDL collision effect from linear rolling models (affine per fluent)."""
    if not _rolling_lr_ready(lr_models):
        return None
    from agents.pddl.pddl_files.events.learn_events import rolling_lr_uses_extended_features

    # Extended LR uses speed/trig features ENHSP cannot express in collision effects.
    if rolling_lr_uses_extended_features(lr_models):
        return None
    lines = []
    for var in ROLLING_LR_OUTPUT_VARS:
        fluent = _COLLISION_PDDL_FLUENT[var]
        entry = lr_models[var]
        coef = entry.get("coef")
        intercept = entry.get("intercept")
        if coef is None or intercept is None:
            model = entry.get("model")
            if model is None:
                return None
            coef = model.coef_
            intercept = float(model.intercept_)
        rhs = format_affine_assign_rhs(coef, intercept)
        lines.append(f"(assign ({fluent} ?b) {rhs})")
    lines.append("(assign (mod) 2)")
    return "\n            ".join(lines)


def _predict_rolling_lr_post_state_from_models(
    lr_models: dict, x: float, y_pddl: float, vx: float, vy: float,
):
    """Predict post-contact state from explicit 4-feature LR models."""
    if not _rolling_lr_ready(lr_models):
        return None

    sanitized = _sanitize_rolling_lr_inputs(x, y_pddl, vx, vy)
    if sanitized is None:
        return None
    x, y_pddl, vx, vy = sanitized

    pre = {
        "x": float(x),
        "y": _pddl_y_to_learning(y_pddl),
        "v_x": float(vx),
        "v_y": float(vy),
    }
    predicted = _rolling_lr_predict_safe(lr_models, pre)
    if predicted is None:
        return None

    out_vx, out_vy = _clamp_platform_post_velocity(
        predicted["v_x"], predicted["v_y"], float(vx), float(vy),
    )
    predicted["v_x"] = out_vx
    predicted["v_y"] = out_vy
    return predicted


def predict_rolling_lr_post_state(kb: dict, x: float, y_pddl: float, vx: float, vy: float):
    """
    Predict post-contact bird state using the same LR selection as ENHSP inject.
    Returns dict with x, y, v_x, v_y in PDDL coordinates, or None if unavailable.
    """
    if not kb:
        return None
    lr_models, _ = _planning_surface_lr_models(kb)
    if lr_models is None:
        return None
    return _predict_rolling_lr_post_state_from_models(lr_models, x, y_pddl, vx, vy)


def _build_collision_ground_effect_m5(collision_vars):
    """Ground rolling uses the same post-contact semantics as platform slide."""
    return _build_collision_effect_m5(collision_vars, bounce_mode="platform")


def _build_collision_effect_m5(collision_vars, bounce_mode="ground"):
    """
    Build M5 piecewise collision effect for PDDL injection.

    bounce_mode:
      - 'ground' / 'platform': surface roll — retain vx/vy, no bounce_count bump
        (slide contacts must not count toward explode_bird's >= 3 limit).

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
        lines.append("(assign (mod) 2)")

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


def predict_surface_collision_post_state(
    kb: dict,
    x: float,
    y_pddl: float,
    vx: float,
    vy: float,
    m5_min_samples: int = PDDL_PLATFORM_M5_MIN_SAMPLES,
):
    """
    Predict post-contact state using the same LR → M5 priority as ENHSP inject.
    Inputs/outputs use PDDL coordinates for y and vy.
    """
    kind, meta = _surface_physics_kind(kb, m5_min_samples=m5_min_samples)
    if kind == "lr":
        lr_pred = _predict_rolling_lr_post_state_from_models(
            meta["lr_models"], x, y_pddl, vx, vy,
        )
        if lr_pred is not None:
            return lr_pred

    if kind != "m5" or not _platform_kb_has_models(kb):
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
        model = kb["variables"][var]["model"]
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


def predict_platform_collision_post_state(platform_kb: dict, x: float, y_pddl: float,
                                          vx: float, vy: float):
    """Platform contact prediction (M5 gated at PDDL_PLATFORM_M5_MIN_SAMPLES)."""
    return predict_surface_collision_post_state(
        platform_kb, x, y_pddl, vx, vy,
        m5_min_samples=PDDL_PLATFORM_M5_MIN_SAMPLES,
    )


def _platform_bounds(plat: dict):
    half_w = plat["w"] / 2
    half_h = plat["h"] / 2
    return (
        plat["x"] - half_w,
        plat["x"] + half_w,
        plat["y"] - half_h,
        plat["y"] + half_h,
    )


def _platform_surface_y(plat: dict, br: float) -> float:
    """Bird center y when resting on platform top (matches collision_platform guard)."""
    _, _, _, top = _platform_bounds(plat)
    return top + br * PLATFORM_SURFACE_RADIUS_FACTOR


def _bird_on_platform_span(x: float, br: float, plat: dict) -> bool:
    left, right, _, _ = _platform_bounds(plat)
    margin = br * PLATFORM_SLIDE_EDGE_MARGIN
    return (left - margin) <= x <= (right + margin)


def _ground_surface_y(br: float) -> float:
    return PLAYFIELD_FLOOR_Y + br * PLATFORM_SURFACE_RADIUS_FACTOR


def _rotated_aabb_extents(width: float, height: float, angle_deg: float):
    """Full width/height of the axis-aligned bbox of a rotated rectangle."""
    import math
    rad = math.radians(float(angle_deg))
    c, s = abs(math.cos(rad)), abs(math.sin(rad))
    w, h = float(width), float(height)
    return w * c + h * s, w * s + h * c


def block_bbox_from_game_object(block, margin_px: float = BLOCK_PLANNING_MARGIN_PX) -> dict:
    """
    Conservative block center and AABB in PDDL coords for ENHSP / forward sim.

    Uses polygon vertices when available. When the block is rotated but only MBR
    width/height are known, expands to the true rotated AABB. Adds margin_px on
    each side so trajectories with small execution error still register block contact.
    """
    angle = float(getattr(block, "angle", 0) or 0)
    margin = max(0.0, float(margin_px))

    if hasattr(block, "vertices") and block.vertices and len(block.vertices) >= 2:
        vertices = np.asarray(block.vertices, dtype=float)
        x_coords = vertices[:, 0]
        y_coords = vertices[:, 1]
        width = float(np.max(x_coords) - np.min(x_coords))
        height = float(np.max(y_coords) - np.min(y_coords))
        center_x = float((np.max(x_coords) + np.min(x_coords)) / 2)
        center_y_screen = float((np.max(y_coords) + np.min(y_coords)) / 2)
    else:
        width = float(block.width)
        height = float(block.height)
        center_x = float(block.X + block.width / 2)
        center_y_screen = float(block.Y + block.height / 2)
        if abs(angle) > 0.5:
            width, height = _rotated_aabb_extents(width, height, angle)

    width += 2.0 * margin
    height += 2.0 * margin

    return {
        "x_block": center_x,
        "y_block": 640.0 - center_y_screen,
        "block_width": width,
        "block_height": height,
    }


def platform_bbox_from_game_object(platform) -> dict:
    """
    Platform center and size in PDDL coords from ground-truth geometry.

    Prefer polygon vertices (float) over the GameObject MBR, which truncates to int
    and stores width/height on swapped axes in Rectangle.
    """
    if hasattr(platform, "vertices") and platform.vertices and len(platform.vertices) >= 2:
        vertices = np.asarray(platform.vertices, dtype=float)
        x_coords = vertices[:, 0]
        y_coords = vertices[:, 1]
        width = float(np.max(x_coords) - np.min(x_coords))
        height = float(np.max(y_coords) - np.min(y_coords))
        center_x = float((np.max(x_coords) + np.min(x_coords)) / 2)
        center_y_screen = float((np.max(y_coords) + np.min(y_coords)) / 2)
    else:
        # Rectangle stores screen x-extent in .height and y-extent in .width.
        width = float(platform.height)
        height = float(platform.width)
        center_x = float(platform.X + width / 2)
        center_y_screen = float(platform.Y + height / 2)

    return {
        "x_platform": center_x,
        "y_platform": 640.0 - center_y_screen,
        "platform_width": width,
        "platform_height": height,
    }


# Hysteresis: snap platform_top to pig_surface whenever they differ by no more
# than this many pixels. Purpose is to eat sub-pixel vision noise on the top
# vertex (see run_20260810_064920 level 1: 1-pixel wobble on the hill top
# flipped the ENHSP plan from a winning 84.5° dial to a losing 36.0° dial).
#
# History: run_20260812_114238 train-17 (t02_00023) lost with `diff = -3`
# where the old 1-px window did not fire and ENHSP declared the problem
# unsolvable. run_20260812_155211 tried widening this window to `pig_radius`
# (~3.5 px) — that "correction" also snapped down every normal pig-on-hill
# scene where SB routinely draws pigs with 2-3 px of visual overlap on the
# supporting hill, and 6+ levels (#2, #3, #4, #10, #14, #19, #22) that had
# been won on attempt 1 regressed to 2+ attempts. Total training attempts
# ballooned 46 → 57, the SB 200-min time budget expired before any test
# level was played, and #17 recovered by luck (favorable vision jitter),
# not by the widening. So the widening is reverted; #17-class multi-pixel
# vision jitter is left to the retry-diversify / ballistic-fallback path.
_PIG_STAND_SNAP_PX = 1.0
# Drop hill platforms whose horizontal span does not overlap the bird→pig corridor.
# run_20260815_093511 test #25 (t02_00070): platform_1 at x≈391 steered ENHSP toward
# high-arc over-flight plans while the pig stand is at x≈228; removing far hills
# leaves the stand line plus any mid-corridor obstacles only.
PLATFORM_CORRIDOR_MARGIN_PX = 15.0


def _block_aabb(block_data: dict):
    """Return left, right, bottom, top for a block entry in problem_data."""
    x = float(block_data["x_block"])
    y = float(block_data["y_block"])
    half_w = float(block_data["block_width"]) / 2.0
    half_h = float(block_data["block_height"]) / 2.0
    return x - half_w, x + half_w, y - half_h, y + half_h


def identify_sheltering_blocks(problem_data: dict) -> set:
    """
    Blocks that cap or wall the pig in the bird→pig corridor (template-4 pattern).

    Hill/platform templates (t01/t02) rarely satisfy this; keeps their pessimistic
    ENHSP platform model unchanged.
    """
    pig_key = next((k for k in problem_data if k.startswith("pig_")), None)
    bird_key = next((k for k in problem_data if k.startswith("bird_")), None)
    if not pig_key:
        return set()

    pig = problem_data[pig_key]
    px = float(pig["x_pig"])
    py = float(pig["y_pig"])
    pr = float(pig.get("pig_radius", 3.5))
    pig_stand = py - pr

    bx = float(problem_data[bird_key]["x_bird"]) if bird_key else px
    br = float(problem_data[bird_key].get("bird_radius", 4.0)) if bird_key else 0.0
    corridor_left = min(bx, px) - br - PLATFORM_CORRIDOR_MARGIN_PX
    corridor_right = max(bx, px) + pr + PLATFORM_CORRIDOR_MARGIN_PX

    sheltering = set()
    for key, val in problem_data.items():
        if not key.startswith("block_"):
            continue
        left, right, bottom, top = _block_aabb(val)
        x_overlap_pig = left - BLOCK_SHELTER_X_PAD_PX <= px <= right + BLOCK_SHELTER_X_PAD_PX
        caps_pig = bottom >= pig_stand - 5.0
        walls_corridor = (
            right >= corridor_left
            and left <= corridor_right
            and top >= pig_stand - 10.0
            and bottom <= py + pr + 15.0
        )
        if (x_overlap_pig and caps_pig) or walls_corridor:
            sheltering.add(key)
    return sheltering


def block_sheltered_layout(problem_data: dict) -> bool:
    return bool(identify_sheltering_blocks(problem_data))


def apply_block_shelter_adjustments(problem_data: dict, debug: bool = False) -> dict:
    """Tighten sheltering block AABB (Pass B) so ENHSP can plan trajectories that
    skirt past the shelter block — critical on t04 layouts where the raw conservative
    AABB blocks every angle. Also lowers block_life and raises bird_block_damage so
    the planner treats a bird→shelter collision as a definitive hard stop rather
    than a bounce-off; no crush/destroy shortcut is exposed (see
    base_domain.pddl — shelter_block_crush_pig was removed after run_20260816_181229
    where 44+ Pass B plans predicted 'hit block → crush' but sim/game confirmed
    zero pig kills)."""
    sheltering = identify_sheltering_blocks(problem_data)
    if not sheltering:
        return problem_data

    shrink = 2.0 * max(0.0, BLOCK_PLANNING_MARGIN_PX - BLOCK_SHELTER_MARGIN_PX)
    for key in sheltering:
        block = dict(problem_data[key])
        block["block_width"] = max(4.0, float(block["block_width"]) - shrink)
        block["block_height"] = max(4.0, float(block["block_height"]) - shrink)
        block["block_life"] = max(
            0.2, float(block.get("block_life", 0.75)) * BLOCK_SHELTER_LIFE_FACTOR,
        )
        block["block_stability"] = min(float(block.get("block_stability", 1.0)), 0.5)
        block["_sheltering_block"] = True
        problem_data[key] = block
        if debug:
            print(
                f"[BLOCK DEBUG] {key}: shelters pig — tightened AABB "
                f"({block['block_width']:.1f}x{block['block_height']:.1f}), "
                f"life={block['block_life']:.2f}, damage={BLOCK_SHELTER_DAMAGE}"
            )
    return problem_data


def should_use_planning_pessimistic_platform(problem_data: dict) -> bool:
    """
    Use hard-stop platform collision for ENHSP on hill/slide-sensitive layouts only.

    Block-sheltered levels (typical t04) keep the slide/M5 platform model so ENHSP
    can plan platform-contact kills; t02 hill levels stay pessimistic.
    """
    if not any(k.startswith("platform_") for k in problem_data):
        return False
    return not block_sheltered_layout(problem_data)


def _extend_platforms_for_pig(platforms: list, px: float, py: float, pr: float):
    """Align platform AABB with the pig stand line (symmetric snap + upward bump).

    - If ``|pig_surface - top| <= _PIG_STAND_SNAP_PX`` snap ``top`` to
      ``pig_surface`` (both up and down). Kills the branch flip that made ENHSP
      plans non-reproducible across replays.
    - Else if ``pig_surface > top`` still extend upward as before (unbounded lift).
    - Otherwise leave the platform alone (genuinely taller than the pig line —
      a wall, not a stand line, OR the standard SB pig-on-hill visual overlap).
    """
    if px is None or py is None or not platforms:
        return platforms
    extended = []
    for plat in platforms:
        p = dict(plat)
        left, right, _bottom, top = _platform_bounds(p)
        if left - pr <= px <= right + pr:
            pig_surface = py - pr
            diff = pig_surface - top
            if abs(diff) <= _PIG_STAND_SNAP_PX:
                if diff != 0:
                    p["h"] = p["h"] + diff
                    p["y"] = p["y"] + diff / 2
            elif diff > 0:
                p["h"] = p["h"] + diff
                p["y"] = p["y"] + diff / 2
        extended.append(p)
    return extended


def extend_platforms_in_problem_data(
    problem_data: dict, debug: bool = False, apply_shelter: bool = False,
) -> dict:
    """
    Align platform tops with the pig stand line when geometry is still slightly low.

    Primary bounds come from polygon vertices in get_platforms(); this pass only
    bumps the AABB upward when a pig standing on the hill is above the computed top.

    ``apply_shelter`` toggles the block-shelter adjustments (tightened AABB,
    reduced ``block_life``). Left off by default so vision→PDDL preserves the
    raw geometry; callers can turn it on for the shelter-mode ENHSP pass.
    """
    pig_key = next((k for k in problem_data if k.startswith("pig_")), None)
    if not pig_key:
        return problem_data

    pig = problem_data[pig_key]
    px = float(pig["x_pig"])
    py = float(pig["y_pig"])
    pr = float(pig.get("pig_radius", 3.5))

    if apply_shelter:
        apply_block_shelter_adjustments(problem_data, debug=debug)

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
                f"[PLATFORM DEBUG] {key}: safety adjust to pig stand line "
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

    return filter_platforms_for_corridor(problem_data, debug=debug)


def filter_platforms_for_corridor(problem_data: dict, debug: bool = False) -> dict:
    """Remove platforms outside the slingshot→pig horizontal corridor.

    Keeps:
      - the pig stand platform (pig x within platform span ± pig_radius), and
      - any platform whose horizontal bounds overlap
        [min(bird_x, pig_x), max(bird_x, pig_x)] expanded by radii/margin.

    Forward sim and ENHSP share ``problem_data``; distant decorative hills that
    never lie on a direct shot path are dropped so ENHSP does not plan high-arc
    routes around irrelevant geometry.
    """
    bird_key = next((k for k in problem_data if k.startswith("bird_")), None)
    pig_key = next((k for k in problem_data if k.startswith("pig_")), None)
    if not bird_key or not pig_key:
        return problem_data

    bx = float(problem_data[bird_key]["x_bird"])
    br = float(problem_data[bird_key].get("bird_radius", 4.0))
    px = float(problem_data[pig_key]["x_pig"])
    pr = float(problem_data[pig_key].get("pig_radius", 3.5))
    margin = float(PLATFORM_CORRIDOR_MARGIN_PX)

    corridor_left = min(bx, px) - br - margin
    corridor_right = max(bx, px) + pr + margin

    drop_keys = []
    for key, val in problem_data.items():
        if not key.startswith("platform_"):
            continue
        plat = {
            "x": float(val["x_platform"]),
            "y": float(val["y_platform"]),
            "w": float(val["platform_width"]),
            "h": float(val["platform_height"]),
        }
        left, right, _bottom, _top = _platform_bounds(plat)

        pig_on_stand = left - pr <= px <= right + pr
        overlaps_corridor = right >= corridor_left and left <= corridor_right
        if pig_on_stand or overlaps_corridor:
            continue

        drop_keys.append(key)
        if debug:
            print(
                f"[PLATFORM DEBUG] {key}: dropped (outside bird→pig corridor "
                f"[{corridor_left:.1f}, {corridor_right:.1f}]; "
                f"platform x=[{left:.1f}, {right:.1f}], pig x={px:.1f})"
            )

    for key in drop_keys:
        del problem_data[key]

    if debug and drop_keys:
        kept = [k for k in problem_data if k.startswith("platform_")]
        print(
            f"[PLATFORM DEBUG] Corridor filter: kept {len(kept)} platform(s) "
            f"{kept}, dropped {len(drop_keys)}"
        )

    return problem_data


def _platform_collision_reach(br: float, margin_mult: float = PLATFORM_MARGIN_MULT) -> float:
    """Effective bird reach for platform contact (aligned with game event detection)."""
    return (float(br) + PLATFORM_COLLISION_EPSILON) * margin_mult


def _bird_overlaps_platform(x: float, y: float, br: float, plat: dict,
                            margin_mult: float = PLATFORM_MARGIN_MULT) -> bool:
    """
    Circle-vs-AABB closest-point test — same geometry as ``is_platform_collision``
    in event_conditions.py (screen coords) after PDDL y-up bounds mapping.
    """
    left, right, bottom, top = _platform_bounds(plat)
    closest_x = max(left, min(x, right))
    closest_y = max(bottom, min(y, top))
    dx = x - closest_x
    dy = y - closest_y
    reach = _platform_collision_reach(br, margin_mult)
    return dx * dx + dy * dy <= reach * reach


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


def _default_ground_roll_post_state(x: float, y: float, vx: float, vy: float, br: float):
    """
    Heuristic ground roll when platform/ground KB is cold.
    Same speed retention as platform slide (_default_platform_slide_post_state).
    """
    pre_speed = math.hypot(vx, vy)
    if pre_speed < 5.0:
        return None

    ratio = PLATFORM_DEFAULT_SPEED_RATIO
    out_vx = vx * ratio
    out_vy = vy * PLATFORM_DEFAULT_VY_RATIO
    if abs(out_vx) < 8.0:
        out_vx = math.copysign(max(pre_speed * ratio, 8.0), vx if abs(vx) > 1e-6 else 1.0)
    surface_y = PLAYFIELD_FLOOR_Y + br * PLATFORM_SURFACE_RADIUS_FACTOR
    out_vx, out_vy = _clamp_platform_post_velocity(out_vx, out_vy, vx, vy)
    return {"y": surface_y, "v_x": out_vx, "v_y": out_vy}


def _apply_ground_contact(collision_kb: dict, x: float, y: float,
                          vx: float, vy: float, br: float, debug: bool = False):
    """
    Apply ground contact as surface rolling — LR first, then M5/heuristic fallback.
    Returns (new_x, new_y, new_vx, new_vy, roll_continued, hard_stop).
    """
    predicted = predict_surface_collision_post_state(
        collision_kb or {}, x, y, vx, vy, m5_min_samples=0,
    )
    if predicted is None:
        predicted = _default_ground_roll_post_state(x, y, vx, vy, br)
        if predicted is None:
            if debug:
                print("[SIM DEBUG]   Ground hard stop (roll unavailable; speed too low)")
            return x, y, vx, vy, False, True
        if debug:
            print("[SIM DEBUG]   Bootstrap ground roll (KB cold; surface heuristic)")
    elif debug:
        post_speed = math.hypot(predicted["v_x"], predicted["v_y"])
        print(
            f"[SIM DEBUG]   Learned post-ground roll: y={predicted['y']:.1f}, "
            f"vx={predicted['v_x']:.1f}, vy={predicted['v_y']:.1f}, speed={post_speed:.1f}"
        )

    post_speed = math.hypot(predicted["v_x"], predicted["v_y"])
    roll = post_speed > 5.0
    out_x = predicted.get("x", x)
    out_vx = predicted["v_x"]
    out_vy = predicted["v_y"]
    # Learned one-shot samples often carry vy>0; rolling phase uses horizontal slide only.
    if roll:
        out_vy = 0.0
    return out_x, predicted["y"], out_vx, out_vy, roll, not roll


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
        # Cold-start KB: the old hard stop rejected every pig-on-hill ENHSP plan
        # whose arc grazes the ramp before the pig center (run_20260826: t05_00077
        # @ 66.5° stops at (361,395) with pig at (359.5,387) — 8.6 px short).
        # Heuristic hill slide lets the sim continue and reach the pig, matching
        # what the game does when the bird rolls up the stand line.
        predicted = _default_platform_slide_post_state(plat, x, y, vx, vy, br)
        if predicted is None:
            if debug:
                print(
                    "[SIM DEBUG]   Bootstrap platform hard stop "
                    "(slide heuristic unavailable; speed too low)"
                )
            return x, y, vx, vy, False, True
        if debug:
            print("[SIM DEBUG]   Bootstrap platform slide (KB cold; hill heuristic)")

    post_speed = math.hypot(predicted["v_x"], predicted["v_y"])
    slide = post_speed > 5.0
    out_x = predicted.get("x", x)
    return out_x, predicted["y"], predicted["v_x"], predicted["v_y"], slide, not slide

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
ANGLE_BIAS_DEGREES = 0
DEG_TO_RAD = 0.01745329252
# Slingshot pull clamp (screen-space aim); separate from ENHSP dial min (may be negative).
GAME_PULL_MIN = 15.0
GAME_PULL_MAX = 85.0
# Shallow ENHSP dial angles are hard to reproduce; platform levels use a higher floor.
# NOTE: In practice this value is dominated by the calibrator's executable dial floor
# (~25° with a full calibration curve), so the constant is effectively inert unless
# raised well above it. A previous experiment at 35° regressed win rate 18/21 → 13/21
# because _level_has_platforms fires on any scene with a hill in the scenery — not
# only when the pig is on it — forcing steep lobs on levels that need direct shots.
# Keep at 20° until the platform-floor trigger is narrowed to pig-on-hill geometry.
PLANNER_MIN_ANGLE_PLATFORM = 20.0


class AngleCalibrator:
    """
    Convert PDDL planner dial angle to game slingshot pull angle.

    PDDL flight uses (dial - ANGLE_BIAS_DEGREES). The slingshot does not obey a fixed
    offset: shallow pulls lose much more angle than steep ones. This class inverts
    (game_pull → measured launch) so measured ≈ PDDL flight angle.

    Force-dependence:
        Same pull produces different measured launch angles at different forces —
        the game's short-pullback release physics vary with pullback distance.
        For example (from run logs):
            pull=17.9°, force=1.00 → measured=20.2°
            pull=17.9°, force=0.85 → measured=13.4°  (−7° at same pull)
        We keep two bootstrap curves — force≈1.0 (``_BOOTSTRAP``) and force≈0.85
        (``_BOOTSTRAP_PARTIAL``) — and linearly blend them by ``force`` in
        [0.85, 1.0]. Force outside that band uses the nearer curve (conservative:
        below 0.85 keeps the partial-force curve rather than extrapolating).
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

    # Bootstrap (game_pull, measured_θ) at force≈0.85, derived from clean-execution
    # observations in run_20260819_112711 + run_20260816_181229 (|angle-err|≤6°,
    # unclamped pulls). At shallow pulls the game produces a much flatter launch
    # for partial force; at pull ≥ 51° the two curves converge, so beyond that we
    # inherit the force=1.0 shape (measured values pinned to _BOOTSTRAP).
    _BOOTSTRAP_PARTIAL = (
        (17.9, 13.4),
        (24.0, 22.7),
        (29.0, 26.0),
        (35.0, 28.0),
        (39.0, 32.0),
        (44.0, 42.0),
        (51.0, 50.9),
        (59.0, 58.8),
        (75.5, 73.9),
        (81.5, 77.9),
    )

    # Force at which _BOOTSTRAP_PARTIAL was measured. Between this and 1.0 we
    # linearly blend the two curves; outside this band we clamp to the nearer
    # curve rather than extrapolating.
    _PARTIAL_FORCE_ANCHOR = 0.85

    def __init__(self):
        self._samples = []

    def pddl_flight_angle(self, pddl_dial: float) -> float:
        return float(pddl_dial) - ANGLE_BIAS_DEGREES

    def raw_game_pull_for_pddl_dial(
        self,
        pddl_dial: float,
        force: float = 1.0,
    ) -> float:
        """Unclamped slingshot pull (degrees) for a PDDL dial target flight angle."""
        return self._invert_measured_to_pull(self.pddl_flight_angle(pddl_dial), force=force)

    def measured_flight_for_pddl_dial(
        self,
        pddl_dial: float,
        min_pull: float = GAME_PULL_MIN,
        max_pull: float = GAME_PULL_MAX,
        force: float = 1.0,
    ) -> float:
        """Launch angle the game will produce for this PDDL dial (after pull clamping)."""
        flight_target = self.pddl_flight_angle(pddl_dial)
        # Steep SB6.6 lobs: game flight tracks the BamBirds pull angle, not the
        # PDDL flight target. Shipped bootstrap extrapolates to ~79° for an
        # 82.5° dial while observed launches are ~87° (multiple_forces 00033).
        from .bambirds_shot_helper import actual_to_launch_deg, HIGH_ANGLE_BEGIN_RAD
        if math.radians(flight_target) >= HIGH_ANGLE_BEGIN_RAD:
            return actual_to_launch_deg(flight_target)
        game_pull = self.game_pull_for_pddl_dial(
            pddl_dial, min_pull=min_pull, max_pull=max_pull, force=force,
        )
        return self.measured_flight_for_game_pull(
            game_pull, min_pull=min_pull, max_pull=max_pull, force=force,
        )

    def is_dial_executable(
        self,
        pddl_dial: float,
        min_pull: float = GAME_PULL_MIN,
        max_pull: float = GAME_PULL_MAX,
        flight_slack: float = ANGLE_EXECUTION_SLACK_DEG,
        force: float = 1.0,
    ) -> bool:
        """
        True when pull is unclamped and measured launch angle matches PDDL flight target.

        Rejects dials whose inverted pull sits at the slingshot floor/ceiling but cannot
        reach the ENHSP flight angle (e.g. dial 0° clamped to pull 15° → shallow flight).
        With ``force<1.0`` the check uses the force-blended curve, so shallow-angle
        requests that are only reachable at full force are rejected.
        """
        target = self.pddl_flight_angle(pddl_dial)
        raw_pull = self.raw_game_pull_for_pddl_dial(pddl_dial, force=force)
        if not (min_pull <= raw_pull <= max_pull):
            return False
        game_pull = self.game_pull_for_pddl_dial(
            pddl_dial, min_pull=min_pull, max_pull=max_pull, force=force,
        )
        if abs(game_pull - raw_pull) > 1e-3:
            return False
        measured = self.measured_flight_for_pddl_dial(
            pddl_dial, min_pull=min_pull, max_pull=max_pull, force=force,
        )
        return abs(measured - target) <= float(flight_slack)

    def executable_dial_bounds(
        self,
        dial_min: float = -20.0,
        dial_max: float = 89.0,
        dial_step: float = 0.5,
        min_pull: float = GAME_PULL_MIN,
        max_pull: float = GAME_PULL_MAX,
        flight_slack: float = ANGLE_EXECUTION_SLACK_DEG,
        force: float = 1.0,
    ):
        """
        Scan PDDL dial range; return (min_dial, max_dial) with unclamped pulls whose
        measured launch angle matches the PDDL flight target within flight_slack.
        """
        executable = [
            d for d in np.arange(dial_min, dial_max + 1e-6, dial_step)
            if self.is_dial_executable(
                d, min_pull=min_pull, max_pull=max_pull,
                flight_slack=flight_slack, force=force,
            )
        ]
        if not executable:
            return float(dial_min), float(dial_max)
        return float(min(executable)), float(max(executable))

    def game_pull_for_pddl_dial(
        self,
        pddl_dial: float,
        min_pull: float = GAME_PULL_MIN,
        max_pull: float = GAME_PULL_MAX,
        force: float = 1.0,
    ) -> float:
        target = self.pddl_flight_angle(pddl_dial)
        pull = self._invert_measured_to_pull(target, force=force)
        return max(min_pull, min(max_pull, pull))

    def measured_flight_for_game_pull(
        self,
        game_pull: float,
        min_pull: float = GAME_PULL_MIN,
        max_pull: float = GAME_PULL_MAX,
        force: float = 1.0,
    ) -> float:
        """Measured launch angle (degrees) for a slingshot pull (after clamping)."""
        pull = max(min_pull, min(max_pull, float(game_pull)))
        pulls, measured = self._get_curve(force=force)
        if len(pulls) < 2:
            return pull
        return float(np.interp(pull, pulls, measured))

    def sim_dial_for_game_execution(
        self,
        pddl_dial: float,
        min_pull: float = GAME_PULL_MIN,
        max_pull: float = GAME_PULL_MAX,
        force: float = 1.0,
    ) -> float:
        """
        PDDL dial to pass to forward sim so trajectory matches in-game launch.

        Delegates to ``measured_flight_for_pddl_dial`` (bootstrap at low/mid
        angles; BamBirds pull-angle model above ``HIGH_ANGLE_BEGIN_RAD``) so
        sim and ``pddl_shot_to_release_point`` stay aligned on steep lobs.
        """
        measured = self.measured_flight_for_pddl_dial(
            pddl_dial, min_pull=min_pull, max_pull=max_pull, force=force,
        )
        return measured + ANGLE_BIAS_DEGREES

    def record_shot(
        self,
        pddl_dial: float,
        game_pull: float,
        measured_deg: float,
        force: float = 1.0,
        raw_pull: float = None,
    ) -> None:
        """Append a calibration sample (full-force shots only)."""
        if force < 0.95:
            return
        pull = float(game_pull)
        # Clamped pulls do not reflect the requested dial → flight mapping.
        if pull <= GAME_PULL_MIN + 0.5 or pull >= GAME_PULL_MAX - 0.5:
            return
        if raw_pull is not None and abs(float(raw_pull) - pull) > 1e-3:
            return
        self._samples.append((pull, float(measured_deg)))

    # A learned sample only overrides bootstrap within this pull neighborhood.
    # This prevents a single wild measurement (e.g. pull=50→meas=8) from
    # wiping out large stretches of the shipped calibration.
    _LEARNED_OVERRIDE_PULL_WINDOW = 5.0

    def _build_curve(self, bootstrap=None, include_samples: bool = True):
        """
        Merge bootstrap + samples into a monotonic pull→measured curve.

        ``bootstrap`` selects which bootstrap tuple to use (defaults to the
        force≈1.0 ``_BOOTSTRAP``). Learned samples are only merged into the
        force≈1.0 curve — ``record_shot`` requires force ≥ 0.95, so they do not
        represent the force≈0.85 regime and must not corrupt that curve.

        Learned samples take precedence over LOCALLY conflicting bootstrap
        (within _LEARNED_OVERRIDE_PULL_WINDOW degrees): a stale bootstrap
        point near a learned sample is dropped, on the theory that the
        running game state is more authoritative than the shipped calibration.
        Learned samples that break monotonicity against learned neighbors, or
        that disagree with bootstrap far outside their pull neighborhood, are
        skipped (likely noise or clamped shots).
        """
        if bootstrap is None:
            bootstrap = self._BOOTSTRAP
        merged = {}  # pull -> (measured, is_learned)
        for pull, meas in bootstrap:
            merged[float(pull)] = (float(meas), False)
        if include_samples:
            for pull, meas in self._samples:
                p, m = float(pull), float(meas)
                if p in merged:
                    prev_m, prev_learned = merged[p]
                    if prev_learned:
                        merged[p] = (0.5 * (prev_m + m), True)
                    else:
                        merged[p] = (m, True)
                else:
                    merged[p] = (m, True)

        # Enforce monotonic (non-decreasing) measured vs. pull, walking pulls
        # ascending. When a point would break monotonicity:
        #   - A learned point may drop nearby bootstrap (within the pull
        #     window), reflecting a fresher local reading.
        #   - Otherwise the offending point is skipped.
        window = self._LEARNED_OVERRIDE_PULL_WINDOW
        pulls_sorted = sorted(merged.keys())
        clean = []  # list of (pull, measured, is_learned)
        for p in pulls_sorted:
            m, is_learned = merged[p]
            while clean and clean[-1][1] > m + 1e-6:
                last_p, _, last_learned = clean[-1]
                if is_learned and not last_learned and (p - last_p) <= window:
                    clean.pop()  # drop stale bootstrap in local neighborhood
                else:
                    break
            if not clean or m >= clean[-1][1] - 1e-6:
                clean.append((p, m, is_learned))
            # else: skip this point to preserve monotonicity.

        pulls = np.array([p for p, _, _ in clean], dtype=float)
        measured = np.array([m for _, m, _ in clean], dtype=float)
        return pulls, measured

    def _blend_weight(self, force: float) -> float:
        """
        Weight for the force=1.0 curve. Returns 1.0 for force≥1.0, 0.0 for
        force≤anchor, linear between. Below-anchor forces clamp to the partial
        curve rather than extrapolating (we have no evidence for force<0.85).
        """
        f = float(force)
        if f >= 1.0:
            return 1.0
        if f <= self._PARTIAL_FORCE_ANCHOR:
            return 0.0
        return (f - self._PARTIAL_FORCE_ANCHOR) / (1.0 - self._PARTIAL_FORCE_ANCHOR)

    def _get_curve(self, force: float = 1.0):
        """(pulls, measured) curve for a given force, blending the two bootstraps.

        For force≥1.0 this returns the full-force curve (with learned samples).
        For force≤anchor this returns the partial-force curve (no learned mixing).
        In between, we evaluate both curves at a common pull axis and linearly
        blend the measured values by ``_blend_weight(force)``.
        """
        w = self._blend_weight(force)
        if w >= 1.0 - 1e-9:
            return self._build_curve(bootstrap=self._BOOTSTRAP, include_samples=True)
        if w <= 1e-9:
            return self._build_curve(
                bootstrap=self._BOOTSTRAP_PARTIAL, include_samples=False,
            )
        pulls_full, meas_full = self._build_curve(
            bootstrap=self._BOOTSTRAP, include_samples=True,
        )
        pulls_part, meas_part = self._build_curve(
            bootstrap=self._BOOTSTRAP_PARTIAL, include_samples=False,
        )
        # Common pull axis: union of both curves' pulls, restricted to the
        # overlap so np.interp stays inside both.
        lo = float(max(pulls_full[0], pulls_part[0]))
        hi = float(min(pulls_full[-1], pulls_part[-1]))
        axis_pulls = sorted(
            set(float(p) for p in pulls_full if lo <= p <= hi)
            | set(float(p) for p in pulls_part if lo <= p <= hi)
            | {lo, hi}
        )
        pulls_axis = np.array(axis_pulls, dtype=float)
        m_full = np.interp(pulls_axis, pulls_full, meas_full)
        m_part = np.interp(pulls_axis, pulls_part, meas_part)
        blended = (1.0 - w) * m_part + w * m_full
        # Re-enforce monotonicity after blending (both inputs are monotonic, so
        # the blend should be too, but np.interp + float noise can nudge equal
        # values by ε — clamp to non-decreasing).
        for i in range(1, len(blended)):
            if blended[i] < blended[i - 1]:
                blended[i] = blended[i - 1]
        return pulls_axis, blended

    def _get_inversion_curve(self, force: float = 1.0):
        """Measured (non-decreasing) → pull for dial inversion."""
        pulls, measured = self._get_curve(force=force)
        if len(pulls) < 2:
            return measured, pulls

        pairs = sorted(zip(measured, pulls))
        m_out, p_out = [], []
        for m, p in pairs:
            if m_out and m < m_out[-1] - 1e-6:
                continue
            if m_out and abs(m - m_out[-1]) < 1e-6:
                p_out[-1] = 0.5 * (p_out[-1] + p)
            else:
                m_out.append(m)
                p_out.append(p)
        return np.array(m_out, dtype=float), np.array(p_out, dtype=float)

    def _invert_measured_to_pull(self, target_measured: float, force: float = 1.0) -> float:
        measured, pulls = self._get_inversion_curve(force=force)
        target = float(target_measured)
        if len(measured) < 2:
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


def pddl_flight_angle_deg(dial_deg: float) -> float:
    """Launch angle used by base_domain trig: ``(angle - angle_bias)``."""
    return float(dial_deg) - ANGLE_BIAS_DEGREES


def pddl_planned_launch_state(
    dial_deg: float,
    ref_x: float,
    ref_y: float,
    force: float,
    v_bird: float,
    force_lr_model=None,
    speed_at_force=None,
) -> dict:
    """
    PDDL launch state immediately after pa-twang — position and velocity the
    domain assigns for this angle and force.
    """
    _, cosine, sinus = _initial_angle_trig(dial_deg)
    launch_x, launch_y = pddl_bird_position_after_pa_twang(ref_x, ref_y, dial_deg)
    speed = effective_launch_speed(
        v_bird, force, angle_deg=pddl_flight_angle_deg(dial_deg),
        speed_at_force=speed_at_force, force_lr_model=force_lr_model,
    )
    return {
        "launch_x": launch_x,
        "launch_y": launch_y,
        "vx": speed * cosine,
        "vy": speed * sinus,
        "speed": speed,
        "flight_angle_deg": pddl_flight_angle_deg(dial_deg),
        "angle_deg": float(dial_deg),
        "force": float(force),
    }


# ── Mode B — actuator resolution knob ────────────────────────────────────
# See agents/pddl/CALIBRATION_MODE_B_ACTUATOR_RESOLUTION.md.
#
# The default ``mag = sling.height * 5 * v_portion`` in
# ``SimpleTrajectoryPlanner.find_release_point_partial_power`` gives only
# ~0.2 px per degree of angular resolution when ``sling.height`` is 3–4 px
# (as vision reports for SB6.6). Integer-rounding then discretises the
# reachable launch angles into ~5° bins.
#
# ``MODE_B_MAG_MULTIPLIER`` scales the pullback distance for the PDDL
# agent only (BaselineAgent still uses the raw formula). Larger values
# improve angular resolution but risk overshooting the game's max-pull
# distance — beyond which pulling harder no longer increases velocity
# (SB6.6 clamps) or worse, becomes an invalid shot.
#
# Start at 1.0 (no change). Bump progressively (2 → 4 → 8) and watch
# [MODE B DIAG] logs for velocity behaviour. When observed ``|v|`` stops
# tracking ``force`` linearly (i.e., clamped at v_max regardless of
# multiplier), we've found max_pull; use the largest safe value.
# 4.0 ≈ 4× finer angular bins vs baseline (run_20260822_190314 used 1.0).
MODE_B_MAG_MULTIPLIER = 4.0
# Floor pullback (px) at force=1.0 when vision under-reports sling.height (~3–4 px
# vs ~18 px observed in [MODE B DIAG] / game DragDetails). Without this, multiplier
# alone cannot help when base_mag = sling_h * 5 * v_portion collapses to ~2 px.
MODE_B_MIN_MAG_AT_FULL_FORCE = 90.0


def mode_b_scaled_mag(sling_height: float, force: float) -> float:
    """Pullback magnitude (px) used by ``pddl_shot_to_release_point``."""
    v_portion = pddl_force_to_v_portion(force)
    base_mag = float(sling_height) * 5.0 * v_portion
    force_clamped = max(FORCE_MIN, min(FORCE_MAX, float(force)))
    return max(
        base_mag * MODE_B_MAG_MULTIPLIER,
        MODE_B_MIN_MAG_AT_FULL_FORCE * force_clamped,
    )


def mode_b_angular_resolution_deg(sling_height: float, force: float = 1.0) -> float:
    """Smallest dial change (°) from a 1 px release-point step at ``scaled_mag``."""
    mag = mode_b_scaled_mag(sling_height, force)
    if mag <= 0:
        return float("inf")
    return 90.0 / mag


def mode_b_dial_collapse_map(sling, dial_min: float, dial_max: float, force: float = 1.0):
    """
    Map each integer release pixel to the dials that collapse onto it.

    Returns ``(pixel_to_dials, resolution_deg, scaled_mag)`` for Mode B diagnostics.
    """
    from .bambirds_shot_helper import actual_to_launch
    mag = mode_b_scaled_mag(float(getattr(sling, "height", 0) or 0), force)
    pixel_to_dials: dict[tuple[int, int], list[float]] = {}
    step = 0.5
    d = float(dial_min)
    while d <= float(dial_max) + 1e-9:
        flight_rad = math.radians(pddl_flight_angle_deg(d))
        corrected_rad = actual_to_launch(flight_rad)
        pt = pddl_release_point_with_mag(sling, corrected_rad, mag)
        key = (int(pt.X), int(pt.Y))
        pixel_to_dials.setdefault(key, []).append(round(d, 1))
        d += step
    resolution = mode_b_angular_resolution_deg(float(getattr(sling, "height", 0) or 0), force)
    return pixel_to_dials, resolution, mag


def pddl_release_point_with_mag(sling, theta_rad: float, mag_px: float):
    """
    Screen release point for a bird with fully-controlled pullback distance.

    Mirrors ``SimpleTrajectoryPlanner.find_release_point_partial_power`` but
    lets the caller specify ``mag_px`` directly (in pixels) instead of
    deriving it from ``sling.height * 5 * v_portion``. This is the primary
    knob for Mode B calibration work — see the constant above and the
    Mode B design doc.
    """
    from utils.point2D import Point2D
    ref_x = round(sling.X + 0.45 * sling.width)
    ref_y = round(sling.Y + 0.35 * sling.width)
    rel_x = round(ref_x - mag_px * math.cos(theta_rad))
    rel_y = round(ref_y + mag_px * math.sin(theta_rad))
    return Point2D(rel_x, rel_y)


def pddl_shot_to_release_point(tp, sling, dial_deg: float, force: float):
    """
    Map PDDL angle+force to a game release point.

    The domain's dial-minus-angle_bias is the *actual flight angle* we want
    the bird to produce. But the game's sling launches slightly flatter than
    the requested pull angle (angle-dependent, larger deficit at low angles).
    Apply BamBirds' ``actual_to_launch`` correction so what we ask the game
    to fire at compensates for the deficit, and the observed flight matches
    what the PDDL domain simulates.

    Mode B: pullback distance ``mag`` is scaled by
    ``MODE_B_MAG_MULTIPLIER`` for finer angular resolution.

    Reference: BamBirds ShotHelper.actualToLaunch (see bambirds_shot_helper).
    """
    from .bambirds_shot_helper import actual_to_launch
    flight_angle_rad = math.radians(pddl_flight_angle_deg(dial_deg))
    corrected_launch_rad = actual_to_launch(flight_angle_rad)
    v_portion = pddl_force_to_v_portion(force)
    sling_h = float(getattr(sling, "height", 0) or 0)
    scaled_mag = mode_b_scaled_mag(sling_h, force)
    px_per_deg = scaled_mag / 90.0
    base_mag = sling_h * 5.0 * v_portion
    force_clamped = max(FORCE_MIN, min(FORCE_MAX, float(force)))
    print(
        f"[MODE B DIAG] sling.height={sling_h:.1f}px  v_portion={v_portion:.3f}  "
        f"base_mag={base_mag:.1f}px  scaled_mag={scaled_mag:.1f}px "
        f"(x{MODE_B_MAG_MULTIPLIER}, floor={MODE_B_MIN_MAG_AT_FULL_FORCE * force_clamped:.1f})  "
        f"angular_res={px_per_deg:.2f} px/° (~{mode_b_angular_resolution_deg(sling_h, force):.2f}°/px)"
    )
    return pddl_release_point_with_mag(sling, corrected_launch_rad, scaled_mag)


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
        # T1.2: ENHSP requires every numeric fluent it evaluates to have a
        # defined initial value. Missing inits caused inconsistent heuristic
        # estimates on some layouts (see SPEED MISMATCH cluster in
        # run_20260822_082303.log).
        f"(= (points_score) 0)",
        f"(= (mod) 0)",
    ]
    # T1.2 (continued): per-bird velocity fluents. These are undefined until
    # pa-twang fires; ENHSP's numeric heuristic then treats them as unknown and
    # prunes the flying branch. Initialising to 0 is consistent with a bird at
    # rest on the sling.
    for object_name in problem_data:
        if object_name.startswith('bird_'):
            initial_state.append(f"(= (vx_bird {object_name}) 0)")
            initial_state.append(f"(= (vy_bird {object_name}) 0)")
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
                block_data = problem_data[other_object]
                damage = (
                    BLOCK_SHELTER_DAMAGE
                    if block_data.get("_sheltering_block")
                    else DEFAULT_BIRD_BLOCK_DAMAGE
                )
                initial_state.append(
                    f'(= (bird_block_damage {object} {other_object}) {damage})'
                )

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


def _inject_surface_sentinel(
    content: str,
    sentinel: str,
    kb: dict,
    *,
    placeholder_body: str,
    m5_min_samples: int = 0,
    m5_bounce_mode: str = "platform",
    planning_optimistic_slide: bool = False,
):
    """Inject LR → M5 → placeholder at sentinel. Returns (content, inject_label)."""
    if sentinel not in content:
        return content, "skipped"

    kind, meta = _surface_physics_kind(kb, m5_min_samples=m5_min_samples)
    n_samples = meta.get("n_samples", 0)

    if kind == "lr":
        body = _build_rolling_lr_effect(meta["lr_models"])
        vx_r2 = meta.get("vx_r2")
        tag = meta.get("lr_tag", "lr")
        _collision_inject_log(
            f"Injected LR {m5_bounce_mode} rolling effect ({n_samples} samples, "
            f"{tag}, v_x R²={vx_r2:.3f})"
            if vx_r2 is not None
            else f"Injected LR {m5_bounce_mode} rolling effect ({n_samples} samples, {tag})"
        )
        if planning_optimistic_slide and m5_bounce_mode == "platform" and body:
            body = _append_extra_planning_slide_x(body)
            _collision_inject_log(
                f"Appended optimistic slide x-advance (+{PLATFORM_PLANNING_SLIDE_SEC}s·vx)"
            )
        return content.replace(sentinel, body), "lr"

    if kind == "m5":
        variables = (kb or {}).get("variables") or {}
        m5_body = _build_collision_effect_m5(variables, bounce_mode=m5_bounce_mode)
        if m5_body is not None:
            _collision_inject_log(
                f"Injected M5 {m5_bounce_mode} collision effect ({n_samples} samples)"
            )
            if planning_optimistic_slide and m5_bounce_mode == "platform":
                m5_body = _append_extra_planning_slide_x(m5_body)
                _collision_inject_log(
                    f"Appended optimistic slide x-advance (+{PLATFORM_PLANNING_SLIDE_SEC}s·vx)"
                )
            return content.replace(sentinel, m5_body), "m5"

    _log_surface_placeholder_reason(kb, n_samples, m5_min_samples)
    if planning_optimistic_slide and m5_bounce_mode == "platform":
        placeholder_body = _append_extra_planning_slide_x(placeholder_body)
    return content.replace(sentinel, placeholder_body), "placeholder"


def _log_surface_placeholder_reason(kb: dict, n_samples: int, m5_min_samples: int):
    pddl_lr = (kb or {}).get("lr_models_pddl") or {}
    lr_vx_r2 = _rolling_lr_vx_r2(pddl_lr)
    if _rolling_lr_ready(pddl_lr) and lr_vx_r2 is not None:
        if lr_vx_r2 >= PDDL_PLATFORM_LR_OVERFIT_R2 and n_samples < PDDL_PLATFORM_M5_MIN_SAMPLES:
            reason = f"overfit at n={n_samples} (v_x R²={lr_vx_r2:.3f})"
        else:
            sim_vx_r2 = _rolling_lr_vx_r2((kb or {}).get("lr_models") or {})
            reason = (
                f"pddl v_x R²={lr_vx_r2:.3f} < {PDDL_PLATFORM_LR_MIN_VX_R2}"
                f" (sim v_x R²={sim_vx_r2:.3f})"
                if sim_vx_r2 is not None
                else f"pddl v_x R²={lr_vx_r2:.3f} < {PDDL_PLATFORM_LR_MIN_VX_R2}"
            )
        _collision_inject_log(f"Platform/ground LR rejected ({reason}) — slide placeholder")
    elif m5_min_samples > 0 and n_samples < m5_min_samples:
        _collision_inject_log(
            f"M5 deferred ({n_samples}/{m5_min_samples} samples) — slide placeholder"
        )
    elif n_samples > 0:
        _collision_inject_log(f"M5 not ready ({n_samples} samples) — slide placeholder")
    else:
        _collision_inject_log("LR/M5 not available yet — slide placeholder")


def inject_domain_file(
    path: str,
    world_model: WorldModel,
    defer_ground_m5: bool = False,
    planning_pessimistic_platform: bool = False,
    planning_optimistic_slide: bool = False,
):
    """
    Inject learned collision models into the PDDL domain file.

    Ground: kb['collision'] → {SE-collision-ground-effect}
    Platform: kb['platform_collision'] → {SE-collision-platform-effect}

    Priority (aligned with forward sim): LR → M5 → slide placeholder.
    Returns inject labels {'ground': ..., 'platform': ...} for logging.
    """
    _collision_inject_log(f"inject_domain_file input={path!r}")

    ground_roll_placeholder = _build_ground_roll_placeholder()
    collision_kb = world_model.kb.get("collision") or {}
    platform_kb = world_model.kb.get("platform_collision") or {}
    inject_labels = {"ground": "skipped", "platform": "skipped"}

    with open(path, "r") as file:
        new_content = file.read()

    ground_sentinel = "{SE-collision-ground-effect}"
    if ground_sentinel in new_content:
        if defer_ground_m5:
            _collision_inject_log(
                "Deferring ground M5/LR for planning — using ground roll placeholder"
            )
            new_content = new_content.replace(ground_sentinel, ground_roll_placeholder)
            inject_labels["ground"] = "deferred"
        else:
            new_content, inject_labels["ground"] = _inject_surface_sentinel(
                new_content,
                ground_sentinel,
                collision_kb,
                placeholder_body=ground_roll_placeholder,
                m5_min_samples=0,
                m5_bounce_mode="platform",
            )

    platform_sentinel = "{SE-collision-platform-effect}"
    if platform_sentinel in new_content:
        if planning_pessimistic_platform:
            _collision_inject_log(
                "Planning pessimistic platform collision — hard stop for ENHSP "
                "(forward sim unchanged)"
            )
            new_content = new_content.replace(
                platform_sentinel, PLATFORM_COLLISION_HARD_STOP
            )
            inject_labels["platform"] = "hard_stop"
        else:
            new_content, inject_labels["platform"] = _inject_surface_sentinel(
                new_content,
                platform_sentinel,
                platform_kb,
                placeholder_body=_build_platform_slide_placeholder(),
                m5_min_samples=PDDL_PLATFORM_M5_MIN_SAMPLES,
                m5_bounce_mode="platform",
                planning_optimistic_slide=planning_optimistic_slide,
            )

    base_dir = os.path.dirname(path)
    base_name = os.path.splitext(os.path.basename(path))[0]
    output_path = os.path.join(base_dir, f"{base_name}_modified.pddl")

    with open(output_path, "w") as file:
        file.write(new_content)

    _collision_inject_log(f"Saved to {output_path}")
    return inject_labels


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
    angle_deg: float = None,
    speed_at_force=None,
    force_lr_model=None,
) -> float:
    """
    Launch speed for a PDDL force fraction (0.2–1.0).

    Mirrors base_domain pa-twang: ``v_launch = v_bird * v_bird_multiplier`` (linear in force).
    Optional ``speed_at_force(v_full, force, angle_deg)`` hook for callers
    that want angle-aware velocity prediction (e.g., BamBirds' angle-to-
    velocity polynomial in ``bambirds_shot_helper.py``). The hook may
    also be called with the legacy 2-arg signature ``(v_full, force)``
    for backward compatibility.
    """
    force = max(FORCE_MIN, min(FORCE_MAX, float(force)))
    if speed_at_force is not None:
        try:
            try:
                return float(speed_at_force(v_full, force, angle_deg))
            except TypeError:
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
    ground_kb: dict = None,
    speed_at_force=None,
    force_lr_model=None,
):
    """
    Forward-simulate flight after pa-twang using base_domain flying + placeholder ground bounce.

    Returns whether the shot path touches the playfield floor (bird center y <=
    PLAYFIELD_FLOOR_Y + GROUND_CONTACT_CENTER_SLACK) before termination, aligned with
    game ground_collision events (segments GROUND_LEVEL + epsilon).
    Checks pig-kill FIRST each step, then platforms/blocks. Platform contact uses
    circle-vs-AABB reach ``bird_radius + PLATFORM_COLLISION_EPSILON`` (matches game
    ``is_platform_collision``) so grazing ramp/stand contacts register before a
    late-arc false pig-kill. Same-step coincident pig+platform contact still wins
    via the pig-kill check at platform contact points.
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
        v, force, angle_deg=pddl_flight_angle_deg(angle_deg),
        speed_at_force=speed_at_force, force_lr_model=force_lr_model,
    )
    vx = launch_speed * cosine
    vy = launch_speed * sinus

    pig = problem_data.get(pig_key) if pig_key else None
    pr = float(pig.get("pig_radius", 3.5)) if pig else 3.5
    px = float(pig["x_pig"]) if pig else None
    py = float(pig["y_pig"]) if pig else None

    # Collect platform data (sim-only AABB margin — game hills use MBR slightly
    # larger than vision polygon bounds; ENHSP problem file stays unchanged).
    _sim_plat_pad = PLATFORM_SIM_AABB_MARGIN_PX
    platforms = []
    blocks = []
    for key, val in problem_data.items():
        if key.startswith("platform_"):
            platforms.append({
                "name": key,
                "x": float(val["x_platform"]),
                "y": float(val["y_platform"]),
                "w": float(val["platform_width"]) + 2.0 * _sim_plat_pad,
                "h": float(val["platform_height"]) + 2.0 * _sim_plat_pad,
            })
        elif key.startswith("block_"):
            # ``damage`` mirrors write_problem_file: sheltering blocks get
            # BLOCK_SHELTER_DAMAGE, everything else DEFAULT_BIRD_BLOCK_DAMAGE.
            # ``life`` and ``stability`` default to the same fallbacks used
            # elsewhere in this module so behaviour is unchanged when the
            # problem file omits them.
            _damage = (
                BLOCK_SHELTER_DAMAGE if val.get("_sheltering_block")
                else DEFAULT_BIRD_BLOCK_DAMAGE
            )
            blocks.append({
                "name": key,
                "x": float(val["x_block"]),
                "y": float(val["y_block"]),
                "w": float(val["block_width"]),
                "h": float(val["block_height"]),
                "life": float(val.get("block_life", 0.75)),
                "stability": float(val.get("block_stability", 1.0)),
                "damage": float(_damage),
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
    slide_phase_active = False
    slide_platform = None
    ground_slide_active = False
    ground_slide_stuck_steps = 0
    bounce_count = 0
    platforms_responded = set()
    trajectory = [(x, y)]  # Store trajectory for visualization
    step = 0

    for step in range(max_steps):
        if x > 800 or bounce_count >= 3:
            break

        # Continuous ground slide after first ground impulse (no LR/M5 re-fire per step).
        if ground_slide_active:
            nx = x + vx * dt
            surface_y = _ground_surface_y(br)
            ny = surface_y

            if pig is not None and not pig_killed:
                for sx, sy in ((x, y), (nx, ny)):
                    if not _bird_kills_pig(sx, sy, px, py, br, pr):
                        continue
                    if not _slide_pig_kill_credible(
                        sx, sy, vx, vy, px, py, br, pr,
                    ):
                        continue
                    pig_killed = True
                    x, y = sx, sy
                    if debug:
                        print(
                            f"[SIM DEBUG] Step {step}: PIG HIT (ground slide) "
                            f"at ({x:.1f}, {y:.1f})"
                        )
                    break
                if pig_killed:
                    break

            if abs(nx - x) < GROUND_SLIDE_STUCK_EPS:
                ground_slide_stuck_steps += 1
            else:
                ground_slide_stuck_steps = 0

            if (
                abs(vx) >= GROUND_SLIDE_MIN_SPEED
                and ground_slide_stuck_steps < GROUND_SLIDE_STUCK_MAX_STEPS
                and nx <= 800
            ):
                x, y = nx, surface_y
                vy = 0.0
                if GROUND_SLIDE_FRICTION_PER_S > 0.0:
                    vx *= max(0.0, 1.0 - GROUND_SLIDE_FRICTION_PER_S * dt)
                if step % 5 == 0:
                    trajectory.append((x, y))
                continue

            ground_slide_active = False
            if debug:
                print(
                    f"[SIM DEBUG] Step {step}: ground slide ended at x={x:.1f}, "
                    f"vx={vx:.1f}, stuck_steps={ground_slide_stuck_steps}"
                )
            bounce_count = 3
            break

        # Continuous platform slide: horizontal motion on surface (vy≈0), not ballistic arc.
        if slide_phase_active and slide_platform is not None:
            nx = x + vx * dt
            surface_y = _platform_surface_y(slide_platform, br)
            ny = surface_y

            if pig is not None and not pig_killed:
                for sx, sy in ((x, y), (nx, ny)):
                    if not _bird_kills_pig(sx, sy, px, py, br, pr):
                        continue
                    if not _slide_pig_kill_credible(
                        sx, sy, vx, vy, px, py, br, pr,
                    ):
                        if debug:
                            print(
                                f"[SIM DEBUG] Step {step}: slide pig proximity rejected "
                                f"at ({sx:.1f}, {sy:.1f}) vx={vx:.1f}"
                            )
                        continue
                    pig_killed = True
                    x, y = sx, sy
                    if debug:
                        print(f"[SIM DEBUG] Step {step}: PIG HIT (slide) at ({x:.1f}, {y:.1f})")
                    break
                if pig_killed:
                    break

            on_span = _bird_on_platform_span(nx, br, slide_platform)
            if on_span and abs(vx) >= PLATFORM_SLIDE_MIN_SPEED:
                x, y = nx, surface_y
                vy = 0.0
                if PLATFORM_SLIDE_FRICTION_PER_S > 0.0:
                    vx *= max(0.0, 1.0 - PLATFORM_SLIDE_FRICTION_PER_S * dt)
                if step % 5 == 0:
                    trajectory.append((x, y))
                continue

            slide_phase_active = False
            slide_platform = None
            x, y = nx, surface_y
            vy = 0.0
            if debug and not on_span:
                print(
                    f"[SIM DEBUG] Step {step}: left platform slide at x={x:.1f}, "
                    f"vx={vx:.1f}"
                )
            elif debug:
                print(
                    f"[SIM DEBUG] Step {step}: platform slide stopped at x={x:.1f}, "
                    f"vx={vx:.1f}"
                )

        nx = x + vx * dt
        ny = y + vy * dt
        nvy = vy - gravity * dt

        # Pig-kill BEFORE any obstacle check. When a pig sits on top of a
        # platform (very common in phy_q templates: single_force_t02_00004,
        # t02_00023, ...), reaching the pig requires the bird center to
        # descend into the platform's collision margin at almost the same
        # integration step. ENHSP's numeric semantics fire (pig_dead) in
        # that state; the sim must too, otherwise every "pig-on-hill" plan
        # is falsely rejected as a platform hit. Ordering swap makes this
        # explicit: any same-step pig contact wins over platform/block.
        if pig is not None and not pig_killed:
            for sx, sy in ((x, y), (nx, ny)):
                if not _bird_kills_pig(sx, sy, px, py, br, pr):
                    continue
                if slide_phase_active and not _slide_pig_kill_credible(
                    sx, sy, vx, vy, px, py, br, pr,
                ):
                    if debug:
                        print(
                            f"[SIM DEBUG] Step {step}: slide pig proximity rejected "
                            f"at ({sx:.1f}, {sy:.1f}) vx={vx:.1f}"
                        )
                    continue
                pig_killed = True
                x, y = sx, sy
                if debug:
                    print(f"[SIM DEBUG] Step {step}: PIG HIT at ({x:.1f}, {y:.1f})")
                break
            if pig_killed:
                break

        # Platform / block checks run only if pig-kill did NOT fire this step.
        platform_handled = False
        for plat in platforms:
            if plat["name"] in platforms_responded:
                continue
            hit, cx, cy = _sweep_platform_contact(x, y, nx, ny, br, plat)
            if not hit:
                continue

            # Same contact rule as mid-air and PDDL: dist <= bird_r + pig_r - epsilon.
            if pig is not None and not pig_killed:
                if _bird_kills_pig(cx, cy, px, py, br, pr):
                    pig_killed = True
                    x, y = cx, cy
                    platform_hit = True
                    platform_hit_name = plat["name"]
                    platform_hit_pos = (cx, cy)
                    if debug:
                        print(
                            f"[SIM DEBUG] Step {step}: COINCIDENT PIG KILL at "
                            f"platform '{plat['name']}' contact ({x:.1f}, {y:.1f})"
                        )
                    break

            if pig_killed:
                break

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
                slide_phase_active = False
                slide_platform = None
                platform_handled = True
                break

            if slide:
                platform_slide_continued = True
                slide_phase_active = True
                slide_platform = plat
                y = _platform_surface_y(plat, br)
                vy = 0.0
            else:
                slide_phase_active = False
                slide_platform = None
            platform_handled = True
            break

        if pig_killed:
            break

        # Block collision: replicate base_domain_modified.pddl events
        # ``collision_block`` (punch-through) and ``collision_block_bounce_off``
        # (hard stop). The bird punches through when its kinetic energy
        # exceeds the block's life-squared, scaled by bird_block_damage:
        #
        #     life² <= (vx² + vy²) * damage²
        #
        # Previously we always treated block overlap as a hard stop; that
        # made the Python sim reject any ENHSP plan that flies through a
        # low-life ice/wood block on the way to the pig (see
        # `[SIM/REAL] MISMATCH: sim predicted miss but level WON`
        # fingerprint on ~10 wins in run_20260822_190314.log).
        _hit_this_step = None
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
                    _hit_this_step = (blk, sx, sy)
                    break
            if _hit_this_step is not None:
                break

        if _hit_this_step is not None:
            blk, sx, sy = _hit_this_step
            life = blk["life"]
            damage = blk["damage"]
            ke = vx * vx + vy * vy
            can_punch = (life * life) <= ke * (damage * damage)
            if can_punch:
                # ``collision_block``: block breaks, bird continues with a
                # velocity penalty. The exact impulse loss isn't in the
                # domain; ``_v_scale`` of 0.85 drains ~28 % of KE, matching
                # the "loses some speed" behaviour observers report from
                # AB physics. Recompute ``nvy`` because gravity was already
                # applied to the un-damped vy above.
                _v_scale = 0.85
                vx *= _v_scale
                vy *= _v_scale
                nvy = vy - gravity * dt
                blocks = [b for b in blocks if b["name"] != blk["name"]]
                if debug:
                    print(
                        f"[SIM DEBUG] Step {step}: BLOCK PUNCH-THROUGH "
                        f"'{blk['name']}' at ({sx:.1f}, {sy:.1f}) "
                        f"life={life:.2f}² <= ke={ke:.0f} * dmg²={damage:.2f}² "
                        f"→ v *= {_v_scale}"
                    )
            else:
                # ``collision_block_bounce_off``: hard stop, matches prior
                # behaviour so plans that hit heavy stone still fail.
                block_collision = True
                block_hit_name = blk["name"]
                if debug:
                    print(
                        f"[SIM DEBUG] Step {step}: BLOCK BOUNCE-OFF '{blk['name']}' "
                        f"at ({sx:.1f}, {sy:.1f}) "
                        f"life={life:.2f}² > ke={ke:.0f} * dmg²={damage:.2f}²"
                    )
                bounce_count = 3

        if block_collision:
            break

        if platform_handled:
            if step % 5 == 0:
                trajectory.append((x, y))
            if slide_phase_active:
                continue

        x = nx
        y = ny
        vy = nvy

        if step % 5 == 0:
            trajectory.append((x, y))

        ground_contact_y = _ground_surface_y(br)
        on_ground = y <= ground_contact_y + GROUND_CONTACT_CENTER_SLACK
        if (
            on_ground
            and not ground_slide_active
            and math.hypot(vx, vy) > 5.0
            and vy < 0.5
        ):
            ground_touches += 1
            if debug:
                print(
                    f"[SIM DEBUG] Step {step}: GROUND roll contact at ({x:.1f}, {y:.1f}), "
                    f"surface_y={ground_contact_y:.1f}"
                )
            x, y, vx, vy, roll, hard_stop = _apply_ground_contact(
                ground_kb or platform_kb or {}, x, y, vx, vy, br, debug=debug
            )
            trajectory.append((x, y))
            if hard_stop:
                bounce_count = 3
                slide_phase_active = False
                slide_platform = None
                break
            if roll:
                platform_slide_continued = True
                slide_phase_active = False
                slide_platform = None
                ground_slide_active = True
                ground_slide_stuck_steps = 0
                y = ground_contact_y
                vy = 0.0
            else:
                slide_phase_active = False
                slide_platform = None
                bounce_count = 3
                break
            if step % 5 == 0:
                trajectory.append((x, y))
            continue

        if y <= ground_contact_y and vy <= 0:
            # Safety: do not integrate below the playfield when falling or stopped.
            y = max(y, ground_contact_y)
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
