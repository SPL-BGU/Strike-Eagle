"""
Port of BamBirds' ShotHelper.actualToLaunch / launchToActual.

BamBirds (Uni Bamberg, AIBIRDS finalist) discovered — and every
competitive AIBIRDS agent has since confirmed — that a game's slingshot
produces a launch trajectory whose angle differs from the pull angle by
an angle-dependent amount. They model it with a piecewise quadratic:

    fChangeAngle(θ) = a·θ² + b·θ + c    (radians)
    actualToLaunch(θ_desired) = θ_desired + fChangeAngle(θ_desired)
    launchToActual(θ_requested) = inverse via bisection

Two regimes with different coefficients:
  * "low"       → θ < highAngleBegin
  * "very high" → θ ≥ highAngleBegin

Coefficients below are copied verbatim from BamBirds' Settings.java
``USE_NEW_SLING_DETECTION`` branch (the more accurate of the two
calibrations they ship). They were measured against classic AngryBirds /
ScienceBirds; ScienceBirds 6.6 may differ, but this is a strong prior
that is right at high angles (where our own data already agrees within
~0.1°) and gives us a principled correction at low angles.

Reference:
  https://github.com/dwolter/BamBirds
  common/src/main/java/de/uniba/sme/bambirds/common/utils/ShotHelper.java
  common/src/main/java/de/uniba/sme/bambirds/common/utils/Settings.java
"""

from __future__ import annotations

import math
from typing import Tuple

# ── BamBirds Settings.java (USE_NEW_SLING_DETECTION=true, RedBird) ────────
_HIGH_ANGLE_BEGIN_RED_DEG = 74.476
HIGH_ANGLE_BEGIN_RAD = math.radians(_HIGH_ANGLE_BEGIN_RED_DEG)

LOW_ANGLE_CHANGE = (-0.0230, -7.871e-4, 0.0540)
HIGH_ANGLE_CHANGE_RED = (-6.8544, 19.1149, -13.2502)

LOW_ANGLE_VELOCITY = (0.0473, -0.1756, 2.8654)
HIGH_ANGLE_VELOCITY_RED = (42.1667, -124.9211, 93.8631)

# BamBirds initial scale factor (velocity multiplier). Adjusted per shot
# in ``recalculate_scaling_factor`` once real velocities are observed.
DEFAULT_SCALE_FACTOR = 1.005

# SB6.6 cold-start scene scale (px/s per BamBirds velocity unit). Online
# ``bambirds_scale`` converges to ~68–75 from clean shots; until the first
# sample the forward sim uses this prior so steep lobs aren't simulated at
# domain v_bird×force (~180) while the game launches ~200+ px/s.
DEFAULT_SCENE_SCALE_SB66 = 75.0


def _fn(w: Tuple[float, float, float], x: float) -> float:
    """BamBirds' polynomial: w[0]*x² + w[1]*x + w[2]."""
    return w[0] * x * x + w[1] * x + w[2]


def _is_very_high(theta_rad: float) -> bool:
    return theta_rad >= HIGH_ANGLE_BEGIN_RAD


def _f_change_angle(theta_rad: float) -> float:
    """Angle-correction quadratic (radians)."""
    if _is_very_high(theta_rad):
        return _fn(HIGH_ANGLE_CHANGE_RED, theta_rad)
    return _fn(LOW_ANGLE_CHANGE, theta_rad)


def actual_to_launch(theta_actual_rad: float) -> float:
    """
    Requested game pull-angle that produces ``theta_actual_rad`` as the
    resulting parabola-launch angle.

    Mirrors ShotHelper.actualToLaunch.
    """
    return theta_actual_rad + _f_change_angle(theta_actual_rad)


def launch_to_actual(theta_launch_rad: float) -> float:
    """
    Inverse of ``actual_to_launch``: what the game will actually launch
    at when we pull to ``theta_launch_rad``.

    Mirrors ShotHelper.launchToActual (bisection on a monotone map).
    """
    upper = math.radians(86.0)
    lower = math.radians(-25.0)
    while (upper - lower) > 1e-4:
        mid = lower + (upper - lower) / 2.0
        err_up = abs(theta_launch_rad - actual_to_launch(mid + 1e-5))
        err_dn = abs(theta_launch_rad - actual_to_launch(mid - 1e-5))
        if err_up > err_dn:
            upper = mid
        else:
            lower = mid
    return upper


def actual_to_launch_deg(theta_actual_deg: float) -> float:
    return math.degrees(actual_to_launch(math.radians(theta_actual_deg)))


def launch_to_actual_deg(theta_launch_deg: float) -> float:
    return math.degrees(launch_to_actual(math.radians(theta_launch_deg)))


def _f_launch_velocity(theta_rad: float) -> float:
    """Velocity at requested launch angle (BamBirds units, before scale).

    SB6.6 note: the two-piece BamBirds polynomial was calibrated on classic
    AngryBirds. Its ``HIGH_ANGLE_VELOCITY_RED`` branch returns absurdly
    small values in [~1.5, ~2.8] units for θ ∈ [74.5°, 89°] — multiplied
    by our learned ``bambirds_scale ≈ 68`` this predicts |v| ≈ 100–140
    while the game actually launches |v| ≈ 170–195 (see run
    ``run_20260822_203932.log``, `[BAMBIRDS SCALE]` outlier lines at
    θ ∈ {81°, 81.5°, 82°}, all with obs_scale ≈ 114–120 and rejected).
    That mismatch made the Python sim reject perfectly good ENHSP plans
    for high-lob shots (advisory: pig=False, block=False, ground=True on
    every attempt of L6, L7, L9, L13). Falling back to the LOW_ANGLE
    polynomial across the whole executable range gives ~2.7-3.0 unit
    outputs — matching what SB6.6's near-constant-velocity sling actually
    produces and restoring sim/ENHSP agreement.
    """
    return _fn(LOW_ANGLE_VELOCITY, theta_rad)


def angle_to_velocity(theta_rad: float, scale_factor: float = DEFAULT_SCALE_FACTOR) -> float:
    """Predicted launch speed (BamBirds units). Multiply by scene scale for px/s."""
    return scale_factor * _f_launch_velocity(theta_rad)


def recalculate_scaling_factor(
    v_observed: float,
    theta_launch_rad: float,
    current_scale: float = DEFAULT_SCALE_FACTOR,
) -> float:
    """
    Online scale-factor refit — BamBirds' ``recalculateScalingFactor``.

    Skips extreme angles (bad SNR) and outlier ratios. Blends slow-angle
    updates with prior for stability.
    """
    if theta_launch_rad < math.radians(5.0) or theta_launch_rad > math.radians(83.0):
        return current_scale
    predicted = _f_launch_velocity(theta_launch_rad)
    if predicted <= 0.0:
        return current_scale
    ratio = v_observed / predicted
    # Guard against NaN and >10 % jumps (matches BamBirds).
    if ratio != ratio or ratio > 1.1 or ratio < 0.9:
        return current_scale
    if math.radians(50.0) < theta_launch_rad < HIGH_ANGLE_BEGIN_RAD:
        # Mid-angle sweet spot — trust the measurement fully.
        return ratio
    # Otherwise blend with prior (BamBirds 60/40 mix).
    return ratio * 0.6 + current_scale * 0.4
