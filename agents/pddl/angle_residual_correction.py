"""
Simple dial residual correction (SB6.6 slingshot).

Calibration: ε(d) = θ_meas − θ* when commanding dial d at force≈1 (θ* ≈ dial).
Observed flight θ_meas ≈ d + ε(d) within the calibrated dial band.

Live file ``angle_residual_calibration.json`` is updated over time from
[SHOT DIAG POST]; bootstrap from ``event_determinism_results.json`` if live
file is missing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Calibration shots used force=1.0; partial-force pulls use a different curve.
FULL_FORCE_MIN = 0.95

DEFAULT_LIVE_CALIBRATION_PATH = "angle_residual_calibration.json"
DEFAULT_BOOTSTRAP_PATH = "event_determinism_results.json"

MERGE_DIAL_TOL_DEG = 0.6
OBSERVE_ALPHA = 0.25
MAX_ABS_EPS_DEG = 25.0


class DialAngleResidualTable:
    """Piecewise-linear ε(dial) from calibration shots."""

    def __init__(
        self,
        dials: Sequence[float],
        epsilons: Sequence[float],
        update_counts: Optional[Sequence[int]] = None,
    ):
        pairs = sorted(
            (float(d), float(e), int(update_counts[i]) if update_counts else 0)
            for i, (d, e) in enumerate(zip(dials, epsilons))
        )
        if not pairs:
            self._dials = np.array([], dtype=float)
            self._eps = np.array([], dtype=float)
            self._n_updates = np.array([], dtype=int)
        else:
            self._dials = np.array([p[0] for p in pairs], dtype=float)
            self._eps = np.array([p[1] for p in pairs], dtype=float)
            self._n_updates = np.array([p[2] for p in pairs], dtype=int)

    @property
    def n_knots(self) -> int:
        return len(self._dials)

    @property
    def dial_min(self) -> float:
        return float(self._dials[0]) if self.n_knots else 0.0

    @property
    def dial_max(self) -> float:
        return float(self._dials[-1]) if self.n_knots else 0.0

    def epsilon_at_dial(self, dial: float) -> float:
        if self.n_knots == 0:
            return 0.0
        d = float(dial)
        if self.n_knots == 1:
            return float(self._eps[0])
        return float(np.interp(d, self._dials, self._eps))

    def predicted_flight_deg(self, command_dial: float) -> float:
        d = float(command_dial)
        return d + self.epsilon_at_dial(d)

    def calibrated_flight_range(self) -> Tuple[float, float]:
        if self.n_knots == 0:
            return 0.0, 0.0
        lo = self.predicted_flight_deg(self.dial_min)
        hi = self.predicted_flight_deg(self.dial_max)
        return (min(lo, hi), max(lo, hi))

    def dial_for_target_flight(
        self,
        target_flight_deg: float,
        min_deg: float = 5.0,
        max_deg: float = 89.0,
        tol: float = 0.08,
    ) -> Optional[Tuple[float, float]]:
        if self.n_knots == 0:
            return None
        target = float(target_flight_deg)
        f_lo, f_hi = self.calibrated_flight_range()
        if target < f_lo - tol or target > f_hi + tol:
            return None

        lo_d, hi_d = self.dial_min, self.dial_max
        for _ in range(48):
            mid = 0.5 * (lo_d + hi_d)
            if self.predicted_flight_deg(mid) < target:
                lo_d = mid
            else:
                hi_d = mid
        d_cmd = 0.5 * (lo_d + hi_d)
        d_cmd = max(self.dial_min, min(self.dial_max, d_cmd))
        d_cmd = max(min_deg, min(max_deg, d_cmd))
        return d_cmd, self.epsilon_at_dial(d_cmd)

    def observe(
        self,
        command_dial: float,
        epsilon_deg: float,
        *,
        merge_tol: float = MERGE_DIAL_TOL_DEG,
        alpha: float = OBSERVE_ALPHA,
    ) -> Tuple[str, float]:
        """
        EMA-update ε at command_dial, or add a new knot.

        Returns (action, new_epsilon) where action is 'update' or 'add'.
        """
        dial = round(float(command_dial), 2)
        eps = float(epsilon_deg)
        for i in range(self.n_knots):
            if abs(float(self._dials[i]) - dial) <= merge_tol:
                old = float(self._eps[i])
                n = int(self._n_updates[i])
                a = alpha if n > 0 else 1.0
                new = (1.0 - a) * old + a * eps
                self._eps[i] = new
                self._n_updates[i] = n + 1
                return "update", new
        self._dials = np.append(self._dials, dial)
        self._eps = np.append(self._eps, eps)
        self._n_updates = np.append(self._n_updates, 1)
        order = np.argsort(self._dials)
        self._dials = self._dials[order]
        self._eps = self._eps[order]
        self._n_updates = self._n_updates[order]
        return "add", eps

    def to_live_dict(self) -> dict:
        knots = []
        for i in range(self.n_knots):
            knots.append(
                {
                    "dial_deg": float(self._dials[i]),
                    "angle_err_deg": float(self._eps[i]),
                    "updates": int(self._n_updates[i]),
                }
            )
        return {"version": 1, "knots": knots}

    def save_live(self, path: str) -> None:
        p = Path(path)
        p.write_text(json.dumps(self.to_live_dict(), indent=2), encoding="utf-8")


def load_live_calibration_json(path: str) -> DialAngleResidualTable:
    p = Path(path)
    if not p.is_file():
        return DialAngleResidualTable([], [])
    data = json.loads(p.read_text(encoding="utf-8"))
    knots = data.get("knots") or []
    dials, eps, counts = [], [], []
    for k in knots:
        if k.get("dial_deg") is None or k.get("angle_err_deg") is None:
            continue
        dials.append(float(k["dial_deg"]))
        eps.append(float(k["angle_err_deg"]))
        counts.append(int(k.get("updates") or 0))
    return DialAngleResidualTable(dials, eps, counts)


def load_residual_table_from_event_det_json(path: str) -> DialAngleResidualTable:
    """
    Load ε per dial from ``event_determinism_results.json``.

    Uses ``angle_err_deg`` from the first shot of each angle result.
    """
    p = Path(path)
    if not p.is_file():
        return DialAngleResidualTable([], [])
    data = json.loads(p.read_text(encoding="utf-8"))
    dials: List[float] = []
    epsilons: List[float] = []
    for entry in data.get("results") or []:
        dial = entry.get("dial_deg")
        shots = entry.get("shots") or []
        if dial is None or not shots:
            continue
        err = shots[0].get("angle_err_deg")
        if err is None:
            continue
        dials.append(float(dial))
        epsilons.append(float(err))
    return DialAngleResidualTable(dials, epsilons)


def load_or_bootstrap_angle_residual_table(
    live_path: str = DEFAULT_LIVE_CALIBRATION_PATH,
    bootstrap_path: str = DEFAULT_BOOTSTRAP_PATH,
) -> Tuple[DialAngleResidualTable, str]:
    """
    Prefer live calibration file; else seed from event-determinism JSON.

    Returns (table, source_description).
    """
    live = Path(live_path)
    if live.is_file():
        table = load_live_calibration_json(live_path)
        if table.n_knots:
            return table, str(live_path)
    boot = Path(bootstrap_path)
    if boot.is_file():
        table = load_residual_table_from_event_det_json(bootstrap_path)
        if table.n_knots:
            return table, f"{bootstrap_path} (bootstrap)"
    return DialAngleResidualTable([], []), "none"
