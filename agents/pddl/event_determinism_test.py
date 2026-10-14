"""
Event determinism test for Science Birds.

For each of 20 dial angles where the BamBirds pull→flight model implies
large angle deviation, fire the same shot on the same level 5 times and
compare detected segment events (type + frame index) and WON/LOST.

Controlled by PDDLAgent ``event_determinism_mode`` (default on; CLI: ``--no-event-determinism`` to disable).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from agents.pddl.pddl_files.bambirds_shot_helper import (
    actual_to_launch_deg,
    launch_to_actual_deg,
)

DEFAULT_REPEATS_PER_ANGLE = 5
DEFAULT_ANGLE_COUNT = 20


def bambirds_high_deviation_dials(
    count: int = DEFAULT_ANGLE_COUNT,
    min_deg: float = 10.0,
    max_deg: float = 85.0,
    step: float = 0.25,
    min_spacing_deg: float = 2.0,
) -> List[float]:
    """
    Pick dial angles where BamBirds' mapping diverges most from dial ≈ flight.

    Score = max(|launch_to_actual(dial) − dial|, |actual_to_launch(flight) − dial|)
    with flight = launch_to_actual(dial).
    """
    scored: List[Tuple[float, float]] = []
    d = min_deg
    while d <= max_deg + 1e-9:
        flight = launch_to_actual_deg(d)
        corr = abs(flight - d)
        pull_err = abs(actual_to_launch_deg(flight) - d)
        scored.append((max(corr, pull_err), d))
        d += step
    scored.sort(key=lambda x: (-x[0], x[1]))
    chosen: List[float] = []
    for score, dial in scored:
        if any(abs(dial - c) < min_spacing_deg for c in chosen):
            continue
        chosen.append(dial)
        if len(chosen) >= count:
            break
    chosen.sort()
    return [round(x, 1) for x in chosen]


def event_fingerprint(
    event_indexes_by_event: dict,
    game_won: bool,
) -> Tuple[bool, Tuple[Tuple[str, int], ...]]:
    """Canonical (won, sorted list of (event_type, frame)) for comparison."""
    items: List[Tuple[str, int]] = []
    for etype in sorted(event_indexes_by_event.keys()):
        for idx in event_indexes_by_event[etype]:
            items.append((str(etype), int(idx)))
    return (bool(game_won), tuple(items))


@dataclass
class EventDeterminismShot:
    repeat_index: int
    dial_deg: float
    bambirds_flight_deg: float
    game_won: bool
    fingerprint: Tuple[bool, Tuple[Tuple[str, int], ...]]
    angle_err_deg: Optional[float] = None
    event_indexes_by_event: Dict[str, List[int]] = field(default_factory=dict)


@dataclass
class EventDeterminismAngleResult:
    dial_deg: float
    bambirds_flight_deg: float
    shots: List[EventDeterminismShot] = field(default_factory=list)
    deterministic: bool = False
    note: str = ""


class EventDeterminismRunner:
    """State machine: 20 angles × 5 repeats on one level."""

    def __init__(
        self,
        angles: Optional[Sequence[float]] = None,
        repeats: int = DEFAULT_REPEATS_PER_ANGLE,
        level_path: str = "",
    ):
        self.angles = list(angles or bambirds_high_deviation_dials())
        self.repeats = max(1, int(repeats))
        self.level_path = level_path
        self.angle_index = 0
        self.repeat_index = 0
        self.angle_results: List[EventDeterminismAngleResult] = []
        self._current_group: List[EventDeterminismShot] = []

    @property
    def complete(self) -> bool:
        return self.angle_index >= len(self.angles)

    def current_dial_angle(self) -> float:
        if self.complete:
            return float(self.angles[-1])
        return float(self.angles[self.angle_index])

    def current_bambirds_flight_deg(self) -> float:
        return launch_to_actual_deg(self.current_dial_angle())

    def record_shot(
        self,
        event_indexes_by_event: dict,
        game_won: bool,
        angle_err_deg: Optional[float] = None,
    ) -> None:
        dial = self.current_dial_angle()
        fp = event_fingerprint(event_indexes_by_event, game_won)
        shot = EventDeterminismShot(
            repeat_index=self.repeat_index,
            dial_deg=dial,
            bambirds_flight_deg=launch_to_actual_deg(dial),
            game_won=game_won,
            fingerprint=fp,
            angle_err_deg=angle_err_deg,
            event_indexes_by_event={
                k: list(v) for k, v in event_indexes_by_event.items()
            },
        )
        self._current_group.append(shot)
        self._log_shot(shot)

    def _log_shot(self, shot: EventDeterminismShot) -> None:
        n_events = len(shot.fingerprint[1])
        ae = (
            f" angle_err={shot.angle_err_deg:+.2f}°"
            if shot.angle_err_deg is not None
            else ""
        )
        print(
            f"[EVENT-DET] angle {self.angle_index + 1}/{len(self.angles)} "
            f"repeat {self.repeat_index + 1}/{self.repeats} | "
            f"dial={shot.dial_deg:.1f}° BamBirds flight≈{shot.bambirds_flight_deg:.1f}° | "
            f"{'WIN' if shot.game_won else 'LOSS'} | events={n_events}{ae}"
        )
        print(f"[EVENT-DET]   fingerprint={shot.fingerprint[1]}")

    def advance_after_level_finished(self) -> Tuple[bool, bool]:
        """
        Call once per finished level (single bird).

        Returns (should_retry_same_level, protocol_complete).
        """
        self.repeat_index += 1
        if self.repeat_index < self.repeats:
            return True, False

        result = self._finalize_angle_group()
        self.angle_results.append(result)
        self._current_group = []
        self.repeat_index = 0
        self.angle_index += 1

        if self.angle_index >= len(self.angles):
            self.print_final_report()
            return False, True
        return True, False

    def _finalize_angle_group(self) -> EventDeterminismAngleResult:
        dial = self.angles[self.angle_index]
        flight = launch_to_actual_deg(dial)
        group = list(self._current_group)
        fps = [s.fingerprint for s in group]
        deterministic = len(set(fps)) == 1
        note = "all 5 runs identical (events + outcome)" if deterministic else self._diff_note(group)
        tag = "DETERMINISTIC" if deterministic else "NON-DETERMINISTIC"
        print(
            f"\n[EVENT-DET] === {tag} dial={dial:.1f}° "
            f"(BamBirds flight≈{flight:.1f}°) ===\n[EVENT-DET] {note}\n"
        )
        return EventDeterminismAngleResult(
            dial_deg=dial,
            bambirds_flight_deg=flight,
            shots=group,
            deterministic=deterministic,
            note=note,
        )

    @staticmethod
    def _diff_note(shots: List[EventDeterminismShot]) -> str:
        if len(shots) < 2:
            return "insufficient shots"
        ref = shots[0].fingerprint
        lines = []
        for s in shots[1:]:
            if s.fingerprint == ref:
                continue
            lines.append(
                f"repeat {s.repeat_index + 1} differs: outcome "
                f"{'WIN' if s.fingerprint[0] else 'LOSS'} vs "
                f"{'WIN' if ref[0] else 'LOSS'}; "
                f"events {s.fingerprint[1]} vs {ref[1]}"
            )
        return "; ".join(lines) if lines else "fingerprints differ"

    def print_angle_catalog(self) -> None:
        print("\n[EVENT-DET] BamBirds high-deviation dial angles (20):")
        for i, d in enumerate(self.angles, 1):
            f = launch_to_actual_deg(d)
            print(
                f"  {i:2d}. dial={d:5.1f}° → BamBirds flight≈{f:5.1f}° "
                f"(Δ={f - d:+.1f}°)"
            )
        print(
            f"[EVENT-DET] Plan: {len(self.angles)} angles × {self.repeats} repeats "
            f"= {len(self.angles) * self.repeats} shots on one level.\n"
        )

    def print_final_report(self) -> None:
        n_det = sum(1 for r in self.angle_results if r.deterministic)
        n = len(self.angle_results)
        overall = n_det == n and n > 0
        print("\n" + "=" * 72)
        print("EVENT DETERMINISM TEST — FINAL REPORT")
        print("=" * 72)
        print(f"Level: {self.level_path or '(unknown)'}")
        print(f"Angles tested: {n}/{len(self.angles)}")
        print(f"Deterministic angles: {n_det}/{n}")
        print(
            f"Overall simulation events: "
            f"{'DETERMINISTIC' if overall else 'NON-DETERMINISTIC'}"
        )
        print("-" * 72)
        for r in self.angle_results:
            tag = "OK" if r.deterministic else "FAIL"
            print(
                f"  [{tag}] dial={r.dial_deg:.1f}° flight≈{r.bambirds_flight_deg:.1f}° — {r.note}"
            )
        print("=" * 72 + "\n")

    def save_json(self, path: str) -> None:
        payload: Dict[str, Any] = {
            "level_path": self.level_path,
            "repeats_per_angle": self.repeats,
            "angles": self.angles,
            "results": [],
        }
        for r in self.angle_results:
            payload["results"].append(
                {
                    "dial_deg": r.dial_deg,
                    "bambirds_flight_deg": r.bambirds_flight_deg,
                    "deterministic": r.deterministic,
                    "note": r.note,
                    "shots": [
                        {
                            "repeat_index": s.repeat_index,
                            "game_won": s.game_won,
                            "fingerprint": list(s.fingerprint[1]),
                            "angle_err_deg": s.angle_err_deg,
                            "events": s.event_indexes_by_event,
                        }
                        for s in r.shots
                    ],
                }
            )
        n_det = sum(1 for r in self.angle_results if r.deterministic)
        payload["summary"] = {
            "deterministic_angles": n_det,
            "total_angles": len(self.angle_results),
            "overall_deterministic": n_det == len(self.angle_results)
            and len(self.angle_results) > 0,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[EVENT-DET] Wrote {path}")
