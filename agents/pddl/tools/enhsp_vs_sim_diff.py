"""
ENHSP vs. Python-sim divergence analyzer.

Given a `problem.pddl` and `solution.pddl` produced by an ENHSP run (default:
the live files in agents/pddl/pddl_files/), this script:

  1. Parses the problem's numeric init into a ``problem_data`` dict.
  2. Recovers the planner's chosen (dial, force) from ``solution.pddl``.
  3. Computes ENHSP's OWN ballistic prediction (analytical closed-form from
     the ``flying`` process in ``base_domain.pddl``): flight-angle after bias,
     launch offset from ``pa-twang``, ``v_bird * v_bird_multiplier`` speed,
     constant gravity. Reports the min bird-pig center distance along the arc
     and whether it falls inside ``bird_radius + pig_radius - PIG_HIT_EPSILON``.
  4. Runs ``simulate_pddl_shot_plan`` on the same inputs and reports what it
     says about pig kill, platform / block collisions, and ground touches.
  5. Prints a side-by-side "who says what" table plus a plain-English verdict
     for where the divergence lives (launch origin / collision predicate /
     platform stop / block obstruction).

Run with the project venv:

    venv\\Scripts\\python.exe -m agents.pddl.tools.enhsp_vs_sim_diff

or with explicit files:

    venv\\Scripts\\python.exe -m agents.pddl.tools.enhsp_vs_sim_diff \\
        --problem agents/pddl/pddl_files/problem.pddl \\
        --solution agents/pddl/pddl_files/solution.pddl
"""
from __future__ import annotations

import argparse
import io
import math
import os
import re
import sys
from pathlib import Path

# Windows console defaults to cp1252 which chokes on ° and → in our output.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

# Skip the heavy agents/__init__.py (which pulls in the whole game-client
# chain: computer_vision, sciencebirds, ...) by pre-registering empty package
# shells so pddl_parser's ``from agents.pddl.pddl_files.world_model.params
# import Params`` resolves without executing the real __init__.py files.
import types
for pkg_name, pkg_dir in [
    ("agents", REPO_ROOT / "agents"),
    ("agents.pddl", REPO_ROOT / "agents" / "pddl"),
    ("agents.pddl.pddl_files", REPO_ROOT / "agents" / "pddl" / "pddl_files"),
]:
    mod = types.ModuleType(pkg_name)
    mod.__path__ = [str(pkg_dir)]
    sys.modules[pkg_name] = mod

from agents.pddl.pddl_files import pddl_parser  # noqa: E402

ANGLE_BIAS_DEGREES = pddl_parser.ANGLE_BIAS_DEGREES
FORCE_MAX = pddl_parser.FORCE_MAX
FORCE_RATE = pddl_parser.FORCE_RATE
PIG_HIT_EPSILON = pddl_parser.PIG_HIT_EPSILON
simulate_pddl_shot_plan = pddl_parser.simulate_pddl_shot_plan
pddl_bird_position_after_pa_twang = pddl_parser.pddl_bird_position_after_pa_twang


# --------------------------------------------------------------------------- #
# Parsers                                                                     #
# --------------------------------------------------------------------------- #

_FLUENT_1ARG = re.compile(
    r"\(\s*=\s*\(\s*([a-zA-Z_][\w-]*)\s+([a-zA-Z_][\w-]*)\s*\)\s*([-+0-9.eE]+)\s*\)"
)
_FLUENT_0ARG = re.compile(
    r"\(\s*=\s*\(\s*([a-zA-Z_][\w-]*)\s*\)\s*([-+0-9.eE]+)\s*\)"
)
_OBJ_LINE = re.compile(r"\b([a-zA-Z_][\w]*)\s*-\s*(bird|pig|block|platform|external_agent)")


def parse_problem_pddl(text: str):
    """Return (problem_data, globals) where problem_data mirrors the agent's dict."""
    objects: dict[str, str] = {}  # name -> type
    for m in _OBJ_LINE.finditer(text):
        objects[m.group(1)] = m.group(2)

    problem_data: dict[str, dict] = {name: {} for name in objects}
    globals_dict: dict[str, float] = {}

    for m in _FLUENT_1ARG.finditer(text):
        fluent, obj, val = m.group(1), m.group(2), float(m.group(3))
        if obj in problem_data:
            problem_data[obj][fluent] = val
        else:
            globals_dict.setdefault(f"{fluent}:{obj}", val)

    for m in _FLUENT_0ARG.finditer(text):
        fluent, val = m.group(1), float(m.group(2))
        globals_dict[fluent] = val

    return problem_data, globals_dict


def parse_solution(text: str, globals_dict: dict[str, float]):
    """Return (dial_deg, force, plan_lines)."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    plan_lines = [l for l in lines if "pa-twang" in l or "set_force" in l]
    max_angle = float(globals_dict.get("max_angle", 85.5))
    angle_rate = float(globals_dict.get("angle_rate", 0.5))

    set_force_t = None
    twang_t = None
    for l in plan_lines:
        m = re.match(r"([-+0-9.eE]+)\s*:\s*\((\S+)", l)
        if not m:
            continue
        t = float(m.group(1))
        op = m.group(2).lower()
        if op == "set_force":
            set_force_t = t
        elif op == "pa-twang":
            twang_t = t

    if twang_t is None:
        raise ValueError("No pa-twang action in solution.pddl")

    dial = max_angle - twang_t * angle_rate
    force = FORCE_MAX - (set_force_t or 0.0) * FORCE_RATE
    force = max(0.2, min(1.0, force))
    return dial, force, plan_lines


# --------------------------------------------------------------------------- #
# ENHSP's own closed-form ballistic prediction                                #
# --------------------------------------------------------------------------- #

def enhsp_ballistic_trace(problem_data, dial_deg, force, dt=0.01, t_max=6.0):
    """
    Replay what the PDDL ``flying`` process integrates, in closed form.
    Uses the same launch-offset from ``pa-twang`` and constant gravity —
    no collisions, just the raw parabola ENHSP reasons about.
    """
    bird = next(v for k, v in problem_data.items() if k.startswith("bird_"))
    pig = next((v for k, v in problem_data.items() if k.startswith("pig_")), None)

    ref_x, ref_y = float(bird["x_bird"]), float(bird["y_bird"])
    v_bird = float(bird.get("v_bird", 180.0))
    br = float(bird.get("bird_radius", 4.0))

    flight_deg = dial_deg - ANGLE_BIAS_DEGREES
    launch_x, launch_y = pddl_bird_position_after_pa_twang(ref_x, ref_y, dial_deg)
    speed = v_bird * float(force)
    vx = speed * math.cos(math.radians(flight_deg))
    vy = speed * math.sin(math.radians(flight_deg))

    g = 86.01  # planner uses gravity fluent — we read it below if present
    for k, v in problem_data.items():
        if isinstance(v, dict) and "gravity" in v:
            g = float(v["gravity"])

    trace = []
    steps = int(t_max / dt)
    min_dist = float("inf")
    min_state = None
    for i in range(steps + 1):
        t = i * dt
        x = launch_x + vx * t
        y = launch_y + vy * t - 0.5 * g * t * t
        trace.append((t, x, y))
        if pig is not None:
            px = float(pig["x_pig"]); py = float(pig["y_pig"])
            d = math.hypot(x - px, y - py)
            if d < min_dist:
                min_dist = d
                min_state = (t, x, y)
        if y < 360.0:  # ground
            break

    pr = float(pig.get("pig_radius", 3.5)) if pig else 0.0
    contact_radius = max(0.0, br + pr - PIG_HIT_EPSILON)
    ballistic_kill = pig is not None and contact_radius > 0 and min_dist <= contact_radius
    return {
        "flight_deg": flight_deg,
        "launch": (launch_x, launch_y),
        "launch_vel": (vx, vy),
        "speed": speed,
        "gravity": g,
        "trace": trace,
        "min_dist_to_pig": min_dist,
        "min_dist_state": min_state,
        "contact_radius": contact_radius,
        "ballistic_kill": ballistic_kill,
    }


# --------------------------------------------------------------------------- #
# Collision-path scanner (walks ENHSP's ballistic arc, flags first obstacle)  #
# --------------------------------------------------------------------------- #

def scan_obstacles(problem_data, trace, br):
    """
    Walk the ENHSP ballistic arc and report the FIRST object (block, platform,
    ground) the bird center passes into, using AABB overlap with br margin.
    """
    obstacles = []
    for name, obj in problem_data.items():
        if name.startswith("block_"):
            cx = float(obj["x_block"]); cy = float(obj["y_block"])
            w = float(obj["block_width"]); h = float(obj["block_height"])
            obstacles.append(("block", name, cx - w/2, cx + w/2, cy - h/2, cy + h/2))
        elif name.startswith("platform_"):
            cx = float(obj["x_platform"]); cy = float(obj["y_platform"])
            w = float(obj["platform_width"]); h = float(obj["platform_height"])
            obstacles.append(("platform", name, cx - w/2, cx + w/2, cy - h/2, cy + h/2))

    for (t, x, y) in trace:
        if y - br <= 360.0:
            return ("ground", "-", t, x, y)
        for kind, name, xl, xr, yb, yt in obstacles:
            if (xl - br) <= x <= (xr + br) and (yb - br) <= y <= (yt + br):
                return (kind, name, t, x, y)
    return None


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument(
        "--problem",
        default=str(REPO_ROOT / "agents" / "pddl" / "pddl_files" / "problem.pddl"),
    )
    ap.add_argument(
        "--solution",
        default=str(REPO_ROOT / "agents" / "pddl" / "pddl_files" / "solution.pddl"),
    )
    ap.add_argument("--dial", type=float, default=None, help="Override dial degrees")
    ap.add_argument("--force", type=float, default=None, help="Override force [0.2, 1.0]")
    args = ap.parse_args()

    problem_text = Path(args.problem).read_text(encoding="utf-8")
    problem_data, globals_dict = parse_problem_pddl(problem_text)

    if args.dial is not None:
        dial = args.dial
        force = args.force if args.force is not None else 1.0
        plan_lines = ["(user-override)"]
    else:
        solution_text = Path(args.solution).read_text(encoding="utf-8")
        dial, force, plan_lines = parse_solution(solution_text, globals_dict)

    print("=" * 78)
    print("ENHSP vs Python-sim divergence analysis")
    print("=" * 78)
    print(f"Problem : {args.problem}")
    print(f"Solution: {args.solution}")
    print(f"Plan    : {' | '.join(plan_lines)}")
    print(f"Recovered dial={dial:.2f}°  force={force:.3f}  "
          f"(bias={ANGLE_BIAS_DEGREES}° → flight={dial - ANGLE_BIAS_DEGREES:.2f}°)")

    bird = next(v for k, v in problem_data.items() if k.startswith("bird_"))
    pig = next((v for k, v in problem_data.items() if k.startswith("pig_")), None)
    print(f"\nBird init  : x={bird['x_bird']:.1f}  y={bird['y_bird']:.1f}  "
          f"v_bird={bird.get('v_bird', 180.0):.2f}  r={bird.get('bird_radius', 4.0):.1f}")
    if pig:
        print(f"Pig        : x={pig['x_pig']:.1f}  y={pig['y_pig']:.1f}  "
              f"r={pig.get('pig_radius', 3.5):.1f}")
    for name, obj in problem_data.items():
        if name.startswith("block_"):
            print(f"Block {name}: c=({obj['x_block']:.1f},{obj['y_block']:.1f}) "
                  f"size {obj['block_width']:.1f}x{obj['block_height']:.1f}  "
                  f"life={obj.get('block_life', '?')}")
        elif name.startswith("platform_"):
            print(f"Platform {name}: c=({obj['x_platform']:.1f},{obj['y_platform']:.1f}) "
                  f"size {obj['platform_width']:.1f}x{obj['platform_height']:.1f}")

    # ---- ENHSP's own closed-form ballistic ---- #
    print("\n" + "-" * 78)
    print("[A] ENHSP's own ballistic (no collisions, closed-form from base_domain flying)")
    print("-" * 78)
    ball = enhsp_ballistic_trace(problem_data, dial, force)
    lx, ly = ball["launch"]
    vx, vy = ball["launch_vel"]
    print(f"Launch offset from pa-twang: bird_init ({bird['x_bird']:.1f},{bird['y_bird']:.1f}) "
          f"→ launch ({lx:.2f}, {ly:.2f})")
    print(f"Launch velocity            : |v|={ball['speed']:.2f}  "
          f"vx={vx:.2f}  vy={vy:.2f}  gravity={ball['gravity']:.2f}")
    if ball["min_dist_state"]:
        t, x, y = ball["min_dist_state"]
        print(f"Closest approach to pig    : t={t:.3f}s  bird=({x:.2f},{y:.2f})  "
              f"d={ball['min_dist_to_pig']:.2f} (contact={ball['contact_radius']:.2f})")
    print(f"Ballistic pig-kill?        : "
          f"{'YES' if ball['ballistic_kill'] else 'NO'}"
          f"  (min_dist {ball['min_dist_to_pig']:.2f} "
          f"{'<=' if ball['ballistic_kill'] else '>'} {ball['contact_radius']:.2f})")

    br = float(bird.get("bird_radius", 4.0))
    first_hit = scan_obstacles(problem_data, ball["trace"], br)
    if first_hit:
        kind, name, t, x, y = first_hit
        print(f"First obstacle on arc      : {kind} {name}  @ t={t:.3f}s  ({x:.2f},{y:.2f})")
    else:
        print(f"First obstacle on arc      : none within {ball['trace'][-1][0]:.2f}s")

    # ---- Python forward sim ---- #
    print("\n" + "-" * 78)
    print("[B] Python simulate_pddl_shot_plan (same inputs)")
    print("-" * 78)
    gravity = ball["gravity"]
    sim = simulate_pddl_shot_plan(
        problem_data, dial, gravity=gravity, force=force, dt=0.01, max_steps=10000,
        debug=True,
    )
    print(f"Sim result: pig_killed={sim.get('pig_killed_in_sim')}  "
          f"platform_collision={sim.get('platform_collision')}  "
          f"block_collision={sim.get('block_collision')}  "
          f"ground_touches={sim.get('ground_touches')}  "
          f"uses_ground_collision={sim.get('uses_ground_collision')}")
    if sim.get("platform_hit_point"):
        print(f"  Sim platform impact: {sim['platform_hit_point']}")

    # ---- Verdict ---- #
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    sim_kill = bool(sim.get("pig_killed_in_sim"))
    ballistic_pig_t = ball["min_dist_state"][0] if ball["min_dist_state"] else None

    coincident = False
    if first_hit and ballistic_pig_t is not None:
        # Same-instant contact (within 0.1 s) between an obstacle and the pig.
        # ENHSP's numeric semantics may fire (pig_dead) at that instant too, but
        # the sim's obstacle check runs first and terminates before pig-kill.
        coincident = abs(first_hit[2] - ballistic_pig_t) < 0.10

    path_clear = ball["ballistic_kill"] and (
        first_hit is None or (
            ballistic_pig_t is not None and first_hit[2] > ballistic_pig_t + 0.10
        )
    )
    enhsp_kill = path_clear or coincident

    print(f"ENHSP would say pig dies?  : ballistic={ball['ballistic_kill']}  "
          f"path-clear={'yes' if path_clear else 'no'}"
          + (f"  coincident-with-{first_hit[0]}"
             if coincident and not path_clear else ""))
    print(f"Sim says pig dies?         : {sim_kill}")

    if coincident and not path_clear and not sim_kill:
        kind = first_hit[0]
        print("\n>>> DIVERGENCE: COINCIDENT-CONTACT.")
        print(f"    ENHSP's ballistic hits the pig (min_dist "
              f"{ball['min_dist_to_pig']:.2f} <= {ball['contact_radius']:.2f}) at")
        print(f"    t={ballistic_pig_t:.3f}s — the SAME instant it touches "
              f"{kind} '{first_hit[1]}'.")
        print("    ENHSP's numeric-events semantics let (pig_dead) fire in that")
        print(f"    state, but the sim's {kind} check runs first in the step loop")
        print("    (with a 1.1x bird_radius margin) and terminates before pig-kill")
        print("    is registered.")
        print("    Fix candidates:")
        print("      * In simulate_pddl_shot_plan: check pig-kill BEFORE platform/block")
        print("        at each step, so a same-step pig contact wins.")
        print(f"      * Or reduce the {kind} detection margin from 1.1*br to 1.0*br,")
        print("        matching the domain's collision predicate exactly.")
        print("      * In the domain: give collision_pig_kill event higher priority")
        print("        (e.g. tighter numeric precondition) than collision_platform.")
        return

    if enhsp_kill and not sim_kill:
        print("\n>>> DIVERGENCE: ENHSP thinks pig dies, sim disagrees.")
        if sim.get("platform_collision") and first_hit and first_hit[0] != "platform":
            print("    Cause: SIM's platform collision fires but ENHSP's ballistic arc")
            print("    passes over the platform. Check platform_collision event")
            print("    y-margin in base_domain vs sim (bird radius handling / hill "
                  "top-edge extension in extend_platforms_in_problem_data).")
        elif sim.get("block_collision") and first_hit and first_hit[0] != "block":
            print("    Cause: SIM hits a block ENHSP's ballistic doesn't. Check block")
            print("    AABB conservativeness (apply_block_shelter_adjustments) or")
            print("    swept-segment margin (1.1 * bird_radius) in sim.")
        elif sim.get("uses_ground_collision"):
            print("    Cause: SIM bounces off ground short of pig; ENHSP's collision_ground")
            print("    numeric M5 in base_domain_modified.pddl may model a different bounce.")
        else:
            print("    Cause: Contact radius or launch origin differs. Compare ")
            print("    pddl_bird_position_after_pa_twang() output vs domain's ")
            print("    pa-twang effect (dec/inc x_bird by 16*cosine, y_bird by 12*sinus).")
    elif not enhsp_kill and sim_kill:
        print("\n>>> DIVERGENCE (inverted): sim optimistic, ENHSP pessimistic.")
        print("    ENHSP's ballistic misses (bad plan) but sim's collision model kills anyway.")
    elif enhsp_kill and sim_kill:
        print("\n>>> AGREEMENT: both say pig dies. If the ADVISORY still fired, look")
        print("    at whether ENHSP's actual satisficing plan used a MULTI-hit path")
        print("    (e.g. bounce off block) that neither this ballistic check nor")
        print("    the sim replicates identically. Inspect the full solution.pddl.")
    else:
        print("\n>>> AGREEMENT: both say pig survives. Plan is genuinely bad —")
        print("    look at why ENHSP's satisficing search accepted it (heuristic")
        print("    picked ricochet / TNT / block-collapse path? check base_domain")
        print("    for collision_block_bounce_off and explode_block preconditions).")


if __name__ == "__main__":
    main()
