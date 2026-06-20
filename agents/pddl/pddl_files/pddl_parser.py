from string import Template

import numpy as np
import os

from agents.pddl.pddl_files.world_model.params import Params
from agents.pddl.pddl_files.world_model.world_model import WorldModel
from agents.pddl.pddl_files.events.learn_events import m5_collision_leaves_for_pddl

# Print [COLLISION-INJECT] lines during domain injection.
COLLISION_INJECT_DEBUG = True


def _collision_inject_log(msg: str) -> None:
    if COLLISION_INJECT_DEBUG:
        print(f"[COLLISION-INJECT] {msg}")


_COLLISION_VAR_ORDER = ("y", "v_x", "v_y")
_COLLISION_PDDL_FLUENT = {"y": "y_bird", "v_x": "vx_bird", "v_y": "vy_bird"}
_PDDL_AFFINE_VARS = ("x_bird", "y_bird", "vx_bird", "vy_bird")


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
    """
    Build M5 piecewise collision effect for PDDL injection.
    
    Returns effect body string, or None if M5 models not available yet.
    """
    per_var_leaves = {}
    for vn in _COLLISION_VAR_ORDER:
        vd = collision_vars.get(vn, {})
        mc = vd.get("model_comparison")
        m5 = mc.get("m5_model") if isinstance(mc, dict) else None
        
        leaves = m5_collision_leaves_for_pddl(m5, debug=COLLISION_INJECT_DEBUG, var_label=vn)
        if not leaves:
            _collision_inject_log(f"M5 not ready for '{vn}'")
            return None
        
        _collision_inject_log(f"M5 '{vn}': {len(leaves)} leaf(s)")
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
    
    lines.append("(assign (bounce_count ?b) (+ (bounce_count ?b) 1))")
    body = "\n            ".join(lines)
    _collision_inject_log(f"M5 injection: {len(lines)} statements")
    return body

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


# Angle bias correction (in degrees) - compensates for slingshot mechanics
# Positive bias means the actual shot goes less steep than commanded
ANGLE_BIAS_DEGREES = 0  # Based on empirical measurements
DEG_TO_RAD = 0.01745329252


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
                  min_angle: float = -4, max_angle: float = 85):
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
        f"(= (ground_x_damper) 0.5)"
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
                       min_angle: float = -4, max_angle: float = 85):
    objects, initial_state, goals = generate_pddl(problem_data, init_angle, angel_rate, world_model, min_angle, max_angle)
    problem = problem_template.substitute({"objects": objects, "initial": initial_state, "goal": goals})
    with open(path, 'w') as file:
        file.write(problem)


def inject_domain_file(path: str, world_model: WorldModel):
    """
    Inject learned M5 collision models into the PDDL domain file.
    
    Always uses M5 piecewise (when/assign) for collision effects.
    """
    _collision_inject_log(f"inject_domain_file input={path!r}")

    with open(path, "r") as file:
        new_content = file.read()

    collision_vars = world_model.kb["collision"]["variables"]
    ground_sentinel = "{SE-collision-ground-effect}"
    
    if ground_sentinel in new_content:
        m5_body = _build_collision_ground_effect_m5(collision_vars)
        if m5_body is not None:
            new_content = new_content.replace(ground_sentinel, m5_body)
            _collision_inject_log("Injected M5 piecewise collision effect")
        else:
            _collision_inject_log("M5 not available yet, using placeholder 0.0")
            new_content = new_content.replace(ground_sentinel, 
                "(assign (y_bird ?b) 0.0)\n            "
                "(assign (vy_bird ?b) 0.0)\n            "
                "(assign (vx_bird ?b) 0.0)\n            "
                "(assign (bounce_count ?b) (+ (bounce_count ?b) 1))")

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


def action_filter(line):
    return 'pa-twang' in line


def parse_action(line, init_angle, angel_rate):
    """init_angle = starting slingshot angle (max_angle); angle decreases by angel_rate per plan time unit."""
    n = float(line.split(':')[0])
    return 'shoot', init_angle - n * angel_rate


def parse_solution_to_actions(solution_path: str, init_angle, angel_rate):
    with open(solution_path) as solution_file:
        lines = solution_file.readlines()
        actions = list(filter(action_filter, lines))
        actions = list(map(lambda l: parse_action(l, init_angle, angel_rate), actions))
        return actions


# Replan bird ref when planned angle differs from the guess used for pa-twang inverse.
ANGLE_REPLAN_THRESHOLD_DEG = 0.5


def simulate_pddl_shot_plan(
    problem_data: dict,
    angle_deg: float,
    gravity: float = None,
    dt: float = 0.01,
    max_steps: int = 10000,
    debug: bool = False,
):
    """
    Forward-simulate flight after pa-twang using base_domain flying + placeholder ground bounce.

    Returns whether the shot path touches ground (y <= 0) before termination, matching
  collision_ground in the PDDL model when M5 collision is not injected.
    Also checks platform collisions using expanded margins (2.5/3.0 multipliers) matching PDDL domain.
    
    Now also returns the full trajectory for visualization.
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
    vx = v * cosine
    vy = v * sinus

    pig = problem_data.get(pig_key) if pig_key else None
    pr = float(pig.get("pig_radius", 3.5)) if pig else 3.5
    px = float(pig["x_pig"]) if pig else None
    py = float(pig["y_pig"]) if pig else None

    # Collect platform data
    platforms = []
    for key, val in problem_data.items():
        if key.startswith("platform_"):
            platforms.append({
                "name": key,
                "x": float(val["x_platform"]),
                "y": float(val["y_platform"]),
                "w": float(val["platform_width"]),
                "h": float(val["platform_height"]),
            })

    if debug:
        print(f"\n[SIM DEBUG] Simulating shot at angle={angle_deg:.1f}°")
        print(f"[SIM DEBUG] Bird: ref=({ref_x:.1f}, {ref_y:.1f}), launch=({launch_x:.1f}, {launch_y:.1f}), v={v:.1f}, radius={br:.1f}")
        print(f"[SIM DEBUG] Pig: ({px:.1f}, {py:.1f}), radius={pr:.1f}")
        print(f"[SIM DEBUG] Platforms: {len(platforms)}")
        for p in platforms:
            print(f"[SIM DEBUG]   {p['name']}: center=({p['x']:.1f}, {p['y']:.1f}), size={p['w']:.1f}x{p['h']:.1f}")
            print(f"[SIM DEBUG]     bounds: left={p['x']-p['w']/2:.1f}, right={p['x']+p['w']/2:.1f}, bottom={p['y']-p['h']/2:.1f}, top={p['y']+p['h']/2:.1f}")

    ground_touches = 0
    pig_killed = False
    platform_hit = False
    platform_hit_name = None
    platform_hit_pos = None
    bounce_count = 0
    trajectory = [(x, y)]  # Store trajectory for visualization
    step = 0

    for step in range(max_steps):
        if pig is not None and not pig_killed:
            dx = x - px
            dy = y - py
            if (dx * dx + dy * dy) <= (br + pr) ** 2:
                pig_killed = True
                if debug:
                    print(f"[SIM DEBUG] Step {step}: PIG HIT at ({x:.1f}, {y:.1f})")
                break

        # Platform collision check using PDDL domain margins (2.5/3.0 multipliers)
        for plat in platforms:
            plat_left = plat["x"] - plat["w"] / 2
            plat_right = plat["x"] + plat["w"] / 2
            plat_bottom = plat["y"] - plat["h"] / 2
            plat_top = plat["y"] + plat["h"] / 2
            
            # Check AABB overlap with expanded bird radius (matching PDDL 1.5 multipliers)
            # Bird effective bounds: [x - br*1.5, x + br*1.5] x [y - br*1.5, y + br*1.5]
            if (x - br * 1.5 <= plat_right and
                x + br * 1.5 >= plat_left and
                y + br * 1.5 >= plat_bottom and
                y - br * 1.5 <= plat_top):
                platform_hit = True
                platform_hit_name = plat["name"]
                platform_hit_pos = (x, y)
                if debug:
                    print(f"[SIM DEBUG] Step {step}: PLATFORM HIT '{plat['name']}' at bird pos ({x:.1f}, {y:.1f})")
                    print(f"[SIM DEBUG]   Platform bounds: [{plat_left:.1f}, {plat_right:.1f}] x [{plat_bottom:.1f}, {plat_top:.1f}]")
                    print(f"[SIM DEBUG]   Bird expanded bounds: [{x-br*2.5:.1f}, {x+br*3.0:.1f}] x [{y-br*2.5:.1f}, {y+br*3.0:.1f}]")
                vx, vy = 0.0, 0.0
                bounce_count = 3
                break
        
        if platform_hit:
            break

        if x > 800 or bounce_count >= 3:
            break

        x += vx * dt
        y += vy * dt
        vy -= gravity * dt
        
        # Store trajectory point every 5 steps for efficiency
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

    # Add final position
    trajectory.append((x, y))
    
    if debug:
        print(f"[SIM DEBUG] Simulation ended at step {step}: pos=({x:.1f}, {y:.1f})")
        print(f"[SIM DEBUG] Result: pig_killed={pig_killed}, platform_hit={platform_hit}, ground_touches={ground_touches}")

    return {
        "uses_ground_collision": ground_touches > 0,
        "pig_killed_in_sim": pig_killed,
        "ground_touches": ground_touches,
        "platform_collision": platform_hit,
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
