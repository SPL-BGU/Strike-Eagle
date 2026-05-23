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

def generate_pddl(problem_data: dict, init_angle, angel_rate, world_model: WorldModel):
    objects = list()
    goals = list()
    initial_state = [
        f"(= (angle) 90)",
        f"(= (angle_rad) {np.pi / 2})",
        f"(= (angle_rate) {angel_rate})",
        f"(= (angle_bias) {ANGLE_BIAS_DEGREES})",  # Bias correction for slingshot
        "(= (cosine) 0 )",
        "(= (sinus) 1 )",
        f"(= (bounce_count) 0)",
        f"(= (gravity) {world_model.hyperparams_values[Params.gravity]})",
        f"(= (active_bird) 0)",
        f"(= (ground_y_damper) 0.1)"
        f"(= (ground_x_damper) 0.5)"
    ]
    for object, object_data in problem_data.items():
        objects.append(f"{object} - {object.split('_')[0]}")
        if "pig" in object:
            goals.append(f"(pig_dead {object})")
        for pred, val in object_data.items():
            initial_state.append(f"(= ({pred} {object}) {val})")
        # relations
        for other_object in problem_data:
            if 'bird' in object and 'block' in other_object:
                initial_state.append(f'(= (bird_block_damage {object} {other_object}) 0.01)')

    objects_str = "\n".join(objects)
    goals_str = "\n".join(goals)
    initial_state_str = "\n".join(initial_state)
    return objects_str, initial_state_str, goals_str


def write_problem_file(path: str, problem_data: dict, init_angle: float, angel_rate: float, world_model: WorldModel):
    objects, initial_state, goals = generate_pddl(problem_data, init_angle, angel_rate, world_model)
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
    n = float(line.split(':')[0])
    return 'shoot', 90 - n * angel_rate


def parse_solution_to_actions(solution_path: str, init_angle, angel_rate):
    with open(solution_path) as solution_file:
        lines = solution_file.readlines()
        actions = list(filter(action_filter, lines))
        actions = list(map(lambda l: parse_action(l, init_angle, angel_rate), actions))
        return actions
