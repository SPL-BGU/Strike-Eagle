from string import Template

import numpy as np
import os
import re

from agents.pddl.pddl_files.world_model.params import Params
from agents.pddl.pddl_files.world_model.world_model import WorldModel

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
    Inject learned collision models into the PDDL domain file.
    
    Always uses the General (Linear) model for PDDL injection.
    CART models are used only for comparison, not for injection.
    """
    with open(path, "r") as file:
        content = file.read()
    new_content = content

    for variable_name, variable_data in world_model.kb["collision"]["variables"].items():
        placeholder = "{{SE-collision-{}}}".format(variable_name)
        
        if variable_data["model"] is None:
            new_content = new_content.replace(placeholder, "0.0")
            continue
        
        model = variable_data["model"]
        coefs = model.coef_
        bias = model.intercept_
        vars = ['x_bird', 'y_bird', 'vx_bird', 'vy_bird']

        threshold = 1e-8
        terms = [
            f"(* {c:.4f} ({v} ?b))"
            for c, v in zip(coefs, vars)
            if abs(c) >= threshold
        ]

        def nest_terms(term_list):
            if not term_list:
                return "0.0"
            if len(term_list) == 1:
                return term_list[0]
            return f"(+ {term_list[0]} {nest_terms(term_list[1:])})"

        nested_terms = nest_terms(terms[:4])

        if abs(bias) > threshold:
            equation = f"(+ {bias:.4f} {nested_terms})"
        else:
            equation = nested_terms if nested_terms else "0.0"
        
        new_content = new_content.replace(placeholder, equation)
    
    # Replace any remaining placeholders with 0 as fallback
    remaining_placeholders = re.findall(r'\{\{SE-collision-[^}]+\}\}', new_content)
    for placeholder in remaining_placeholders:
        new_content = new_content.replace(placeholder, "0.0")

    # Save modified file
    base_dir = os.path.dirname(path)
    base_name = os.path.splitext(os.path.basename(path))[0]
    output_path = os.path.join(base_dir, f"{base_name}_modified.pddl")

    with open(output_path, "w") as file:
        file.write(new_content)

    print(f"Modified file saved to: {output_path}")
    print("Injection complete.")


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
