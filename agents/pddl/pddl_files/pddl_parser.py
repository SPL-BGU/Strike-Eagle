from string import Template

import numpy as np
import os

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


def generate_pddl(problem_data: dict, init_angle, angel_rate, world_model: WorldModel):
    objects = list()
    goals = list()
    initial_state = [
        f"(= (angle) 90)",
        f"(= (angle_rad) {np.pi / 2})",
        f"(= (angle_rate) {angel_rate})",
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
    with open(path, "r") as file:
        content = file.read()
    new_content = content

    for variable_name, variable_data in world_model.kb["collision"]["variables"].items():
        if variable_data["model"] == None:
            continue
        coefs = variable_data["model"].coef_
        bias = variable_data["model"].intercept_
        vars = ['x_bird', 'y_bird', 'vx_bird', 'vy_bird']

        # === Step 2: generate the PDDL effect ===
        threshold = 1e-8  # you can adjust this threshold if needed
        terms = [
            f"(* {c:.4f} ({v} ?b))"
            for c, v in zip(coefs, vars)
            if abs(c) >= threshold
        ]
        # terms = [f"(* {0 if abs(c) < threshold else c} ({v} ?b))" for c, v in zip(coeffs, vars)] // for explainability

        def nest_terms(term_list):
            if not term_list:
                return "0.0"
            if len(term_list) == 1:
                return term_list[0]
            return f"(+ {term_list[0]} {nest_terms(term_list[1:])})"

        nested_terms = nest_terms(terms[:4])

        # Wrap bias with the nested terms
        if abs(bias) > threshold:
            equation = f"(+ {bias} {nested_terms})"
        else:
            equation = nested_terms
        # === Step 4: replace the tag ===
        new_content = new_content.replace("{{SE-collision-{}}}".format(variable_name), equation)

    # === Save modified file in same folder ===
    base_dir = os.path.dirname(path)
    base_name = os.path.splitext(os.path.basename(path))[0]
    output_path = os.path.join(base_dir, f"{base_name}_modified.pddl")

    with open(output_path, "w") as file:
        file.write(new_content)

    print(f"Modified file saved to: {output_path}")
    # === Step 5: save to a new file (or overwrite if you prefer) ==
    print("Injection complete.")


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
