from string import Template
problem_template= Template("""(define (problem sample_problem)
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

def generate_pddl(problem_data: dict,init_angle,angel_rate):
    objects = list()
    goals = list()
    initial_state= [
        f"(= (angle) {init_angle})",
        f"(= (angle_rate) {angel_rate})",
        f"(= (bounce_count) 0)",
        f"(= (gravity) 10)",
        f"(= (active_bird) 0)"
    ]
    for object,object_data in problem_data.items():
        objects.append(f"{object} - {object.split('_')[0]}")
        if "pig" in object:
            goals.append(f"(pig_dead {object})")
        for pred, val in object_data.items():
            initial_state.append(f"(= ({pred} {object}) {val})")
    objects_str = "\n".join(objects)
    goals_str = "\n".join(goals)
    initial_state_str = "\n".join(initial_state)
    return objects_str, initial_state_str, goals_str


def write_problem_file(path:str, problem_data:dict,init_angle,angel_rate):
    objects, initial_state, goals = generate_pddl(problem_data,init_angle,angel_rate)
    problem = problem_template.substitute({"objects":objects, "initial":initial_state, "goal": goals })
    with open(path,'w') as file:
        file.write(problem)
