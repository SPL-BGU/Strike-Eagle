import numpy as np

from agents.pddl.pddl_files.world_model import WorldModel


def gridsearch(initial_value,delta,params,world_model : WorldModel):
    # Given true data for testing
    true_distance = 100  # in meters
    initial_velocity = 0  # assuming object starts from rest
    time_of_fall = 4.5  # seconds
    g_vlaues = np.linspace(world_model.gravity)
    # Define the range of gravity values to search over
    g_values = np.linspace(world_model.gravity - delta , world_model.gravity + delta, 100)

    best_g = None
    min_error = float('inf')

    # Grid search loop
    for g in g_values:
        predicted_distance = simulate_motion(g, initial_velocity, time_of_fall)
        error = calculate_error(true_distance, predicted_distance)

        if error < min_error:
            min_error = error
            best_g = g

    print(f"Best estimate for gravity: {best_g}")
    print(f"Minimum error: {min_error}")
