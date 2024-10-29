import numpy
import numpy as np
from matplotlib import pyplot as plt

from agents.pddl.pddl_files.world_model import WorldModel
from numpy.polynomial.polynomial import Polynomial

from agents.pddl.trajectory_parser import construct_trajectory


def calculate_error(observed: Polynomial, estimated:Polynomial):
    domain = [
        max(observed.domain[0], estimated.domain[0]),
        min(observed.domain[1], estimated.domain[1]),
        # 350
    ]
    values = np.linspace(domain[0],
                         domain[1], 100)
    return np.average(
        np.abs(observed(values) - estimated(values)))  # average cancel not matching size domain comparision


def grid_search(observed_trajectory: numpy.ndarray,
                estimated_trajectory: numpy.ndarray,
                assumed_values: list,
                simulating_function,
                n_values: int = 32,
                delta: float = .1,
                visualize=True)->list:
    observed_trajectory_polynom = Polynomial.fit(observed_trajectory[:, 0], observed_trajectory[:, 1], 2, )

    values_options = [
        np.linspace(value * (1 - delta),
                    value * (1 + delta), n_values)
        for value in assumed_values
    ]
    grid_values = np.stack(np.meshgrid(*values_options)).T
    grid_values = grid_values.reshape(-1, grid_values.shape[-1])

    predicated_trajectories_over_grid = [
        simulating_function(observed_trajectory, values) for
        values in grid_values
    ]
    predicated_trajectories_over_grid_polynoms = [
        Polynomial.fit(estimated_trajectory[:, 0], estimated_trajectory[:, 1], 2) for estimated_trajectory in
        predicated_trajectories_over_grid
    ]

    errors = [calculate_error(observed_trajectory_polynom, predicated_trajectory_polynom) for
              predicated_trajectory_polynom in predicated_trajectories_over_grid_polynoms]
    selected_values_index = np.argmin(errors)

    if visualize:
        plt.plot(observed_trajectory[:, 0], observed_trajectory[:, 1], marker='o', color='blue')
        plt.plot(estimated_trajectory[:, 0], estimated_trajectory[:, 1], marker='x', color='red')
        plt.plot(predicated_trajectories_over_grid[selected_values_index][:, 0],
                 predicated_trajectories_over_grid[selected_values_index][:, 1], marker='.', color='green')
        plt.axis('equal')  # Equal scaling for x and y axes
        plt.show()
    return grid_values[selected_values_index]
