import math

import numpy
import numpy as np
from matplotlib import pyplot as plt
from SALib.sample import saltelli
from SALib.analyze import sobol
from agents.pddl.pddl_files.world_model.world_model import WorldModel
from numpy.polynomial.polynomial import Polynomial


def calculate_current_error(observed: Polynomial, estimated: Polynomial):
    domain = [
        max(observed.domain[0], estimated.domain[0]),
        min(observed.domain[1], estimated.domain[1]),
        # 350
    ]
    values = np.linspace(domain[0],
                         domain[1], 100)
    return np.mean(
        np.abs(observed(values) - estimated(values)))  # average cancel not matching size domain comparision


def calculate_aggregative_erros(grid_value: tuple, current_error: list, kb):
    aggregative_error_list = list()

    weights = 1 / 2 ** np.array(range(4))
    if len(kb) == 1:
        return current_error

    grid_value_list = [tuple(row) for row in grid_value]
    for i, value in enumerate(grid_value_list):
        error_list = [d[value] for d in kb if value in d]
        error_list.append(current_error[i])
        aggregative_error_list.append(np.average(error_list, weights=weights[:len(error_list)]))
    return aggregative_error_list


def get_resid(x, y, degree):
    coeffs, stats = Polynomial.fit(x, y, degree, full=True)

    # Diagnostics
    resid, rank, sv, rcond = stats

    # Check rank sufficiency
    if rank < degree + 1:
        print("Rank is insufficient. Consider reducing the polynomial degree.")

    # Analyze singular values
    threshold = rcond * max(sv)
    unstable_singular_values = [v for v in sv if v < threshold]

    if unstable_singular_values:
        print("Warning: Singular values indicate numerical instability.")

    return resid[0]


def get_poly_rank(x, y, max_rank=5, threshold=0.1):
    resids = [get_resid(x, y, degree) for degree in range(0, max_rank)]

    resids_diff = - np.diff(resids)

    condition = resids_diff < threshold

    rank = np.argmax(condition) if np.any(condition) else 3

    return rank


def get_problem(world_model: WorldModel, delta: float = 0.3):
    params = world_model.hyperparams_values
    param_values = [[np.floor(param * (1 - delta)), np.floor(param * (1 + delta))] for param in params.values()]
    return {
        'num_vars': len(params),
        'names': list(params.keys()),
        'bounds': param_values
    }


def get_params_sensitivity(observed_trajectory: numpy.ndarray,
                           estimated_trajectory: numpy.ndarray,
                           world_model: WorldModel,
                           rank,
                           simulating_function,
                           delta: float = 0.3):
    problem = get_problem(world_model, delta)

    param_values = saltelli.sample(problem, N=16)

    param_values = [dict(zip(problem["names"],i)) for i in param_values]
    param_values, errors = grid_search(
        observed_trajectory,
        estimated_trajectory,
        param_values,
        rank,
        simulating_function,
        visualize= False
    )

    pivot_params = param_values[np.argmin(errors)]

    return sobol.analyze(problem,np.array(errors),print_to_console=True)['ST'],pivot_params


def get_param_values(
        params: dict,
        delta: float,
        precision: float):
    values_options = [
        np.arange(math.floor(value * (1 - delta)),
                  math.floor(value * (1 + delta)), precision)
        for value in params.values()
    ]
    # Generate posebilites
    grid_values = np.stack(np.meshgrid(*values_options)).T
    grid_values = grid_values.reshape(-1, grid_values.shape[-1])

    return [dict(zip(params.keys(), i)) for i in grid_values]

def grid_search(observed_trajectory: numpy.ndarray,
                estimated_trajectory: numpy.ndarray,
                param_values: list,
                rank,
                simulating_function,
                visualize=True):

    observed_trajectory_polynom = Polynomial.fit(observed_trajectory[:, 0], observed_trajectory[:, 1], rank )

    predicated_trajectories_over_grid = [
        simulating_function(observed_trajectory, values) for
        values in param_values
    ]

    # Generate polynomials
    predicated_trajectories_over_grid_polynoms = [
        Polynomial.fit(estimated_trajectory[:, 0], estimated_trajectory[:, 1], 2) for estimated_trajectory in
        predicated_trajectories_over_grid
    ]

    # Calculate errors
    errors = [calculate_current_error(observed_trajectory_polynom, predicated_trajectory_polynom) for
              predicated_trajectory_polynom in predicated_trajectories_over_grid_polynoms]

    selected_values_index = np.argmin(errors)

    if visualize:
        plt.plot(observed_trajectory[:,
                 0], observed_trajectory[:, 1], marker='o', color='blue')
        plt.plot(estimated_trajectory[:, 0], estimated_trajectory[:, 1], marker='x', color='red')
        plt.plot(predicated_trajectories_over_grid[selected_values_index][:, 0],
                 predicated_trajectories_over_grid[selected_values_index][:, 1], marker='.', color='green')
        plt.axis('equal')  # Equal scaling for x and y axes
        plt.show()

    return param_values, errors,
