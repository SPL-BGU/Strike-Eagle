import math

import numpy
import numpy as np
from matplotlib import pyplot as plt
from SALib.sample import saltelli
from SALib.analyze import sobol
from agents.pddl.pddl_files.world_model.world_model import WorldModel
from numpy.polynomial.polynomial import Polynomial
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


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
    """
    Fit a polynomial and return residual. Handles numerical errors gracefully.
    """
    try:
        # Diagnostics
        coeffs, resid, rank, sv, rcond = np.polyfit(x, y, degree, full=True)

        # Check rank sufficiency
        if rank < degree + 1:
            print("Rank is insufficient. Consider reducing the polynomial degree.")

        # Analyze singular values
        if len(sv) > 0 and max(sv) > 0:
            threshold = rcond * max(sv)
            unstable_singular_values = [v for v in sv if v < threshold]

            if unstable_singular_values:
                print("Warning: Singular values indicate numerical instability.")

        residual_sum_squares = resid[0] if len(resid) > 0 else 0.0

        return residual_sum_squares, Polynomial(coeffs[::-1])
    except (np.linalg.LinAlgError, ValueError) as e:
        # If polynomial fitting fails, return a high residual and a constant polynomial
        # This will cause get_poly_rank to skip this degree
        print(f"Warning: Polynomial fitting failed for degree {degree}: {e}")
        # Return a constant polynomial (mean value) as fallback
        mean_val = np.nanmean(y) if len(y) > 0 else 0.0
        return float('inf'), Polynomial([mean_val])


def get_poly_rank(x, y, max_rank=5, threshold=1):
    """
    Automatically select optimal polynomial degree.
    Original simple implementation.
    """
    resids_coeff = np.array([get_resid(x, y, degree) for degree in range(0, max_rank)])

    resids,polys = resids_coeff.T

    resids_diff = - np.diff(resids)

    condition = resids_diff < threshold

    rank = np.argmax(condition) if np.any(condition) else 3

    return rank, polys[rank]


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


def compute_derivatives(poly_x: Polynomial, poly_y: Polynomial, time_samples: np.ndarray, 
                        x_values: np.ndarray = None, y_values: np.ndarray = None):
    """
    Compute derivatives (xdot, xddot, ydot, yddot) for all time samples.
    
    Step 2 of the learning process: Sample the fitted curves and compute derivatives.
    Uses numerical differentiation for more accurate results, with polynomial derivatives as fallback.
    
    Parameters:
    -----------
    poly_x : Polynomial
        Fitted polynomial for x(t) trajectory
    poly_y : Polynomial
        Fitted polynomial for y(t) trajectory
    time_samples : np.ndarray
        Array of time values to sample the derivatives at
    x_values : np.ndarray, optional
        Actual x values from the polynomial (if provided, used for numerical differentiation)
    y_values : np.ndarray, optional
        Actual y values from the polynomial (if provided, used for numerical differentiation)
        
    Returns:
    --------
    tuple : (xdot, xddot, ydot, yddot)
        Arrays of derivative values for each time sample
        - xdot: first derivative of x(t) at each time sample
        - xddot: second derivative of x(t) at each time sample
        - ydot: first derivative of y(t) at each time sample
        - yddot: second derivative of y(t) at each time sample
    """
    try:
        # Get x and y values if not provided
        if x_values is None:
            x_values = np.array([poly_x(t) for t in time_samples])
        if y_values is None:
            y_values = np.array([poly_y(t) for t in time_samples])
        
        # Compute numerical derivatives for better accuracy
        # Use central differences for interior points, forward/backward for boundaries
        dt = time_samples[1] - time_samples[0] if len(time_samples) > 1 else 1.0
        
        # First derivatives (velocity)
        xdot = np.gradient(x_values, dt)  # Numerical derivative: dx/dt
        ydot = np.gradient(y_values, dt)  # Numerical derivative: dy/dt
        
        # Second derivatives (acceleration)
        xddot = np.gradient(xdot, dt)  # Numerical derivative: d²x/dt²
        yddot = np.gradient(ydot, dt)  # Numerical derivative: d²y/dt²
        
        # Validate: compare with polynomial derivatives for sanity check
        # (but use numerical derivatives as primary)
        try:
            poly_x_deriv = poly_x.deriv()
            poly_y_deriv = poly_y.deriv()
            poly_x_deriv2 = poly_x.deriv(2)
            poly_y_deriv2 = poly_y.deriv(2)
            
            # Compute polynomial derivatives for comparison
            xdot_poly = np.array([poly_x_deriv(t) for t in time_samples])
            ydot_poly = np.array([poly_y_deriv(t) for t in time_samples])
            xddot_poly = np.array([poly_x_deriv2(t) for t in time_samples])
            yddot_poly = np.array([poly_y_deriv2(t) for t in time_samples])
            
            # Check for large discrepancies (warn if difference is significant)
            xdot_diff = np.abs(xdot - xdot_poly)
            if np.max(xdot_diff) > 10.0:  # Threshold for warning
                print(f"Warning: Large discrepancy between numerical and polynomial xdot derivatives (max diff: {np.max(xdot_diff):.2f})")
                # Use polynomial derivative if numerical seems problematic
                if np.any(np.isnan(xdot)) or np.any(np.isinf(xdot)):
                    xdot = xdot_poly
                    xddot = xddot_poly
                    ydot = ydot_poly
                    yddot = yddot_poly
        except:
            # If polynomial derivative fails, stick with numerical
            pass
        
        # Replace any NaN or inf values with 0 (fallback)
        xdot = np.nan_to_num(xdot, nan=0.0, posinf=0.0, neginf=0.0)
        xddot = np.nan_to_num(xddot, nan=0.0, posinf=0.0, neginf=0.0)
        ydot = np.nan_to_num(ydot, nan=0.0, posinf=0.0, neginf=0.0)
        yddot = np.nan_to_num(yddot, nan=0.0, posinf=0.0, neginf=0.0)
        
        return xdot, xddot, ydot, yddot
    except Exception as e:
        print(f"Error in compute_derivatives: {e}. Trying polynomial derivatives as fallback.")
        try:
            # Fallback to polynomial derivatives
            poly_x_deriv = poly_x.deriv()
            poly_x_deriv2 = poly_x.deriv(2)
            poly_y_deriv = poly_y.deriv()
            poly_y_deriv2 = poly_y.deriv(2)
            
            xdot = np.array([poly_x_deriv(t) for t in time_samples])
            xddot = np.array([poly_x_deriv2(t) for t in time_samples])
            ydot = np.array([poly_y_deriv(t) for t in time_samples])
            yddot = np.array([poly_y_deriv2(t) for t in time_samples])
            
            xdot = np.nan_to_num(xdot, nan=0.0, posinf=0.0, neginf=0.0)
            xddot = np.nan_to_num(xddot, nan=0.0, posinf=0.0, neginf=0.0)
            ydot = np.nan_to_num(ydot, nan=0.0, posinf=0.0, neginf=0.0)
            yddot = np.nan_to_num(yddot, nan=0.0, posinf=0.0, neginf=0.0)
            return xdot, xddot, ydot, yddot
        except:
            print("Error in compute_derivatives fallback. Returning zero derivatives.")
            n_samples = len(time_samples)
            return (np.zeros(n_samples), np.zeros(n_samples), 
                    np.zeros(n_samples), np.zeros(n_samples))


def fit_state_transition(state_current: np.ndarray, state_previous: np.ndarray, max_degree: int = 5, threshold: float = 1.0):
    """
    Fit a polynomial transition function to predict current state from previous state.
    
    Step 3 of the learning process: Fit polynomial functions to predict state variables.
    For single-feature inputs, uses get_poly_rank() to automatically select the optimal degree.
    For multi-feature inputs, uses sklearn's PolynomialFeatures with automatic degree selection.
    
    Parameters:
    -----------
    state_current : np.ndarray
        Array of current state values (target values for prediction)
        Shape: (n_samples,)
    state_previous : np.ndarray
        Array of previous state values (input features for prediction)
        Shape: (n_samples, n_features) where n_features can be 1, 2, or 3
    max_degree : int, default=5
        Maximum polynomial degree to consider
    threshold : float, default=1.0
        Threshold for polynomial rank selection (passed to get_poly_rank)
        
    Returns:
    --------
    dict
        Dictionary with keys:
        - "model": Regression model (LinearRegression for multi-feature, wrapper for single-feature)
        - "polynomial": Polynomial object representing the transition function
    """
    # Validate inputs
    if len(state_current) == 0 or len(state_previous) == 0:
        print("Warning: Empty input data in fit_state_transition. Returning constant model.")
        mean_val = np.nanmean(state_current) if len(state_current) > 0 else 0.0
        constant_poly = Polynomial([mean_val])
        # Create a simple wrapper model for single-feature case
        class ConstantModel:
            def predict(self, X):
                return np.full(len(X), mean_val)
        return {"model": ConstantModel(), "polynomial": constant_poly}
    
    # Check for NaN or inf values
    if np.any(np.isnan(state_current)) or np.any(np.isnan(state_previous)) or \
       np.any(np.isinf(state_current)) or np.any(np.isinf(state_previous)):
        print("Warning: NaN or inf values detected in fit_state_transition. Returning constant model.")
        mean_val = np.nanmean(state_current)
        constant_poly = Polynomial([mean_val])
        class ConstantModel:
            def predict(self, X):
                return np.full(len(X), mean_val)
        return {"model": ConstantModel(), "polynomial": constant_poly}
    
    # Ensure state_previous is 2D
    if state_previous.ndim == 1:
        state_previous = state_previous.reshape(-1, 1)
    
    n_features = state_previous.shape[1]
    
    if n_features == 1:
        # Single feature: use get_poly_rank for automatic degree selection
        x_input = state_previous.flatten()
        try:
            rank, poly = get_poly_rank(x_input, state_current, max_rank=max_degree, threshold=threshold)
            # Create a wrapper model that uses the polynomial for prediction
            class PolynomialModel:
                def __init__(self, polynomial):
                    self.polynomial = polynomial
                def predict(self, X):
                    if isinstance(X, np.ndarray) and X.ndim > 1:
                        X = X.flatten()
                    return np.array([self.polynomial(x) for x in X])
            return {"model": PolynomialModel(poly), "polynomial": poly}
        except Exception as e:
            print(f"Error in fit_state_transition (single feature): {e}. Returning constant model.")
            mean_val = np.nanmean(state_current)
            constant_poly = Polynomial([mean_val])
            class ConstantModel:
                def predict(self, X):
                    return np.full(len(X) if hasattr(X, '__len__') else 1, mean_val)
            return {"model": ConstantModel(), "polynomial": constant_poly}
    else:
        # Multiple features: use sklearn's PolynomialFeatures with degree selection
        # Try different degrees and select the best one based on residual
        best_model = None
        best_degree = 1
        best_residual = float('inf')
        best_poly_features = None
        
        try:
            for degree in range(1, max_degree + 1):
                try:
                    # Create polynomial features
                    poly_features = PolynomialFeatures(degree=degree, include_bias=True)
                    X_poly = poly_features.fit_transform(state_previous)
                    
                    # Check for NaN or inf in polynomial features
                    if np.any(np.isnan(X_poly)) or np.any(np.isinf(X_poly)):
                        print(f"Warning: NaN/inf in polynomial features for degree {degree}. Skipping.")
                        continue
                    
                    # Fit linear regression
                    model = LinearRegression()
                    model.fit(X_poly, state_current)
                    
                    # Calculate residual (sum of squared errors)
                    y_pred = model.predict(X_poly)
                    residual = np.sum((state_current - y_pred) ** 2)
                    
                    # Check if residual is valid
                    if not np.isfinite(residual):
                        print(f"Warning: Invalid residual for degree {degree}. Skipping.")
                        continue
                    
                    # Check if this degree is better
                    if degree == 1:
                        best_model = model
                        best_degree = degree
                        best_residual = residual
                        best_poly_features = poly_features
                    else:
                        # Check if improvement is significant (similar to get_poly_rank logic)
                        improvement = best_residual - residual
                        if improvement > threshold:
                            best_model = model
                            best_degree = degree
                            best_residual = residual
                            best_poly_features = poly_features
                        else:
                            # No significant improvement, use previous degree
                            break
                except Exception as e:
                    print(f"Warning: Error fitting degree {degree}: {e}. Trying next degree.")
                    continue
            
            # If no model was successfully fitted, return a simple constant model
            if best_model is None:
                print("Warning: Could not fit polynomial model. Returning constant model.")
                mean_val = np.nanmean(state_current)
                constant_poly = Polynomial([mean_val])
                class ConstantModel:
                    def predict(self, X):
                        n = len(X) if hasattr(X, '__len__') and not isinstance(X, (str, bytes)) else 1
                        return np.full(n, mean_val)
                return {"model": ConstantModel(), "polynomial": constant_poly}
            
            # Store poly_features in the model for later use
            best_model.poly_features = best_poly_features
            
            # Create a polynomial representation by fitting to model predictions
            # Sample the input space and fit a polynomial to the model's predictions
            try:
                # Create a sample grid of inputs
                input_ranges = []
                for i in range(n_features):
                    col = state_previous[:, i]
                    input_ranges.append(np.linspace(np.min(col), np.max(col), min(50, len(state_current))))
                
                # Generate sample points
                if n_features == 2:
                    # For 2 features, create a meshgrid
                    X1, X2 = np.meshgrid(input_ranges[0], input_ranges[1])
                    sample_inputs = np.column_stack([X1.ravel(), X2.ravel()])
                elif n_features == 3:
                    # For 3 features, sample along each dimension
                    # Use a simpler approach: sample along the first feature
                    sample_inputs = np.column_stack([
                        input_ranges[0],
                        np.mean(state_previous[:, 1]) * np.ones(len(input_ranges[0])),
                        np.mean(state_previous[:, 2]) * np.ones(len(input_ranges[0]))
                    ])
                else:
                    # For more features, use mean values for other dimensions
                    sample_inputs = np.column_stack([
                        input_ranges[0],
                        *[np.mean(state_previous[:, i]) * np.ones(len(input_ranges[0])) 
                          for i in range(1, n_features)]
                    ])
                
                # Get model predictions
                X_poly_sample = best_poly_features.transform(sample_inputs)
                y_pred_sample = best_model.predict(X_poly_sample)
                
                # Fit a polynomial to the first feature vs predictions
                # This gives us a polynomial representation (simplified, but useful)
                x_sample = sample_inputs[:, 0]
                poly_rep = Polynomial.fit(x_sample, y_pred_sample, deg=min(best_degree, 5))
                
            except Exception as e:
                print(f"Warning: Could not create polynomial representation: {e}. Using constant.")
                mean_val = np.nanmean(state_current)
                poly_rep = Polynomial([mean_val])
            
            return {"model": best_model, "polynomial": poly_rep}
        except Exception as e:
            print(f"Error in fit_state_transition (multi-feature): {e}. Returning constant model.")
            mean_val = np.nanmean(state_current)
            constant_poly = Polynomial([mean_val])
            class ConstantModel:
                def predict(self, X):
                    n = len(X) if hasattr(X, '__len__') and not isinstance(X, (str, bytes)) else 1
                    return np.full(n, mean_val)
            return {"model": ConstantModel(), "polynomial": constant_poly}