import numpy as np
from scipy.optimize import minimize
from scipy.optimize import curve_fit


def solve_difference(theta,observed_traj, estimated_traj):
    # Sample time points and unknown trajectory
    t_values = np.arange(300)

    # Initial guess for the parameters of the known equation
    x0 = estimated_traj[0][0]
    y0 = estimated_traj[0][1]
    v = 175.9259
    g_init = 60

    params = [x0, y0, v, theta,estimated_traj]
    # Optimize using SciPy's minimize function
    result = minimize(
        objective_function,
        g_init,
        args=(params, t_values, observed_traj), method='BFGS'
    )
    # The optimized parameters
    optimized_params = result.x
    print("Optimized parameters:", optimized_params)


def model_equation(g, params, t):
    x0, y0, v, theta,_ = params
    x = x0 + v * np.cos(theta) * t/100
    y = y0 + v * np.sin(theta) * t/100 - g * .5 * (t/100) ** 2  # estimated gravity
    return polynomial(x,*curve_fit(polynomial,x,y)[0])


def y_equation(g,params,t):
    x0, y0, v, theta, _ = params
    y = y0 + v * np.sin(theta) * t / 100 - g * .5 * (t / 100) ** 2  # estimated gravity
    return y

def polynomial(x, a, b, c):
    return a * x**2 + b * x + c

def objective_function(gravity, params, t, observed_traj):
    funced_model_traj = model_equation(gravity, params, t)
    funced_observed_traj = polynomial(observed_traj[:,0],*curve_fit(polynomial,observed_traj[:,0],observed_traj[:,1])[0])
    return np.sum((funced_model_traj - funced_observed_traj)**2)  # Sum of squared errors
