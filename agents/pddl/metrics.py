import numpy as np
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error,root_mean_squared_error


def resample_trajectory(traj, num_points):
    """Resample a trajectory to `num_points` using linear interpolation."""
    original_idx = np.linspace(0, 1, len(traj))
    target_idx = np.linspace(0, 1, num_points)
    interpolator = interp1d(original_idx, traj, axis=0)
    return interpolator(target_idx)


def calculate_rmse(observed_trajectory, estimated_trajectory):
    N = 100

    max_distance =np.max(observed_trajectory,axis=0)[0]

    condition = estimated_trajectory[:, 0] > max_distance

    indices = np.where(condition)[0]

    # Return the last matching index or len(observed_trajectory) if none found
    last_index = indices[-1] if indices.size > 0 else len(estimated_trajectory)

    observed_trajectory_resampled = resample_trajectory(observed_trajectory, N)
    estimated_trajectory_resampled = resample_trajectory(estimated_trajectory[:last_index], N)

    rmse = root_mean_squared_error(observed_trajectory_resampled, estimated_trajectory_resampled)

    return rmse
