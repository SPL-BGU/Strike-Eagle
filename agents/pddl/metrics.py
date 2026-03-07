import numpy as np
from scipy.interpolate import interp1d, CubicSpline
from sklearn.metrics import mean_squared_error,root_mean_squared_error
import matplotlib.pyplot as plt


def resample_trajectory(traj, num_points):
    """Resample a trajectory to `num_points` using linear interpolation."""
    original_idx = np.linspace(0, 1, len(traj))
    target_idx = np.linspace(0, 1, num_points)
    interpolator = interp1d(original_idx, traj, axis=0)
    return interpolator(target_idx)


# Angle bias for RMSE calculation only (does not affect planning)
# Set to 0 to disable - empirical testing showed inconsistent/negligible benefit
RMSE_ANGLE_BIAS_DEGREES = 0


def compare_rmse_with_and_without_bias(observed_trajectory, estimated_trajectory):
    """
    Compare RMSE with and without bias correction to see the effect.
    
    Args:
        observed_trajectory: (N, 2) array with [x, y] columns
        estimated_trajectory: (M, 2) array with [x, y] columns
        
    Returns:
        dict with both RMSE values
    """
    rmse_without_bias = calculate_rmse(observed_trajectory, estimated_trajectory, apply_bias_correction=False)
    rmse_with_bias = calculate_rmse(observed_trajectory, estimated_trajectory, apply_bias_correction=True)
    
    improvement = rmse_without_bias - rmse_with_bias
    percent_improvement = (improvement / rmse_without_bias * 100) if rmse_without_bias != 0 else 0
    
    print("=" * 60)
    print("RMSE BIAS CORRECTION COMPARISON")
    print("=" * 60)
    print(f"  Bias setting: {RMSE_ANGLE_BIAS_DEGREES}°")
    print("-" * 60)
    print(f"  RMSE without bias correction: {rmse_without_bias:.4f}")
    print(f"  RMSE with bias correction:    {rmse_with_bias:.4f}")
    print("-" * 60)
    print(f"  Improvement: {improvement:.4f} ({percent_improvement:.2f}%)")
    
    if rmse_with_bias < rmse_without_bias:
        print(f"  ✅ Bias correction HELPS (reduces RMSE)")
    elif rmse_with_bias > rmse_without_bias:
        print(f"  ❌ Bias correction HURTS (increases RMSE)")
        print(f"     Try changing RMSE_ANGLE_BIAS_DEGREES sign or value")
    else:
        print(f"  → No change (bias might be 0 or trajectories identical)")
    print("=" * 60)
    
    return {
        'rmse_without_bias': rmse_without_bias,
        'rmse_with_bias': rmse_with_bias,
        'improvement': improvement,
        'percent_improvement': percent_improvement
    }


def calculate_rmse(observed_trajectory, estimated_trajectory, trim_start_percent=0, trim_end_percent=0, 
                   apply_bias_correction=True):
    """
    Calculate RMSE between observed and estimated trajectories.
    
    Args:
        observed_trajectory: (N, 2) array with [x, y] columns
        estimated_trajectory: (M, 2) array with [x, y] columns
        trim_start_percent: Percentage of trajectory to trim from start (default 0%)
        trim_end_percent: Percentage of trajectory to trim from end (default 0%)
        apply_bias_correction: If True, adjust estimated trajectory for slingshot bias (default True)
        
    Returns:
        RMSE value
    """
    N = 1000
    
    # Apply bias correction to estimated trajectory
    # The estimated trajectory assumes ideal physics, but the slingshot shoots ~1.5° less steep
    # We shift the estimated trajectory's Y values to account for this
    if apply_bias_correction and RMSE_ANGLE_BIAS_DEGREES != 0:
        estimated_trajectory = _apply_trajectory_bias_correction(
            estimated_trajectory, 
            observed_trajectory,
            RMSE_ANGLE_BIAS_DEGREES
        )
    
    # Get the actual x-range of observed trajectory
    obs_x_min = np.min(observed_trajectory[:, 0])
    obs_x_max = np.max(observed_trajectory[:, 0])
    
    # Apply trimming to avoid noisy start/end frames
    x_range = obs_x_max - obs_x_min
    trimmed_x_min = obs_x_min + (x_range * trim_start_percent / 100)
    trimmed_x_max = obs_x_max - (x_range * trim_end_percent / 100)
    
    # Filter both trajectories to the trimmed x-range
    obs_mask = (observed_trajectory[:, 0] >= trimmed_x_min) & (observed_trajectory[:, 0] <= trimmed_x_max)
    est_mask = (estimated_trajectory[:, 0] >= trimmed_x_min) & (estimated_trajectory[:, 0] <= trimmed_x_max)
    
    obs_filtered = observed_trajectory[obs_mask]
    est_filtered = estimated_trajectory[est_mask]
    
    # Ensure we have points to work with
    if len(obs_filtered) == 0 or len(est_filtered) == 0:
        return float('inf')
    
    # Resample based on x-coordinate (arc length) rather than index
    # This ensures better alignment
    obs_x = obs_filtered[:, 0]
    est_x = est_filtered[:, 0]
    
    # Create common x-coordinates for interpolation
    common_x = np.linspace(trimmed_x_min, trimmed_x_max, N)
    
    # Interpolate y-coordinates
    obs_y_interp = np.interp(common_x, obs_x, obs_filtered[:, 1])
    est_y_interp = np.interp(common_x, est_x, est_filtered[:, 1])

    # Stack for RMSE calculation
    obs_resampled = np.column_stack([common_x, obs_y_interp])
    est_resampled = np.column_stack([common_x, est_y_interp])
    
    rmse = root_mean_squared_error(obs_resampled, est_resampled)
    
    return rmse


def _apply_trajectory_bias_correction(estimated_trajectory, observed_trajectory, bias_degrees):
    """
    Adjust the estimated trajectory to account for slingshot angle bias.
    
    The slingshot shoots at a slightly different angle than commanded.
    This function shifts the estimated trajectory vertically to approximate
    what the slingshot actually produces.
    
    For a trajectory with angle bias:
    - Less steep angle (positive bias) = bird goes higher (less negative Y in screen coords)
    - The Y shift increases with distance from launch (proportional to x distance)
    
    Args:
        estimated_trajectory: (N, 2) array with [x, y] columns
        observed_trajectory: (M, 2) array with [x, y] columns (used for reference)
        bias_degrees: Angle bias in degrees (positive = less steep = higher trajectory)
        
    Returns:
        Corrected estimated trajectory
    """
    if len(estimated_trajectory) == 0:
        return estimated_trajectory
    
    # Get launch point
    launch_x = estimated_trajectory[0, 0]
    launch_y = estimated_trajectory[0, 1]
    
    # Calculate x distance from launch for each point
    x_distance = estimated_trajectory[:, 0] - launch_x
    
    # Convert bias to radians
    bias_rad = np.radians(bias_degrees)
    
    # The Y shift due to angle change is approximately:
    # dy = dx * tan(bias) for small angles
    # For screen coordinates (Y increases downward):
    # - Positive bias (less steep) means the bird goes HIGHER (less positive Y)
    # - So we SUBTRACT the shift
    y_shift = x_distance * np.tan(bias_rad)
    
    # Apply correction
    corrected = estimated_trajectory.copy()
    corrected[:, 1] = corrected[:, 1] - y_shift  # Subtract because screen Y is inverted
    
    return corrected


def check_sampling_uniformity(observed_trajectory, plot=True):
    """
    Check how uniform the sampling is in the observed trajectory.
    
    If x-velocity is constant, uniform time sampling should result in 
    uniform Δx values between consecutive points.
    
    Args:
        observed_trajectory: (N, 2) array with [x, y] columns
        plot: If True, show a histogram of Δx values
        
    Returns:
        dict with statistics about the sampling uniformity
    """
    x_coords = observed_trajectory[:, 0]
    
    # Calculate consecutive x differences
    delta_x = np.diff(x_coords)
    
    # Calculate statistics
    mean_dx = np.mean(delta_x)
    std_dx = np.std(delta_x)
    min_dx = np.min(delta_x)
    max_dx = np.max(delta_x)
    cv = (std_dx / mean_dx * 100) if mean_dx != 0 else float('inf')  # Coefficient of Variation
    
    # Print results
    print("=" * 50)
    print("SAMPLING UNIFORMITY ANALYSIS")
    print("=" * 50)
    print(f"Number of points: {len(x_coords)}")
    print(f"X range: [{x_coords.min():.2f}, {x_coords.max():.2f}]")
    print("-" * 50)
    print(f"Mean Δx:  {mean_dx:.4f}")
    print(f"Std Δx:   {std_dx:.4f}")
    print(f"Min Δx:   {min_dx:.4f}")
    print(f"Max Δx:   {max_dx:.4f}")
    print(f"Coefficient of Variation: {cv:.2f}%")
    print("-" * 50)
    
    # Interpret uniformity
    if cv < 5:
        uniformity = "VERY UNIFORM"
    elif cv < 15:
        uniformity = "REASONABLY UNIFORM"
    else:
        uniformity = "NON-UNIFORM"
    print(f"Assessment: {uniformity}")
    print("=" * 50)
    
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Histogram of Δx values
        axes[0].hist(delta_x, bins=30, edgecolor='black', alpha=0.7)
        axes[0].axvline(mean_dx, color='red', linestyle='--', label=f'Mean: {mean_dx:.4f}')
        axes[0].set_xlabel('Δx')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distribution of X-coordinate Differences')
        axes[0].legend()
        
        # Δx over index (to see if there's a pattern)
        axes[1].plot(delta_x, marker='.', markersize=2, linestyle='-', linewidth=0.5)
        axes[1].axhline(mean_dx, color='red', linestyle='--', label=f'Mean: {mean_dx:.4f}')
        axes[1].set_xlabel('Point Index')
        axes[1].set_ylabel('Δx')
        axes[1].set_title('Δx Over Trajectory')
        axes[1].legend()
        
        plt.tight_layout()
        plt.show()
    
    return {
        'mean_dx': mean_dx,
        'std_dx': std_dx,
        'min_dx': min_dx,
        'max_dx': max_dx,
        'coefficient_of_variation': cv,
        'uniformity': uniformity,
        'delta_x': delta_x
    }


def compare_truncation_methods(observed_trajectory, estimated_trajectory):
    """
    Compare RMSE using different truncation methods:
    - Option 1: X-range of observed trajectory (current behavior)
    - Option 3: Minimum overlap of both trajectories
    
    Args:
        observed_trajectory: (N, 2) array with [x, y] columns
        estimated_trajectory: (M, 2) array with [x, y] columns
        
    Returns:
        dict with both RMSE values and comparison info
    """
    N = 1000
    
    # ============ OPTION 1: Observed X-range (current) ============
    obs_x_min_1 = np.min(observed_trajectory[:, 0])
    obs_x_max_1 = np.max(observed_trajectory[:, 0])
    
    obs_mask_1 = (observed_trajectory[:, 0] >= obs_x_min_1) & (observed_trajectory[:, 0] <= obs_x_max_1)
    est_mask_1 = (estimated_trajectory[:, 0] >= obs_x_min_1) & (estimated_trajectory[:, 0] <= obs_x_max_1)
    
    obs_filtered_1 = observed_trajectory[obs_mask_1]
    est_filtered_1 = estimated_trajectory[est_mask_1]
    
    common_x_1 = np.linspace(obs_x_min_1, obs_x_max_1, N)
    obs_y_1 = np.interp(common_x_1, obs_filtered_1[:, 0], obs_filtered_1[:, 1])
    est_y_1 = np.interp(common_x_1, est_filtered_1[:, 0], est_filtered_1[:, 1])
    
    rmse_option1 = root_mean_squared_error(
        np.column_stack([common_x_1, obs_y_1]),
        np.column_stack([common_x_1, est_y_1])
    )
    
    # ============ OPTION 3: Minimum overlap ============
    obs_x_min = np.min(observed_trajectory[:, 0])
    obs_x_max = np.max(observed_trajectory[:, 0])
    est_x_min = np.min(estimated_trajectory[:, 0])
    est_x_max = np.max(estimated_trajectory[:, 0])
    
    # Find the overlap region
    overlap_x_min = max(obs_x_min, est_x_min)
    overlap_x_max = min(obs_x_max, est_x_max)
    
    if overlap_x_min >= overlap_x_max:
        print("ERROR: No overlap between trajectories!")
        return {'error': 'No overlap'}
    
    obs_mask_3 = (observed_trajectory[:, 0] >= overlap_x_min) & (observed_trajectory[:, 0] <= overlap_x_max)
    est_mask_3 = (estimated_trajectory[:, 0] >= overlap_x_min) & (estimated_trajectory[:, 0] <= overlap_x_max)
    
    obs_filtered_3 = observed_trajectory[obs_mask_3]
    est_filtered_3 = estimated_trajectory[est_mask_3]
    
    common_x_3 = np.linspace(overlap_x_min, overlap_x_max, N)
    obs_y_3 = np.interp(common_x_3, obs_filtered_3[:, 0], obs_filtered_3[:, 1])
    est_y_3 = np.interp(common_x_3, est_filtered_3[:, 0], est_filtered_3[:, 1])
    
    rmse_option3 = root_mean_squared_error(
        np.column_stack([common_x_3, obs_y_3]),
        np.column_stack([common_x_3, est_y_3])
    )
    
    # ============ RESULTS ============
    difference = abs(rmse_option1 - rmse_option3)
    percent_diff = (difference / rmse_option1 * 100) if rmse_option1 != 0 else 0
    
    print("=" * 70)
    print("TRUNCATION METHOD COMPARISON")
    print("=" * 70)
    print("\nX-RANGE INFO:")
    print(f"  Observed trajectory:    x ∈ [{obs_x_min:.1f}, {obs_x_max:.1f}]  (width: {obs_x_max - obs_x_min:.1f})")
    print(f"  Estimated trajectory:   x ∈ [{est_x_min:.1f}, {est_x_max:.1f}]  (width: {est_x_max - est_x_min:.1f})")
    print(f"  Overlap region:         x ∈ [{overlap_x_min:.1f}, {overlap_x_max:.1f}]  (width: {overlap_x_max - overlap_x_min:.1f})")
    print("-" * 70)
    print("\nCOMPARISON RANGES:")
    print(f"  Option 1 (Observed):    x ∈ [{obs_x_min_1:.1f}, {obs_x_max_1:.1f}]  (width: {obs_x_max_1 - obs_x_min_1:.1f})")
    print(f"  Option 3 (Overlap):     x ∈ [{overlap_x_min:.1f}, {overlap_x_max:.1f}]  (width: {overlap_x_max - overlap_x_min:.1f})")
    print("-" * 70)
    print("\nRMSE RESULTS:")
    print(f"  Option 1 (Observed X-range):  {rmse_option1:.6f}")
    print(f"  Option 3 (Minimum Overlap):   {rmse_option3:.6f}")
    print("-" * 70)
    print(f"  Absolute Difference:          {difference:.6f}")
    print(f"  Percent Difference:           {percent_diff:.2f}%")
    print("=" * 70)
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Full trajectories with regions highlighted
    axes[0, 0].plot(observed_trajectory[:, 0], observed_trajectory[:, 1], 'b-', linewidth=2, label='Observed')
    axes[0, 0].plot(estimated_trajectory[:, 0], estimated_trajectory[:, 1], 'r-', linewidth=2, label='Estimated')
    axes[0, 0].axvline(obs_x_min, color='blue', linestyle='--', alpha=0.5)
    axes[0, 0].axvline(obs_x_max, color='blue', linestyle='--', alpha=0.5)
    axes[0, 0].axvline(est_x_min, color='red', linestyle='--', alpha=0.5)
    axes[0, 0].axvline(est_x_max, color='red', linestyle='--', alpha=0.5)
    axes[0, 0].axvspan(overlap_x_min, overlap_x_max, alpha=0.2, color='green', label='Overlap region')
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('Y')
    axes[0, 0].set_title('Full Trajectories with X-ranges')
    axes[0, 0].legend()
    axes[0, 0].invert_yaxis()
    
    # Plot 2: Option 1 comparison region
    axes[0, 1].plot(common_x_1, obs_y_1, 'b-', linewidth=2, label='Observed (interp)')
    axes[0, 1].plot(common_x_1, est_y_1, 'r-', linewidth=2, label='Estimated (interp)')
    axes[0, 1].fill_between(common_x_1, obs_y_1, est_y_1, alpha=0.3, color='purple')
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('Y')
    axes[0, 1].set_title(f'Option 1: Observed X-range\nRMSE = {rmse_option1:.4f}')
    axes[0, 1].legend()
    axes[0, 1].invert_yaxis()
    
    # Plot 3: Option 3 comparison region
    axes[1, 0].plot(common_x_3, obs_y_3, 'b-', linewidth=2, label='Observed (interp)')
    axes[1, 0].plot(common_x_3, est_y_3, 'r-', linewidth=2, label='Estimated (interp)')
    axes[1, 0].fill_between(common_x_3, obs_y_3, est_y_3, alpha=0.3, color='purple')
    axes[1, 0].set_xlabel('X')
    axes[1, 0].set_ylabel('Y')
    axes[1, 0].set_title(f'Option 3: Minimum Overlap\nRMSE = {rmse_option3:.4f}')
    axes[1, 0].legend()
    axes[1, 0].invert_yaxis()
    
    # Plot 4: Error along trajectory for both options
    error_1 = np.abs(obs_y_1 - est_y_1)
    error_3 = np.abs(obs_y_3 - est_y_3)
    axes[1, 1].plot(common_x_1, error_1, 'b-', linewidth=1, label=f'Option 1 (mean: {np.mean(error_1):.2f})')
    axes[1, 1].plot(common_x_3, error_3, 'g-', linewidth=1, label=f'Option 3 (mean: {np.mean(error_3):.2f})')
    axes[1, 1].set_xlabel('X')
    axes[1, 1].set_ylabel('|Y error|')
    axes[1, 1].set_title('Absolute Error Along Trajectory')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()
    
    return {
        'rmse_option1': rmse_option1,
        'rmse_option3': rmse_option3,
        'absolute_difference': difference,
        'percent_difference': percent_diff,
        'option1_range': (obs_x_min_1, obs_x_max_1),
        'option3_range': (overlap_x_min, overlap_x_max),
        'overlap_width': overlap_x_max - overlap_x_min
    }


def compare_interpolation_methods(observed_trajectory, estimated_trajectory):
    """
    Compare RMSE using linear interpolation vs cubic spline interpolation.
    
    This helps determine if the interpolation method significantly affects
    the RMSE calculation for trajectory comparison.
    
    Args:
        observed_trajectory: (N, 2) array with [x, y] columns
        estimated_trajectory: (M, 2) array with [x, y] columns
        
    Returns:
        dict with both RMSE values and the difference
    """
    N = 1000
    
    # Get the actual x-range of observed trajectory
    obs_x_min = np.min(observed_trajectory[:, 0])
    obs_x_max = np.max(observed_trajectory[:, 0])
    
    # Filter both trajectories to the same x-range
    obs_mask = (observed_trajectory[:, 0] >= obs_x_min) & (observed_trajectory[:, 0] <= obs_x_max)
    est_mask = (estimated_trajectory[:, 0] >= obs_x_min) & (estimated_trajectory[:, 0] <= obs_x_max)
    
    obs_filtered = observed_trajectory[obs_mask]
    est_filtered = estimated_trajectory[est_mask]
    
    # Ensure we have points to work with
    if len(obs_filtered) == 0 or len(est_filtered) == 0:
        return {'error': 'No points in range'}
    
    obs_x = obs_filtered[:, 0]
    obs_y = obs_filtered[:, 1]
    est_x = est_filtered[:, 0]
    est_y = est_filtered[:, 1]
    
    # Create common x-coordinates for interpolation
    common_x = np.linspace(obs_x_min, obs_x_max, N)
    
    # ============ LINEAR INTERPOLATION ============
    obs_y_linear = np.interp(common_x, obs_x, obs_y)
    est_y_linear = np.interp(common_x, est_x, est_y)
    
    obs_resampled_linear = np.column_stack([common_x, obs_y_linear])
    est_resampled_linear = np.column_stack([common_x, est_y_linear])
    
    rmse_linear = root_mean_squared_error(obs_resampled_linear, est_resampled_linear)
    
    # ============ CUBIC SPLINE INTERPOLATION ============
    # Need to handle potential duplicate x values for spline
    # Remove duplicates by keeping first occurrence
    _, obs_unique_idx = np.unique(obs_x, return_index=True)
    _, est_unique_idx = np.unique(est_x, return_index=True)
    
    obs_x_unique = obs_x[np.sort(obs_unique_idx)]
    obs_y_unique = obs_y[np.sort(obs_unique_idx)]
    est_x_unique = est_x[np.sort(est_unique_idx)]
    est_y_unique = est_y[np.sort(est_unique_idx)]
    
    # Create cubic splines
    obs_spline = CubicSpline(obs_x_unique, obs_y_unique)
    est_spline = CubicSpline(est_x_unique, est_y_unique)
    
    obs_y_cubic = obs_spline(common_x)
    est_y_cubic = est_spline(common_x)
    
    obs_resampled_cubic = np.column_stack([common_x, obs_y_cubic])
    est_resampled_cubic = np.column_stack([common_x, est_y_cubic])
    
    rmse_cubic = root_mean_squared_error(obs_resampled_cubic, est_resampled_cubic)
    
    # ============ RESULTS ============
    difference = abs(rmse_linear - rmse_cubic)
    percent_diff = (difference / rmse_linear * 100) if rmse_linear != 0 else 0
    
    print("=" * 60)
    print("INTERPOLATION METHOD COMPARISON")
    print("=" * 60)
    print(f"RMSE (Linear Interpolation):       {rmse_linear:.6f}")
    print(f"RMSE (Cubic Spline Interpolation): {rmse_cubic:.6f}")
    print("-" * 60)
    print(f"Absolute Difference:               {difference:.6f}")
    print(f"Percent Difference:                {percent_diff:.4f}%")
    print("-" * 60)
    
    if percent_diff < 1:
        assessment = "NEGLIGIBLE - Linear interpolation is fine"
    elif percent_diff < 5:
        assessment = "SMALL - Either method is acceptable"
    else:
        assessment = "SIGNIFICANT - Consider using cubic spline"
    
    print(f"Assessment: {assessment}")
    print("=" * 60)
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Both trajectories with interpolated points
    axes[0, 0].scatter(obs_filtered[:, 0], obs_filtered[:, 1], s=10, alpha=0.5, label='Observed (raw)')
    axes[0, 0].scatter(est_filtered[:, 0], est_filtered[:, 1], s=10, alpha=0.5, label='Estimated (raw)')
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('Y')
    axes[0, 0].set_title('Raw Trajectories')
    axes[0, 0].legend()
    axes[0, 0].invert_yaxis()
    
    # Plot 2: Linear vs Cubic for observed trajectory
    axes[0, 1].plot(common_x, obs_y_linear, 'b-', linewidth=1, label='Linear', alpha=0.7)
    axes[0, 1].plot(common_x, obs_y_cubic, 'r--', linewidth=1, label='Cubic', alpha=0.7)
    axes[0, 1].scatter(obs_x, obs_y, s=5, c='black', alpha=0.3, label='Raw points')
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('Y')
    axes[0, 1].set_title('Observed: Linear vs Cubic Interpolation')
    axes[0, 1].legend()
    axes[0, 1].invert_yaxis()
    
    # Plot 3: Difference between linear and cubic (observed)
    y_diff_obs = obs_y_linear - obs_y_cubic
    axes[1, 0].plot(common_x, y_diff_obs, 'g-', linewidth=1)
    axes[1, 0].axhline(0, color='black', linestyle='--', linewidth=0.5)
    axes[1, 0].set_xlabel('X')
    axes[1, 0].set_ylabel('Y difference (Linear - Cubic)')
    axes[1, 0].set_title(f'Interpolation Difference (Observed)\nMax diff: {np.max(np.abs(y_diff_obs)):.4f}')
    
    # Plot 4: RMSE contribution along trajectory
    error_linear = (obs_y_linear - est_y_linear) ** 2
    error_cubic = (obs_y_cubic - est_y_cubic) ** 2
    axes[1, 1].plot(common_x, error_linear, 'b-', linewidth=1, label='Linear', alpha=0.7)
    axes[1, 1].plot(common_x, error_cubic, 'r--', linewidth=1, label='Cubic', alpha=0.7)
    axes[1, 1].set_xlabel('X')
    axes[1, 1].set_ylabel('Squared Error')
    axes[1, 1].set_title('Squared Error Along Trajectory')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()
    
    return {
        'rmse_linear': rmse_linear,
        'rmse_cubic': rmse_cubic,
        'absolute_difference': difference,
        'percent_difference': percent_diff,
        'assessment': assessment
    }


def analyze_error_by_position(observed_trajectory, estimated_trajectory, num_segments=10):
    """
    Analyze RMSE contribution by position along the trajectory.
    
    This helps identify if errors are concentrated at the start, middle, or end
    of the trajectory, which can reveal systematic issues with the model.
    
    Args:
        observed_trajectory: (N, 2) array with [x, y] columns
        estimated_trajectory: (M, 2) array with [x, y] columns
        num_segments: Number of segments to divide the trajectory into
        
    Returns:
        dict with per-segment RMSE values and analysis
    """
    N = 1000
    
    # Get the actual x-range of observed trajectory
    obs_x_min = np.min(observed_trajectory[:, 0])
    obs_x_max = np.max(observed_trajectory[:, 0])
    
    # Filter both trajectories to the same x-range
    obs_mask = (observed_trajectory[:, 0] >= obs_x_min) & (observed_trajectory[:, 0] <= obs_x_max)
    est_mask = (estimated_trajectory[:, 0] >= obs_x_min) & (estimated_trajectory[:, 0] <= obs_x_max)
    
    obs_filtered = observed_trajectory[obs_mask]
    est_filtered = estimated_trajectory[est_mask]
    
    if len(obs_filtered) == 0 or len(est_filtered) == 0:
        return {'error': 'No points in range'}
    
    obs_x = obs_filtered[:, 0]
    est_x = est_filtered[:, 0]
    
    # Create common x-coordinates for interpolation
    common_x = np.linspace(obs_x_min, obs_x_max, N)
    
    # Interpolate y-coordinates
    obs_y = np.interp(common_x, obs_x, obs_filtered[:, 1])
    est_y = np.interp(common_x, est_x, est_filtered[:, 1])
    
    # Calculate per-point squared error
    squared_error = (obs_y - est_y) ** 2
    absolute_error = np.abs(obs_y - est_y)
    
    # Calculate overall RMSE
    overall_rmse = np.sqrt(np.mean(squared_error))
    
    # Divide into segments and calculate RMSE for each
    segment_size = N // num_segments
    segment_rmse = []
    segment_ranges = []
    segment_contribution = []
    
    for i in range(num_segments):
        start_idx = i * segment_size
        end_idx = (i + 1) * segment_size if i < num_segments - 1 else N
        
        segment_sq_error = squared_error[start_idx:end_idx]
        segment_rmse_val = np.sqrt(np.mean(segment_sq_error))
        segment_rmse.append(segment_rmse_val)
        
        # X range for this segment
        x_start = common_x[start_idx]
        x_end = common_x[end_idx - 1]
        segment_ranges.append((x_start, x_end))
        
        # Contribution to overall RMSE (as percentage)
        contribution = np.sum(segment_sq_error) / np.sum(squared_error) * 100
        segment_contribution.append(contribution)
    
    # Find problematic segments
    mean_segment_rmse = np.mean(segment_rmse)
    std_segment_rmse = np.std(segment_rmse)
    
    # Identify segments with high error (> 1.5 std above mean)
    high_error_threshold = mean_segment_rmse + 1.5 * std_segment_rmse
    high_error_segments = [i for i, rmse in enumerate(segment_rmse) if rmse > high_error_threshold]
    
    # Print results
    print("=" * 80)
    print("ERROR ANALYSIS BY POSITION")
    print("=" * 80)
    print(f"\nOverall RMSE: {overall_rmse:.4f}")
    print(f"X range: [{obs_x_min:.1f}, {obs_x_max:.1f}] (width: {obs_x_max - obs_x_min:.1f})")
    print("-" * 80)
    print(f"\n{'Segment':<10} {'X Range':<20} {'RMSE':<12} {'Contribution':<15} {'Status':<10}")
    print("-" * 80)
    
    for i in range(num_segments):
        x_start, x_end = segment_ranges[i]
        rmse_val = segment_rmse[i]
        contrib = segment_contribution[i]
        
        # Determine status
        if rmse_val > high_error_threshold:
            status = "⚠️ HIGH"
        elif rmse_val < mean_segment_rmse - std_segment_rmse:
            status = "✓ LOW"
        else:
            status = "NORMAL"
        
        position_label = ""
        if i == 0:
            position_label = " (START)"
        elif i == num_segments - 1:
            position_label = " (END)"
        
        print(f"{i+1:<10} [{x_start:>6.1f}, {x_end:>6.1f}]{position_label:<8} {rmse_val:<12.4f} {contrib:<14.1f}% {status:<10}")
    
    print("-" * 80)
    print(f"\nSegment RMSE Statistics:")
    print(f"  Mean:  {mean_segment_rmse:.4f}")
    print(f"  Std:   {std_segment_rmse:.4f}")
    print(f"  Min:   {np.min(segment_rmse):.4f} (Segment {np.argmin(segment_rmse) + 1})")
    print(f"  Max:   {np.max(segment_rmse):.4f} (Segment {np.argmax(segment_rmse) + 1})")
    
    # Analysis summary
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    
    start_rmse = segment_rmse[0]
    end_rmse = segment_rmse[-1]
    middle_rmse = np.mean(segment_rmse[1:-1]) if num_segments > 2 else segment_rmse[0]
    
    if start_rmse > high_error_threshold:
        print("⚠️  HIGH ERROR AT START: Initial conditions may be incorrect")
        print("    - Check launch position, velocity, or angle estimation")
    
    if end_rmse > high_error_threshold:
        print("⚠️  HIGH ERROR AT END: Model diverges over time")
        print("    - Physics parameters (gravity, drag) may be inaccurate")
        print("    - Or trajectory truncation point is inconsistent")
    
    if len(high_error_segments) == 0:
        print("✓  Error is evenly distributed - no systematic issues detected")
    
    # Trend analysis
    first_half_rmse = np.mean(segment_rmse[:num_segments//2])
    second_half_rmse = np.mean(segment_rmse[num_segments//2:])
    
    if second_half_rmse > first_half_rmse * 1.5:
        print(f"\n📈 ERROR GROWS over trajectory: {first_half_rmse:.4f} → {second_half_rmse:.4f}")
        print("   This suggests cumulative error (physics model drift)")
    elif first_half_rmse > second_half_rmse * 1.5:
        print(f"\n📉 ERROR DECREASES over trajectory: {first_half_rmse:.4f} → {second_half_rmse:.4f}")
        print("   This is unusual - check initial conditions")
    else:
        print(f"\n→  Error is relatively stable across trajectory")
    
    print("=" * 80)
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Trajectories
    axes[0, 0].plot(common_x, obs_y, 'b-', linewidth=2, label='Observed')
    axes[0, 0].plot(common_x, est_y, 'r-', linewidth=2, label='Estimated')
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('Y')
    axes[0, 0].set_title('Trajectories Comparison')
    axes[0, 0].legend()
    axes[0, 0].invert_yaxis()
    
    # Add segment boundaries
    for i, (x_start, x_end) in enumerate(segment_ranges):
        if i > 0:
            axes[0, 0].axvline(x_start, color='gray', linestyle=':', alpha=0.5)
    
    # Plot 2: Error along trajectory
    axes[0, 1].plot(common_x, absolute_error, 'purple', linewidth=1)
    axes[0, 1].fill_between(common_x, 0, absolute_error, alpha=0.3, color='purple')
    axes[0, 1].axhline(np.mean(absolute_error), color='red', linestyle='--', label=f'Mean: {np.mean(absolute_error):.2f}')
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('|Y error|')
    axes[0, 1].set_title('Absolute Error Along Trajectory')
    axes[0, 1].legend()
    
    # Plot 3: RMSE per segment (bar chart)
    segment_labels = [f'{i+1}' for i in range(num_segments)]
    colors = ['red' if i in high_error_segments else 'steelblue' for i in range(num_segments)]
    bars = axes[1, 0].bar(segment_labels, segment_rmse, color=colors, edgecolor='black')
    axes[1, 0].axhline(mean_segment_rmse, color='green', linestyle='--', label=f'Mean: {mean_segment_rmse:.2f}')
    axes[1, 0].axhline(high_error_threshold, color='red', linestyle=':', label=f'High threshold: {high_error_threshold:.2f}')
    axes[1, 0].set_xlabel('Segment')
    axes[1, 0].set_ylabel('RMSE')
    axes[1, 0].set_title('RMSE by Segment')
    axes[1, 0].legend()
    
    # Add labels for start/end
    axes[1, 0].text(0, segment_rmse[0] + 0.5, 'START', ha='center', fontsize=8)
    axes[1, 0].text(num_segments-1, segment_rmse[-1] + 0.5, 'END', ha='center', fontsize=8)
    
    # Plot 4: Contribution percentage (pie chart style as horizontal bar)
    axes[1, 1].barh(segment_labels, segment_contribution, color=colors, edgecolor='black')
    axes[1, 1].set_xlabel('Contribution to Total Error (%)')
    axes[1, 1].set_ylabel('Segment')
    axes[1, 1].set_title('Error Contribution by Segment')
    axes[1, 1].axvline(100/num_segments, color='green', linestyle='--', label=f'Even distribution: {100/num_segments:.1f}%')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()
    
    return {
        'overall_rmse': overall_rmse,
        'segment_rmse': segment_rmse,
        'segment_ranges': segment_ranges,
        'segment_contribution': segment_contribution,
        'high_error_segments': high_error_segments,
        'mean_segment_rmse': mean_segment_rmse,
        'std_segment_rmse': std_segment_rmse,
        'start_rmse': start_rmse,
        'end_rmse': end_rmse,
        'common_x': common_x,
        'absolute_error': absolute_error
    }


def analyze_launch_angle(observed_trajectory, commanded_angle_deg=None, num_initial_frames=20):
    """
    Analyze the instantaneous angle along the trajectory to detect
    slingshot effects or angle discrepancies.
    
    Args:
        observed_trajectory: (N, 2) array with [x, y] columns
        commanded_angle_deg: The angle you told the model to shoot (optional)
        num_initial_frames: Number of initial frames to focus on
        
    Returns:
        dict with angle analysis
    """
    x = observed_trajectory[:, 0]
    y = observed_trajectory[:, 1]
    
    # Calculate velocity components (finite differences)
    dx = np.diff(x)
    dy = np.diff(y)
    
    # Calculate instantaneous angle (in degrees)
    # Note: y-axis is inverted in screen coordinates, so we negate dy
    angles_rad = np.arctan2(-dy, dx)  # negative dy because screen y increases downward
    angles_deg = np.degrees(angles_rad)
    
    # Calculate velocity magnitude at each point
    velocity_mag = np.sqrt(dx**2 + dy**2)
    
    # Focus on initial frames
    initial_angles = angles_deg[:num_initial_frames]
    initial_velocities = velocity_mag[:num_initial_frames]
    
    # Statistics
    first_angle = angles_deg[0]
    mean_initial_angle = np.mean(initial_angles[:5])  # First 5 frames
    angle_after_launch = np.mean(angles_deg[5:15]) if len(angles_deg) > 15 else np.mean(angles_deg[5:])
    
    # Detect angle change rate in initial frames
    angle_changes = np.diff(initial_angles)
    
    # Expected angle change due to gravity (rough estimate)
    # For constant g: dθ/dt ≈ -g·cos²(θ)/v
    # With typical values, expect ~0.5-2 deg per frame
    
    # Detect anomalies
    large_angle_changes = np.where(np.abs(angle_changes) > 5)[0]  # > 5 deg change
    
    # Print results
    print("=" * 70)
    print("LAUNCH ANGLE ANALYSIS")
    print("=" * 70)
    
    if commanded_angle_deg is not None:
        print(f"\nCommanded angle:        {commanded_angle_deg:.2f}°")
        print(f"Actual first frame:     {first_angle:.2f}°")
        print(f"Discrepancy:            {first_angle - commanded_angle_deg:.2f}°")
    else:
        print(f"\nFirst frame angle:      {first_angle:.2f}°")
    
    print(f"\nMean angle (frames 1-5):   {mean_initial_angle:.2f}°")
    print(f"Mean angle (frames 6-15):  {angle_after_launch:.2f}°")
    print(f"Change after launch:       {angle_after_launch - mean_initial_angle:.2f}°")
    
    print("-" * 70)
    print(f"\nINITIAL FRAMES ANALYSIS (first {num_initial_frames} frames):")
    print(f"{'Frame':<8} {'Angle (°)':<12} {'Δ Angle (°)':<14} {'Velocity':<12} {'Status':<15}")
    print("-" * 70)
    
    for i in range(min(num_initial_frames, len(angles_deg))):
        angle = angles_deg[i]
        vel = velocity_mag[i]
        
        if i == 0:
            delta = "-"
            status = "LAUNCH"
        else:
            delta_val = angles_deg[i] - angles_deg[i-1]
            delta = f"{delta_val:+.2f}"
            
            if abs(delta_val) > 5:
                status = "⚠️ LARGE CHANGE"
            elif abs(delta_val) > 3:
                status = "Notable"
            else:
                status = "Normal"
        
        print(f"{i+1:<8} {angle:<12.2f} {delta:<14} {vel:<12.2f} {status:<15}")
    
    print("-" * 70)
    
    if len(large_angle_changes) > 0:
        print(f"\n⚠️  ANOMALIES DETECTED at frames: {large_angle_changes + 1}")
        print("    Large angle changes (>5°) may indicate slingshot effect")
    else:
        print("\n✓  No large angle discontinuities detected in initial frames")
    
    # Check if angle stabilizes after initial frames
    late_angle_std = np.std(np.diff(angles_deg[10:])) if len(angles_deg) > 20 else np.std(np.diff(angles_deg))
    early_angle_std = np.std(angle_changes[:10]) if len(angle_changes) > 10 else np.std(angle_changes)
    
    print(f"\nAngle change variability:")
    print(f"  Early (frames 1-10):  std = {early_angle_std:.3f}°")
    print(f"  Late (frames 10+):    std = {late_angle_std:.3f}°")
    
    if early_angle_std > late_angle_std * 2:
        print("  → Early trajectory is MORE UNSTABLE (possible slingshot effect)")
    else:
        print("  → Trajectory is stable throughout")
    
    print("=" * 70)
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Trajectory with initial frames highlighted
    axes[0, 0].plot(x, y, 'b-', linewidth=1, alpha=0.5, label='Full trajectory')
    axes[0, 0].scatter(x[:num_initial_frames], y[:num_initial_frames], c=range(num_initial_frames), 
                       cmap='Reds', s=50, zorder=5, label=f'First {num_initial_frames} frames')
    axes[0, 0].scatter(x[0], y[0], c='green', s=100, marker='*', zorder=10, label='Launch point')
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('Y')
    axes[0, 0].set_title('Trajectory with Initial Frames Highlighted')
    axes[0, 0].legend()
    axes[0, 0].invert_yaxis()
    
    # Plot 2: Angle over entire trajectory
    frame_numbers = np.arange(1, len(angles_deg) + 1)
    axes[0, 1].plot(frame_numbers, angles_deg, 'b-', linewidth=1)
    axes[0, 1].axvline(num_initial_frames, color='red', linestyle='--', alpha=0.5, label=f'Frame {num_initial_frames}')
    if commanded_angle_deg is not None:
        axes[0, 1].axhline(commanded_angle_deg, color='green', linestyle='--', label=f'Commanded: {commanded_angle_deg}°')
    axes[0, 1].set_xlabel('Frame')
    axes[0, 1].set_ylabel('Angle (degrees)')
    axes[0, 1].set_title('Instantaneous Angle Over Trajectory')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Angle change (derivative) - focus on initial frames
    axes[1, 0].bar(range(1, len(angle_changes[:num_initial_frames]) + 1), 
                   angle_changes[:num_initial_frames], color='steelblue', edgecolor='black')
    axes[1, 0].axhline(0, color='black', linewidth=0.5)
    axes[1, 0].axhline(5, color='red', linestyle=':', label='±5° threshold')
    axes[1, 0].axhline(-5, color='red', linestyle=':')
    axes[1, 0].set_xlabel('Frame')
    axes[1, 0].set_ylabel('Angle Change (degrees)')
    axes[1, 0].set_title(f'Frame-to-Frame Angle Change (First {num_initial_frames} frames)')
    axes[1, 0].legend()
    
    # Plot 4: Velocity magnitude over initial frames
    axes[1, 1].plot(range(1, len(initial_velocities) + 1), initial_velocities, 'g-o', markersize=4)
    axes[1, 1].set_xlabel('Frame')
    axes[1, 1].set_ylabel('Velocity Magnitude (pixels/frame)')
    axes[1, 1].set_title(f'Velocity Magnitude (First {num_initial_frames} frames)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'angles_deg': angles_deg,
        'first_angle': first_angle,
        'mean_initial_angle': mean_initial_angle,
        'angle_after_launch': angle_after_launch,
        'large_angle_changes': large_angle_changes,
        'velocity_magnitudes': velocity_mag,
        'commanded_angle': commanded_angle_deg,
        'discrepancy': first_angle - commanded_angle_deg if commanded_angle_deg else None
    }


def get_robust_launch_angle(observed_trajectory, num_points=10, commanded_angle_deg=None):
    """
    Calculate launch angle by fitting a line/parabola to the first few points,
    which is more robust to pixel quantization noise.
    
    Args:
        observed_trajectory: (N, 2) array with [x, y] columns
        num_points: Number of initial points to use for fitting
        commanded_angle_deg: The angle you commanded (optional, for comparison)
        
    Returns:
        dict with robust angle estimate
    """
    x = observed_trajectory[:num_points, 0]
    y = observed_trajectory[:num_points, 1]
    
    # Fit a line: y = mx + b
    coeffs_linear = np.polyfit(x, y, 1)
    slope_linear = coeffs_linear[0]
    
    # Convert slope to angle (accounting for inverted y-axis)
    # slope = dy/dx, angle = arctan(-slope) for screen coords where y increases downward
    angle_linear_rad = np.arctan(-slope_linear)
    angle_linear_deg = np.degrees(angle_linear_rad)
    
    # Quadratic fit for better accuracy (accounts for gravity)
    coeffs_quad = np.polyfit(x, y, 2)
    # Initial slope from quadratic: dy/dx at x[0] = 2*a*x[0] + b
    initial_slope_quad = 2 * coeffs_quad[0] * x[0] + coeffs_quad[1]
    angle_quad_rad = np.arctan(-initial_slope_quad)
    angle_quad_deg = np.degrees(angle_quad_rad)
    
    # Calculate frame-by-frame angles for comparison
    dx = np.diff(observed_trajectory[:num_points, 0])
    dy = np.diff(observed_trajectory[:num_points, 1])
    frame_angles = np.degrees(np.arctan2(-dy, dx))
    mean_frame_angle = np.mean(frame_angles)
    std_frame_angle = np.std(frame_angles)
    
    # Print results
    print("=" * 70)
    print("ROBUST LAUNCH ANGLE ESTIMATION")
    print("=" * 70)
    print(f"\nUsing first {num_points} points to reduce pixel quantization noise:")
    print("-" * 70)
    print(f"  Linear fit angle:       {angle_linear_deg:.2f}°")
    print(f"  Quadratic fit angle:    {angle_quad_deg:.2f}°  (accounts for gravity)")
    print(f"  Mean frame-by-frame:    {mean_frame_angle:.2f}° ± {std_frame_angle:.2f}°")
    print("-" * 70)
    
    if commanded_angle_deg is not None:
        print(f"\n  Commanded angle:        {commanded_angle_deg:.2f}°")
        print(f"  Discrepancy (linear):   {angle_linear_deg - commanded_angle_deg:+.2f}°")
        print(f"  Discrepancy (quad):     {angle_quad_deg - commanded_angle_deg:+.2f}°")
        
        # Check if it's a sign/coordinate issue
        if abs(angle_linear_deg + commanded_angle_deg) < 10:
            print(f"\n  ⚠️  SIGN MISMATCH DETECTED!")
            print(f"      Your angle conventions may be inverted.")
            print(f"      Try using {-commanded_angle_deg:.2f}° or check coordinate system.")
    
    print("=" * 70)
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Trajectory with fits
    x_fit = np.linspace(x[0], x[-1], 100)
    y_linear_fit = np.polyval(coeffs_linear, x_fit)
    y_quad_fit = np.polyval(coeffs_quad, x_fit)
    
    axes[0].scatter(x, y, c='blue', s=50, zorder=5, label='Observed points')
    axes[0].plot(x_fit, y_linear_fit, 'r-', linewidth=2, label=f'Linear fit ({angle_linear_deg:.1f}°)')
    axes[0].plot(x_fit, y_quad_fit, 'g--', linewidth=2, label=f'Quadratic fit ({angle_quad_deg:.1f}°)')
    
    # Draw angle indicators from first point
    arrow_len = (x[-1] - x[0]) * 0.3
    # Linear angle arrow
    axes[0].annotate('', xy=(x[0] + arrow_len * np.cos(np.radians(angle_linear_deg)), 
                             y[0] - arrow_len * np.sin(np.radians(angle_linear_deg))),
                     xytext=(x[0], y[0]),
                     arrowprops=dict(arrowstyle='->', color='red', lw=2))
    
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].set_title(f'First {num_points} Points with Fitted Lines')
    axes[0].legend()
    axes[0].invert_yaxis()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Frame-by-frame angles with fitted value
    axes[1].bar(range(1, len(frame_angles) + 1), frame_angles, color='steelblue', 
                edgecolor='black', alpha=0.7, label='Frame-by-frame')
    axes[1].axhline(angle_linear_deg, color='red', linestyle='-', linewidth=2, 
                    label=f'Linear fit: {angle_linear_deg:.1f}°')
    axes[1].axhline(angle_quad_deg, color='green', linestyle='--', linewidth=2, 
                    label=f'Quadratic fit: {angle_quad_deg:.1f}°')
    if commanded_angle_deg is not None:
        axes[1].axhline(commanded_angle_deg, color='orange', linestyle=':', linewidth=2, 
                        label=f'Commanded: {commanded_angle_deg:.1f}°')
    axes[1].set_xlabel('Frame')
    axes[1].set_ylabel('Angle (degrees)')
    axes[1].set_title('Frame-by-Frame Angles vs Fitted Estimates')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'linear_angle': angle_linear_deg,
        'quadratic_angle': angle_quad_deg,
        'mean_frame_angle': mean_frame_angle,
        'std_frame_angle': std_frame_angle,
        'linear_coeffs': coeffs_linear,
        'quadratic_coeffs': coeffs_quad,
        'commanded_angle': commanded_angle_deg,
        'discrepancy_linear': angle_linear_deg - commanded_angle_deg if commanded_angle_deg else None,
        'discrepancy_quad': angle_quad_deg - commanded_angle_deg if commanded_angle_deg else None
    }
