from matplotlib import patches, pyplot as plt
from matplotlib.path import Path
import numpy as np
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error
from numpy.polynomial import Polynomial

from agents.pddl.trajectory_parser import groundtruth_trajectory_parser
from agents.pddl.pddl_files.events.event_conditions import is_hit, is_platform_collision, is_ground_collision


def get_object_visuallization(object_trajectory):
    object_name, trajectory = object_trajectory
    path_codes = [Path.MOVETO] + [Path.LINETO] * (len(trajectory) - 1)
    path = Path(trajectory, path_codes)
    colorMap = {
        "stone": 'gray',
        "bird": 'red',
        "pig": 'green'
    }
    color = 'black'
    for object, mapped_color in colorMap.items():
        if object.lower() in object_name.lower():
            color = mapped_color

    patch = patches.PathPatch(path, edgecolor=color, facecolor="none", lw=2)
    return patch


#
def visualize_trajectory(model, target_class, raw_trajectories):
    trajectories,_ = groundtruth_trajectory_parser(raw_trajectories, model, target_class)
    patches = list(map(get_object_visuallization, trajectories.items()))
    # invert y axis
    fig, ax = plt.subplots()
    for patch in patches:
        ax.add_patch(patch)
    ax.set_xlim(0, 480)
    ax.set_ylim(300, 640)
    plt.show()


def visualize_compare(observed_trajectory, estimated_trajectory, changed_trajectoty=None):
    plt.figure()
    plt.plot(observed_trajectory[:, 0], observed_trajectory[:, 1], marker='o', color='blue')
    plt.plot(estimated_trajectory[:, 0], estimated_trajectory[:, 1], marker='x', color='red')
    if np.all(changed_trajectoty!=None):
        plt.plot(changed_trajectoty[:, 0], changed_trajectoty[:, 1], marker='x', color='green')
    plt.axis('equal')  # Equal scaling for x and y axes
    plt.show()


def visualize_rmse(rmse_values, suggested_rmse_values=None):
    """
    Visualize RMSE values over time, optionally comparing with suggested RMSE values.
    
    Parameters:
    -----------
    rmse_values : list or np.ndarray
        Primary RMSE values to plot
    suggested_rmse_values : list or np.ndarray, optional
        Secondary RMSE values to compare against (shown in orange)
    """
    time_steps = list(range(len(rmse_values)))

    plt.figure(figsize=(8, 4))
    plt.plot(time_steps, rmse_values, marker='o', color="blue", label="RMSE")
    
    if suggested_rmse_values is not None:
        plt.plot(time_steps, suggested_rmse_values, marker='o', color="orange", label="Suggested RMSE")
        plt.legend()
    
    plt.title('RMSE Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('RMSE')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def visualize_rmse_vs_suggsted(rmse_values, suggested_rmse_values):
    """Deprecated: Use visualize_rmse(rmse_values, suggested_rmse_values) instead."""
    import warnings
    warnings.warn(
        "visualize_rmse_vs_suggsted is deprecated, use visualize_rmse(rmse_values, suggested_rmse_values) instead",
        DeprecationWarning,
        stacklevel=2
    )
    visualize_rmse(rmse_values, suggested_rmse_values)


def visuallize_wins_percentage(wins):
    # Compute cumulative win percentage
    cumulative_wins = np.cumsum(wins)
    games_played = np.arange(1, len(wins) + 1)
    win_percentage = (cumulative_wins / games_played) * 100

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(games_played, win_percentage, marker='o', linestyle='-')
    plt.ylim(-5, 105)
    plt.xlabel('Game Number')
    plt.ylabel('Win Percentage (%)')
    plt.title('Win Percentage Over Time')
    plt.grid(True)
    plt.xticks(games_played)

    plt.show()


def plot_errors(errors, aggragive_errors):
    # Generate main data
    i = list(range(1, len(errors) + 1))

    # Plot the main line
    plt.plot(i, errors, marker='o', linestyle='-', color='b', label="errors")
    plt.plot(i, aggragive_errors, marker='x', linestyle='--', label="aggragive_errors")

    plt.legend()

    # Show grid and plot
    plt.grid(True)
    plt.show()


def plot_score(score):
    # Generate main data
    i = list(range(1, len(score) + 1))

    # Plot the main line
    plt.plot(i, score, marker='o', linestyle='-', color='b', label="errors")

    plt.legend()

    # Show grid and plot
    plt.grid(True)
    plt.show()


def visualize_euclidean_errors(timestamps, errors, event_indexes=None, highlight_last_n=20, title="Euclidean Error Over Time"):
    """
    Visualize euclidean error at each timestamp.
    
    Parameters:
    -----------
    timestamps : np.ndarray
        Array of timestamps in seconds
    errors : np.ndarray
        Array of euclidean errors at each timestamp
    event_indexes : list, optional
        List of event frame indexes to mark on the plot
    highlight_last_n : int
        Number of last frames to highlight (default 20)
    title : str
        Plot title (default "Euclidean Error Over Time")
    """
    if len(timestamps) == 0 or len(errors) == 0:
        print("Warning: No data to visualize")
        return
    
    plt.figure(figsize=(12, 6))
    
    # Determine which points are in the last N frames
    if len(timestamps) > highlight_last_n:
        last_n_start_idx = len(timestamps) - highlight_last_n
        regular_mask = np.arange(len(timestamps)) < last_n_start_idx
        last_n_mask = np.arange(len(timestamps)) >= last_n_start_idx
        
        # Plot regular frames
        plt.plot(timestamps[regular_mask], errors[regular_mask], 
                marker='o', linestyle='-', color='blue', markersize=3, 
                label='Error', alpha=0.7)
        
        # Highlight last N frames
        plt.plot(timestamps[last_n_mask], errors[last_n_mask], 
                marker='o', linestyle='-', color='red', markersize=5, 
                label=f'Last {highlight_last_n} frames', linewidth=2)
    else:
        # If trajectory is shorter than highlight_last_n, plot all in red
        plt.plot(timestamps, errors, marker='o', linestyle='-', 
                color='red', markersize=5, label='Error', linewidth=2)
    
    # Mark event locations if provided
    if event_indexes is not None and len(event_indexes) > 0:
        # Convert event indexes to timestamps (assuming same frame_rate)
        frame_rate = timestamps[1] - timestamps[0] if len(timestamps) > 1 else 0.02
        event_times = np.array(event_indexes) * frame_rate
        # Only mark events that are within our timestamp range
        if len(timestamps) > 0:
            valid_mask = (event_times >= timestamps[0]) & (event_times <= timestamps[-1])
            valid_event_times = event_times[valid_mask]
            if len(valid_event_times) > 0:
                event_errors = np.interp(valid_event_times, timestamps, errors)
                plt.scatter(valid_event_times, event_errors, 
                           color='green', marker='x', s=100, linewidths=3,
                           label='Events', zorder=5)
    
    # Fit linear and quadratic models to analyze error growth pattern
    if len(timestamps) > 2:
        # Fit linear model: error = a*t + b
        linear_poly = Polynomial.fit(timestamps, errors, deg=1)
        linear_fit = linear_poly(timestamps)
        linear_r2 = 1 - np.sum((errors - linear_fit)**2) / np.sum((errors - np.mean(errors))**2)
        
        # Fit quadratic model: error = a*t² + b*t + c
        quadratic_poly = Polynomial.fit(timestamps, errors, deg=2)
        quadratic_fit = quadratic_poly(timestamps)
        quadratic_r2 = 1 - np.sum((errors - quadratic_fit)**2) / np.sum((errors - np.mean(errors))**2)
        
        # Plot fitted curves
        plt.plot(timestamps, linear_fit, '--', color='orange', alpha=0.7, 
                label=f'Linear fit (R²={linear_r2:.3f})', linewidth=2)
        plt.plot(timestamps, quadratic_fit, '--', color='purple', alpha=0.7, 
                label=f'Quadratic fit (R²={quadratic_r2:.3f})', linewidth=2)
        
        # Print analysis
        print(f"\nError Growth Analysis:")
        print(f"  Linear fit R²: {linear_r2:.4f}")
        print(f"  Quadratic fit R²: {quadratic_r2:.4f}")
        if abs(linear_r2 - quadratic_r2) < 0.01:
            print(f"  → Error growth is approximately LINEAR (velocity error dominates)")
        elif quadratic_r2 > linear_r2:
            print(f"  → Error growth is QUADRATIC (gravity error dominates)")
        else:
            print(f"  → Error growth is LINEAR (velocity error dominates)")
    
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Euclidean Error (pixels)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()


def visualize_starting_point_offset(observed_trajectory, estimated_trajectory, 
                                    show_first_n_points=10, frame_rate=0.02,
                                    pddl_bird_pos=None, pddl_ref_pos=None, angle=None):
    """
    Visualize starting point offset between observed and estimated trajectories.
    
    This helps diagnose if the first point of observed trajectory matches
    the starting point used in the estimated trajectory, and if PDDL bird
    position aligns with the actual starting positions.
    
    Parameters:
    -----------
    observed_trajectory : np.ndarray
        Observed trajectory array of shape (N, 2)
    estimated_trajectory : np.ndarray
        Estimated trajectory array of shape (M, 2)
    show_first_n_points : int
        Number of initial points to highlight and label (default: 10)
    frame_rate : float
        Frame rate in seconds (default: 0.02 for 50 fps)
    pddl_bird_pos : tuple or None
        PDDL bird starting position (x, y) AFTER pa-twang action is applied
        If None, will not be shown
    pddl_ref_pos : tuple or None
        PDDL reference point (x, y) BEFORE pa-twang action (initial position in problem file)
        If None, will not be shown
    angle : float or None
        Launch angle in degrees (for display purposes)
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # ========== Plot 1: Full Trajectory View ==========
    ax1 = axes[0]
    
    # Plot full trajectories
    ax1.plot(observed_trajectory[:, 0], observed_trajectory[:, 1], 
             'o-', color='blue', markersize=4, alpha=0.6, label='Observed', linewidth=1.5)
    ax1.plot(estimated_trajectory[:, 0], estimated_trajectory[:, 1], 
             'x-', color='red', markersize=4, alpha=0.6, label='Estimated', linewidth=1.5)
    
    # Highlight starting points
    obs_start = observed_trajectory[0]
    est_start = estimated_trajectory[0]
    
    ax1.scatter(obs_start[0], obs_start[1], s=200, color='blue', 
               marker='*', edgecolors='black', linewidths=2, 
               label='Observed Start', zorder=10)
    ax1.scatter(est_start[0], est_start[1], s=200, color='red', 
               marker='*', edgecolors='black', linewidths=2, 
               label='Estimated Start', zorder=10)
    
    # Add PDDL reference point (before pa-twang) if provided
    if pddl_ref_pos is not None:
        ref_x, ref_y = pddl_ref_pos
        ax1.scatter(ref_x, ref_y, s=150, color='lightgreen', 
                   marker='s', edgecolors='black', linewidths=1.5, 
                   label='PDDL Ref Point (Before pa-twang)', zorder=9, alpha=0.7)
    
    # Add PDDL bird position (after pa-twang) if provided
    if pddl_bird_pos is not None:
        pddl_x, pddl_y = pddl_bird_pos
        ax1.scatter(pddl_x, pddl_y, s=200, color='green', 
                   marker='s', edgecolors='black', linewidths=2, 
                   label='PDDL Bird Pos (After pa-twang)', zorder=10)
    
    # Draw line between starting points
    offset_distance = np.sqrt(np.sum((obs_start - est_start)**2))
    ax1.plot([obs_start[0], est_start[0]], [obs_start[1], est_start[1]], 
             'k--', linewidth=2, alpha=0.5, label=f'Offset: {offset_distance:.2f} px')
    
    # Draw lines to PDDL points if provided
    if pddl_ref_pos is not None and pddl_bird_pos is not None:
        ref_x, ref_y = pddl_ref_pos
        pddl_x, pddl_y = pddl_bird_pos
        # Draw line from ref point to after pa-twang position
        ax1.plot([ref_x, pddl_x], [ref_y, pddl_y], 
                 'g:', linewidth=1, alpha=0.5, 
                 label='pa-twang offset', zorder=8)
    
    if pddl_bird_pos is not None:
        pddl_x, pddl_y = pddl_bird_pos
        pddl_to_obs = np.sqrt((obs_start[0] - pddl_x)**2 + (obs_start[1] - pddl_y)**2)
        ax1.plot([pddl_x, obs_start[0]], [pddl_y, obs_start[1]], 
                 'g--', linewidth=1.5, alpha=0.4, 
                 label=f'PDDL(after)→Obs: {pddl_to_obs:.2f} px')
    
    # Highlight first N points
    n_points = min(show_first_n_points, len(observed_trajectory), len(estimated_trajectory))
    ax1.scatter(observed_trajectory[:n_points, 0], observed_trajectory[:n_points, 1],
               s=100, color='blue', marker='o', edgecolors='cyan', linewidths=1.5, 
               alpha=0.8, zorder=5)
    ax1.scatter(estimated_trajectory[:n_points, 0], estimated_trajectory[:n_points, 1],
               s=100, color='red', marker='x', linewidths=2, alpha=0.8, zorder=5)
    
    # Label first few points
    for i in range(min(5, n_points)):
        ax1.annotate(f'O{i}', (observed_trajectory[i, 0], observed_trajectory[i, 1]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8, color='blue',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        ax1.annotate(f'E{i}', (estimated_trajectory[i, 0], estimated_trajectory[i, 1]),
                    xytext=(5, -15), textcoords='offset points', fontsize=8, color='red',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    ax1.set_xlabel('X (pixels)', fontsize=12)
    ax1.set_ylabel('Y (pixels)', fontsize=12)
    ax1.set_title('Full Trajectory Comparison\n(Starting Points Highlighted)', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # ========== Plot 2: Zoomed View of Starting Region ==========
    ax2 = axes[1]
    
    # Calculate zoom region (around starting points)
    margin = 50  # pixels margin
    x_min = min(obs_start[0], est_start[0]) - margin
    x_max = max(obs_start[0], est_start[0]) + margin
    y_min = min(obs_start[1], est_start[1]) - margin
    y_max = max(obs_start[1], est_start[1]) + margin
    
    # Include PDDL points in zoom region if provided
    if pddl_bird_pos is not None:
        pddl_x, pddl_y = pddl_bird_pos
        x_min = min(x_min, pddl_x) - margin
        x_max = max(x_max, pddl_x) + margin
        y_min = min(y_min, pddl_y) - margin
        y_max = max(y_max, pddl_y) + margin
    
    if pddl_ref_pos is not None:
        ref_x, ref_y = pddl_ref_pos
        x_min = min(x_min, ref_x) - margin
        x_max = max(x_max, ref_x) + margin
        y_min = min(y_min, ref_y) - margin
        y_max = max(y_max, ref_y) + margin
    
    # Expand to include first N points
    if len(observed_trajectory) > n_points:
        x_min = min(x_min, np.min(observed_trajectory[:n_points, 0])) - margin
        x_max = max(x_max, np.max(observed_trajectory[:n_points, 0])) + margin
        y_min = min(y_min, np.min(observed_trajectory[:n_points, 1])) - margin
        y_max = max(y_max, np.max(observed_trajectory[:n_points, 1])) + margin
    
    # Plot zoomed trajectories
    ax2.plot(observed_trajectory[:n_points, 0], observed_trajectory[:n_points, 1], 
             'o-', color='blue', markersize=6, label='Observed', linewidth=2)
    ax2.plot(estimated_trajectory[:n_points, 0], estimated_trajectory[:n_points, 1], 
             'x-', color='red', markersize=6, label='Estimated', linewidth=2)
    
    # Highlight starting points
    ax2.scatter(obs_start[0], obs_start[1], s=300, color='blue', 
               marker='*', edgecolors='black', linewidths=3, 
               label='Observed Start', zorder=10)
    ax2.scatter(est_start[0], est_start[1], s=300, color='red', 
               marker='*', edgecolors='black', linewidths=3, 
               label='Estimated Start', zorder=10)
    
    # Add PDDL reference point (before pa-twang) if provided
    if pddl_ref_pos is not None:
        ref_x, ref_y = pddl_ref_pos
        ax2.scatter(ref_x, ref_y, s=200, color='lightgreen', 
                   marker='s', edgecolors='black', linewidths=2, 
                   label='PDDL Ref Point (Before pa-twang)', zorder=9, alpha=0.7)
    
    # Add PDDL bird position (after pa-twang) if provided
    if pddl_bird_pos is not None:
        pddl_x, pddl_y = pddl_bird_pos
        ax2.scatter(pddl_x, pddl_y, s=300, color='green', 
                   marker='s', edgecolors='black', linewidths=3, 
                   label='PDDL Bird Pos (After pa-twang)', zorder=10)
        
        # Draw line from ref to after pa-twang
        if pddl_ref_pos is not None:
            ref_x, ref_y = pddl_ref_pos
            ax2.plot([ref_x, pddl_x], [ref_y, pddl_y], 
                     'g:', linewidth=2, alpha=0.6, 
                     label='pa-twang offset', zorder=8)
    
    # Draw offset line with distance annotation
    ax2.plot([obs_start[0], est_start[0]], [obs_start[1], est_start[1]], 
             'k--', linewidth=2, alpha=0.7)
    mid_x = (obs_start[0] + est_start[0]) / 2
    mid_y = (obs_start[1] + est_start[1]) / 2
    ax2.annotate(f'{offset_distance:.2f} px', 
                (mid_x, mid_y),
                xytext=(0, 0), textcoords='offset points',
                fontsize=11, fontweight='bold', color='black',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8),
                ha='center', va='center')
    
    # Label all visible points
    for i in range(n_points):
        ax2.annotate(f'O{i}', (observed_trajectory[i, 0], observed_trajectory[i, 1]),
                    xytext=(8, 8), textcoords='offset points', fontsize=9, color='blue',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8))
        ax2.annotate(f'E{i}', (estimated_trajectory[i, 0], estimated_trajectory[i, 1]),
                    xytext=(8, -18), textcoords='offset points', fontsize=9, color='red',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.8))
    
    # Calculate and show velocity vectors for first point
    if len(observed_trajectory) > 1:
        obs_vx = (observed_trajectory[1, 0] - observed_trajectory[0, 0]) / frame_rate
        obs_vy = (observed_trajectory[1, 1] - observed_trajectory[0, 1]) / frame_rate
        ax2.arrow(obs_start[0], obs_start[1], obs_vx * frame_rate * 5, obs_vy * frame_rate * 5,
                 head_width=5, head_length=3, fc='blue', ec='blue', alpha=0.6, linewidth=2)
    
    if len(estimated_trajectory) > 1:
        est_vx = (estimated_trajectory[1, 0] - estimated_trajectory[0, 0]) / frame_rate
        est_vy = (estimated_trajectory[1, 1] - estimated_trajectory[0, 1]) / frame_rate
        ax2.arrow(est_start[0], est_start[1], est_vx * frame_rate * 5, est_vy * frame_rate * 5,
                 head_width=5, head_length=3, fc='red', ec='red', alpha=0.6, linewidth=2)
    
    ax2.set_xlabel('X (pixels)', fontsize=12)
    ax2.set_ylabel('Y (pixels)', fontsize=12)
    ax2.set_title(f'Zoomed View: First {n_points} Points\n(Velocity Vectors Shown)', 
                 fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)
    ax2.axis('equal')
    
    plt.tight_layout()
    
    # Print diagnostic information
    print("\n" + "="*60)
    print("STARTING POINT OFFSET DIAGNOSTICS")
    print("="*60)
    print(f"Observed starting point: ({obs_start[0]:.2f}, {obs_start[1]:.2f})")
    print(f"Estimated starting point: ({est_start[0]:.2f}, {est_start[1]:.2f})")
    print(f"Offset distance: {offset_distance:.2f} pixels")
    print(f"Offset X: {obs_start[0] - est_start[0]:.2f} pixels")
    print(f"Offset Y: {obs_start[1] - est_start[1]:.2f} pixels")
    
    if pddl_ref_pos is not None:
        ref_x, ref_y = pddl_ref_pos
        print(f"\nPDDL Reference Point (Before pa-twang): ({ref_x:.2f}, {ref_y:.2f})")
        if angle is not None:
            print(f"  Angle: {angle:.2f}°")
    
    if pddl_bird_pos is not None:
        pddl_x, pddl_y = pddl_bird_pos
        pddl_to_obs = np.sqrt((obs_start[0] - pddl_x)**2 + (obs_start[1] - pddl_y)**2)
        pddl_to_est = np.sqrt((est_start[0] - pddl_x)**2 + (est_start[1] - pddl_y)**2)
        print(f"\nPDDL Bird Position (After pa-twang): ({pddl_x:.2f}, {pddl_y:.2f})")
        if pddl_ref_pos is not None:
            ref_x, ref_y = pddl_ref_pos
            patwang_offset = np.sqrt((pddl_x - ref_x)**2 + (pddl_y - ref_y)**2)
            print(f"  pa-twang offset from ref: {patwang_offset:.2f} pixels")
            print(f"    (x: {ref_x - pddl_x:.2f}, y: {ref_y - pddl_y:.2f})")
        print(f"  PDDL(after) → Observed: {pddl_to_obs:.2f} pixels")
        print(f"  PDDL(after) → Estimated: {pddl_to_est:.2f} pixels")
        if pddl_to_obs < 5:
            print(f"  ✓ PDDL bird position (after pa-twang) matches observed trajectory start!")
        else:
            print(f"  ⚠️  PDDL bird position (after pa-twang) differs from observed trajectory start!")
    
    if len(observed_trajectory) > 1 and len(estimated_trajectory) > 1:
        obs_vx = (observed_trajectory[10, 0] - observed_trajectory[0, 0]) / (frame_rate * 10)
        obs_vy = (observed_trajectory[10, 1] - observed_trajectory[0, 1]) / (frame_rate * 10)
        est_vx = (estimated_trajectory[10, 0] - estimated_trajectory[0, 0]) / (frame_rate * 10)
        est_vy = (estimated_trajectory[10, 1] - estimated_trajectory[0, 1]) / (frame_rate * 10)
        
        obs_vel = np.sqrt(obs_vx**2 + obs_vy**2)
        est_vel = np.sqrt(est_vx**2 + est_vy**2)
        obs_angle = np.arctan2(obs_vy, obs_vx) * 180 / np.pi
        est_angle = np.arctan2(est_vy, est_vx) * 180 / np.pi
        
        print(f"\nInitial Velocity Comparison:")
        print(f"  Observed: {obs_vel:.2f} px/s at {obs_angle:.2f}°")
        print(f"  Estimated: {est_vel:.2f} px/s at {est_angle:.2f}°")
        print(f"  Velocity difference: {abs(obs_vel - est_vel):.2f} px/s")
        print(f"  Angle difference: {abs(obs_angle - est_angle):.2f}°")
    
    print("="*60 + "\n")
    
    plt.show()


def full_trajectory_comparison(observed_trajectory, estimated_trajectory, frame_rate=0.02, n_frames=20):
    """
    Full trajectory comparison with 4 subplots:
    1. Full estimated vs observed trajectory
    2. First N frames comparison (zoomed)
    3. Last N frames comparison (zoomed)
    4. Accumulating RMSE error over trajectory
    
    Uses X-aligned interpolation (same method as calculate_rmse in metrics.py)
    to properly compare trajectories that may have different frame counts or
    starting positions.
    
    Parameters:
    -----------
    observed_trajectory : np.ndarray
        Observed trajectory array of shape (N, 2) with [x, y] columns
    estimated_trajectory : np.ndarray
        Estimated trajectory array of shape (M, 2) with [x, y] columns
    frame_rate : float
        Frame rate in seconds (default: 0.02 for 50 fps)
    n_frames : int
        Number of frames to show in zoomed views (default: 20)
    """
    N_INTERP = 1000  # Number of interpolation points (same as calculate_rmse)
    
    # Get X-range overlap between trajectories
    obs_x_min, obs_x_max = np.min(observed_trajectory[:, 0]), np.max(observed_trajectory[:, 0])
    est_x_min, est_x_max = np.min(estimated_trajectory[:, 0]), np.max(estimated_trajectory[:, 0])
    
    # Use observed trajectory's X-range (same as calculate_rmse)
    x_min, x_max = obs_x_min, obs_x_max
    
    # Filter trajectories to X-range
    obs_mask = (observed_trajectory[:, 0] >= x_min) & (observed_trajectory[:, 0] <= x_max)
    est_mask = (estimated_trajectory[:, 0] >= x_min) & (estimated_trajectory[:, 0] <= x_max)
    
    obs_filtered = observed_trajectory[obs_mask]
    est_filtered = estimated_trajectory[est_mask]
    
    # Create common X-coordinates for interpolation
    common_x = np.linspace(x_min, x_max, N_INTERP)
    
    # Interpolate Y-coordinates at common X positions
    obs_y_interp = np.interp(common_x, obs_filtered[:, 0], obs_filtered[:, 1])
    est_y_interp = np.interp(common_x, est_filtered[:, 0], est_filtered[:, 1])
    
    # Create aligned trajectory arrays
    obs = np.column_stack([common_x, obs_y_interp])
    est = np.column_stack([common_x, est_y_interp])
    min_len = N_INTERP
    
    # Calculate per-point Y errors (since X is aligned)
    y_errors = np.abs(obs_y_interp - est_y_interp)
    frame_errors = y_errors  # For X-aligned comparison, Y error is the main metric
    
    # Calculate accumulating RMSE
    accumulating_rmse = np.zeros(min_len)
    for i in range(1, min_len + 1):
        accumulating_rmse[i-1] = np.sqrt(np.mean(frame_errors[:i] ** 2))
    
    # Create figure with 4 subplots (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('Full Trajectory Comparison: Observed vs Estimated (X-Aligned)', fontsize=16, fontweight='bold')
    
    # ========== Plot 1: Full Trajectory ==========
    ax1 = axes[0, 0]
    # Plot original trajectories (not interpolated) for visual clarity
    ax1.plot(observed_trajectory[:, 0], observed_trajectory[:, 1], 'o-', color='blue', markersize=3, 
             linewidth=1.5, alpha=0.7, label='Observed')
    ax1.plot(estimated_trajectory[:, 0], estimated_trajectory[:, 1], 'x-', color='red', markersize=3, 
             linewidth=1.5, alpha=0.7, label='Estimated')
    
    # Mark start and end points
    ax1.scatter(observed_trajectory[0, 0], observed_trajectory[0, 1], s=150, color='blue', marker='*', 
                edgecolors='black', linewidths=2, zorder=10, label='Obs Start')
    ax1.scatter(estimated_trajectory[0, 0], estimated_trajectory[0, 1], s=150, color='red', marker='*', 
                edgecolors='black', linewidths=2, zorder=10, label='Est Start')
    ax1.scatter(observed_trajectory[-1, 0], observed_trajectory[-1, 1], s=100, color='blue', marker='s', 
                edgecolors='black', linewidths=2, zorder=10)
    ax1.scatter(estimated_trajectory[-1, 0], estimated_trajectory[-1, 1], s=100, color='red', marker='s', 
                edgecolors='black', linewidths=2, zorder=10)
    
    # Calculate overall RMSE (Y-error based, X-aligned)
    overall_rmse = np.sqrt(np.mean(frame_errors ** 2))
    ax1.set_xlabel('X (pixels)', fontsize=11)
    ax1.set_ylabel('Y (pixels)', fontsize=11)
    ax1.set_title(f'Full Trajectory\nOverall RMSE: {overall_rmse:.2f} pixels (X-aligned, {N_INTERP} points)', fontsize=12)
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # ========== Plot 2: First N Points (Zoomed) - X-Aligned ==========
    ax2 = axes[0, 1]
    
    # Calculate how many interpolation points correspond to n_frames worth of X range
    n_interp_first = min(n_frames * (N_INTERP // max(len(observed_trajectory), len(estimated_trajectory))), N_INTERP // 5)
    n_interp_first = max(n_interp_first, 50)  # At least 50 points
    
    # Use first portion of X range
    x_range_first = x_min + (x_max - x_min) * 0.2  # First 20% of X range
    first_mask = common_x <= x_range_first
    
    obs_first_x = common_x[first_mask]
    obs_first_y = obs_y_interp[first_mask]
    est_first_y = est_y_interp[first_mask]
    first_errors = frame_errors[first_mask]
    
    ax2.plot(obs_first_x, obs_first_y, 'o-', color='blue', markersize=4, 
             linewidth=2, label='Observed', markevery=max(1, len(obs_first_x)//20))
    ax2.plot(obs_first_x, est_first_y, 'x-', color='red', markersize=4, 
             linewidth=2, label='Estimated', markevery=max(1, len(obs_first_x)//20))
    
    # Highlight start points
    ax2.scatter(obs_first_x[0], obs_first_y[0], s=200, color='blue', marker='*', 
                edgecolors='black', linewidths=2, zorder=10, label='Obs Start')
    ax2.scatter(obs_first_x[0], est_first_y[0], s=200, color='red', marker='*', 
                edgecolors='black', linewidths=2, zorder=10, label='Est Start')
    
    first_rmse = np.sqrt(np.mean(first_errors ** 2)) if len(first_errors) > 0 else 0
    start_offset = abs(obs_first_y[0] - est_first_y[0]) if len(obs_first_y) > 0 else 0
    ax2.set_xlabel('X (pixels)', fontsize=11)
    ax2.set_ylabel('Y (pixels)', fontsize=11)
    ax2.set_title(f'First 20% of X-Range (Zoomed)\nRMSE: {first_rmse:.2f} px | Start Y-Offset: {start_offset:.2f} px', fontsize=12)
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # ========== Plot 3: Last 20% of X-Range (Zoomed) ==========
    ax3 = axes[1, 0]
    
    # Use last portion of X range
    x_range_last = x_max - (x_max - x_min) * 0.2  # Last 20% of X range
    last_mask = common_x >= x_range_last
    
    obs_last_x = common_x[last_mask]
    obs_last_y = obs_y_interp[last_mask]
    est_last_y = est_y_interp[last_mask]
    last_errors = frame_errors[last_mask]
    
    ax3.plot(obs_last_x, obs_last_y, 'o-', color='blue', markersize=4, 
             linewidth=2, label='Observed', markevery=max(1, len(obs_last_x)//20))
    ax3.plot(obs_last_x, est_last_y, 'x-', color='red', markersize=4, 
             linewidth=2, label='Estimated', markevery=max(1, len(obs_last_x)//20))
    
    # Highlight end points
    ax3.scatter(obs_last_x[-1], obs_last_y[-1], s=200, color='blue', marker='s', 
                edgecolors='black', linewidths=2, zorder=10, label='Obs End')
    ax3.scatter(obs_last_x[-1], est_last_y[-1], s=200, color='red', marker='s', 
                edgecolors='black', linewidths=2, zorder=10, label='Est End')
    
    last_rmse = np.sqrt(np.mean(last_errors ** 2)) if len(last_errors) > 0 else 0
    end_offset = abs(obs_last_y[-1] - est_last_y[-1]) if len(obs_last_y) > 0 else 0
    ax3.set_xlabel('X (pixels)', fontsize=11)
    ax3.set_ylabel('Y (pixels)', fontsize=11)
    ax3.set_title(f'Last 20% of X-Range (Zoomed)\nRMSE: {last_rmse:.2f} px | End Y-Offset: {end_offset:.2f} px', fontsize=12)
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')
    
    # ========== Plot 4: Accumulating RMSE Error ==========
    ax4 = axes[1, 1]
    
    # Use X position instead of frame number for X-aligned comparison
    ax4.plot(common_x, accumulating_rmse, 'b-', linewidth=2, label='Accumulating RMSE')
    ax4.fill_between(common_x, 0, accumulating_rmse, alpha=0.2, color='blue')
    
    # Also plot per-point Y error for reference
    ax4.plot(common_x, frame_errors, 'r-', linewidth=1, alpha=0.5, label='Per-point Y Error')
    
    # Mark 20% and 80% X positions
    x_20pct = x_min + (x_max - x_min) * 0.2
    x_80pct = x_min + (x_max - x_min) * 0.8
    ax4.axvline(x_20pct, color='green', linestyle='--', alpha=0.7, label=f'20% X ({x_20pct:.0f})')
    ax4.axvline(x_80pct, color='orange', linestyle='--', alpha=0.7, label=f'80% X ({x_80pct:.0f})')
    
    # Add horizontal line for final RMSE
    ax4.axhline(overall_rmse, color='purple', linestyle=':', linewidth=2, 
                label=f'Final RMSE: {overall_rmse:.2f}')
    
    # Fit linear to see error growth pattern
    if len(common_x) > 10:
        linear_fit = np.polyfit(common_x, accumulating_rmse, 1)
        linear_vals = np.polyval(linear_fit, common_x)
        ax4.plot(common_x, linear_vals, 'g--', linewidth=1.5, alpha=0.7, 
                label=f'Linear trend (slope: {linear_fit[0]:.3f})')
    
    ax4.set_xlabel('X Position (pixels)', fontsize=11)
    ax4.set_ylabel('RMSE (pixels)', fontsize=11)
    ax4.set_title(f'Accumulating RMSE Over X-Range\nError Growth: {accumulating_rmse[-1] - accumulating_rmse[0]:.2f} px', fontsize=12)
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(x_min, x_max)
    ax4.set_ylim(0, max(np.max(frame_errors), np.max(accumulating_rmse)) * 1.1)
    
    plt.tight_layout()
    
    # Print summary statistics
    print("\n" + "=" * 70)
    print("TRAJECTORY COMPARISON SUMMARY (X-Aligned Method)")
    print("=" * 70)
    print(f"Interpolation points: {N_INTERP}")
    print(f"X-range: [{x_min:.1f}, {x_max:.1f}] pixels (width: {x_max - x_min:.1f})")
    print("-" * 70)
    print(f"Overall RMSE:          {overall_rmse:.2f} pixels")
    print(f"First 20% RMSE:        {first_rmse:.2f} pixels")
    print(f"Last 20% RMSE:         {last_rmse:.2f} pixels")
    print("-" * 70)
    print(f"Start Y-offset:        {start_offset:.2f} pixels")
    print(f"End Y-offset:          {end_offset:.2f} pixels")
    print("-" * 70)
    print(f"Min Y-error:           {np.min(frame_errors):.2f} pixels (at X={common_x[np.argmin(frame_errors)]:.1f})")
    print(f"Max Y-error:           {np.max(frame_errors):.2f} pixels (at X={common_x[np.argmax(frame_errors)]:.1f})")
    print(f"Mean Y-error:          {np.mean(frame_errors):.2f} pixels")
    print("-" * 70)
    
    # Error trend analysis
    if last_rmse > first_rmse * 1.5:
        print("ERROR TREND: Error INCREASES over trajectory (model diverges)")
    elif first_rmse > last_rmse * 1.5:
        print("ERROR TREND: Error DECREASES over trajectory (initial conditions issue)")
    else:
        print("ERROR TREND: Error is relatively STABLE across trajectory")
    
    print("=" * 70 + "\n")
    
    plt.show()
    
    return {
        'overall_rmse': overall_rmse,
        'first_20pct_rmse': first_rmse,
        'last_20pct_rmse': last_rmse,
        'start_y_offset': start_offset,
        'end_y_offset': end_offset,
        'y_errors': frame_errors,
        'accumulating_rmse': accumulating_rmse,
        'common_x': common_x,
        'min_error': np.min(frame_errors),
        'max_error': np.max(frame_errors),
        'mean_error': np.mean(frame_errors),
        'x_range': (x_min, x_max)
    }


def debug_is_hit_last_frames(objects_features, groundtruth_objects, n_frames=20):
    """
    Debug visualization for is_hit detection in the last N frames.
    Shows bird/pig positions, distances, and whether is_hit was triggered.
    Also notifies if bird or pig is missing in any frame.
    
    Parameters:
    -----------
    objects_features : dict
        Dictionary of object features from getSegmentsEvents, e.g.:
        {"redBird_0": [{"x": ..., "y": ..., "v_x": ..., ...}, ...], "pig_0": [...]}
    groundtruth_objects : dict
        Dictionary of groundtruth object properties
    n_frames : int
        Number of last frames to analyze (default: 20)
    """
    # Build frames list (same structure as check_events)
    max_time = max(len(traj) for traj in objects_features.values())
    
    # Stretch trajectories to match max length
    features_copy = {k: list(v) for k, v in objects_features.items()}
    for obj, traj in features_copy.items():
        while len(traj) < max_time:
            traj.append(traj[-1])
    
    frames = []
    for frame_values in zip(*features_copy.values()):
        frame_dict = dict(zip(features_copy.keys(), frame_values))
        frames.append(frame_dict)
    
    total_frames = len(frames)
    start_idx = max(0, total_frames - n_frames)
    last_frames = list(range(start_idx, total_frames))
    
    print("\n" + "=" * 80)
    print(f"DEBUG: is_hit DETECTION - LAST {len(last_frames)} FRAMES (frames {start_idx} to {total_frames-1})")
    print("=" * 80)
    
    # Collect data for plotting
    frame_indices = []
    bird_positions = []
    pig_positions = []
    distances = []
    is_hit_results = []
    missing_objects = []
    
    r_bird = 3.5
    r_pig = 3.5
    collision_threshold = r_bird + r_pig  # = 7.0
    
    for i in last_frames:
        frame = frames[i]
        frame_indices.append(i)
        
        bird_exists = "redBird_0" in frame
        pig_exists = "pig_0" in frame
        
        missing = []
        if not bird_exists:
            missing.append("redBird_0")
        if not pig_exists:
            missing.append("pig_0")
        missing_objects.append(missing)
        
        if bird_exists and pig_exists:
            bird = frame["redBird_0"]
            pig = frame["pig_0"]
            
            x_bird, y_bird = bird["x"], bird["y"]
            x_pig, y_pig = pig["x"], pig["y"]
            
            bird_positions.append((x_bird, y_bird))
            pig_positions.append((x_pig, y_pig))
            
            dist = np.sqrt((x_bird - x_pig) ** 2 + (y_bird - y_pig) ** 2)
            distances.append(dist)
            
            # Call actual is_hit function
            hit_result = is_hit(frames, groundtruth_objects, i)
            is_hit_results.append(hit_result)
        else:
            bird_positions.append(None)
            pig_positions.append(None)
            distances.append(None)
            is_hit_results.append(False)
    
    # Print detailed frame-by-frame analysis
    print(f"\n{'Frame':<8} {'Bird Pos':<20} {'Pig Pos':<20} {'Distance':<12} {'Threshold':<12} {'is_hit':<10} {'Notes'}")
    print("-" * 100)
    
    for idx, (frame_i, bird_pos, pig_pos, dist, hit, missing) in enumerate(
        zip(frame_indices, bird_positions, pig_positions, distances, is_hit_results, missing_objects)
    ):
        if missing:
            missing_str = f"MISSING: {', '.join(missing)}"
            print(f"{frame_i:<8} {'N/A':<20} {'N/A':<20} {'N/A':<12} {collision_threshold:<12.2f} {str(hit):<10} {missing_str}")
        else:
            bird_str = f"({bird_pos[0]:.2f}, {bird_pos[1]:.2f})"
            pig_str = f"({pig_pos[0]:.2f}, {pig_pos[1]:.2f})"
            
            if dist <= collision_threshold:
                note = "*** COLLISION DETECTED ***"
            elif dist <= collision_threshold * 1.5:
                note = "Close!"
            else:
                note = ""
            
            print(f"{frame_i:<8} {bird_str:<20} {pig_str:<20} {dist:<12.2f} {collision_threshold:<12.2f} {str(hit):<10} {note}")
    
    # Summary
    print("\n" + "-" * 80)
    print("SUMMARY:")
    
    # Check for missing objects
    frames_with_missing = [(frame_indices[i], missing_objects[i]) for i in range(len(missing_objects)) if missing_objects[i]]
    if frames_with_missing:
        print(f"\n⚠️  MISSING OBJECTS in {len(frames_with_missing)} frames:")
        for frame_i, missing in frames_with_missing:
            print(f"   Frame {frame_i}: Missing {', '.join(missing)}")
    
    valid_distances = [d for d in distances if d is not None]
    if valid_distances:
        min_dist = min(valid_distances)
        min_dist_frame = frame_indices[distances.index(min_dist)]
        print(f"\nMinimum distance: {min_dist:.2f} at frame {min_dist_frame}")
        print(f"Collision threshold (r_bird + r_pig): {collision_threshold:.2f}")
        
        if min_dist <= collision_threshold:
            print(f"✓ Collision SHOULD be detected (min_dist <= threshold)")
        else:
            print(f"✗ No collision expected (min_dist > threshold)")
            print(f"  Gap to threshold: {min_dist - collision_threshold:.2f} units")
    
    hits_detected = sum(is_hit_results)
    print(f"\nis_hit returned True in {hits_detected} frames")
    
    if hits_detected == 0 and valid_distances and min(valid_distances) <= collision_threshold:
        print("\n⚠️  BUG: Distance suggests collision but is_hit returned False!")
        print("   Possible issues:")
        print("   1. Object names mismatch (expecting 'redBird_0' and 'pig_0')")
        print("   2. Radius values too small")
        print("   3. Frame indexing issue")
    
    print("=" * 80 + "\n")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'is_hit Debug: Last {len(last_frames)} Frames', fontsize=14, fontweight='bold')
    
    # Plot 1: Bird and Pig trajectories
    ax1 = axes[0, 0]
    valid_bird = [(i, p) for i, p in zip(frame_indices, bird_positions) if p is not None]
    valid_pig = [(i, p) for i, p in zip(frame_indices, pig_positions) if p is not None]
    
    if valid_bird:
        bird_x = [p[0] for _, p in valid_bird]
        bird_y = [p[1] for _, p in valid_bird]
        ax1.plot(bird_x, bird_y, 'ro-', markersize=8, linewidth=2, label='Bird')
        ax1.scatter(bird_x[-1], bird_y[-1], s=200, color='red', marker='*', 
                   edgecolors='black', zorder=10, label='Bird End')
        
        # Draw bird radius at last position
        circle_bird = plt.Circle((bird_x[-1], bird_y[-1]), r_bird, 
                                  fill=False, color='red', linestyle='--', linewidth=2)
        ax1.add_patch(circle_bird)
    
    if valid_pig:
        pig_x = [p[0] for _, p in valid_pig]
        pig_y = [p[1] for _, p in valid_pig]
        ax1.plot(pig_x, pig_y, 'go-', markersize=8, linewidth=2, label='Pig')
        ax1.scatter(pig_x[-1], pig_y[-1], s=200, color='green', marker='*', 
                   edgecolors='black', zorder=10, label='Pig End')
        
        # Draw pig radius at last position
        circle_pig = plt.Circle((pig_x[-1], pig_y[-1]), r_pig, 
                                 fill=False, color='green', linestyle='--', linewidth=2)
        ax1.add_patch(circle_pig)
    
    ax1.set_xlabel('X', fontsize=11)
    ax1.set_ylabel('Y', fontsize=11)
    ax1.set_title('Bird & Pig Positions (Last N Frames)', fontsize=12)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Plot 2: Distance over frames
    ax2 = axes[0, 1]
    valid_dist_data = [(i, d) for i, d in zip(frame_indices, distances) if d is not None]
    if valid_dist_data:
        dist_frames = [i for i, _ in valid_dist_data]
        dist_values = [d for _, d in valid_dist_data]
        
        ax2.plot(dist_frames, dist_values, 'b-o', markersize=6, linewidth=2, label='Distance')
        ax2.axhline(collision_threshold, color='red', linestyle='--', linewidth=2, 
                   label=f'Threshold ({collision_threshold:.1f})')
        ax2.fill_between(dist_frames, 0, collision_threshold, alpha=0.2, color='red', 
                        label='Collision Zone')
        
        # Mark frames where is_hit was True
        hit_frames = [frame_indices[i] for i in range(len(is_hit_results)) if is_hit_results[i]]
        hit_distances = [distances[i] for i in range(len(is_hit_results)) if is_hit_results[i] and distances[i] is not None]
        if hit_frames and hit_distances:
            ax2.scatter(hit_frames, hit_distances, s=150, color='yellow', marker='*', 
                       edgecolors='black', linewidths=2, zorder=10, label='is_hit=True')
    
    ax2.set_xlabel('Frame', fontsize=11)
    ax2.set_ylabel('Distance', fontsize=11)
    ax2.set_title('Bird-Pig Distance vs Collision Threshold', fontsize=12)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: is_hit result per frame
    ax3 = axes[1, 0]
    hit_colors = ['green' if h else 'red' for h in is_hit_results]
    # Mark missing frames differently
    for i, missing in enumerate(missing_objects):
        if missing:
            hit_colors[i] = 'gray'
    
    ax3.bar(frame_indices, [1] * len(frame_indices), color=hit_colors, edgecolor='black')
    ax3.set_xlabel('Frame', fontsize=11)
    ax3.set_ylabel('is_hit Result', fontsize=11)
    ax3.set_title('is_hit Detection per Frame (Green=True, Red=False, Gray=Missing)', fontsize=12)
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(['False', 'True'])
    ax3.grid(True, alpha=0.3, axis='x')
    
    # Plot 4: Zoomed view of closest approach
    ax4 = axes[1, 1]
    if valid_bird and valid_pig and valid_distances:
        min_dist_idx = distances.index(min(valid_distances))
        min_frame = frame_indices[min_dist_idx]
        
        # Get a few frames around minimum distance
        context = 3
        start = max(0, min_dist_idx - context)
        end = min(len(frame_indices), min_dist_idx + context + 1)
        
        for i in range(start, end):
            if bird_positions[i] is not None and pig_positions[i] is not None:
                alpha = 1.0 if i == min_dist_idx else 0.4
                bx, by = bird_positions[i]
                px, py = pig_positions[i]
                
                ax4.scatter(bx, by, s=100, color='red', alpha=alpha, marker='o')
                ax4.scatter(px, py, s=100, color='green', alpha=alpha, marker='s')
                
                if i == min_dist_idx:
                    # Draw collision radii
                    circle_b = plt.Circle((bx, by), r_bird, fill=False, color='red', 
                                         linestyle='--', linewidth=2)
                    circle_p = plt.Circle((px, py), r_pig, fill=False, color='green', 
                                         linestyle='--', linewidth=2)
                    ax4.add_patch(circle_b)
                    ax4.add_patch(circle_p)
                    
                    # Draw line between centers
                    ax4.plot([bx, px], [by, py], 'k--', linewidth=2, alpha=0.7)
                    
                    # Annotate distance
                    mid_x, mid_y = (bx + px) / 2, (by + py) / 2
                    ax4.annotate(f'd={min(valid_distances):.2f}', (mid_x, mid_y),
                               fontsize=10, fontweight='bold',
                               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
        ax4.set_xlabel('X', fontsize=11)
        ax4.set_ylabel('Y', fontsize=11)
        ax4.set_title(f'Closest Approach (Frame {min_frame})\nDistance: {min(valid_distances):.2f}, Threshold: {collision_threshold:.2f}', fontsize=12)
        ax4.legend(['Bird', 'Pig'], loc='best')
        ax4.grid(True, alpha=0.3)
        ax4.axis('equal')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'frame_indices': frame_indices,
        'distances': distances,
        'is_hit_results': is_hit_results,
        'missing_objects': missing_objects,
        'min_distance': min(valid_distances) if valid_distances else None,
        'collision_threshold': collision_threshold,
        'hits_detected': hits_detected
    }


def debug_all_events_full_trajectory(objects_features, groundtruth_objects):
    """
    Debug visualization for ALL event detection across the ENTIRE trajectory.
    Shows bird trajectory with all detected events marked, plus detailed analysis
    of platform collision detection.
    
    Parameters:
    -----------
    objects_features : dict
        Dictionary of object features from getSegmentsEvents
    groundtruth_objects : dict
        Dictionary of groundtruth object properties (dimensions)
    """
    import math
    
    # Build frames list (same structure as check_events)
    max_time = max(len(traj) for traj in objects_features.values())
    
    # Stretch trajectories to match max length
    features_copy = {k: list(v) for k, v in objects_features.items()}
    for obj, traj in features_copy.items():
        while len(traj) < max_time:
            traj.append(traj[-1])
    
    frames = []
    for frame_values in zip(*features_copy.values()):
        frame_dict = dict(zip(features_copy.keys(), frame_values))
        frames.append(frame_dict)
    
    total_frames = len(frames)
    
    print("\n" + "=" * 100)
    print(f"DEBUG: ALL EVENTS - FULL TRAJECTORY ({total_frames} frames)")
    print("=" * 100)
    
    # Print available objects
    print(f"\nAvailable objects in trajectory: {list(objects_features.keys())}")
    print(f"Groundtruth objects (dimensions): {list(groundtruth_objects.keys())}")
    
    # Check for platforms
    platform_names = [name for name in objects_features.keys() if "hill" in name.lower()]
    print(f"Detected platforms (containing 'hill'): {platform_names if platform_names else 'NONE'}")
    
    # Collect event data for all frames
    ground_collision_frames = []
    hit_frames = []
    platform_collision_frames = []
    
    bird_positions = []
    pig_positions = []
    platform_data = []  # List of (frame_idx, platform_name, platform_bounds, bird_dist)
    
    r_bird = 3.5  # Same as in is_platform_collision
    
    for i in range(total_frames):
        frame = frames[i]
        
        # Get bird position
        if "redBird_0" in frame:
            bird = frame["redBird_0"]
            bird_positions.append((bird["x"], bird["y"]))
        else:
            bird_positions.append(None)
        
        # Get pig position
        if "pig_0" in frame:
            pig = frame["pig_0"]
            pig_positions.append((pig["x"], pig["y"]))
        else:
            pig_positions.append(None)
        
        # Check ground collision
        if is_ground_collision(frames, groundtruth_objects, i):
            ground_collision_frames.append(i)
        
        # Check hit (bird-pig collision)
        if is_hit(frames, groundtruth_objects, i):
            hit_frames.append(i)
        
        # Check platform collision
        if is_platform_collision(frames, groundtruth_objects, i):
            platform_collision_frames.append(i)
        
        # Detailed platform analysis for each frame
        if "redBird_0" in frame:
            bird = frame["redBird_0"]
            x_b, y_b = bird["x"], bird["y"]
            
            for pname in platform_names:
                if pname in frame:
                    platform = frame[pname]
                    props = groundtruth_objects.get(pname, [0, 0])
                    
                    x_p, y_p = platform["x"], platform["y"]
                    w = props[0] if isinstance(props, (list, tuple)) and len(props) > 0 else 0
                    h = props[1] if isinstance(props, (list, tuple)) and len(props) > 1 else 0
                    
                    # Calculate bounds (assuming center)
                    left = x_p - w / 2
                    right = x_p + w / 2
                    top = y_p - h / 2
                    bottom = y_p + h / 2
                    
                    # Closest point on rect
                    closest_x = max(left, min(x_b, right))
                    closest_y = max(top, min(y_b, bottom))
                    
                    # Distance
                    dist = math.sqrt((x_b - closest_x)**2 + (y_b - closest_y)**2)
                    
                    platform_data.append({
                        'frame': i,
                        'platform': pname,
                        'bird_pos': (x_b, y_b),
                        'platform_center': (x_p, y_p),
                        'platform_size': (w, h),
                        'bounds': (left, right, top, bottom),
                        'closest_point': (closest_x, closest_y),
                        'distance': dist,
                        'collision': dist <= r_bird
                    })
    
    # Print event summary
    print(f"\n{'='*60}")
    print("EVENT DETECTION SUMMARY")
    print(f"{'='*60}")
    print(f"Ground collisions detected: {len(ground_collision_frames)} at frames {ground_collision_frames}")
    print(f"Bird-Pig hits detected:     {len(hit_frames)} at frames {hit_frames}")
    print(f"Platform collisions:        {len(platform_collision_frames)} at frames {platform_collision_frames}")
    
    # Detailed platform collision analysis
    if platform_data:
        print(f"\n{'='*60}")
        print("PLATFORM COLLISION DETAILS")
        print(f"{'='*60}")
        
        # Find frames where collision was detected
        collision_platform_data = [p for p in platform_data if p['collision']]
        
        if collision_platform_data:
            print(f"\nPlatform collisions detected in {len(collision_platform_data)} frame-platform pairs:")
            print(f"\n{'Frame':<8} {'Platform':<15} {'Bird Pos':<20} {'Platform Center':<20} {'Size (WxH)':<15} {'Distance':<10} {'Threshold':<10}")
            print("-" * 110)
            
            for p in collision_platform_data[:30]:  # Show first 30
                bird_str = f"({p['bird_pos'][0]:.1f}, {p['bird_pos'][1]:.1f})"
                plat_str = f"({p['platform_center'][0]:.1f}, {p['platform_center'][1]:.1f})"
                size_str = f"{p['platform_size'][0]:.1f} x {p['platform_size'][1]:.1f}"
                print(f"{p['frame']:<8} {p['platform']:<15} {bird_str:<20} {plat_str:<20} {size_str:<15} {p['distance']:<10.2f} {r_bird:<10.2f}")
            
            if len(collision_platform_data) > 30:
                print(f"... and {len(collision_platform_data) - 30} more")
        
        # Show minimum distances to platforms
        print(f"\n{'='*60}")
        print("MINIMUM DISTANCES TO PLATFORMS")
        print(f"{'='*60}")
        
        for pname in platform_names:
            pdata = [p for p in platform_data if p['platform'] == pname]
            if pdata:
                min_dist_entry = min(pdata, key=lambda x: x['distance'])
                print(f"\n{pname}:")
                print(f"  Min distance: {min_dist_entry['distance']:.2f} at frame {min_dist_entry['frame']}")
                print(f"  Platform center: ({min_dist_entry['platform_center'][0]:.1f}, {min_dist_entry['platform_center'][1]:.1f})")
                print(f"  Platform size: {min_dist_entry['platform_size'][0]:.1f} x {min_dist_entry['platform_size'][1]:.1f}")
                print(f"  Bird pos at min: ({min_dist_entry['bird_pos'][0]:.1f}, {min_dist_entry['bird_pos'][1]:.1f})")
                print(f"  Collision threshold: {r_bird}")
                if min_dist_entry['distance'] <= r_bird:
                    print(f"  *** COLLISION DETECTED ***")
                else:
                    print(f"  Gap to collision: {min_dist_entry['distance'] - r_bird:.2f}")
    else:
        print("\nNo platforms found for detailed analysis.")
    
    # Check for potential issues
    print(f"\n{'='*60}")
    print("POTENTIAL ISSUES")
    print(f"{'='*60}")
    
    issues = []
    
    if not platform_names:
        issues.append("- No platforms detected (looking for 'hill' in object names)")
    
    for pname in platform_names:
        if pname not in groundtruth_objects:
            issues.append(f"- Platform '{pname}' has no dimensions in groundtruth_objects")
        else:
            props = groundtruth_objects[pname]
            if not isinstance(props, (list, tuple)) or len(props) < 2:
                issues.append(f"- Platform '{pname}' has invalid dimensions format: {props}")
            elif props[0] == 0 or props[1] == 0:
                issues.append(f"- Platform '{pname}' has zero dimension: {props}")
    
    if "redBird_0" not in objects_features:
        issues.append("- Bird 'redBird_0' not found in trajectory")
    
    if issues:
        for issue in issues:
            print(issue)
    else:
        print("No obvious issues detected.")
    
    print("=" * 100 + "\n")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Event Detection Debug - Full Trajectory', fontsize=14, fontweight='bold')
    
    # Plot 1: Full trajectory with all events marked
    ax1 = axes[0, 0]
    
    # Plot bird trajectory
    valid_bird = [(i, p) for i, p in enumerate(bird_positions) if p is not None]
    if valid_bird:
        bird_x = [p[0] for _, p in valid_bird]
        bird_y = [p[1] for _, p in valid_bird]
        ax1.plot(bird_x, bird_y, 'b-', linewidth=1, alpha=0.5, label='Bird trajectory')
        ax1.scatter(bird_x[0], bird_y[0], s=150, color='blue', marker='*', 
                   edgecolors='black', zorder=10, label='Start')
    
    # Plot pig position (usually stationary)
    valid_pig = [(i, p) for i, p in enumerate(pig_positions) if p is not None]
    if valid_pig:
        # Just plot the first pig position as they're usually stationary
        pig_x, pig_y = valid_pig[0][1]
        ax1.scatter(pig_x, pig_y, s=200, color='green', marker='o', 
                   edgecolors='black', zorder=10, label='Pig')
    
    # Plot platforms
    for pname in platform_names:
        if pname in groundtruth_objects and pname in frames[0]:
            props = groundtruth_objects[pname]
            platform = frames[0][pname]
            x_p, y_p = platform["x"], platform["y"]
            w = props[0] if isinstance(props, (list, tuple)) and len(props) > 0 else 10
            h = props[1] if isinstance(props, (list, tuple)) and len(props) > 1 else 10
            
            rect = plt.Rectangle((x_p - w/2, y_p - h/2), w, h, 
                                 fill=True, facecolor='brown', edgecolor='black', 
                                 alpha=0.5, linewidth=2, label=f'Platform: {pname}')
            ax1.add_patch(rect)
    
    # Mark events
    if ground_collision_frames and valid_bird:
        gc_x = [bird_positions[i][0] for i in ground_collision_frames if bird_positions[i]]
        gc_y = [bird_positions[i][1] for i in ground_collision_frames if bird_positions[i]]
        ax1.scatter(gc_x, gc_y, s=150, color='orange', marker='v', 
                   edgecolors='black', linewidths=2, zorder=11, label='Ground collision')
    
    if hit_frames and valid_bird:
        hit_x = [bird_positions[i][0] for i in hit_frames if bird_positions[i]]
        hit_y = [bird_positions[i][1] for i in hit_frames if bird_positions[i]]
        ax1.scatter(hit_x, hit_y, s=150, color='red', marker='x', 
                   linewidths=3, zorder=11, label='Pig hit')
    
    if platform_collision_frames and valid_bird:
        pc_x = [bird_positions[i][0] for i in platform_collision_frames if bird_positions[i]]
        pc_y = [bird_positions[i][1] for i in platform_collision_frames if bird_positions[i]]
        ax1.scatter(pc_x, pc_y, s=150, color='purple', marker='s', 
                   edgecolors='black', linewidths=2, zorder=11, label='Platform collision')
    
    ax1.set_xlabel('X', fontsize=11)
    ax1.set_ylabel('Y', fontsize=11)
    ax1.set_title('Full Trajectory with Events', fontsize=12)
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Plot 2: Event timeline
    ax2 = axes[0, 1]
    
    # Create event bars
    event_types = ['Ground Collision', 'Pig Hit', 'Platform Collision']
    event_frames_list = [ground_collision_frames, hit_frames, platform_collision_frames]
    colors = ['orange', 'red', 'purple']
    
    for idx, (event_name, event_frames, color) in enumerate(zip(event_types, event_frames_list, colors)):
        if event_frames:
            ax2.scatter(event_frames, [idx] * len(event_frames), s=100, color=color, 
                       marker='|', linewidths=3, label=event_name)
    
    ax2.set_xlabel('Frame', fontsize=11)
    ax2.set_ylabel('Event Type', fontsize=11)
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(event_types)
    ax2.set_xlim(-5, total_frames + 5)
    ax2.set_title('Event Timeline', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.legend(loc='upper right', fontsize=9)
    
    # Plot 3: Distance to platforms over time
    ax3 = axes[1, 0]
    
    if platform_data:
        for pname in platform_names:
            pdata = [p for p in platform_data if p['platform'] == pname]
            if pdata:
                p_frames = [p['frame'] for p in pdata]
                p_dists = [p['distance'] for p in pdata]
                ax3.plot(p_frames, p_dists, '-', linewidth=2, label=pname)
        
        ax3.axhline(r_bird, color='red', linestyle='--', linewidth=2, label=f'Collision threshold ({r_bird})')
        ax3.fill_between([0, total_frames], 0, r_bird, alpha=0.2, color='red')
        
        # Mark collision frames
        if platform_collision_frames:
            collision_dists = []
            for f in platform_collision_frames:
                frame_data = [p for p in platform_data if p['frame'] == f and p['collision']]
                if frame_data:
                    collision_dists.append(frame_data[0]['distance'])
                else:
                    collision_dists.append(0)
            ax3.scatter(platform_collision_frames, collision_dists, s=100, color='purple', 
                       marker='*', zorder=10, label='Collision detected')
    
    ax3.set_xlabel('Frame', fontsize=11)
    ax3.set_ylabel('Distance to Platform', fontsize=11)
    ax3.set_title('Bird Distance to Platforms', fontsize=12)
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Zoomed view of platform collision area (if any)
    ax4 = axes[1, 1]
    
    if platform_collision_frames and platform_data:
        # Get first collision frame
        first_collision = platform_collision_frames[0]
        collision_data = [p for p in platform_data if p['frame'] == first_collision and p['collision']]
        
        if collision_data:
            cd = collision_data[0]
            
            # Plot platform
            left, right, top, bottom = cd['bounds']
            w = right - left
            h = bottom - top
            rect = plt.Rectangle((left, top), w, h, 
                                 fill=True, facecolor='brown', edgecolor='black', 
                                 alpha=0.5, linewidth=2, label='Platform')
            ax4.add_patch(rect)
            
            # Plot bird with radius
            bx, by = cd['bird_pos']
            ax4.scatter(bx, by, s=200, color='blue', marker='o', 
                       edgecolors='black', zorder=10, label='Bird')
            circle = plt.Circle((bx, by), r_bird, fill=False, color='blue', 
                               linestyle='--', linewidth=2, label=f'Bird radius ({r_bird})')
            ax4.add_patch(circle)
            
            # Plot closest point
            cx, cy = cd['closest_point']
            ax4.scatter(cx, cy, s=100, color='red', marker='x', 
                       linewidths=3, zorder=10, label='Closest point')
            
            # Draw line from bird to closest point
            ax4.plot([bx, cx], [by, cy], 'r--', linewidth=2, alpha=0.7)
            
            # Annotate distance
            mid_x, mid_y = (bx + cx) / 2, (by + cy) / 2
            ax4.annotate(f'd={cd["distance"]:.2f}', (mid_x, mid_y),
                        fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
            
            ax4.set_xlabel('X', fontsize=11)
            ax4.set_ylabel('Y', fontsize=11)
            ax4.set_title(f'Platform Collision Detail (Frame {first_collision})\n{cd["platform"]}', fontsize=12)
            ax4.legend(loc='best', fontsize=9)
            ax4.grid(True, alpha=0.3)
            ax4.axis('equal')
            
            # Set appropriate limits
            margin = max(w, h, r_bird * 3)
            ax4.set_xlim(left - margin, right + margin)
            ax4.set_ylim(top - margin, bottom + margin)
    else:
        ax4.text(0.5, 0.5, 'No platform collisions detected', 
                ha='center', va='center', fontsize=14, transform=ax4.transAxes)
        ax4.set_title('Platform Collision Detail', fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'total_frames': total_frames,
        'ground_collision_frames': ground_collision_frames,
        'hit_frames': hit_frames,
        'platform_collision_frames': platform_collision_frames,
        'platform_data': platform_data,
        'objects_in_trajectory': list(objects_features.keys()),
        'platforms_found': platform_names
    }


def visualize_ground_collision_detection(bird_trajectory, bird_features, event_indexes_by_event, 
                                          world_model=None, angle=None):
    """
    Visualize ground collision detection and compare predicted vs actual post-collision trajectory.
    
    Parameters:
    -----------
    bird_trajectory : np.ndarray or list
        Full bird trajectory [(x,y), ...]
    bird_features : list of dict
        Features at each frame: [{x, y, v_x, v_y, a_x, a_y}, ...]
    event_indexes_by_event : dict
        Event indexes by type, e.g., {"ground_collision": [45, 102], ...}
    world_model : WorldModel, optional
        World model with learned KB for prediction
    angle : float, optional
        Launch angle in degrees (for trajectory reconstruction)
    """
    from agents.pddl.trajectory_parser import construct_trajectory
    
    bird_trajectory = np.array(bird_trajectory)
    collision_frames = event_indexes_by_event.get("ground_collision", [])
    
    if len(collision_frames) == 0:
        print("No ground collisions detected!")
        return
    
    GROUND_LEVEL = 360  # Ground level in game coordinates
    n_collisions = len(collision_frames)
    
    # Transform Y: multiply by -1 so trajectory shows correctly
    # Ground level will be at y=0, bird flying high = positive y values
    traj_y_natural = abs((bird_trajectory[:, 1] - GROUND_LEVEL))
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Ground Collision Detection Analysis\n{n_collisions} collision(s) detected at frames: {collision_frames}', 
                 fontsize=14, fontweight='bold')
    
    # =========================================================================
    # Plot 1: Full trajectory with collision points marked
    # =========================================================================
    ax1 = axes[0, 0]
    
    # Plot full trajectory with transformed Y (natural coordinates)
    ax1.plot(bird_trajectory[:, 0], traj_y_natural, 'b-', linewidth=1.5, 
             alpha=0.7, label='Bird trajectory')
    
    # Mark collision points
    for i, coll_frame in enumerate(collision_frames):
        if coll_frame < len(bird_trajectory):
            coll_x = bird_trajectory[coll_frame, 0]
            coll_y_natural = traj_y_natural[coll_frame]
            coll_y_screen = bird_trajectory[coll_frame, 1]
            ax1.scatter(coll_x, coll_y_natural, s=200, c='red', marker='*', zorder=10,
                       edgecolors='black', linewidths=2,
                       label=f'Collision {i+1} (frame {coll_frame})' if i == 0 else f'Collision {i+1}')
            
            # Add annotation with original screen Y for reference
            ax1.annotate(f'Frame {coll_frame}\ny={coll_y_screen:.1f}', 
                        (coll_x, coll_y_natural), textcoords="offset points",
                        xytext=(10, 10), fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Draw ground level (at y=0 in natural coords)
    ax1.axhline(y=0, color='brown', linestyle='--', linewidth=2, label='Ground (y=0)')
    
    ax1.set_xlabel('X Position', fontsize=11)
    ax1.set_ylabel('Height above ground (pixels)', fontsize=11)
    ax1.set_title('Full Trajectory with Collision Points\n(Natural view: bird arcs UP then falls DOWN to ground)', fontsize=12)
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # =========================================================================
    # Plot 2: Y position over time with collision detection
    # =========================================================================
    ax2 = axes[0, 1]
    
    # Extract y values from features
    y_values = [f['y'] for f in bird_features]
    frames = list(range(len(y_values)))
    
    ax2.plot(frames, y_values, 'b-', linewidth=2, label='Y position (from ground)')
    ax2.axhline(y=0, color='brown', linestyle='--', linewidth=2, label='Ground level (y=0)')
    ax2.axhline(y=3, color='orange', linestyle=':', linewidth=1, label='Detection threshold (ε=3)')
    ax2.axhline(y=-3, color='orange', linestyle=':', linewidth=1)
    
    # Mark collision frames
    for coll_frame in collision_frames:
        if coll_frame < len(y_values):
            ax2.axvline(x=coll_frame, color='red', linestyle='-', linewidth=2, alpha=0.7)
            ax2.scatter(coll_frame, y_values[coll_frame], s=150, c='red', marker='o', zorder=10)
            
            # Show pre/post collision y values
            if coll_frame > 0:
                ax2.annotate(f'pre: {y_values[coll_frame-1]:.2f}\nat: {y_values[coll_frame]:.2f}\npost: {y_values[coll_frame+1]:.2f}' if coll_frame+1 < len(y_values) else f'pre: {y_values[coll_frame-1]:.2f}\nat: {y_values[coll_frame]:.2f}',
                            (coll_frame, y_values[coll_frame]), 
                            textcoords="offset points", xytext=(15, 0), fontsize=8,
                            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    ax2.set_xlabel('Frame', fontsize=11)
    ax2.set_ylabel('Y Position (from ground)', fontsize=11)
    ax2.set_title('Y Position Over Time\n(Collision: y[i-1]>ε AND y[i]≤ε AND v_y<0)', fontsize=12)
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # =========================================================================
    # Plot 3: Velocity components around collision
    # =========================================================================
    ax3 = axes[1, 0]
    
    vx_values = [f['v_x'] for f in bird_features]
    vy_values = [f['v_y'] for f in bird_features]
    
    ax3.plot(frames, vx_values, 'g-', linewidth=2, label='v_x (horizontal)')
    ax3.plot(frames, vy_values, 'purple', linewidth=2, label='v_y (vertical)')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Mark collision frames and show velocity changes
    for coll_frame in collision_frames:
        ax3.axvline(x=coll_frame, color='red', linestyle='-', linewidth=2, alpha=0.7)
        
        if coll_frame > 0 and coll_frame + 1 < len(vx_values):
            pre_vx, pre_vy = vx_values[coll_frame], vy_values[coll_frame]
            post_vx, post_vy = vx_values[coll_frame + 1], vy_values[coll_frame + 1]
            
            # Annotate velocity change
            ax3.annotate(f'Δv_x: {post_vx - pre_vx:.1f}\nΔv_y: {post_vy - pre_vy:.1f}',
                        (coll_frame, max(pre_vy, post_vy)), 
                        textcoords="offset points", xytext=(15, 10), fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8),
                        arrowprops=dict(arrowstyle='->', color='red'))
    
    ax3.set_xlabel('Frame', fontsize=11)
    ax3.set_ylabel('Velocity', fontsize=11)
    ax3.set_title('Velocity Components Over Time', fontsize=12)
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # =========================================================================
    # Plot 4: Pre vs Post collision state comparison + predicted trajectory
    # =========================================================================
    ax4 = axes[1, 1]
    
    if len(collision_frames) > 0:
        first_coll = collision_frames[0]
        
        # Show pre and post collision states
        if first_coll > 0 and first_coll + 1 < len(bird_features):
            pre_state = bird_features[first_coll]
            post_state = bird_features[first_coll + 1]
            
            # Create comparison table
            table_data = [
                ['Variable', 'Pre-Collision', 'Post-Collision', 'Change'],
                ['x', f'{pre_state["x"]:.2f}', f'{post_state["x"]:.2f}', f'{post_state["x"]-pre_state["x"]:+.2f}'],
                ['y', f'{pre_state["y"]:.2f}', f'{post_state["y"]:.2f}', f'{post_state["y"]-pre_state["y"]:+.2f}'],
                ['v_x', f'{pre_state["v_x"]:.2f}', f'{post_state["v_x"]:.2f}', f'{post_state["v_x"]-pre_state["v_x"]:+.2f}'],
                ['v_y', f'{pre_state["v_y"]:.2f}', f'{post_state["v_y"]:.2f}', f'{post_state["v_y"]-pre_state["v_y"]:+.2f}'],
            ]
            
            # Add damping ratios
            if abs(pre_state["v_x"]) > 0.1:
                vx_ratio = post_state["v_x"] / pre_state["v_x"]
                table_data.append(['v_x ratio', '-', '-', f'{vx_ratio:.3f}'])
            if abs(pre_state["v_y"]) > 0.1:
                vy_ratio = post_state["v_y"] / pre_state["v_y"]
                table_data.append(['v_y ratio', '-', '-', f'{vy_ratio:.3f}'])
            
            # Display as text
            ax4.axis('off')
            table = ax4.table(cellText=table_data[1:], colLabels=table_data[0],
                             loc='center', cellLoc='center',
                             colColours=['lightblue']*4)
            table.auto_set_font_size(False)
            table.set_fontsize(11)
            table.scale(1.2, 1.8)
            
            # Add title with physical interpretation
            vy_sign_change = "YES ✓" if (pre_state["v_y"] * post_state["v_y"]) < 0 else "NO ✗"
            ax4.set_title(f'Collision State Analysis (Frame {first_coll})\n'
                         f'v_y sign reversal: {vy_sign_change}', fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed collision info
    print("\n" + "="*70)
    print("GROUND COLLISION DETECTION DETAILS")
    print("="*70)
    for i, coll_frame in enumerate(collision_frames):
        if coll_frame > 0 and coll_frame + 1 < len(bird_features):
            pre = bird_features[coll_frame]
            post = bird_features[coll_frame + 1]
            print(f"\nCollision {i+1} at frame {coll_frame}:")
            print(f"  PRE:  x={pre['x']:.2f}, y={pre['y']:.2f}, v_x={pre['v_x']:.2f}, v_y={pre['v_y']:.2f}")
            print(f"  POST: x={post['x']:.2f}, y={post['y']:.2f}, v_x={post['v_x']:.2f}, v_y={post['v_y']:.2f}")
            if abs(pre['v_y']) > 0.1:
                print(f"  v_y damping ratio: {post['v_y']/pre['v_y']:.3f} (expected: negative, ~-0.5)")
            if abs(pre['v_x']) > 0.1:
                print(f"  v_x friction ratio: {post['v_x']/pre['v_x']:.3f} (expected: positive, ~0.8)")
    print("="*70)
    
    return {
        'collision_frames': collision_frames,
        'bird_features': bird_features,
        'bird_trajectory': bird_trajectory
    }


def visualize_post_collision_trajectory(bird_trajectory, bird_features, event_indexes_by_event,
                                         world_model, kb=None):
    """
    Deprecated: Use visualize_post_collision_trajectory_v2 instead.
    
    V2 improvements:
    - Uses v_x, v_y directly instead of angle/magnitude conversion (avoids precision loss)
    - Multi-frame velocity calculation to avoid quantization noise
    - Skips first 2 post-collision frames where bird may still be at ground level
    
    Parameters:
    -----------
    bird_trajectory : np.ndarray or list
        Full bird trajectory [(x,y), ...]
    bird_features : list of dict
        Features at each frame
    event_indexes_by_event : dict
        Event indexes by type
    world_model : WorldModel
        World model with physics parameters
    kb : dict, optional
        Knowledge base with learned collision models
    """
    import warnings
    warnings.warn(
        "visualize_post_collision_trajectory is deprecated, use visualize_post_collision_trajectory_v2 instead",
        DeprecationWarning,
        stacklevel=2
    )
    from agents.pddl.trajectory_parser import construct_trajectory
    from agents.pddl.pddl_files.world_model.params import Params
    
    bird_trajectory = np.array(bird_trajectory)
    collision_frames = event_indexes_by_event.get("ground_collision", [])
    
    if len(collision_frames) == 0:
        print("No ground collisions to analyze!")
        return
    
    first_coll = collision_frames[0]
    
    if first_coll + 1 >= len(bird_features):
        print("Not enough data after collision!")
        return
    
    # Get actual post-collision trajectory
    actual_post_traj = bird_trajectory[first_coll + 1:]
    
    # Get states with CORRECTED velocities (matching the learning code)
    FRAME_RATE = 0.02
    pre_features = bird_features[first_coll]
    post_features = bird_features[first_coll + 1]
    prev_features = bird_features[first_coll - 1] if first_coll > 0 else pre_features
    
    # PRE-COLLISION state: velocity BEFORE collision (backward difference)
    pre_state = pre_features.copy()
    pre_state['v_x'] = (pre_features['x'] - prev_features['x']) / FRAME_RATE
    pre_state['v_y'] = (pre_features['y'] - prev_features['y']) / FRAME_RATE
    
    # POST-COLLISION state: velocity AFTER collision (forward from collision point)
    post_state = post_features.copy()
    post_state['v_x'] = (post_features['x'] - pre_features['x']) / FRAME_RATE
    post_state['v_y'] = (post_features['y'] - pre_features['y']) / FRAME_RATE
    
    # Predict using learned model if KB available
    predicted_post_state = None
    if kb is not None and "collision" in kb:
        predicted_post_state = {}
        
        for var_name in ["v_x", "v_y", "y"]:
            model = kb["collision"]["variables"].get(var_name, {}).get("model")
            if model is not None:
                from sklearn.preprocessing import PolynomialFeatures
                X = np.array([[pre_state["x"], pre_state["y"], pre_state["v_x"], pre_state["v_y"]]])
                poly = PolynomialFeatures(degree=1, include_bias=False)
                X_poly = poly.fit_transform(X)
                predicted_post_state[var_name] = model.predict(X_poly)[0]
            else:
                predicted_post_state[var_name] = post_state[var_name]
        
        predicted_post_state["x"] = post_state["x"]  # x doesn't change much in collision
    
    # Construct predicted trajectory from post-collision state
    # Calculate angle from velocity
    import math
    post_vx = predicted_post_state["v_x"] if predicted_post_state else post_state["v_x"]
    post_vy = predicted_post_state["v_y"] if predicted_post_state else post_state["v_y"]
    post_angle = math.degrees(math.atan2(post_vy, post_vx))
    post_velocity = math.sqrt(post_vx**2 + post_vy**2)
    
    # Create a temporary world model with post-collision velocity
    from agents.pddl.pddl_files.world_model.world_model import WorldModel
    post_world_model = WorldModel({
        Params.gravity: world_model.hyperparams_values[Params.gravity],
        Params.velocity: post_velocity
    })
    
    # Get starting point (post-collision position)
    GROUND_LEVEL = 360
    start_x = post_state["x"]
    start_y = GROUND_LEVEL - post_state["y"]  # Convert back to screen coords
    
    # Construct predicted trajectory
    if len(actual_post_traj) > 0:
        limit = np.max(actual_post_traj[:, 0])
    else:
        limit = start_x + 200
    
    predicted_traj = construct_trajectory(
        [start_x, start_y],
        post_angle,
        post_world_model,
        limit,
        prt=False,
        integration_method='rk4'
    )
    
    # Also construct with learned model predictions if available
    learned_predicted_traj = None
    if predicted_post_state:
        learned_post_vy = predicted_post_state["v_y"]
        learned_post_vx = predicted_post_state["v_x"]
        learned_post_angle = math.degrees(math.atan2(learned_post_vy, learned_post_vx))
        learned_post_velocity = math.sqrt(learned_post_vx**2 + learned_post_vy**2)
        
        learned_world_model = WorldModel({
            Params.gravity: world_model.hyperparams_values[Params.gravity],
            Params.velocity: learned_post_velocity
        })
        
        learned_start_y = GROUND_LEVEL - predicted_post_state["y"]
        
        learned_predicted_traj = construct_trajectory(
            [start_x, learned_start_y],
            learned_post_angle,
            learned_world_model,
            limit,
            prt=False,
            integration_method='rk4'
        )
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Post-Collision Trajectory Analysis', fontsize=14, fontweight='bold')
    
    # Plot 1: Compare observed vs predicted trajectory (FOCUSED on collision area)
    ax1 = axes[0]
    
    # Transform Y: use absolute value so Y is always positive (height above ground)
    actual_post_y_natural = np.abs(actual_post_traj[:, 1] - GROUND_LEVEL)
    
    # Only show a small portion of pre-collision trajectory (last N frames before collision)
    n_pre_frames = min(20, first_coll)  # Show last 20 frames before collision
    pre_traj_focused = bird_trajectory[first_coll - n_pre_frames:first_coll + 1]
    pre_traj_y_natural = np.abs(pre_traj_focused[:, 1] - GROUND_LEVEL)
    
    # Plot focused pre-collision trajectory
    ax1.plot(pre_traj_focused[:, 0], pre_traj_y_natural, 'b-', linewidth=2, alpha=0.7, label='Pre-collision')
    
    # Plot post-collision OBSERVED trajectory (full)
    ax1.plot(actual_post_traj[:, 0], actual_post_y_natural, 'g-', linewidth=2.5, 
             label='Post-collision (OBSERVED)')
    
    # Plot PREDICTED trajectory from learned model (full)
    learned_y_natural = None
    if learned_predicted_traj is not None and len(learned_predicted_traj) > 0:
        learned_y_natural = np.abs(learned_predicted_traj[:, 1] - GROUND_LEVEL)
        ax1.plot(learned_predicted_traj[:, 0], learned_y_natural, 'r--', linewidth=2,
                 label='Post-collision (PREDICTED by model)')
    
    # Mark collision point prominently
    coll_point = bird_trajectory[first_coll]
    coll_y_natural = np.abs(coll_point[1] - GROUND_LEVEL)
    ax1.scatter(coll_point[0], coll_y_natural, s=300, c='red', marker='*', zorder=10,
               edgecolors='black', linewidths=2, label=f'Collision (frame {first_coll})')
    
    # Ground level at y=0 in natural coords
    ax1.axhline(y=0, color='brown', linestyle='--', linewidth=2, alpha=0.5, label='Ground')
    
    # Set axis limits: X focuses on collision area, Y shows full range (positive)
    x_min = coll_point[0] - 50  # 50 pixels before collision
    x_max = actual_post_traj[-1, 0] + 20
    if learned_predicted_traj is not None and len(learned_predicted_traj) > 0:
        x_max = max(x_max, learned_predicted_traj[-1, 0] + 20)
    
    # Y: from 0 (ground) to max height in trajectories
    y_min = -2  # Slightly below ground for visual clarity
    y_max = max(np.max(actual_post_y_natural), np.max(pre_traj_y_natural)) + 10
    if learned_y_natural is not None:
        y_max = max(y_max, np.max(learned_y_natural) + 10)
    
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    
    ax1.set_xlabel('X Position', fontsize=11)
    ax1.set_ylabel('Height above ground (pixels)', fontsize=11)
    ax1.set_title('Observed vs Predicted Trajectory\n(Green=Observed, Red=Model Prediction)', fontsize=12)
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: State comparison table with collision physics
    ax2 = axes[1]
    ax2.axis('off')
    
    # Calculate collision physics ratios (pre_state already defined above)
    vx_ratio = post_state['v_x'] / pre_state['v_x'] if abs(pre_state['v_x']) > 0.1 else 0
    vy_ratio = post_state['v_y'] / pre_state['v_y'] if abs(pre_state['v_y']) > 0.1 else 0
    
    table_data = [
        ['State', 'Pre-Coll', 'Actual Post', 'Learned Post', 'Diff', 'Ratio (post/pre)']
    ]
    
    for var in ['x', 'y', 'v_x', 'v_y']:
        pre_val = pre_state[var]
        actual_val = post_state[var]
        
        # Calculate ratio for velocities
        if var in ['v_x', 'v_y'] and abs(pre_val) > 0.1:
            ratio = f'{actual_val/pre_val:.2f}'
        else:
            ratio = '-'
        
        if predicted_post_state and var in predicted_post_state:
            learned_val = predicted_post_state[var]
            diff = learned_val - actual_val
            table_data.append([var, f'{pre_val:.2f}', f'{actual_val:.2f}', 
                              f'{learned_val:.2f}', f'{diff:+.2f}', ratio])
        else:
            table_data.append([var, f'{pre_val:.2f}', f'{actual_val:.2f}', 'N/A', 'N/A', ratio])
    
    table = ax2.table(cellText=table_data[1:], colLabels=table_data[0],
                     loc='upper center', cellLoc='center',
                     colColours=['lightblue']*6)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.1, 2.0)
    
    # Add physics interpretation text below table
    physics_text = f"""
Collision Physics Analysis:
- v_x ratio: {vx_ratio:.2f} (horizontal velocity retention)
- v_y ratio: {vy_ratio:.2f} (vertical velocity reversal/damping)

Interpretation:
- v_x ratio ~0.5 means 50% horizontal velocity lost (friction)
- v_y ratio ~-0.67 means bounce with 67% energy retention
  (negative = direction reversed)

Learning Status:
- Diff = 0: Model perfectly memorized this collision
- With more samples, model will learn general bounce physics
"""
    ax2.text(0.5, 0.25, physics_text, transform=ax2.transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
             family='monospace')
    
    ax2.set_title('State Comparison at Collision Point', fontsize=12, pad=20)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary to console
    print("\n" + "="*60)
    print("COLLISION LEARNING SUMMARY")
    print("="*60)
    print(f"Collision detected at frame: {first_coll}")
    print(f"\nPre-collision state:")
    print(f"  Position: ({pre_state['x']:.1f}, {pre_state['y']:.1f})")
    print(f"  Velocity: (v_x={pre_state['v_x']:.1f}, v_y={pre_state['v_y']:.1f})")
    print(f"\nActual post-collision state:")
    print(f"  Position: ({post_state['x']:.1f}, {post_state['y']:.1f})")
    print(f"  Velocity: (v_x={post_state['v_x']:.1f}, v_y={post_state['v_y']:.1f})")
    print(f"\nCollision physics:")
    print(f"  v_x retention: {vx_ratio:.2%}")
    print(f"  v_y reversal:  {vy_ratio:.2%}")
    if predicted_post_state:
        print(f"\nLearned model prediction error:")
        for var in ['v_x', 'v_y', 'y']:
            if var in predicted_post_state:
                diff = predicted_post_state[var] - post_state[var]
                print(f"  {var}: {diff:+.4f}")
    print("="*60)
    
    return {
        'actual_post_traj': actual_post_traj,
        'collision_frame': first_coll,
        'pre_state': pre_state,
        'post_state': post_state,
        'predicted_post_state': predicted_post_state,
        'vx_ratio': vx_ratio,
        'vy_ratio': vy_ratio
    }


def visualize_post_collision_trajectory_v2(bird_trajectory, bird_features, event_indexes_by_event,
                                           world_model, kb=None):
    """
    Compare actual post-collision trajectory with predicted trajectory using learned model.
    
    VERSION 2: Uses direct v_x, v_y values instead of angle/magnitude conversion.
    This avoids coordinate system mismatches and precision loss.
    
    Parameters:
    -----------
    bird_trajectory : np.ndarray or list
        Full bird trajectory [(x,y), ...]
    bird_features : list of dict
        Features at each frame
    event_indexes_by_event : dict
        Event indexes by type
    world_model : WorldModel
        World model with physics parameters
    kb : dict, optional
        Knowledge base with learned collision models
    """
    from agents.pddl.trajectory_parser import construct_trajectory_from_velocity
    from agents.pddl.pddl_files.world_model.params import Params
    import math
    
    bird_trajectory = np.array(bird_trajectory)
    collision_frames = event_indexes_by_event.get("ground_collision", [])
    
    if len(collision_frames) == 0:
        print("No ground collisions to analyze!")
        return
    
    first_coll = collision_frames[0]
    
    # Multi-frame velocity calculation settings (must match pddl_agent.py!)
    FRAME_RATE = 0.02
    VELOCITY_FRAMES = 3  # Use 3 frames for velocity calculation
    POST_OFFSET = 2  # Skip frames where bird is still at ground level
    
    if first_coll < VELOCITY_FRAMES:
        print("Not enough data before collision!")
        return
    
    if first_coll + POST_OFFSET + VELOCITY_FRAMES >= len(bird_features):
        print(f"Not enough data after collision (need at least {POST_OFFSET + VELOCITY_FRAMES} frames)!")
        return
    
    # Get actual post-collision trajectory
    actual_post_traj = bird_trajectory[first_coll + 1:]
    
    # Get states with CORRECTED velocities (matching the learning code in pddl_agent.py)
    # Uses multi-frame velocity calculation to avoid quantization
    pre_features = bird_features[first_coll]
    prev_features = bird_features[first_coll - VELOCITY_FRAMES]
    
    # POST-COLLISION: Skip first POST_OFFSET frames, then measure velocity over VELOCITY_FRAMES
    post_start = first_coll + POST_OFFSET
    post_end = post_start + VELOCITY_FRAMES
    post_features_start = bird_features[post_start]
    post_features_end = bird_features[post_end]
    
    # PRE-COLLISION state: velocity over VELOCITY_FRAMES frames before collision
    pre_state = pre_features.copy()
    pre_dt = VELOCITY_FRAMES * FRAME_RATE
    pre_state['v_x'] = (pre_features['x'] - prev_features['x']) / pre_dt
    pre_state['v_y'] = (pre_features['y'] - prev_features['y']) / pre_dt
    
    # POST-COLLISION state: velocity over VELOCITY_FRAMES frames after bounce starts
    post_state = post_features_start.copy()
    post_dt = VELOCITY_FRAMES * FRAME_RATE
    post_state['v_x'] = (post_features_end['x'] - post_features_start['x']) / post_dt
    post_state['v_y'] = (post_features_end['y'] - post_features_start['y']) / post_dt
    
    # Predict using learned model if KB available
    predicted_post_state = None
    if kb is not None and "collision" in kb:
        from agents.pddl.pddl_files.events.learn_events import PhysicsRatioModel, AngleDependentFrictionModel
        predicted_post_state = {}
        
        for var_name in ["v_x", "v_y", "y"]:
            model = kb["collision"]["variables"].get(var_name, {}).get("model")
            if model is not None:
                # Create input array for prediction
                X = np.array([[pre_state["x"], pre_state["y"], pre_state["v_x"], pre_state["v_y"]]])
                
                if isinstance(model, (PhysicsRatioModel, AngleDependentFrictionModel)):
                    # Physics-based models use X directly (no polynomial transformation)
                    predicted_post_state[var_name] = model.predict(X)[0]
                else:
                    # Linear regression models need polynomial features
                    from sklearn.preprocessing import PolynomialFeatures
                    poly = PolynomialFeatures(degree=1, include_bias=False)
                    X_poly = poly.fit_transform(X)
                    predicted_post_state[var_name] = model.predict(X_poly)[0]
            else:
                predicted_post_state[var_name] = post_state[var_name]
        
        predicted_post_state["x"] = post_state["x"]
    
    # Get gravity from world model
    gravity = world_model.hyperparams_values[Params.gravity]
    
    # COORDINATE SYSTEM ANALYSIS:
    # - bird_trajectory uses SCREEN coords: y=0 at top, y increases downward, ground at y~360
    # - bird_features['y'] uses NATURAL coords: y=0 at ground, y increases upward
    # - Velocity in features is computed from natural coords differences
    # - construct_trajectory_from_velocity works in SCREEN coords
    #
    # In SCREEN coords:
    # - Negative vy = moving UP (y decreasing)
    # - Positive vy = moving DOWN (y increasing)
    # - Gravity should INCREASE vy (pull down), so we need POSITIVE gravity effect on vy
    # - But euler_step does: vy_new = vy - gravity*dt
    # - So for gravity to pull DOWN in screen coords, gravity must be NEGATIVE
    
    GROUND_LEVEL = 360
    start_x = post_state["x"]
    start_y = bird_trajectory[first_coll + 1][1]  # Use actual screen y from trajectory
    
    # Construct predicted trajectory limit
    if len(actual_post_traj) > 0:
        limit = np.max(actual_post_traj[:, 0]) + 50
    else:
        limit = start_x + 200
    
    # Get velocities - use observed values (in natural coords)
    obs_vx = post_state["v_x"]
    obs_vy_natural = post_state["v_y"]  # positive = moving up in natural coords
    
    # Convert to SCREEN coordinates:
    # In natural: positive v_y = up
    # In screen: negative v_y = up (y decreases when moving up)
    screen_vy = -obs_vy_natural
    
    # Gravity in screen coords: needs to be NEGATIVE so that vy_new = vy - (-g)*dt = vy + g*dt
    # This makes vy more positive over time (pulling down in screen coords)
    screen_gravity = -gravity
    
    print(f"\n[V2 DEBUG] Coordinate analysis:")
    print(f"  Natural v_y (from features): {obs_vy_natural:.2f} (positive = up)")
    print(f"  Screen v_y (for trajectory): {screen_vy:.2f} (negative = up)")
    print(f"  World model gravity: {gravity:.2f}")
    print(f"  Screen gravity: {screen_gravity:.2f} (negative for screen coords)")
    print(f"  Start position (screen): ({start_x:.1f}, {start_y:.1f})")
    
    # Construct trajectory using OBSERVED post-collision velocity (direct v_x, v_y)
    observed_predicted_traj = construct_trajectory_from_velocity(
        [start_x, start_y],
        obs_vx,
        screen_vy,
        screen_gravity,  # Use screen-coordinate gravity
        limit,
        prt=False,
        integration_method='rk4'
    )
    
    # Also construct with learned model predictions if available
    learned_predicted_traj = None
    if predicted_post_state:
        learned_vx = predicted_post_state["v_x"]
        learned_vy_natural = predicted_post_state["v_y"]
        learned_screen_vy = -learned_vy_natural  # Convert to screen coords
        
        learned_start_y = GROUND_LEVEL - predicted_post_state["y"]
        
        print(f"  Learned v_x: {learned_vx:.2f}, v_y (natural): {learned_vy_natural:.2f}, v_y (screen): {learned_screen_vy:.2f}")
        
        learned_predicted_traj = construct_trajectory_from_velocity(
            [start_x, learned_start_y],
            learned_vx,
            learned_screen_vy,
            screen_gravity,  # Use screen-coordinate gravity
            limit,
            prt=False,
            integration_method='rk4'
        )
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Post-Collision Trajectory Analysis (V2 - Direct Velocity)', fontsize=14, fontweight='bold')
    
    ax1 = axes[0]
    
    # Transform Y: height above ground (natural coordinates for display)
    actual_post_y_natural = np.abs(actual_post_traj[:, 1] - GROUND_LEVEL)
    
    # Pre-collision trajectory
    n_pre_frames = min(20, first_coll)
    pre_traj_focused = bird_trajectory[first_coll - n_pre_frames:first_coll + 1]
    pre_traj_y_natural = np.abs(pre_traj_focused[:, 1] - GROUND_LEVEL)
    
    # Plot pre-collision
    ax1.plot(pre_traj_focused[:, 0], pre_traj_y_natural, 'b-', linewidth=2, alpha=0.7, label='Pre-collision')
    
    # Plot post-collision OBSERVED trajectory
    ax1.plot(actual_post_traj[:, 0], actual_post_y_natural, 'g-', linewidth=2.5, 
             label='Post-collision (OBSERVED)')
    
    # Plot PREDICTED trajectory from observed velocity (sanity check - should match green)
    if observed_predicted_traj is not None and len(observed_predicted_traj) > 0:
        obs_pred_y_natural = np.abs(observed_predicted_traj[:, 1] - GROUND_LEVEL)
        ax1.plot(observed_predicted_traj[:, 0], obs_pred_y_natural, 'c:', linewidth=2, alpha=0.7,
                 label='Predicted (from observed v_x,v_y)')
    
    # Plot PREDICTED trajectory from learned model
    learned_y_natural = None
    if learned_predicted_traj is not None and len(learned_predicted_traj) > 0:
        learned_y_natural = np.abs(learned_predicted_traj[:, 1] - GROUND_LEVEL)
        ax1.plot(learned_predicted_traj[:, 0], learned_y_natural, 'r--', linewidth=2,
                 label='Predicted (from LEARNED model)')
    
    # Mark collision point
    coll_point = bird_trajectory[first_coll]
    coll_y_natural = np.abs(coll_point[1] - GROUND_LEVEL)
    ax1.scatter(coll_point[0], coll_y_natural, s=300, c='red', marker='*', zorder=10,
               edgecolors='black', linewidths=2, label=f'Collision (frame {first_coll})')
    
    # Ground level
    ax1.axhline(y=0, color='brown', linestyle='--', linewidth=2, alpha=0.5, label='Ground')
    
    # Set axis limits
    x_min = coll_point[0] - 50
    x_max = actual_post_traj[-1, 0] + 20
    if learned_predicted_traj is not None and len(learned_predicted_traj) > 0:
        x_max = max(x_max, learned_predicted_traj[-1, 0] + 20)
    
    y_min = -2
    y_max = max(np.max(actual_post_y_natural), np.max(pre_traj_y_natural)) + 10
    if learned_y_natural is not None:
        y_max = max(y_max, np.max(learned_y_natural) + 10)
    
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    
    ax1.set_xlabel('X Position', fontsize=11)
    ax1.set_ylabel('Height above ground (pixels)', fontsize=11)
    ax1.set_title('Observed vs Predicted Trajectory (V2)\n(Cyan=from observed velocity, Red=from learned model)', fontsize=12)
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: State comparison table
    ax2 = axes[1]
    ax2.axis('off')
    
    vx_ratio = post_state['v_x'] / pre_state['v_x'] if abs(pre_state['v_x']) > 0.1 else 0
    vy_ratio = post_state['v_y'] / pre_state['v_y'] if abs(pre_state['v_y']) > 0.1 else 0
    
    table_data = [
        ['State', 'Pre-Coll', 'Actual Post', 'Learned Post', 'Diff', 'Ratio (post/pre)']
    ]
    
    for var in ['x', 'y', 'v_x', 'v_y']:
        pre_val = pre_state[var]
        actual_val = post_state[var]
        
        if var in ['v_x', 'v_y'] and abs(pre_val) > 0.1:
            ratio = f'{actual_val/pre_val:.2f}'
        else:
            ratio = '-'
        
        if predicted_post_state and var in predicted_post_state:
            learned_val = predicted_post_state[var]
            diff = learned_val - actual_val
            table_data.append([var, f'{pre_val:.2f}', f'{actual_val:.2f}', 
                              f'{learned_val:.2f}', f'{diff:+.2f}', ratio])
        else:
            table_data.append([var, f'{pre_val:.2f}', f'{actual_val:.2f}', 'N/A', 'N/A', ratio])
    
    table = ax2.table(cellText=table_data[1:], colLabels=table_data[0],
                     loc='upper center', cellLoc='center',
                     colColours=['lightblue']*6)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.1, 2.0)
    
    physics_text = f"""
V2 Analysis (Direct v_x, v_y):
- v_x ratio: {vx_ratio:.2f} (horizontal velocity retention)
- v_y ratio: {vy_ratio:.2f} (vertical velocity reversal/damping)

Key difference from V1:
- V1: Converts v_x,v_y -> angle,magnitude -> back to v_x,v_y
- V2: Uses v_x,v_y directly (no conversion loss)

If CYAN line matches GREEN: physics model is correct
If RED matches GREEN: learned model is working
"""
    ax2.text(0.5, 0.25, physics_text, transform=ax2.transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
             family='monospace')
    
    ax2.set_title('State Comparison at Collision Point', fontsize=12, pad=20)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print("\n" + "="*60)
    print("COLLISION LEARNING SUMMARY (V2 - Direct Velocity)")
    print("="*60)
    print(f"Collision detected at frame: {first_coll}")
    print(f"\nPre-collision state (natural coords):")
    print(f"  Position: ({pre_state['x']:.1f}, {pre_state['y']:.1f})")
    print(f"  Velocity: (v_x={pre_state['v_x']:.1f}, v_y={pre_state['v_y']:.1f})")
    print(f"\nActual post-collision state (natural coords):")
    print(f"  Position: ({post_state['x']:.1f}, {post_state['y']:.1f})")
    print(f"  Velocity: (v_x={post_state['v_x']:.1f}, v_y={post_state['v_y']:.1f})")
    print(f"\nCollision physics:")
    print(f"  v_x retention: {vx_ratio:.2%}")
    print(f"  v_y reversal:  {vy_ratio:.2%}")
    if predicted_post_state:
        print(f"\nLearned model prediction error:")
        for var in ['v_x', 'v_y', 'y']:
            if var in predicted_post_state:
                diff = predicted_post_state[var] - post_state[var]
                print(f"  {var}: {diff:+.4f}")
    print("="*60)
    
    return {
        'actual_post_traj': actual_post_traj,
        'collision_frame': first_coll,
        'pre_state': pre_state,
        'post_state': post_state,
        'predicted_post_state': predicted_post_state,
        'vx_ratio': vx_ratio,
        'vy_ratio': vy_ratio
    }


def plot_loo_cv_comparison(kb, event_name="collision", save_path=None):
    """
    Plot LOO-CV (Leave-One-Out Cross-Validation) comparison between 
    General (ElasticNet) and Domain-Specific models over time.
    
    LOO-CV measures how well a model generalizes to unseen data.
    Lower values = better generalization.
    
    Parameters:
        kb: Knowledge base containing event learning data
        event_name: Name of the event to plot (default: "collision")
        save_path: Optional path to save the figure
    """
    if event_name not in kb:
        print(f"Event '{event_name}' not found in KB")
        return
    
    event_data = kb[event_name]
    variables = event_data.get("variables", {})
    
    # Find variables with LOO-CV history
    vars_with_history = []
    for var_name, var_data in variables.items():
        if "loo_cv_history" in var_data and len(var_data["loo_cv_history"]["n_samples"]) > 1:
            vars_with_history.append(var_name)
    
    if not vars_with_history:
        print("No LOO-CV history available yet. Need at least 2 samples.")
        return
    
    # Create subplots - one for each variable with history
    n_vars = len(vars_with_history)
    fig, axes = plt.subplots(1, n_vars, figsize=(6*n_vars, 5))
    
    if n_vars == 1:
        axes = [axes]
    
    # Get current regularization type from config
    try:
        from agents.pddl.pddl_files.events.learn_events import REGULARIZATION_CONFIG
        reg_type = REGULARIZATION_CONFIG.get('type', 'elasticnet').upper()
        reg_name = {'NONE': 'OLS', 'L1': 'Lasso/L1', 'L2': 'Ridge/L2', 'ELASTICNET': 'ElasticNet'}.get(reg_type, reg_type)
    except:
        reg_name = 'ElasticNet'
    
    fig.suptitle(f"LOO-CV Comparison: General ({reg_name}) vs Domain-Specific Models\n"
                 "(Lower = Better Generalization)", fontsize=12, fontweight='bold')
    
    for ax, var_name in zip(axes, vars_with_history):
        history = variables[var_name]["loo_cv_history"]
        n_samples = history["n_samples"]
        general_loo = history["general"]
        general_std = history.get("general_std", [0] * len(general_loo))  # Get std if available
        domain_loo = history["domain"]
        
        # X-axis: number of samples at each measurement point
        x = list(range(1, len(n_samples) + 1))
        
        # Filter out inf values for plotting (include std for general)
        general_valid = [(i, v, s) for i, v, s in zip(x, general_loo, general_std) 
                        if v != float('inf') and v < 1000]
        domain_valid = [(i, v) for i, v in zip(x, domain_loo) if v != float('inf') and v < 1000]
        
        # Plot General (ElasticNet) model with variance bands
        if general_valid:
            gx, gy, gstd = zip(*general_valid)
            gx, gy, gstd = np.array(gx), np.array(gy), np.array(gstd)
            
            # Plot variance band (±1 std) for General model only
            ax.fill_between(gx, gy - gstd, gy + gstd, alpha=0.2, color='blue', 
                           label='General ±1σ')
            
            # Plot main line
            ax.plot(gx, gy, 'b-o', linewidth=2, markersize=8, label='General (ElasticNet)', alpha=0.8)
            
            # Annotate final value with std
            if gstd[-1] > 0:
                ax.annotate(f'{gy[-1]:.2f}±{gstd[-1]:.2f}', (gx[-1], gy[-1]), 
                           textcoords="offset points", xytext=(5, 5), fontsize=9, color='blue')
            else:
                ax.annotate(f'{gy[-1]:.2f}', (gx[-1], gy[-1]), textcoords="offset points", 
                           xytext=(5, 5), fontsize=9, color='blue')
        
        # Plot Domain-Specific model (no variance bands)
        if domain_valid:
            dx, dy = zip(*domain_valid)
            ax.plot(dx, dy, 'r-s', linewidth=2, markersize=8, label='Domain-Specific', alpha=0.8)
            # Annotate final value
            ax.annotate(f'{dy[-1]:.2f}', (dx[-1], dy[-1]), textcoords="offset points", 
                       xytext=(5, -10), fontsize=9, color='red')
        
        # Get current comparison result
        comparison = variables[var_name].get("model_comparison", {})
        winner_name = comparison.get("winner_name", "unknown")
        
        ax.set_xlabel("Training Iteration", fontsize=11)
        ax.set_ylabel("LOO-CV RMSE", fontsize=11)
        ax.set_title(f"{var_name}_after\n(Winner: {winner_name})", fontsize=11)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(x)
        
        # Add sample count as secondary x-axis labels
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"n={n}" for n in n_samples], fontsize=8)
        ax2.set_xlabel("Sample Count", fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"LOO-CV comparison plot saved to: {save_path}")
    
    # Show interactively - block=True waits for user to close the window
    print("\n[LOO-CV] Showing plot... Close the window to continue.")
    plt.show(block=True)
    
    # Print summary
    print("\n" + "="*60)
    print("LOO-CV COMPARISON SUMMARY")
    print("="*60)
    print("LOO-CV = Leave-One-Out Cross-Validation RMSE")
    print("Measures how well the model predicts data it hasn't seen")
    print("Lower value = better generalization (less overfitting)")
    print("-"*60)
    
    for var_name in vars_with_history:
        history = variables[var_name]["loo_cv_history"]
        comparison = variables[var_name].get("model_comparison", {})
        
        general_final = history["general"][-1] if history["general"] else float('inf')
        domain_final = history["domain"][-1] if history["domain"] else float('inf')
        winner = comparison.get("winner_name", "unknown")
        
        print(f"\n{var_name}_after:")
        print(f"  General (ElasticNet) LOO-CV: {general_final:.4f}")
        if domain_final != float('inf'):
            print(f"  Domain-Specific LOO-CV: {domain_final:.4f}")
        else:
            print(f"  Domain-Specific LOO-CV: N/A (no domain model)")
        print(f"  Winner: {winner}")
        
        if domain_final != float('inf') and general_final != float('inf'):
            if general_final < domain_final and domain_final > 0:
                improvement = ((domain_final - general_final) / domain_final) * 100
                print(f"  General is {improvement:.1f}% better")
            elif domain_final < general_final and general_final > 0:
                improvement = ((general_final - domain_final) / general_final) * 100
                print(f"  Domain is {improvement:.1f}% better")
    
    print("="*60)