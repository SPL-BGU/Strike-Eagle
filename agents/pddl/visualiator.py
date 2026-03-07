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


def visualize_rmse(rmse_values):

    time_steps = list(range(len(rmse_values)))

    # Plotting
    plt.figure(figsize=(8, 4))
    plt.plot(time_steps, rmse_values, marker='o',color="blue")
    plt.title('RMSE Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('RMSE')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def visualize_rmse_vs_suggsted(rmse_values,suggested_rmse_values):

    time_steps = list(range(len(rmse_values)))

    # Plotting
    plt.figure(figsize=(8, 4))
    plt.plot(time_steps, rmse_values, marker='o',color="blue")
    plt.plot(time_steps, suggested_rmse_values, marker='o', color="orange")
    plt.title('RMSE Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('RMSE')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


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