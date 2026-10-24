from agents.pddl.pddl_files.world_model.params import Params
from agents.pddl.pddl_files.world_model.world_model import WorldModel
from src.computer_vision.GroundTruthReader import GroundTruthReader
import numpy as np
from src.computer_vision.game_object import GameObject
import math
from agents.pddl.pddl_files.pddl_parser import platform_bbox_from_game_object

def filter_from_entity(entity: GameObject, entity_type: str = None):
    """
    Extract location and dimensions from a game entity.
    
    For platforms (hills), uses polygon vertices via platform_bbox_from_game_object.
    For other entities (birds, pigs, blocks), uses X, Y as center.
    """
    is_platform = entity_type and "hill" in entity_type.lower()
    
    if is_platform:
        bbox = platform_bbox_from_game_object(entity)
        return {
            "location": np.array([bbox["x_platform"], bbox["y_platform"]]),
            "dimension": [bbox["platform_width"], bbox["platform_height"]],
            "platform_geom": bbox,
        }

    dimension = [entity.width, entity.height]
    location = np.array([entity.X, 640 - entity.Y])  # invert y axis
    
    return {
        "location": location,
        "dimension": dimension
    }


def groundtruth_trajectory_parser(
        raw_trajectory,
        model,
        target_class
):
    entity_trajectories = dict()
    entity_dimensions = dict()
    combined_trajectory = [GroundTruthReader(shoot, model, target_class).allObj for shoot in raw_trajectory]
    for temporal_state in combined_trajectory:
        for object_type, object_list in temporal_state.items():
            for i, entity in enumerate(object_list):
                entity_name = f"{object_type}_{i}"
                filtered_entity = filter_from_entity(entity, entity_type=object_type)
                if entity_name not in entity_trajectories.keys():
                    entity_trajectories[entity_name] = []
                entity_trajectories[entity_name].append(filtered_entity["location"])
                if filtered_entity.get("platform_geom") is not None:
                    entity_dimensions[entity_name] = filtered_entity["platform_geom"]
                else:
                    entity_dimensions[entity_name] = filtered_entity["dimension"]

    return entity_trajectories, entity_dimensions


def extract_real_trajectory(
        raw_trajectory,
        alpha: int,
        model,
        target_class):
    groundtruth_trajectories, groundtruth_dimensions = groundtruth_trajectory_parser(raw_trajectory, model, target_class)
    return groundtruth_trajectories, groundtruth_dimensions


def euler_step(state, dt, gravity):
    """Simple Euler integration: O(Δt²) error per step"""
    x, y, vx, vy = state
    return np.array([
        x + dt * vx,
        y + dt * vy,
        vx,  # vx is constant
        vy - dt * gravity
    ])


def midpoint_step(state, dt, gravity):
    """Midpoint method: O(Δt³) error per step"""
    x, y, vx, vy = state
    
    # Calculate midpoint velocity
    vy_mid = vy - (dt / 2) * gravity
    
    # Use midpoint velocity for position update
    return np.array([
        x + dt * vx,
        y + dt * vy_mid,
        vx,  # vx is constant
        vy - dt * gravity
    ])


def rk4_step(state, dt, gravity):
    """Runge-Kutta 4th order: O(Δt⁵) error per step"""
    x, y, vx, vy = state
    
    # Define derivative function: d[state]/dt = f(state, t)
    def f(s):
        sx, sy, svx, svy = s
        return np.array([svx, svy, 0, -gravity])
    
    # RK4 algorithm
    k1 = f(state)
    k2 = f(state + dt/2 * k1)
    k3 = f(state + dt/2 * k2)
    k4 = f(state + dt * k3)
    
    # Weighted average
    state_new = state + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
    
    return state_new


def construct_trajectory_from_velocity(
        starting_point: [float, float],
        v_x: float,
        v_y: float,
        gravity: float,
        limit: float,
        frame_rate=0.02,
        prt=False,
        integration_method='rk4',
        stop_at_ground=False,
        ground_level=280):
    """
    Construct trajectory directly from velocity components (v_x, v_y).
    
    This version avoids the angle/magnitude conversion that can introduce
    coordinate system mismatches and precision loss.
    
    Parameters:
    -----------
    starting_point : [float, float]
        Starting position (x, y)
    v_x : float
        Initial horizontal velocity (pixels/second)
    v_y : float
        Initial vertical velocity (pixels/second, positive = up in natural coords)
    gravity : float
        Gravity acceleration (pixels/second^2)
    limit : float
        Maximum x-coordinate to simulate to
    frame_rate : float
        Time step in seconds (default 0.02 for 50 fps)
    prt : bool
        Whether to print debug information
    integration_method : str
        'euler', 'midpoint', or 'rk4'
    stop_at_ground : bool
        If True, stop trajectory when bird hits ground level (default False)
    ground_level : float
        Ground level in natural coordinates (default 280, which is 640-360)
    
    Returns:
    --------
    trajectory : np.ndarray
        Array of shape (N, 2) containing (x, y) positions
    """
    MAX_FRAMES = 500
    
    trajectory = np.reshape(starting_point, [1, 2])
    
    # State: [x, y, vx, vy] - directly use provided velocities
    state = np.array([float(starting_point[0]), float(starting_point[1]), float(v_x), float(v_y)])
    
    if prt:
        print(f"[construct_trajectory_from_velocity] Starting at ({starting_point[0]:.1f}, {starting_point[1]:.1f})")
        print(f"  v_x={v_x:.2f}, v_y={v_y:.2f}, gravity={gravity:.2f}")
    
    for i in range(1, MAX_FRAMES):
        if prt and i <= 5:
            print(f"  Frame {i}: pos=({state[0]:.1f}, {state[1]:.1f}), vel=({state[2]:.1f}, {state[3]:.1f})")
        
        # Integrate one step based on method
        if integration_method == 'euler':
            state = euler_step(state, frame_rate, gravity)
        elif integration_method == 'midpoint':
            state = midpoint_step(state, frame_rate, gravity)
        elif integration_method == 'rk4':
            state = rk4_step(state, frame_rate, gravity)
        else:
            raise ValueError(f"Unknown integration method: {integration_method}")
        
        # Append position to trajectory
        trajectory = np.vstack([trajectory, state[0:2]])
        
        if state[0] > limit:
            break
        
        if stop_at_ground and state[1] <= ground_level:
            break
    
    return trajectory


# Defaults aligned with collision learning in pddl_agent (50 fps, multi-frame windows)
LAUNCH_ESTIMATE_FRAME_RATE = 0.02
LAUNCH_ESTIMATE_N_VEL = 5
LAUNCH_ESTIMATE_N_POS = 3
LAUNCH_ESTIMATE_SKIP = 0
# Full-segment ballistic fit: grow the window frame by frame while every GT point
# stays within this many px of the fitted parabola (GT is integer-pixel, so ±0.5
# is rounding). The first larger miss is a contact/bounce — stop there.
LAUNCH_FIT_MAX_RESID_PX = 1.5
LAUNCH_FIT_MIN_FRAMES = 12


def _fit_ballistic_prefix(traj: np.ndarray, frame_rate: float):
    """
    Fit x = x0 + vx·t, y = y0 + vy·t − g·t²/2 (g free) over the longest clean
    prefix of ``traj``. Returns None when the segment is too short.

    The 6-frame fit this replaces read shallow launches up to ~7° off
    (re-measured 243 shots, Oct 2026): over a few integer-pixel frames vy is
    mostly rounding noise, and θ error ≈ vx·Δvy/|v|² is largest when vx is.
    """
    if len(traj) < LAUNCH_FIT_MIN_FRAMES:
        return None
    best = None
    for n in range(LAUNCH_FIT_MIN_FRAMES, len(traj) + 1):
        t = np.arange(n, dtype=float) * frame_rate
        A = np.column_stack([np.ones(n), t, -0.5 * t ** 2])
        cy, *_ = np.linalg.lstsq(A, traj[:n, 1], rcond=None)
        cx = np.polyfit(t, traj[:n, 0], 1)
        resid = max(
            np.abs(traj[:n, 1] - A @ cy).max(),
            np.abs(traj[:n, 0] - np.polyval(cx, t)).max(),
        )
        if resid > LAUNCH_FIT_MAX_RESID_PX and best is not None:
            break
        best = {"x0": cx[1], "y0": cy[0], "vx": cx[0], "vy": cy[1], "g": cy[2], "n": n}
    return best


def estimate_launch_from_trajectory(
        trajectory: np.ndarray,
        n_vel: int = LAUNCH_ESTIMATE_N_VEL,
        n_pos: int = LAUNCH_ESTIMATE_N_POS,
        skip: int = LAUNCH_ESTIMATE_SKIP,
        frame_rate: float = LAUNCH_ESTIMATE_FRAME_RATE,
):
    """
    Estimate release position and initial velocity from a GT flight segment.

    Primary: ballistic fit over the longest clean prefix of the segment (after
    skip) — release is the fitted t=0 point, θ = atan2(vy, vx). Gravity is a free
    parameter, so θ does not depend on ``frame_rate``.

    Fallback for segments shorter than LAUNCH_FIT_MIN_FRAMES:
      - Position: mean of the first n_pos points.
      - vx: linear regression of x(t); vy: quadratic fit of y(t) when n_vel >= 2.
    """
    traj = np.asarray(trajectory, dtype=float)
    if traj.ndim != 2 or traj.shape[1] < 2:
        raise ValueError("trajectory must be (N, 2) with N >= 2")

    n_available = len(traj) - skip
    if n_available < 2:
        raise ValueError(f"need at least {skip + 2} trajectory points, got {len(traj)}")

    fit = _fit_ballistic_prefix(traj[skip:], frame_rate)
    if fit is not None:
        vx, vy = float(fit["vx"]), float(fit["vy"])
        return {
            "release": np.array([fit["x0"], fit["y0"]], dtype=float),
            "vx": vx,
            "vy": vy,
            "v_meas": float(np.hypot(vx, vy)),
            "theta_deg": float(np.degrees(np.arctan2(vy, vx))),
            "fit_frames": int(fit["n"]),
            "g_fit": float(fit["g"]),
        }

    n_pos = max(1, min(n_pos, n_available))
    n_vel = max(1, min(n_vel, n_available - 1))

    release = np.mean(traj[skip: skip + n_pos], axis=0)
    t = np.arange(skip, skip + n_vel + 1, dtype=float) * frame_rate
    xs = traj[skip: skip + n_vel + 1, 0]
    ys = traj[skip: skip + n_vel + 1, 1]

    vx = float(np.polyfit(t, xs, 1)[0])

    # Quadratic fit for y: y(t) = a*t² + b*t + c  →  b is vy at t=0.
    # Requires ≥ 3 points (n_vel ≥ 2). Fall back to linear for very short segments.
    if n_vel >= 2:
        y_coeffs = np.polyfit(t, ys, 2)  # [a, b, c], highest-degree first
        vy = float(y_coeffs[1])
    else:
        vy = float(np.polyfit(t, ys, 1)[0])

    v_meas = float(np.hypot(vx, vy))
    theta_deg = float(np.degrees(np.arctan2(vy, vx)))

    return {
        "release": release,
        "vx": vx,
        "vy": vy,
        "v_meas": v_meas,
        "theta_deg": theta_deg,
    }


def construct_trajectory(
        starting_point: [float, float],
        angle: float,
        agent_world_model: WorldModel,
        limit: int,
        frame_rate=.02,
        prt=True,
        integration_method='rk4',
        stop_at_ground=False,
        ground_level=280):
    """
    Construct trajectory with improved numerical integration.
    
    Parameters:
    -----------
    starting_point : [float, float]
        Starting position (x, y)
    angle : float
        Launch angle in degrees
    agent_world_model : WorldModel
        World model containing velocity and gravity parameters
    limit : int
        Maximum x-coordinate to simulate to
    frame_rate : float
        Time step in seconds (default 0.02 for 50 fps)
    prt : bool
        Whether to print debug information
    integration_method : str
        'euler' - Simple Euler (O(Δt²) error, fastest)
        'midpoint' - Midpoint method (O(Δt³) error, good balance)
        'rk4' - Runge-Kutta 4th order (O(Δt⁵) error, most accurate)
    stop_at_ground : bool
        If True, stop trajectory when bird hits ground level (default False)
    ground_level : float
        Ground level in natural coordinates (default 280, which is 640-360)
    
    Returns:
    --------
    trajectory : np.ndarray
        Array of shape (N, 2) containing (x, y) positions
    """
    MAX_FRAMES = 500
    velocity = agent_world_model.hyperparams_values[Params.velocity]
    gravity = agent_world_model.hyperparams_values[Params.gravity]
    
    vx = velocity * math.cos(angle*math.pi/180)
    vy = velocity * math.sin(angle*math.pi/180)
    
    trajectory = np.reshape(starting_point, [1, 2])
    
    # State: [x, y, vx, vy]
    state = np.array([starting_point[0], starting_point[1], vx, vy])
    
    for i in range(1, MAX_FRAMES):
        if prt:
            print(f"vx:{state[2]}\t vy:{state[3]}")
            print(f"location:{state[0:2]}")
        
        # Integrate one step based on method
        if integration_method == 'euler':
            # Simple Euler (original method)
            state = euler_step(state, frame_rate, gravity)
        elif integration_method == 'midpoint':
            # Midpoint method (2nd order, better accuracy)
            state = midpoint_step(state, frame_rate, gravity)
        elif integration_method == 'rk4':
            # Runge-Kutta 4th order (4th order, best accuracy)
            state = rk4_step(state, frame_rate, gravity)
        else:
            raise ValueError(f"Unknown integration method: {integration_method}")
        
        # Append position to trajectory
        trajectory = np.vstack([trajectory, state[0:2]])
        
        if state[0] > limit:
            break
        
        if stop_at_ground and state[1] <= ground_level:
            break
    
    return trajectory
