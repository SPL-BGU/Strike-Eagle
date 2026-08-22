import numpy as np
import ruptures as rpt
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans

from agents.pddl.pddl_files.events.event_conditions import (
    is_ground_collision, is_hit, is_platform_collision, is_block_collision,
)


def getSegmentsPelt(signal, penalty):
    # Sample data (replace with your own x, y data)
    x = signal[:, 0]  # x positions
    y = signal[:, 1]

    # Calculate velocity and acceleration as features for clustering
    dt = 50  # Assuming 1 unit time step (adjust accordingly)
    v_x = np.diff(x) / dt
    v_y = np.diff(y) / dt
    a_x = np.diff(v_x) / dt
    a_y = np.diff(v_y) / dt

    # Combine features (velocity and acceleration)
    features = np.abs(np.column_stack((v_x[:290], v_y[:290], a_x[:290], a_y[:290])))

    # 2. Apply PELT to detect changepoints
    model = "rbf"  # cost model: least squares
    algo = rpt.Pelt(model=model).fit(features)  # penalty controls number of changepoints
    result = algo.predict(pen=penalty)

    # 3. Plot results
    rpt.display(signal, result)
    plt.title("Changepoint Detection using PELT")
    plt.show()


def calculate_features(trajectory):
    """
    Calculate position, velocity, and acceleration features from trajectory.
    
    Note: Coordinate system
    - trajectory y values are in screen coordinates (y increases downward)
    - After subtracting GROUND_LEVEL, positive y = above ground, negative = below
    - v_y negative = moving down (toward ground), v_y positive = moving up
    """
    GROUND_LEVEL = 360
    FRAME_RATE = 0.02  # 50 fps = 0.02 seconds per frame
    
    x = trajectory[:, 0]  # x positions
    y = trajectory[:, 1] - GROUND_LEVEL  # y relative to ground (positive = above ground)

    # Calculate velocity: change in position per second
    v_x = np.diff(x) / FRAME_RATE
    v_y = np.diff(y) / FRAME_RATE
    
    # Calculate acceleration: change in velocity per second
    a_x = np.diff(v_x) / FRAME_RATE
    a_y = np.diff(v_y) / FRAME_RATE

    # Pad arrays to match length (velocity has len-1, acceleration has len-2)
    # Use forward fill for the last values
    v_x = np.append(v_x, v_x[-1] if len(v_x) > 0 else 0)
    v_y = np.append(v_y, v_y[-1] if len(v_y) > 0 else 0)
    a_x = np.append(a_x, [a_x[-1], a_x[-1]] if len(a_x) > 0 else [0, 0])
    a_y = np.append(a_y, [a_y[-1], a_y[-1]] if len(a_y) > 0 else [0, 0])

    min_length = min(len(arr) for arr in [x, y, v_y, v_x, a_x, a_y])
    features = np.column_stack(
        (x[:min_length], y[:min_length], v_x[:min_length], v_y[:min_length], a_x[:min_length], a_y[:min_length]))

    keys = ['x', 'y', 'v_x', 'v_y', 'a_x', 'a_y']
    features_dict_list = [dict(zip(keys, row)) for row in features]

    return features_dict_list


EVENT_KIND_SPECS = [
    {"name": "ground_collision", "func": is_ground_collision},
    {"name": "hit", "func": is_hit},
    {"name": "platform_collision", "func": is_platform_collision},
    {"name": "block_collision", "func": is_block_collision},
]


def _empty_event_indexes():
    """Empty event-indexes dict with the canonical event-name keys."""
    return {spec["name"]: [] for spec in EVENT_KIND_SPECS}


def getSegmentsEvents(groundtruth_trajectories: dict, groundtruth_objects: dict):
    """Compute event indexes from per-object trajectories.

    Returns ``(_empty_event_indexes(), {})`` if either argument is empty or
    every object has an empty trajectory. Science Birds occasionally returns
    zero ground-truth frames for a shot (see run_20260814_082859 crash at
    train-21: SB reported ``got 0 ground truth frames`` and ``check_events``
    called ``max()`` on an empty generator, which killed the agent thread).
    Callers already have a placeholder-trajectory fallback for the
    "no active bird" case (see ``pddl_agent.solve`` line ~1156); this guard
    just lets that fallback take over instead of crashing.
    """
    if not groundtruth_trajectories:
        return _empty_event_indexes(), {}

    objects_features = dict()
    for object_name, traj in groundtruth_trajectories.items():
        if traj is None or len(traj) == 0:
            continue
        objects_features[object_name] = calculate_features(np.stack(traj))

    if not objects_features:
        return _empty_event_indexes(), {}

    event_indexes = check_events(objects_features, groundtruth_objects, EVENT_KIND_SPECS)
    return event_indexes, objects_features


def check_events(objects_features, groundtruth_objects, events: list):
    result = {event["name"]: [] for event in events}

    if not objects_features:
        return result

    non_empty_lens = [len(traj) for traj in objects_features.values() if traj]
    if not non_empty_lens:
        return result

    max_time = max(non_empty_lens)

    # Step 2: Stretch the trajectories to match the max length
    for obj, traj in objects_features.items():
        while len(traj) < max_time:
            traj.append(traj[-1])

    frames = []
    for frame_values in zip(*objects_features.values()):
        frame_dict = dict(zip(objects_features.keys(), frame_values))
        frames.append(frame_dict)

    for i in range(len(frames)):
        for event in events:
            if event["func"](frames,groundtruth_objects,i):
                result[event["name"]].append(i)
    return result


def is_flying(features_dict_list, i):
    epsilon = 2
    return i > 0 and features_dict_list[i]['y'] > epsilon and features_dict_list[i - 1]['y'] > epsilon
