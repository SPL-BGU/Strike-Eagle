import numpy as np
import ruptures as rpt
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans

from agents.pddl.pddl_files.events.event_conditions import is_ground_collision, is_hit, is_platform_collision


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
    GROUND_LEVEL = 360
    # Sample data (replace with your own x, y data)
    x = trajectory[:, 0]  # x positions
    y = trajectory[:, 1] - GROUND_LEVEL

    # Calculate velocity and acceleration as features for clustering
    dt = 20  # Assuming 1 unit time step (adjust accordingly)
    v_x = np.diff(x) / dt
    v_y = np.diff(y) / dt
    a_x = np.diff(v_x) / dt
    a_y = np.diff(v_y) / dt

    min_length = min(len(arr) for arr in [x, y, v_y, v_x, a_x, a_y])
    features = np.column_stack(
        (x[:min_length], y[:min_length], v_x[:min_length], v_y[:min_length], a_x[:min_length], a_y[:min_length]))

    keys = ['x', 'y', 'v_x', 'v_y', 'a_x', 'a_y']
    features_dict_list = [dict(zip(keys, row)) for row in features]

    return features_dict_list


def getSegmentsEvents(groundtruth_trajectories:dict,groundtruth_objects:dict):
    objects_features = dict()
    for object,traj in groundtruth_trajectories.items():
        objects_features[object] = calculate_features(np.stack(traj))
    event_indexes = check_events(objects_features,groundtruth_objects
                                 , [
                                     {
                                         "name": "ground_collision",
                                         "func": is_ground_collision
                                     },
                                     {
                                         "name": "hit",
                                         "func": is_hit
                                     },
                                     {
                                         "name": "platform_collision",
                                         "func": is_platform_collision

                                     }
                                 ]
                                 )

    return event_indexes, objects_features


def check_events(objects_features,groundtruth_objects, events: list):
    result = {event["name"]: [] for event in events}

    max_time = max(len(traj) for traj in objects_features.values())

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
