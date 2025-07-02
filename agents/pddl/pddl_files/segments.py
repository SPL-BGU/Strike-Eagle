import numpy as np
import ruptures as rpt
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN,KMeans



def getSegmentsPelt(signal,penalty):


    # Sample data (replace with your own x, y data)
    x = signal[:,0]  # x positions
    y = signal[:,1]

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



def getSegmentsPreconditions(trajectory):
    GROUND_LEVEL = 360
    # Sample data (replace with your own x, y data)
    x = trajectory[:,0]  # x positions
    y = trajectory[:,1]-GROUND_LEVEL

    # Calculate velocity and acceleration as features for clustering
    dt = 50  # Assuming 1 unit time step (adjust accordingly)
    v_x = np.diff(x) / dt
    v_y = np.diff(y) / dt
    a_x = np.diff(v_x) / dt
    a_y = np.diff(v_y) / dt

    features = np.column_stack((x[:290],y[:290],v_x[:290], v_y[:290], a_x[:290], a_y[:290]))

    keys = ['x', 'y', 'v_x', 'v_y', 'a_x', 'a_y']
    features_dict_list = [dict(zip(keys, row)) for row in features]


    collsions = getGroundCollisions(features_dict_list)

    parts = np.split(trajectory, collsions)

    return parts


def getGroundCollisions(features_dict_list):

    epsilon = 3
    ground_collision_frames = list()
    flying_frames = list()
    for i, f in enumerate(features_dict_list):
        # Check if current y is at or below ground and previous y was above ground (falling)
        if is_ground_collision(features_dict_list,i):
            print(f"Ground collision likely at frame {i}")
            ground_collision_frames.append(i)
        if is_flying(features_dict_list,i):
            print(f"Flying likely at frame {i}")
            flying_frames.append(i)

    return ground_collision_frames


def is_ground_collision(features_dict_list,i):
    epsilon=3
    return i > 0 and features_dict_list[i]['y'] <= epsilon and features_dict_list[i - 1]['y'] > epsilon

def is_flying(features_dict_list,i):
    epsilon=2
    return i > 0 and features_dict_list[i]['y'] > epsilon and features_dict_list[i - 1]['y'] > epsilon

