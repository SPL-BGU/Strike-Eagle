import numpy as np
import ruptures as rpt
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN,KMeans
def getSegments(signal,penalty):


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

def getSegmentsML(signal):


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
    features = np.column_stack((v_x[:290], v_y[:290], a_x[:290], a_y[:290]))

    # Apply DBSCAN clustering
    model = DBSCAN(eps=0.5,min_samples=2).fit(features)
    labels = model.labels_

    # Find the transition between clusters (which may indicate a change in motion)
    # The last cluster (after the transition) corresponds to the sudden change
    impact_idx = np.where(labels != labels[0])[0][-1]  # Find where the cluster changes

    # Output the impact point
    print(f"Sudden change in motion detected at x = {x[impact_idx]}, y = {y[impact_idx]}")

    # Plot the trajectory with color coding based on cluster labels
    plt.figure(figsize=(8, 6))

    # Assign colors based on cluster labels
    scatter = plt.scatter(x[:290], y[:290], c=labels, cmap='viridis', label="Trajectory")

    # Highlight the impact point with a distinct color
    plt.scatter(x[impact_idx], y[impact_idx], color='red', label=f"Impact Point: ({x[impact_idx]}, {y[impact_idx]})",
                zorder=5)

    # Add colorbar to visualize clusters
    plt.colorbar(scatter, label='Cluster ID')

    # Plot labels and title
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Projectile Trajectory with Clustering (Impact Highlighted)')
    plt.legend()

    plt.show()

