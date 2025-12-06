from matplotlib import patches, pyplot as plt
from matplotlib.path import Path
import numpy as np
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error

from agents.pddl.trajectory_parser import groundtruth_trajectory_parser


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


