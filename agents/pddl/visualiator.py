from matplotlib import patches, pyplot as plt
from matplotlib.path import Path

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


def visualize_trajectory(model,target_class,raw_trajectories):
    trajectories = groundtruth_trajectory_parser(raw_trajectories, model, target_class)
    patches = list(map(get_object_visuallization, trajectories.items()))
    # invert y axis
    fig, ax = plt.subplots()
    for patch in patches:
        ax.add_patch(patch)
    ax.set_xlim(0, 480)
    ax.set_ylim(300, 640)
    plt.show()

def plot_errors(errors,aggragive_errors):
    # Generate main data
    i = list(range(1,len(errors)+1))

    # Plot the main line
    plt.plot(i, errors, marker='o', linestyle='-', color='b', label="errors")
    plt.plot(i, aggragive_errors, marker='x', linestyle='--', label="aggragive_errors")

    plt.legend()

    # Show grid and plot
    plt.grid(True)
    plt.show()

def plot_score(score):
    # Generate main data
    i = list(range(1,len(score)+1))

    # Plot the main line
    plt.plot(i, score, marker='o', linestyle='-', color='b', label="errors")

    plt.legend()

    # Show grid and plot
    plt.grid(True)
    plt.show()
