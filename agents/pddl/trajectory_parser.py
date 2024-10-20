import math

from agents.pddl.pddl_files.world_model import WorldModel
from src.computer_vision.GroundTruthReader import GroundTruthReader
import numpy as np
from src.computer_vision.game_object import GameObject


def filter_from_entity(entity: GameObject):
    return np.array([
        entity.X,
        640 - entity.Y  # invert y axis
    ])


def groundtruth_trajectory_parser(
        raw_trajectory,
        model,
        target_class
):
    entity_trajectories = dict()
    combined_trajectory = [GroundTruthReader(shoot, model, target_class).allObj for shoot in raw_trajectory]
    for temporal_state in combined_trajectory:
        for object_type, object_list in temporal_state.items():
            for i, entity in enumerate(object_list):
                entity_name = f"{object_type}_{i}"
                filtered_entity = filter_from_entity(entity)
                if entity_name not in entity_trajectories.keys():
                    entity_trajectories[entity_name] = []
                entity_trajectories[entity_name].append(filtered_entity)
    # todo: select what objects to take
    return entity_trajectories


def extract_real_trajectory(
        raw_trajectory,
        alpha: int,
        model,
        target_class):
    groundtruth_trajectories = groundtruth_trajectory_parser(raw_trajectory, model, target_class)
    red_bird_traj = np.array(groundtruth_trajectories["redBird_0"])
    return red_bird_traj


def construct_trajectory(
        starting_point: [float, float],
        angle: float,
        agent_world_model: WorldModel,
        frames=400,
        frame_rate=.01*2):
    angle_rad = angle / 180 * math.pi
    vx = agent_world_model.v_bird * math.cos(angle_rad)  # consider use the cos usage
    vy = agent_world_model.v_bird * math.sin(angle_rad)
    vx = agent_world_model.v_bird * agent_world_model.taylor_cos(angle_rad,3)  # consider use the cos usage
    vy = agent_world_model.v_bird * agent_world_model.taylor_sin(angle_rad,3)
    trajectory = np.reshape(starting_point, [1, 2])
    for i in range(1, frames):
        trajectory = np.vstack([
            trajectory,
            trajectory[i - 1, :] + [
                frame_rate * vx,
                frame_rate * vy
            ]
        ])

        vy -= frame_rate * agent_world_model.gravity
    return trajectory
# if __name__ == "__main__":
#     extract_estimated_trajectory()
