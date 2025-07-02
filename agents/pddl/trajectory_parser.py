from agents.pddl.pddl_files.world_model.params import Params
from agents.pddl.pddl_files.world_model.world_model import WorldModel
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
        limit: int,
        frame_rate=.02,
        prt= True):
    MAX_FRAMES = 500
    angle_rad = angle * 0.01745329252 # use angle rad as same as used in the pddl domain
    velocity = agent_world_model.hyperparams_values[Params.velocity]
    gravity = agent_world_model.hyperparams_values[Params.gravity]
    vx = velocity * agent_world_model.taylor_cos(angle_rad,4)  # consider use the cos usage
    vy = velocity * agent_world_model.taylor_sin(angle_rad,3)

    if prt:
        print(agent_world_model.taylor_cos(angle_rad,4))
        print(agent_world_model.taylor_sin(angle_rad, 3))
    trajectory = np.reshape(starting_point, [1, 2])

    for i in range(1,MAX_FRAMES):
        if prt:
            print(f"vx:{vx}\t vy:{vy}")
            print(f"location:{trajectory[i - 1, :]}")


        trajectory = np.vstack([
            trajectory,
            trajectory[i - 1, :] + [
                frame_rate * vx,
                frame_rate * vy
            ]
        ])

        vy -= frame_rate * gravity

        if trajectory[-1,0]>limit:
            break

    return trajectory
