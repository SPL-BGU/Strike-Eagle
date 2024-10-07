from src.computer_vision.GroundTruthReader import GroundTruthReader
import numpy as np
import json
from src.computer_vision.game_object import GameObject


def filter_from_entity(entity: GameObject):
    return np.array([
        entity.X,
        640-entity.Y #invert y axis
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

def exttract_estimated_trajectory():

    return np.load('agents/pddl/pddl_files/pddl_example/.npy', allow_pickle=True)
