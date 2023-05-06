import os.path

import numpy as np
import random
from agents import BaselineAgent
from agents.utility import GroundTruthType
from agents.utility.exceptions import *
import pickle
from functools import reduce
import socket
from math import cos, sin, degrees, pi
from agents.utility.vision.relations import *
from src.client.agent_client import AgentClient, GameState, RequestCodes
from src.trajectory_planner.trajectory_planner import SimpleTrajectoryPlanner
from src.computer_vision.GroundTruthReader import GroundTruthReader, NotVaildStateError
from src.computer_vision.game_object import GameObjectType
from src.computer_vision.GroundTruthTest import GroundTruthTest
from src.utils.point2D import Point2D
from agents.utility.degrees import *
from src.computer_vision.game_object import GameObject
from agents.utility.vision.vector_vision import VectorVision
import logging
import time
from agents.utility.vision.vector_vision import VectorVision, end_goal


class OwlerAgent(BaselineAgent):
    """Birds in boots (server/client version)"""

    def __init__(self, agent_ind, agent_configs, min_deg: int = -4, max_deg: int = 78, deg_step: float = 1,
                 learn: bool = False):

        super().__init__(
            agent_ind=agent_ind,
            agent_configs=agent_configs)
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.deg_step = deg_step

        # Override sim speed from 20
        self.sim_speed = 10
        self.ground_truth_type = GroundTruthType.ground_truth_screenshot
        self.learn = True

        self.KB = None

    def train(self):
        """
        * train  on a particular level by shooting birds and study effects
        * @return GameState: the game state after shots.
        """
        # Initialization
        game_state = self.ar.get_game_state()

        # take pre vision
        vision = self._update_reader(self.ground_truth_type.value, self.if_check_gt)
        sling = vision.find_slingshot_mbr()[0]
        # TODO: formulate vision into better domain set
        state_prev = self.formulate_state(vision)
        state_end = end_goal(state_prev)

        # TODO: get deg
        random_deg = get_random_deg(deg_range=(self.min_deg, self.max_deg), step=self.deg_step)
        trajectory = self.shoot_bird_by_angle_with_img_trail(random_deg, sling)
        #
        # # TODO: formulate trajectory
        # # For now just take last image
        state_post = self.formulate_trajectory(state_prev, trajectory)[-1]
        #
        # TODO: compute effects
        state_effect = self.compute_effect(state_prev, state_post)

        #   TODO: update KB
        self.update_kb(vision, state_effect, random_deg)
        time.sleep(2)

    def solve(self):
        """
        * Solve a particular level by shooting birds directly to pigs
        * @return GameState: the game state after shots.
        """
        # Initialization
        # game_state = self.ar.get_game_state()

        # take pre vision
        vision = self._update_reader(self.ground_truth_type.value, self.if_check_gt)
        vector_vision = VectorVision(vision)
        vector_vision.formulate_from_base_vision()
        x = 0

    def formulate_state(self, vision, base_vector_vision=None) -> VectorVision:
        """
        Formulate_image
        """
        vector_vision = VectorVision(vision)
        if base_vector_vision:
            vector_vision.formulate_from_base_vision()
            vector_vision.update_dead(base_vector_vision)
        else:
            vector_vision.formulate_from_base_vision()
        return vector_vision

    def formulate_trajectory(self, base_vision: VectorVision, trajectory: list):
        """
        Formulate an untire trajectory ( list of visions) to trajectory Vector Vision
        :param base_vision: base vision to start from
        :type base_vision: VectorVision
        :param trajectory: trajectory
        :type trajectory:
        :return:
        :rtype:
        """
        vector_trajectory = [base_vision]
        for vision in trajectory:
            vector_trajectory.append(self.formulate_state(vision, vector_trajectory[-1]))
        return vector_trajectory

    def compute_effect(self, prev: VectorVision, post: VectorVision) -> dict:
        """
        Compute the effect by substracting post and prev matrices
        :param prev:
        :type prev:
        :param post:
        :type post:
        :return:
        :rtype:
        """
        # update post matrix
        post.update_matrix()
        return {
            "effect": post.matrix - prev.matrix,
            "frame": 0
        }

    def update_kb(self, state, effect, action,frame=0) -> bool:
        """
        Update KB based on the new information
        :return
            bool depending on if the KB has been updated
        """
        folder = "TrainKB"
        file = "train_data.pickle"
        path = os.path.join(folder, file)
        # Load KB from data
        if os.path.exists(path):
            with open(path, 'rb') as handle:
                data: list = pickle.load(handle)
        else:
            data = list()
        data.append({
                "state": state,
                "action": action,
                "frame": frame,
                "effect": effect
            })
        with open(path, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            return True

    def choose_action(self, current_state):
        """
        Choose an action based on the KB and te current state
        """
