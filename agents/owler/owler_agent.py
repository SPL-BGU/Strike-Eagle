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
from agents.utility.vision.vector_vision import VectorVision


class OwlerAgent(BaselineAgent):
    """Birds in boots (server/client version)"""

    def __init__(self, agent_ind, agent_configs, min_deg: int = -4, max_deg: int = 78, deg_step: float = 0.3,
                 learn: bool = False):

        super().__init__(
            agent_ind=agent_ind,
            agent_configs=agent_configs)
        self.min_deg = min_deg,
        self.max_deg = max_deg,
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

        # TODO: get deg
        random_deg = get_random_deg(deg_range=(self.min_deg, self.max_deg), step=self.deg_step)
        # trajectory = self.shoot_bird_by_angle_with_img_trail(random_deg, sling)
        #
        # # TODO: formulate trajectory
        # # For now just take last image
        # state_post = self.formulate_state(trajectory[-1])
        #
        # # # TODO: compute effects
        # # state_effect = self.compute_effect(state_prev, state_post)

        #   TODO: update KB
        self.update_kb(vision, state_prev, random_deg)

    def solve(self):
        """
        * Solve a particular level by shooting birds directly to pigs
        * @return GameState: the game state after shots.
        """
        # Initialization
        game_state = self.ar.get_game_state()

        # take pre vision
        vision = self._update_reader(self.ground_truth_type.value, self.if_check_gt)
        vector_vision = VectorVision(vision)
        vector_vision.formulate_from_base_vision()
        x = 0

    def take_action(self, state: int, action: int, sling):
        score = self.check_current_level_score()
        # Shoot the bird
        deg = get_deg_from_index(self.min_deg, self.deg_step, action)
        self.shoot_bird_by_angle_with_img_trail(deg, sling)
        # Check score diff
        score = self.check_current_level_score() - score
        # Get next_state
        game_state = self.ar.get_game_state()
        # If the level is solved , go to the next level
        if game_state == GameState.WON:
            raise GameSessionWonException(reward=score)
        if game_state == GameState.LOST:
            raise GameSessionLossException(
                reward=-1000)  # TODO: constant reward for loss, need to think this differently
        next_state = state + 1
        return score, next_state

    def formulate_state(self, vision):
        """
        Formulate_image
        """
        vector_vision = VectorVision(vision)
        vector_vision.formulate_from_base_vision()
        return vector_vision

    def compute_effect(self, prev, post):
        pass

    def update_kb(self, state, effect, action) -> bool:
        """
        Update KB based on the new information
        :return
            bool depending on if the KB has been updated
        """
        path = "TrainKB/train_data.pickle"
        # Load KB from data
        if not os.path.exists(path):
            with open(path, 'rb') as handle:
                data = pickle.load(handle)
        data = effect
        with open(path, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def choose_action(self, current_state):
        """
        Choose an action based on the KB and te current state
        """
