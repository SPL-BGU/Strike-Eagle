import os.path
import time

import numpy as np
import random
from agents import BaselineAgent
from agents.utility import GroundTruthType
import subprocess
import pddl
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
from agents.pddl.pddl_files.pddl_parser import generate_pddl, write_problem_file, parse_solution_to_actions
from src.utils.point2D import Point2D


class PDDLAgent(BaselineAgent):
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
        self.sim_speed = 5
        self.ground_truth_type = GroundTruthType.ground_truth_screenshot
        self.learn = True

        self.KB = None

    def train(self):
        """
        * train  on a particular level by shooting birds and study effects
        * @return GameState: the game state after shots.
        """
        pass

    def solve(self):
        """
        * Solve a particular level by shooting birds directly to pigs
        * @return GameState: the game state after shots.
        """
        ground_truth_type = GroundTruthType.ground_truth_screenshot
        vision = self._update_reader(ground_truth_type.value, self.if_check_gt)
        sling = vision.find_slingshot_mbr()[0]
        sling.width, sling.height = sling.height, sling.width
        # DEFINE PROBLEM
        actions = self.define_problem(vision)
        for action, value in actions:
            release_point = self.tp.find_release_point(sling, value * np.pi / 180)
            batch_gt = self.ar.shoot_and_record_ground_truth(release_point.X, release_point.Y, 0, 0, 1, 0)
            time.sleep(2)

    def define_problem(self, vision: GroundTruthReader):
        """
        Formulate_image
        """
        # GET Problem
        initial_angle = self.min_deg
        angle_rate = self.deg_step
        ground_truth_type = GroundTruthType.ground_truth_screenshot
        vision = self._update_reader(ground_truth_type.value, self.if_check_gt)
        sling = vision.find_slingshot_mbr()[0]

        # Define birds
        BIRD_TYPES = [GameObjectType.REDBIRD, GameObjectType.YELLOWBIRD, GameObjectType.BLACKBIRD,
                      GameObjectType.WHITEBIRD, GameObjectType.BLUEBIRD]
        bird_id = 0
        problem_data = dict()
        birds_types = vision.find_birds()

        # Get ref

        for bird_type, birds in birds_types.items():
            for bird in birds:
                problem_data[f"bird_{bird_id}"] = {
                    "x_bird": bird.X + (min(bird.width, bird.height) / 2),  # move to center
                    "y_bird": 640 - bird.Y + min(bird.width, bird.height) / 2,
                    "bird_id": bird_id,
                    "bird_type": BIRD_TYPES.index(GameObjectType(bird_type)),
                    "m_bird": bird.width * bird.height,  # check this because it is not mandatory
                    "bird_radius": min(bird.width, bird.height) / 2,  # check this
                    "v_bird": 175.9259
                    # "v_bird": 190.5
                }
            bird_id += 1

        # add pigs

        pig_id = 0
        pigs = vision.find_pigs_mbr()
        for pig in pigs:
            temp_pt = pig.get_centre_point()

            # TODO change computer_vision.cv_utils.Rectangle
            # to be more intuitive
            problem_data[f"pig_{pig_id}"] = {
                "x_pig": pig.X + min(pig.width, pig.height) / 2,
                "y_pig": 640 - pig.Y + min(pig.width, pig.height) / 2,
                "m_pig": pig.width * pig.height,  # check this because it is not mandatory
                "pig_radius": min(pig.width, pig.height) / 2,  # check this
                "pig_life": 1  # check

            }
            pig_id += 1

        solution_path = 'agents/pddl/pddl_files/solution.pddl'
        write_problem_file('agents/pddl/pddl_files/problem.pddl', problem_data, 0, 0.2)
        os.chdir('agents/pddl/pddl_files/')
        subprocess.call(
            ['java', '-jar', 'enhsp-20.jar', '-o', 'domain.pddl', '-f', 'problem.pddl', '-sp', 'solution.pddl'])
        os.chdir('../../..')
        actions = parse_solution_to_actions(solution_path, 0, 0.2)
        return actions
