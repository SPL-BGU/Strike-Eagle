import os.path
import numpy as np
import random
from agents import BaselineAgent
from agents.utility import GroundTruthType
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
from agents.pddl.pddl_files.pddl_parser import generate_pddl,write_problem_file


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

        # DEFINE PROBLEM
        self.define_problem(vision)

    def define_problem(self, vision: GroundTruthReader):
        """
        Formulate_image
        """
        # GET Problem
        initial_angle = self.min_deg
        angle_rate = self.deg_step
        active_bird = 0
        bounce_count = 0
        gravity = 10
        # Define birds
        BIRD_TYPES = [GameObjectType.REDBIRD, GameObjectType.YELLOWBIRD, GameObjectType.BLACKBIRD,
                      GameObjectType.WHITEBIRD, GameObjectType.BLUEBIRD]
        bird_id = 0
        problem_data = dict()
        birds_types = vision.find_birds()
        for bird_type, birds in birds_types.items():
            for bird in birds:
                problem_data[f"bird_{bird_id}"] = {
                    "x_bird": bird.X + (min(bird.width, bird.height) / 2), #move to center
                    "y_bird": 640 - bird.Y + min(bird.width, bird.height) / 2,
                    "bird_id": bird_id,
                    "bird_type": BIRD_TYPES.index(GameObjectType(bird_type)),
                    "m_bird": bird.width * bird.height,  # check this because it is not mandatory
                    "bird_radius": min(bird.width, bird.height) / 2,  # check this

                    # "bird_radius": 8,  # check this
                    "v_bird": 175.9259
                }
            bird_id += 1

        # add pigs

        pig_id = 0
        pigs = vision.find_pigs_mbr()
        for pig in pigs:
            problem_data[f"pig_{pig_id}"] = {
                "x_pig": pig.X + (min(pig.width, pig.height) / 2),
                "y_pig": 640 - pig.Y + min(pig.width, pig.height) / 2,
                "m_pig": pig.width * pig.height,  # check this because it is not mandatory
                "pig_radius": min(pig.width, pig.height) / 2,  # check this
                "pig_life": 1  # check

            }
            pig_id += 1

        data_objects = list()
        predicates = list()


        write_problem_file('agents/pddl/pddl_files/problem.pddl',problem_data,0,0.2)
