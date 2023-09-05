import pddl_plus_parser
import os.path
import numpy as np
import random

import unified_planning.model

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
from unified_planning.shortcuts import *
from unified_planning.model import Type
from pddl import parse_domain
from pddl.formatter import problem_to_string
from pddl.core import Problem,Domain,Requirements
from pddl.logic import constants, Predicate
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
        self.sim_speed = 10
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
        # GET DOMAIN
        domain = parse_domain('agents/pddl/pddl_files/domain.pddl')

        requirements = [Requirements.STRIPS, Requirements.TYPING]

        # SET OBJECTS
        objects = []
        for bird_type,birds in vision.find_birds().items():
            objects += [constants(bird_type+str(i),types=[bird_type])[0] for i in range(len(birds))]
        for pig_index in range(len(vision.find_pigs_mbr())):
            objects.append(constants("pig" + str(pig_index),types=['pig'])[0])
        objects.append(constants('slingshot',types=['slingshot'])[0])

        # predicates
        predicate = Predicate("at-location", objects[0], objects[-1]) # bird is at location

        # goal
        goal = Predicate("at-location", objects[0], objects[1])  # bird is at location

        problem = Problem(
            name='angry-birds-level',
            domain=domain,
            requirements=requirements,
            objects=objects,
            init=[predicate],
            goal=goal,

        )
        print(problem_to_string(problem))

