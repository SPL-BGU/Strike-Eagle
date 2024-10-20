import os.path
import time

import numpy as np
from numpy.polynomial.polynomial import Polynomial
from scipy.optimize import minimize
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
import matplotlib.pyplot as plt
import numpy as np
import random
from agents import BaselineAgent
# from agents.pddl.optimizer import solve_difference
from agents.pddl.pddl_files.pddl_objects import get_birds, get_pigs, get_blocks, get_platforms
from agents.pddl.pddl_files.world_model import WorldModel
from agents.pddl.trajectory_parser import extract_real_trajectory, construct_trajectory
from agents.pddl.visualiator import visualize_trajectory
from agents.utility import GroundTruthType
import subprocess
from agents.utility.vision.relations import *
from agents.pddl.pddl_files.pddl_parser import write_problem_file, parse_solution_to_actions


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
        self.sim_speed = 20
        self.visualize = True
        self.ground_truth_type = GroundTruthType.ground_truth_screenshot
        self.learn = True
        self.world_model = WorldModel()

    def solve(self):
        """
        * Solve a particular level by shooting birds directly to pigs
        * @return GameState: the game state after shots.
        """
        ground_truth_type = GroundTruthType.ground_truth_screenshot
        vision = self._update_reader(ground_truth_type.value, self.if_check_gt)
        sling = vision.find_slingshot_mbr()[0]
        sling.width, sling.height = sling.height, sling.width
        actions = self.get_action_to_perform(self.world_model)
        for action, angle in actions:
            release_point = self.tp.find_release_point(sling, angle * np.pi / 180)
            batch_gt = self.ar.shoot_and_record_ground_truth(release_point.X, release_point.Y, 0, 0, 1, 0)
            observed_trajectory = extract_real_trajectory(batch_gt, angle, self.model, self.target_class)
            estimated_trajectory = construct_trajectory(observed_trajectory[0],angle,self.world_model)

            # cut frames - meed to understand how to get this number
            frames = 200
            observed_trajectory = observed_trajectory[:frames]
            estimated_trajectory = estimated_trajectory[:frames]

            # plt.figure()

            # transform into funcitons
            observed_trajectory_polynom = Polynomial.fit(observed_trajectory[:,0],observed_trajectory[:,1],2,)
            estimated_trajectory_polynom = Polynomial.fit(estimated_trajectory[:, 0], estimated_trajectory[:, 1], 2)

            # visual
            # plt.plot(observed_trajectory[:, 0], observed_trajectory[:, 1], marker='o', color='blue')
            # plt.plot(estimated_trajectory[:, 0], estimated_trajectory[:, 1], marker='x', color='red')
            # plt.axis('equal')  # Equal scaling for x and y axes
            # plt.show()

            delta = 20 # use delta as a percentage

            g_values = np.linspace(self.world_model.gravity - delta,
                                   self.world_model.gravity + delta, 100)

            predicated_trajectories_over_g = [
                construct_trajectory(observed_trajectory[0],angle,WorldModel(g))[:frames]for g in g_values
            ]

            predicated_trajectories_over_g_polynoms = [
                Polynomial.fit(estimated_trajectory[:, 0], estimated_trajectory[:, 1], 2) for estimated_trajectory in predicated_trajectories_over_g
            ]

            def calculate_error(observed, estimated):
                domain = [
                    max(observed.domain[0],estimated.domain[0]),
                    min(observed.domain[1], estimated.domain[1]),
                    #350
                ]
                values = np.linspace(domain[0],
                                       domain[1], 100)
                return np.average(np.abs(observed(values) - estimated(values))) # average cancel not matching size domain comparision

            errors = [calculate_error(observed_trajectory_polynom, predicated_trajectory_polynom) for predicated_trajectory_polynom in predicated_trajectories_over_g_polynoms]
            actual_g_index = np.argmin(errors)
            g_new_value = g_values[actual_g_index]

            # visual
            plt.plot(observed_trajectory[:, 0], observed_trajectory[:, 1], marker='o', color='blue')
            plt.plot(estimated_trajectory[:, 0], estimated_trajectory[:, 1], marker='x', color='red')
            plt.plot(predicated_trajectories_over_g[actual_g_index][:, 0], predicated_trajectories_over_g[actual_g_index][:, 1], marker='.', color='green')
            plt.axis('equal')  # Equal scaling for x and y axes
            plt.show()

            # solve_difference(value* np.pi / 180, observed_trajectory, estimated_trajectory)
            # if self.visualize:
            #     visualize_trajectory(self.model, self.target_class, batch_gt)

            # Update world model
            self.world_model.gravity = g_new_value
            time.sleep(5)
            x = 0

    def get_action_to_perform(self,agent_world_model:WorldModel):
        """
        Formulate_image
        """
        # GET Problem
        initial_angle = self.min_deg
        angle_rate = self.deg_step
        ground_truth_type = GroundTruthType.ground_truth_screenshot
        time.sleep(1)
        vision = self._update_reader(ground_truth_type.value, self.if_check_gt)
        sling = vision.find_slingshot_mbr()[0]
        sling.width, sling.height = sling.height, sling.width

        bird_objects = get_birds(vision, sling, self.tp)

        pigs_objects = get_pigs(vision, sling, self.tp)

        block_objects = get_blocks(vision, sling, self.tp)

        platform_objects = get_platforms(vision, sling, self.tp)

        problem_data = bird_objects | pigs_objects | block_objects | platform_objects

        solution_path = 'agents/pddl/pddl_files/solution.pddl'
        write_problem_file('agents/pddl/pddl_files/problem.pddl', problem_data, 0, 0.2,agent_world_model)
        os.chdir('agents/pddl/pddl_files/')
        subprocess.call(
            ['java', '-jar', 'enhsp-20.jar', '-o', 'domain.pddl', '-f', 'problem.pddl', '-sp', 'solution.pddl',
             '-planner', 'sat'
             '-pt'
             # ,'-sjr','solution_path.json'
             ])
        os.chdir('../../..')
        actions = parse_solution_to_actions(solution_path, 0, 0.2)
        return actions
