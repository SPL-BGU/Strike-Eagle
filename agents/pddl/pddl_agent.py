import math
import os.path
import time
import pickle
import numpy as np
from agents import BaselineAgent
from agents.pddl.optimizer import grid_search, get_poly_rank, get_param_values, calculate_aggregative_erros, \
    get_params_sensitivity
from agents.pddl.pddl_files.events.learn_events import update_model_effects
from agents.pddl.pddl_files.pddl_objects import get_birds, get_pigs, get_blocks, get_platforms
from agents.pddl.pddl_files.segments import getSegmentsPelt, getSegmentsEvents
from agents.pddl.pddl_files.world_model.params import Params
from agents.pddl.pddl_files.world_model.process import Process
from agents.pddl.pddl_files.world_model.world_model import WorldModel
from agents.pddl.trajectory_parser import extract_real_trajectory, construct_trajectory
from agents.pddl.visualiator import plot_errors, plot_score, visualize_compare, visuallize_wins_percentage, \
    visualize_rmse
from agents.utility import GroundTruthType
import subprocess
from agents.utility.vision.relations import *
from agents.pddl.pddl_files.pddl_parser import write_problem_file, parse_solution_to_actions, inject_domain_file
from src.client.agent_client import GameState
from agents.pddl.metrics import calculate_rmse
from scipy.interpolate import BSpline, make_interp_spline
from numpy.polynomial import Polynomial


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
        self.visualize = False
        self.ground_truth_type = GroundTruthType.ground_truth_screenshot
        self.learn = True
        self.world_model = WorldModel({
            Params.gravity: 90,
            Params.velocity: 200
        })

        self.kb = {
            "collision": {
                "states": [],
                "variables": {
                    "v_x": {
                        "value": [],
                        "model": None
                    },
                    "v_y": {
                        "value": [],
                        "model": None
                    },
                    "y": {
                        "value": [],
                        "model": None
                    },
                }
            }
        }
        self.x =0
        self.kb_max_size = 3

        # metrics
        self.error_rate = list()
        self.aggravate_error_rate = list()
        self.aggravate_score = list()
        self.rmse = list()
        self.wins = []
        self.c=0

    def solve(self):
        """
        * Solve a particular level by shooting birds directly to pigs
        * @return GameState: the game state after shots.
        """
        ground_truth_type = GroundTruthType.ground_truth_screenshot
        vision = self._update_reader(ground_truth_type.value, self.if_check_gt)

        with open(f"game-{self.c}.pkl", "wb") as f:
            pickle.dump(vision, f)

        sling = vision.find_slingshot_mbr()[0]
        sling.width, sling.height = sling.height, sling.width
        actions = self.get_action_to_perform(self.world_model)[0]
        task, angle = actions

        release_point = self.tp.find_release_point(sling, angle * np.pi / 180)
        batch_gt = self.ar.shoot_and_record_ground_truth(release_point.X, release_point.Y, 0, 0, 1, 0)
        with open(f"batch-{self.c}.pkl", "wb") as f:
            pickle.dump(vision, f)
        self.c += 1
        time.sleep(2)

        # Analyze observed trajectory
        groundtruth_trajectories,_ = extract_real_trajectory(batch_gt, angle, self.model, self.target_class)


        # getSegmentsPelt(observed_trajectory, 30)

        event_indexes, objects_features = getSegmentsEvents(groundtruth_trajectories)

        bird_observed_trajectory = groundtruth_trajectories["redBird_0"]
        bird_observed_features = objects_features["redBird_0"]
        collisions = event_indexes["collision"]
        parts = np.split(bird_observed_trajectory, collisions)

        # LEARN EVENT

        for collision_index in collisions:
            update_model_effects("collision", self.kb, bird_observed_features[collision_index],
                                 bird_observed_features[collision_index + 1])

        # LEARN PROCESS
        bird_observed_trajectory = parts[0]  # override everything else, only learn on part 1

        new_world_model = self.learn_process( bird_observed_trajectory)

        # VISUALIZE
        limit = np.max( bird_observed_trajectory,axis=0)[0]

        estimated_trajectory = construct_trajectory( bird_observed_trajectory[0], angle, self.world_model, limit, prt=False)
        changed_trajectoty = construct_trajectory( bird_observed_trajectory[0], angle, new_world_model, limit, prt=False)

        self.rmse.append(calculate_rmse( bird_observed_trajectory,estimated_trajectory))

        visualize_compare( bird_observed_trajectory, estimated_trajectory, changed_trajectoty)
        visualize_rmse(self.rmse)

        self.wins.append(self.ar.get_game_state() == GameState.WON)
        visuallize_wins_percentage(self.wins)



        if self.ar.get_game_state() == GameState.LOST:
            print(f"Old values- {self.world_model.hyperparams_values} ")
            print(f"New values- gravity: {new_world_model.hyperparams_values} ")

            # Update world model
            self.world_model = new_world_model
            self.world_model.kb = self.kb

        if len(self.aggravate_score) == 0:
            self.aggravate_score.append(self.check_current_level_score())
        else:
            self.aggravate_score.append(self.aggravate_score[-1] + self.check_current_level_score())

        # plot_score(self.aggravate_score)

        time.sleep(3)

    def get_action_to_perform(self, agent_world_model: WorldModel):
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

        bird_objects = get_birds(vision, sling, self.tp, self.world_model)

        pigs_objects = get_pigs(vision, sling, self.tp)

        block_objects = get_blocks(vision, sling, self.tp)

        platform_objects = get_platforms(vision, sling, self.tp)

        problem_data = bird_objects | pigs_objects | block_objects | platform_objects

        solution_path = 'agents/pddl/pddl_files/solution.pddl'
        write_problem_file('agents/pddl/pddl_files/problem.pddl', problem_data, 0, 0.2, agent_world_model)

        if agent_world_model.kb != None:
            inject_domain_file('agents/pddl/pddl_files/base_domain.pddl',agent_world_model)


        domain_path = 'base_domain_modified.pddl' if agent_world_model.kb != None else 'domain.pddl'
        os.chdir('agents/pddl/pddl_files/')
        subprocess.call(
            ['java', '-jar', 'enhsp-20.jar', '-o', domain_path, '-f', 'problem.pddl', '-sp', 'solution.pddl',
             '-planner', 'sat'
                         '-pt'
             # ,'-sjr','solution_path.json'
             ])
        os.chdir('../../..')
        actions = parse_solution_to_actions(solution_path, 0, 0.2)
        return actions

    def learn_process(self, observed_trajectory: np.ndarray):

        # Trim trajectory
        observed_trajectory = observed_trajectory

        function_range = np.array(range(len(observed_trajectory))) / 50

        # Determine polynomial rank of observed
        rank_x, poly_x = get_poly_rank(function_range, observed_trajectory[:, 0])
        rank_y, poly_y = get_poly_rank(function_range, observed_trajectory[:, 1])

        # visualize_compare(observed_trajectory, estimated_trajectory)

        v0_x = poly_x.deriv().coef[0]
        v0_y = poly_y.coef[1]

        new_values = WorldModel(
            {
                Params.gravity: abs(poly_y.deriv(2)(0)),
                Params.velocity: math.sqrt(v0_x ** 2 + v0_y ** 2)
            }
        )

        return new_values

