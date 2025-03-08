import os.path
import time

from agents import BaselineAgent
from agents.pddl.optimizer import grid_search, get_poly_rank, get_param_values, calculate_aggregative_erros, \
    get_params_sensitivity
from agents.pddl.pddl_files.pddl_objects import get_birds, get_pigs, get_blocks, get_platforms
from agents.pddl.pddl_files.world_model.params import Params
from agents.pddl.pddl_files.world_model.process import Process
from agents.pddl.pddl_files.world_model.world_model import WorldModel
from agents.pddl.trajectory_parser import extract_real_trajectory, construct_trajectory
from agents.pddl.visualiator import plot_errors
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
        self.visualize = False
        self.ground_truth_type = GroundTruthType.ground_truth_screenshot
        self.learn = True
        self.world_model = WorldModel({
            Params.gravity: 87.2,
            Params.velocity:175
        })

        self.kb = list()
        self.kb_max_size = 3

        # metrics
        self.error_rate = list()
        self.aggravate_error_rate = list()

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
            estimated_trajectory = construct_trajectory(observed_trajectory[0],angle,self.world_model,prt=False)


            new_values = self.improve_model(observed_trajectory,estimated_trajectory,angle)

            plot_errors(self.error_rate,self.aggravate_error_rate)

            print(f"Old values- {self.world_model.hyperparams_values} ")
            print(f"New values- gravity: {new_values[Params.gravity]},\t v_bird:{new_values[Params.velocity]} ")
            self.world_model = WorldModel(new_values)
            time.sleep(3)

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

        bird_objects = get_birds(vision, sling, self.tp,self.world_model)

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

    def improve_model(self,observed_trajectory:np.ndarray,estimated_trajectory:np.ndarray,angle:float):

        # Trim trajectory
        frames = 150
        observed_trajectory = observed_trajectory[:frames]
        estimated_trajectory = estimated_trajectory[:frames]

        # Determine polynomial rank of observed
        rank = get_poly_rank(observed_trajectory[:, 0], observed_trajectory[:, 1])

        affecting_parmas = self.world_model.get_all_hyper_parameters(Process.flight)
        # Define simulating function
        def simulating_function(observed_trajectory, values):
            return construct_trajectory(observed_trajectory[0], angle, WorldModel(values), prt=False)[:frames]

        # Get best params to improve
        # param_sensitivity,pivot = get_params_sensitivity(
        #     observed_trajectory,
        #     estimated_trajectory,
        #     self.world_model,
        #     rank,
        #     simulating_function
        # )

        # select values based on sensitivity

        param_values = get_param_values(self.world_model.hyperparams_values, 0.05, 0.5)

        param_values, errors = grid_search(
            observed_trajectory,
            estimated_trajectory,
            param_values,
            rank,
            simulating_function
        )

        # Build KB
        self.add_to_kb(param_values, errors)

        # Check aggregative errors
        aggregate_errors = calculate_aggregative_erros(param_values, errors, self.kb)

        # Choose minimal error index
        chosen_index = np.argmin(aggregate_errors)

        # log
        self.error_rate.append(errors[chosen_index])
        self.aggravate_error_rate.append(aggregate_errors[chosen_index])

        new_values = param_values[chosen_index]

        return new_values

    def add_to_kb(self, grid_values, errors):

        current_iteration = dict()
        # Construct to KB
        for grid_value,error in zip(grid_values,errors):
            current_iteration[tuple(grid_value)] = error

        # Add to KB
        if len(self.kb) == self.kb_max_size:
            self.kb.pop()
        self.kb.insert(0,current_iteration)
