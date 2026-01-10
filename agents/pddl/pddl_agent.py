import math
import os.path
import time
import pickle
import numpy as np
from agents import BaselineAgent
from agents.pddl.optimizer import grid_search, get_poly_rank, get_param_values, calculate_aggregative_erros, \
    get_params_sensitivity, compute_derivatives, fit_state_transition
from agents.pddl.pddl_files.events.learn_events import update_model_effects
from agents.pddl.pddl_files.pddl_objects import get_birds, get_pigs, get_blocks, get_platforms
from agents.pddl.pddl_files.segments import getSegmentsPelt, getSegmentsEvents
from agents.pddl.pddl_files.world_model.params import Params
from agents.pddl.pddl_files.world_model.process import Process
from agents.pddl.pddl_files.world_model.world_model import WorldModel
from agents.pddl.trajectory_parser import extract_real_trajectory, construct_trajectory
from agents.pddl.visualiator import plot_errors, plot_score, visualize_compare, visuallize_wins_percentage, \
    visualize_rmse, visualize_rmse_vs_suggsted
from agents.utility import GroundTruthType
import subprocess
from agents.utility.vision.relations import *
from agents.pddl.pddl_files.pddl_parser import write_problem_file, parse_solution_to_actions, inject_domain_file
from src.client.agent_client import GameState
from agents.pddl.metrics import calculate_rmse

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

        # Storage for learned state transition functions
        # Each entry contains a regression model, polynomial representation, string representation, and initial value
        self.learned_transitions = {
            "x": None,      # {"model": regression_model, "polynomial": Polynomial, "string": str, "initial_value": float}
            "y": None,      # {"model": regression_model, "polynomial": Polynomial, "string": str, "initial_value": float}
            "xdot": None,   # {"model": regression_model, "polynomial": Polynomial, "string": str, "initial_value": float}
            "ydot": None,   # {"model": regression_model, "polynomial": Polynomial, "string": str, "initial_value": float}
            "xddot": None,  # {"model": regression_model, "polynomial": Polynomial, "string": str, "initial_value": float}
            "yddot": None   # {"model": regression_model, "polynomial": Polynomial, "string": str, "initial_value": float}
        }

        # metrics
        self.error_rate = list()
        self.aggravate_error_rate = list()
        self.aggravate_score = list()
        self.rmse = list()
        self.suggested_rmse = list()
        self.wins = []
        self.c=0

    def solve(self):
        """
        * Solve a particular level by shooting birds directly to pigs
        * @return GameState: the game state after shots.
        """
        ground_truth_type = GroundTruthType.ground_truth_screenshot

        vision = self._update_reader(ground_truth_type.value, self.if_check_gt)


        sling = vision.find_slingshot_mbr()[0]
        sling.width, sling.height = sling.height, sling.width
        actions = self.get_action_to_perform(self.world_model)[0]
        task, angle = actions

        release_point = self.tp.find_release_point(sling, angle * np.pi / 180)

        batch_gt = self.ar.shoot_and_record_ground_truth(release_point.X, release_point.Y, 0, 0, 1, 0)
        # with open(f"batch-3.pkl", "rb") as f:
        #     batch_gt = pickle.load(f)
        #
        # time.sleep(2)

        # Analyze observed trajectory
        groundtruth_trajectories,groundtruth_objects = extract_real_trajectory(batch_gt, angle, self.model, self.target_class)


        # getSegmentsPelt(observed_trajectory, 30)

        event_indexes_by_event, objects_features = getSegmentsEvents(groundtruth_trajectories,groundtruth_objects)

        bird_observed_trajectory = groundtruth_trajectories["redBird_0"]

        bird_observed_features = objects_features["redBird_0"]

        event_indexes = sorted([val for values in event_indexes_by_event.values() for val in values])

        parts = np.split(bird_observed_trajectory, event_indexes)



        # LEARN EVENT

        collisions = event_indexes_by_event["ground_collision"]

        for collision_index in collisions:
            update_model_effects("collision", self.kb, bird_observed_features[collision_index],
                                 bird_observed_features[collision_index + 1])

        # LEARN PROCESS
        bird_observed_trajectory = parts[0]  # override everything else, only learn on part 1

        # Learn physics parameters (gravity, velocity) - existing method
        new_world_model = self.learn_process( bird_observed_trajectory)
        
        # Learn state transition functions - new method
        # This learns how to predict next state from previous state
        self.learn_process_transitions(bird_observed_trajectory)
        
        # Create a new world model from learned transitions
        self.learned_transition_world_model = self._create_learned_transition_world_model()
        
        # Print both world models
        print(f"\nOriginal World Model: {new_world_model.hyperparams_values}")
        print(f"Learned Transition World Model: {self.learned_transition_world_model.hyperparams_values}")

        # VISUALIZE
        limit = np.max( bird_observed_trajectory,axis=0)[0]

        estimated_trajectory = construct_trajectory( bird_observed_trajectory[0], angle, self.world_model, limit, prt=False)
        suggested_trajectoty = construct_trajectory( bird_observed_trajectory[0], angle, new_world_model, limit, prt=False)

        self.rmse.append(calculate_rmse( bird_observed_trajectory,estimated_trajectory))
        self.suggested_rmse.append(calculate_rmse(bird_observed_trajectory,suggested_trajectoty))

        # visualize_compare( bird_observed_trajectory, estimated_trajectory, suggested_trajectoty)
        # visualize_rmse(self.rmse)
        # visualize_rmse_vs_suggsted(self.rmse,self.suggested_rmse)

        self.wins.append(self.ar.get_game_state() == GameState.WON)
        # visuallize_wins_percentage(self.wins)



        print(f"Old values- {self.world_model.hyperparams_values} ")
        print(f"New values- gravity: {new_world_model.hyperparams_values} ")
            # Update world model
        self.world_model = self.learned_transition_world_model
        self.world_model.kb = self.kb

        if len(self.aggravate_score) == 0:
            self.aggravate_score.append(self.check_current_level_score())
        else:
            self.aggravate_score.append(self.aggravate_score[-1] + self.check_current_level_score())

        # plot_score(self.aggravate_score)
        print(self.rmse)

        time.sleep(5)

    def get_action_to_perform(self, agent_world_model: WorldModel):
        """
        Formulate_image
        """
        # GET Problem
        initial_angle = self.min_deg
        angle_rate = self.deg_step
        ground_truth_type = GroundTruthType.ground_truth_screenshot
        time.sleep(1)
        vision = self._update_reader(ground_truth_type.value,self.if_check_gt)

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
        try:
            subprocess.call(
                ['java', '-jar', 'enhsp-20.jar', '-o', domain_path, '-f', 'problem.pddl', '-sp', 'solution.pddl',
                 '-planner', 'sat'
                             '-pt'
                 # ,'-sjr','solution_path.json'
                 ],timeout=200)
            os.chdir('../../..')
            actions = parse_solution_to_actions(solution_path, 0, 0.2)
        except:
            actions = [("shoot",45)]
            os.chdir('../../..')

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

    def learn_process_transitions(self, observed_trajectory: np.ndarray):
        """
        Learn state transition functions using a 3-step process:
        1. Fit curves to observed x,y positions
        2. Sample curves and compute derivatives (xdot, xddot, ydot, yddot)
        3. Fit polynomial transition functions for each state variable
        
        This method learns how to predict the next state from the previous state,
        rather than learning global physics parameters.
        
        Parameters:
        -----------
        observed_trajectory : np.ndarray
            Observed trajectory points as (x, y) coordinates
            Shape: (n_points, 2)
            
        Returns:
        --------
        None
            Learned transition polynomials are stored in self.learned_transitions
        """

        start_frame = 0
        # ========================================================================
        # STEP 1: Fit curves to observed x,y positions
        # ========================================================================
        # Convert trajectory indices to time values (each frame = 0.02 seconds)
        # So time = index / 50
        # Time starts from 0 for the selected segment
        trajectory_segment = observed_trajectory[start_frame:]
        function_range = np.array(range(len(trajectory_segment))) / 50.0  # Each frame = 0.02 seconds
        
        # Fit polynomials to x(t) and y(t) trajectories
        # get_poly_rank() automatically selects the optimal polynomial degree
        rank_x, poly_x = get_poly_rank(function_range, observed_trajectory[start_frame:, 0])
        rank_y, poly_y = get_poly_rank(function_range, observed_trajectory[start_frame:, 1])
        
        # ========================================================================
        # STEP 2: Sample curves and compute derivatives
        # ========================================================================
        # Get the actual x and y values at each time sample from the fitted polynomials
        x_values = np.array([poly_x(t) for t in function_range])
        y_values = np.array([poly_y(t) for t in function_range])
        
        # # Validate computed values
        # x_values = np.nan_to_num(x_values, nan=0.0, posinf=0.0, neginf=0.0)
        # y_values = np.nan_to_num(y_values, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Compute derivatives for all time samples using numerical differentiation
        # This is more accurate than analytical derivatives of fitted polynomials
        # - xdot(t) = first derivative of x(t) = dx/dt
        # - xddot(t) = second derivative of x(t) = d²x/dt²
        # - ydot(t) = first derivative of y(t) = dy/dt
        # - yddot(t) = second derivative of y(t) = d²y/dt²
        xdot, xddot, ydot, yddot = compute_derivatives(poly_x, poly_y, function_range, 
                                                       x_values=x_values, y_values=y_values)
        
        # ========================================================================
        # STEP 3: Fit polynomial transition functions
        # ========================================================================
        # For each state variable, fit a polynomial that predicts the current value
        # from the previous state values
        
        # Prepare training data: for each time step t from 1 to n-1:
        # - Previous state: state at time t-1
        # - Current state: state at time t (target for prediction)
        n_samples = len(observed_trajectory)
        
        # Need at least 2 samples to create training pairs
        if n_samples < 2:
            print("Warning: Not enough trajectory samples for learning transitions. Skipping.")
            return
        
        try:
            # x(t) = poly(x(t-1), xdot(t-1), xddot(t-1))
            # Previous state features: [x[t-1], xdot[t-1], xddot[t-1]]
            x_prev = np.column_stack([x_values[:-1], xdot[:-1], xddot[:-1]])
            x_curr = x_values[1:]  # Current x values (target)
            
            # Debug: Check if data makes sense
            if len(x_prev) > 0 and len(x_curr) > 0:
                # x(t) should be approximately x(t-1) + xdot(t-1)*dt
                # Check the actual relationship
                x_diff = x_curr - x_prev[:, 0]  # x(t) - x(t-1)
                dt = function_range[1] - function_range[0] if len(function_range) > 1 else 0.02
                expected_diff = xdot[:-1] * dt  # xdot(t-1) * dt
                correlation = np.corrcoef(x_prev[:, 0], x_curr)[0, 1] if len(x_prev) > 1 else 0
                
                print(f"\nDEBUG x(t) relationship:")
                print(f"  Mean x(t) - x(t-1): {np.mean(x_diff):.6f}")
                print(f"  Mean xdot(t-1)*dt: {np.mean(expected_diff):.6f}")
                print(f"  Correlation between x(t-1) and x(t): {correlation:.6f}")
                print(f"  Sample: x_prev[0]={x_prev[0]}, x_curr[0]={x_curr[0]}, diff={x_curr[0]-x_prev[0,0]:.6f}")
                print(f"  If x(t) ≈ x(t-1) + xdot(t-1)*dt, then coefficient for x(t-1) should be ≈ 1.0")
            
            # fit_state_transition returns {"model": regression_model, "polynomial": Polynomial}
            self.learned_transitions["x"] = fit_state_transition(x_curr, x_prev)
            
            # y(t) = poly(y(t-1), ydot(t-1), yddot(t-1))
            # Previous state features: [y[t-1], ydot[t-1], yddot[t-1]]
            y_curr = y_values[1:]  # Current y values (target)
            y_prev = np.column_stack([y_values[:-1], ydot[:-1], yddot[:-1]])
            
            # Debug: Check if data makes sense
            if len(y_prev) > 0 and len(y_curr) > 0:
                y_diff = y_curr - y_prev[:, 0]  # y(t) - y(t-1)
                print(f"\nDEBUG y(t) relationship:")
                print(f"  Mean y(t) - y(t-1): {np.mean(y_diff):.6f}")
                print(f"  Mean ydot(t-1): {np.mean(ydot[:-1]):.6f}")
            
            self.learned_transitions["y"] = fit_state_transition(y_curr, y_prev)
            
            # xdot(t) = poly(xdot(t-1), xddot(t-1))
            # Previous state features: [xdot[t-1], xddot[t-1]]
            xdot_prev = np.column_stack([xdot[:-1], xddot[:-1]])
            xdot_curr = xdot[1:]  # Current xdot values (target)
            self.learned_transitions["xdot"] = fit_state_transition(xdot_curr, xdot_prev)
            
            # ydot(t) = poly(ydot(t-1), yddot(t-1))
            # Previous state features: [ydot[t-1], yddot[t-1]]
            ydot_prev = np.column_stack([ydot[:-1], yddot[:-1]])
            ydot_curr = ydot[1:]  # Current ydot values (target)
            self.learned_transitions["ydot"] = fit_state_transition(ydot_curr, ydot_prev)
            
            # xddot(t) = poly(xddot(t-1))
            # BUT: xddot should be constant (0 - no horizontal acceleration)
            # So we force it to be constant instead of learning a transition function
            if len(xddot) > 0:
                # xddot should be 0 (no horizontal acceleration)
                xddot_constant = 0.0
                
                # Create a constant model for xddot
                class ConstantXddotModel:
                    def __init__(self, constant_value):
                        self.constant_value = constant_value
                    def predict(self, X):
                        if hasattr(X, '__len__') and not isinstance(X, (str, bytes)):
                            return np.full(len(X), self.constant_value)
                        return np.array([self.constant_value])
                
                constant_poly = Polynomial([xddot_constant])
                self.learned_transitions["xddot"] = {
                    "model": ConstantXddotModel(xddot_constant),
                    "polynomial": constant_poly
                }
                
                # Update xddot array to be constant for use in other transitions
                xddot = np.full_like(xddot, xddot_constant)
            else:
                # Fallback if xddot is empty
                self.learned_transitions["xddot"] = None
            
            # yddot(t) = poly(yddot(t-1))
            # BUT: yddot should be constant (gravity doesn't change)
            # So we force it to be constant instead of learning a transition function
            if len(yddot) > 0:
                # Use analytical second derivative from polynomial (should be constant)
                # This is more accurate than numerical derivatives which can have noise
                try:
                    yddot_constant = poly_y.deriv(2)(0)  # Second derivative at t=0 (should be constant)
                except:
                    # Fallback to mean of computed yddot values
                    yddot_constant = np.mean(yddot)
                
                # Create a constant model for yddot
                class ConstantYddotModel:
                    def __init__(self, constant_value):
                        self.constant_value = constant_value
                    def predict(self, X):
                        if hasattr(X, '__len__') and not isinstance(X, (str, bytes)):
                            return np.full(len(X), self.constant_value)
                        return np.array([self.constant_value])
                
                constant_poly = Polynomial([yddot_constant])
                self.learned_transitions["yddot"] = {
                    "model": ConstantYddotModel(yddot_constant),
                    "polynomial": constant_poly
                }
                
                # Update yddot array to be constant for use in other transitions (ydot depends on yddot)
                yddot = np.full_like(yddot, yddot_constant)
            else:
                # Fallback if yddot is empty
                self.learned_transitions["yddot"] = None
            
            # Extract and store initial values (t=0)
            # For constant models (xddot, yddot), use the constant value
            xddot_initial = 0.0
            yddot_initial = 0.0
            if self.learned_transitions.get("xddot") is not None:
                xddot_model = self.learned_transitions["xddot"].get("model")
                if hasattr(xddot_model, 'constant_value'):
                    xddot_initial = round(xddot_model.constant_value, 2)
            if self.learned_transitions.get("yddot") is not None:
                yddot_model = self.learned_transitions["yddot"].get("model")
                if hasattr(yddot_model, 'constant_value'):
                    yddot_initial = round(yddot_model.constant_value, 2)
            
            initial_values = {
                "x": round(x_values[0], 2) if len(x_values) > 0 else 0.0,
                "y": round(y_values[0], 2) if len(y_values) > 0 else 0.0,
                "xdot": round(xdot[0], 2) if len(xdot) > 0 else 0.0,
                "ydot": round(ydot[0], 2) if len(ydot) > 0 else 0.0,
                "xddot": xddot_initial,
                "yddot": yddot_initial
            }
            
            # Store initial values in each transition dictionary
            for var_name in ["x", "y", "xdot", "ydot", "xddot", "yddot"]:
                if self.learned_transitions[var_name] is not None:
                    self.learned_transitions[var_name]["initial_value"] = initial_values[var_name]
            
            # Add string representations for all learned transitions
            # Define feature dependencies for each variable
            feature_dependencies = {
                "x": ["x(t-1)", "xdot(t-1)", "xddot(t-1)"],
                "y": ["y(t-1)", "ydot(t-1)", "yddot(t-1)"],
                "xdot": ["xdot(t-1)", "xddot(t-1)"],
                "ydot": ["ydot(t-1)", "yddot(t-1)"],
                "xddot": ["xddot(t-1)"],
                "yddot": ["yddot(t-1)"]
            }
            
            for var_name in ["x", "y", "xdot", "ydot", "xddot", "yddot"]:
                if self.learned_transitions[var_name] is not None:
                    transition_dict = self.learned_transitions[var_name]
                    features = feature_dependencies.get(var_name, [f"{var_name}(t-1)"])
                    
                    # Check if this is a constant model (for xddot and yddot)
                    model = transition_dict.get("model")
                    if hasattr(model, 'constant_value'):
                        # Constant model - just output the constant value
                        constant_val = round(model.constant_value, 2)
                        transition_dict["string"] = f"{var_name}(t) = {constant_val:.2f}"
                    else:
                        # Regular transition function
                        transition_dict["string"] = self._get_transition_string(
                            var_name, 
                            transition_dict["polynomial"],
                            transition_dict["model"],
                            features
                        )
                    
                    # Add initial value to string representation
                    if "initial_value" in transition_dict:
                        initial_val = transition_dict["initial_value"]
                        transition_dict["string"] += f", {var_name}(0) = {initial_val:.2f}"
                    elif hasattr(model, 'constant_value'):
                        # For constant models, initial value is the same as the constant
                        constant_val = round(model.constant_value, 2)
                        transition_dict["string"] += f", {var_name}(0) = {constant_val:.2f}"
            
            # Print detailed coefficients for all learned transitions
            print("\n" + "="*80)
            print("DETAILED COEFFICIENTS FOR LEARNED TRANSITIONS:")
            print("="*80)
            for var_name in ["x", "y", "xdot", "ydot", "xddot", "yddot"]:
                if self.learned_transitions[var_name] is not None:
                    transition_dict = self.learned_transitions[var_name]
                    model = transition_dict.get("model")
                    features = feature_dependencies.get(var_name, [f"{var_name}(t-1)"])
                    
                    print(f"\n{var_name}(t):")
                    if model is not None:
                        if hasattr(model, 'poly_features') and hasattr(model, 'coef_'):
                            # Multi-feature model with PolynomialFeatures
                            poly_features = model.poly_features
                            coefs = model.coef_
                            intercept = model.intercept_
                            
                            try:
                                feature_names_poly = poly_features.get_feature_names_out()
                            except AttributeError:
                                feature_names_poly = poly_features.get_feature_names()
                            
                            print(f"  Intercept: {intercept:.6f}")
                            print(f"  Total polynomial features: {len(feature_names_poly)}")
                            print(f"  Total coefficients: {len(coefs)}")
                            print(f"  Feature names from PolynomialFeatures: {list(feature_names_poly[:min(10, len(feature_names_poly))])}")  # Show first 10
                            bias_offset = 1 if len(feature_names_poly) > 0 and feature_names_poly[0] == "1" else 0
                            print(f"  Bias offset: {bias_offset}")
                            
                            # Check specifically for the first feature (should be x(t-1) or y(t-1))
                            print(f"  First feature name: '{features[0] if features else 'N/A'}'")
                            print(f"  Looking for coefficient mapping to '{features[0] if features else 'N/A'}'")
                            
                            # IMPORTANT: When PolynomialFeatures has include_bias=True and LinearRegression has fit_intercept=True,
                            # there's a double bias. LinearRegression's coef_ includes coefficients for ALL columns in X_poly,
                            # including the bias column. So coef_[0] corresponds to feature_names_poly[0] which is '1'.
                            # We need to check if LinearRegression actually used the bias column or its own intercept.
                            
                            # Check if first coefficient corresponds to bias column
                            if len(coefs) == len(feature_names_poly):
                                # coef_ includes bias column from PolynomialFeatures
                                # coef_[0] = coefficient for '1' (bias column)
                                # coef_[1] = coefficient for 'x0'
                                # etc.
                                print(f"  NOTE: coef_ includes bias column from PolynomialFeatures")
                                for i, coef in enumerate(coefs):
                                    if i < len(feature_names_poly):
                                        poly_feat_name = feature_names_poly[i]
                                        if poly_feat_name == "1":
                                            print(f"  Coefficient[{i}] -> '{poly_feat_name}' (bias from PolynomialFeatures): {coef:.6f}")
                                        else:
                                            readable_term = self._parse_polynomial_feature(poly_feat_name, features)
                                            if readable_term:
                                                is_first_feature = (readable_term == features[0] if features else False)
                                                marker = " <-- FIRST FEATURE" if is_first_feature else ""
                                                print(f"  Coefficient[{i}] -> '{poly_feat_name}' -> '{readable_term}': {coef:.6f}{marker}")
                                            else:
                                                print(f"  Coefficient[{i}] -> '{poly_feat_name}' (unparseable): {coef:.6f}")
                            else:
                                # coef_ does NOT include bias column (LinearRegression handles it separately)
                                # Use bias_offset as before
                                for i, coef in enumerate(coefs):
                                    poly_idx = i + bias_offset
                                    if poly_idx < len(feature_names_poly):
                                        poly_feat_name = feature_names_poly[poly_idx]
                                        readable_term = self._parse_polynomial_feature(poly_feat_name, features)
                                        if readable_term:
                                            is_first_feature = (readable_term == features[0] if features else False)
                                            marker = " <-- FIRST FEATURE" if is_first_feature else ""
                                            print(f"  Coefficient[{i}] -> '{poly_feat_name}' -> '{readable_term}': {coef:.6f}{marker}")
                                        else:
                                            print(f"  Coefficient[{i}] -> '{poly_feat_name}' (unparseable): {coef:.6f}")
                                    else:
                                        print(f"  Coefficient[{i}]: {coef:.6f} (index {poly_idx} out of range)")
                            
                            # Also print raw coefficient array for debugging
                            print(f"  Raw coefficients array: {coefs[:min(5, len(coefs))]}")  # First 5 coefficients
                        elif hasattr(model, 'coef_'):
                            # Simple linear model
                            coefs = model.coef_
                            intercept = model.intercept_ if hasattr(model, 'intercept_') else 0
                            print(f"  Intercept: {intercept:.6f}")
                            if len(coefs.shape) == 1:
                                for i, coef in enumerate(coefs):
                                    if i < len(features):
                                        print(f"  Coefficient for '{features[i]}': {coef:.6f}")
                                    else:
                                        print(f"  Coefficient[{i}]: {coef:.6f}")
                            else:
                                print(f"  Coefficients shape: {coefs.shape}")
                                print(f"  Coefficients: {coefs}")
                    else:
                        print("  Model is None")
                else:
                    print(f"\n{var_name}(t): Not learned")
            
            # Print all learned transition functions (formatted strings)
            print("\n" + "="*80)
            print("LEARNED STATE TRANSITION FUNCTIONS (FORMATTED):")
            print("="*80)
            for var_name in ["x", "y", "xdot", "ydot", "xddot", "yddot"]:
                if self.learned_transitions[var_name] is not None and "string" in self.learned_transitions[var_name]:
                    print(self.learned_transitions[var_name]["string"])
                else:
                    print(f"{var_name}(t) = Not learned")
            print("="*80 + "\n")
        except Exception as e:
            print(f"Error in learn_process_transitions step 3: {e}")
            print("Some transition functions may not have been learned.")
    
    def _create_learned_transition_world_model(self):
        """
        Create a new WorldModel from learned transitions.
        Only extracts yddot (gravity) and v (velocity from vx and vy).
        
        Returns:
        --------
        WorldModel
            A new WorldModel instance with gravity and velocity from learned transitions
        """
        initial_values = {}
        
        # Get gravity from yddot (constant value)
        if self.learned_transitions.get("yddot") is not None:
            yddot_model = self.learned_transitions["yddot"].get("model")
            if hasattr(yddot_model, 'constant_value'):
                initial_values[Params.gravity] = abs(yddot_model.constant_value)
        
        # Calculate velocity from xdot and ydot initial values
        vx = None
        vy = None
        if self.learned_transitions.get("xdot") is not None and "initial_value" in self.learned_transitions["xdot"]:
            vx = self.learned_transitions["xdot"]["initial_value"]
        if self.learned_transitions.get("ydot") is not None and "initial_value" in self.learned_transitions["ydot"]:
            vy = self.learned_transitions["ydot"]["initial_value"]
        
        if vx is not None and vy is not None:
            initial_values[Params.velocity] = math.sqrt(vx**2 + vy**2)
        
        return WorldModel(initial_values)
    
    def _get_transition_string(self, variable_name: str, poly, model, feature_names: list = None):
        """
        Get string representation of learned transition function showing all feature relationships.
        
        Parameters:
        -----------
        variable_name : str
            Name of the state variable (e.g., "x", "y", "xdot", etc.)
        poly : Polynomial
            The polynomial object representing the transition (simplified)
        model : object
            The regression model (sklearn LinearRegression or PolynomialModel wrapper)
        feature_names : list, optional
            List of feature names for multi-feature transitions (e.g., ["x(t-1)", "xdot(t-1)", "xddot(t-1)"])
            If None, defaults to single feature: [f"{variable_name}(t-1)"]
            
        Returns:
        --------
        str
            String representation showing full relationship with all features
        """
        if poly is None or model is None:
            return f"{variable_name}(t) = Not learned"
        
        try:
            # For multi-feature transitions, extract full polynomial from sklearn model
            if feature_names and len(feature_names) > 1:
                features_str = ", ".join(feature_names)
                
                # Check if model has poly_features attribute (sklearn model)
                if hasattr(model, 'poly_features'):
                    # This is a sklearn LinearRegression with PolynomialFeatures
                    poly_features = model.poly_features
                    coefs = model.coef_
                    intercept = model.intercept_
                    
                    # Get feature names from PolynomialFeatures
                    try:
                        # For sklearn >= 1.0
                        feature_names_poly = poly_features.get_feature_names_out()
                    except AttributeError:
                        # For sklearn < 1.0
                        feature_names_poly = poly_features.get_feature_names()
                    
                    # Map polynomial feature names to our readable names
                    # PolynomialFeatures uses x0, x1, x2 for features
                    # We need to map them to our feature_names
                    terms = []
                    
                    # Process each coefficient
                    # IMPORTANT: When PolynomialFeatures has include_bias=True and LinearRegression has fit_intercept=False,
                    # coef_ includes the bias column. coef_[0] corresponds to feature_names_poly[0] ('1' bias column).
                    # When fit_intercept=True (default), intercept_ is separate and coef_[0] corresponds to feature_names_poly[1].
                    
                    # Check if coef_ length matches feature_names_poly length
                    # If yes, coef_ includes bias column - coef_[0] is for '1', coef_[1] is for 'x0', etc.
                    # If no, LinearRegression handled bias separately and coef_[0] corresponds to feature_names_poly[1]
                    if len(coefs) == len(feature_names_poly):
                        # coef_ includes bias column - use intercept from coef_[0] instead of intercept_
                        intercept_from_coef = coefs[0] if len(coefs) > 0 else 0
                        intercept_rounded = round(intercept_from_coef, 2)
                        if abs(intercept_rounded) > 1e-6:  # Only skip truly negligible intercepts
                            terms.append(f"{intercept_rounded:.2f}")
                        bias_offset = 0
                        start_idx = 1  # Start from index 1 to skip bias column
                    else:
                        # coef_ does NOT include bias column - use intercept_ separately
                        intercept_rounded = round(intercept, 2)
                        if abs(intercept_rounded) > 1e-6:  # Only skip truly negligible intercepts
                            terms.append(f"{intercept_rounded:.2f}")
                        bias_offset = 1 if len(feature_names_poly) > 0 and feature_names_poly[0] == "1" else 0
                        start_idx = 0  # Start from index 0
                    
                    for i in range(start_idx, len(coefs)):
                        # Round to 2 decimal places and drop if zero
                        coef_rounded = round(coefs[i], 2)
                        if abs(coef_rounded) < 1e-6:  # Drop zero coefficients
                            continue
                            
                        poly_idx = i + bias_offset
                        if poly_idx < len(feature_names_poly):
                            poly_feat_name = feature_names_poly[poly_idx]
                            
                            # Skip the bias column '1' - it's already handled above
                            if poly_feat_name == "1":
                                continue
                            
                            # Parse the polynomial feature name (e.g., "x0 x1" -> ["x0", "x1"])
                            # and convert to readable format
                            readable_term = self._parse_polynomial_feature(poly_feat_name, feature_names)
                            if readable_term:
                                # Format: if coefficient is 1.0, show without the "1.00*" prefix for readability
                                if abs(coef_rounded - 1.0) < 1e-6:
                                    terms.append(readable_term)
                                elif abs(coef_rounded + 1.0) < 1e-6:
                                    terms.append(f"-{readable_term}")
                                else:
                                    terms.append(f"{coef_rounded:.2f}*{readable_term}")
                            else:
                                # If we can't parse it, still show it with the raw name
                                if abs(coef_rounded - 1.0) < 1e-6:
                                    terms.append(poly_feat_name)
                                elif abs(coef_rounded + 1.0) < 1e-6:
                                    terms.append(f"-{poly_feat_name}")
                                else:
                                    terms.append(f"{coef_rounded:.2f}*{poly_feat_name}")
                    
                    if not terms:
                        return f"{variable_name}(t) = f({features_str}) = 0"
                    
                    poly_str = " + ".join(terms)
                    return f"{variable_name}(t) = f({features_str}) = {poly_str}"
                else:
                    # Single-feature model wrapped in PolynomialModel
                    # Fall back to polynomial representation
                    coefs = poly.coef
                    terms = []
                    for i, coef in enumerate(coefs):
                        # Round to 2 decimal places and drop if zero
                        coef_rounded = round(coef, 2)
                        if abs(coef_rounded) < 1e-6:  # Drop zero coefficients
                            continue
                        if i == 0:
                            terms.append(f"{coef_rounded:.2f}")
                        elif i == 1:
                            if abs(coef_rounded - 1.0) < 1e-6:
                                terms.append(feature_names[0])
                            elif abs(coef_rounded + 1.0) < 1e-6:
                                terms.append(f"-{feature_names[0]}")
                            else:
                                terms.append(f"{coef_rounded:.2f}*{feature_names[0]}")
                        else:
                            if abs(coef_rounded - 1.0) < 1e-6:
                                terms.append(f"{feature_names[0]}**{i}")
                            elif abs(coef_rounded + 1.0) < 1e-6:
                                terms.append(f"-{feature_names[0]}**{i}")
                            else:
                                terms.append(f"{coef_rounded:.2f}*{feature_names[0]}**{i}")
                    
                    if not terms:
                        return f"{variable_name}(t) = f({features_str}) = 0"
                    
                    poly_str = " + ".join(terms)
                    return f"{variable_name}(t) = f({features_str}) = {poly_str}"
            
            # For single-feature transitions, show full polynomial
            coefs = poly.coef
            terms = []
            for i, coef in enumerate(coefs):
                # Round to 2 decimal places and drop if zero
                coef_rounded = round(coef, 2)
                if abs(coef_rounded) < 1e-6:  # Drop zero coefficients
                    continue
                feature = feature_names[0] if feature_names else f"{variable_name}(t-1)"
                if i == 0:
                    terms.append(f"{coef_rounded:.2f}")
                elif i == 1:
                    if abs(coef_rounded - 1.0) < 1e-6:
                        terms.append(feature)
                    elif abs(coef_rounded + 1.0) < 1e-6:
                        terms.append(f"-{feature}")
                    else:
                        terms.append(f"{coef_rounded:.2f}*{feature}")
                else:
                    if abs(coef_rounded - 1.0) < 1e-6:
                        terms.append(f"{feature}**{i}")
                    elif abs(coef_rounded + 1.0) < 1e-6:
                        terms.append(f"-{feature}**{i}")
                    else:
                        terms.append(f"{coef_rounded:.2f}*{feature}**{i}")
            
            if not terms:
                return f"{variable_name}(t) = 0"
            
            return f"{variable_name}(t) = " + " + ".join(terms)
        except Exception as e:
            return f"{variable_name}(t) = Error generating string: {e}"
    
    def _parse_polynomial_feature(self, poly_feat_name: str, feature_names: list) -> str:
        """
        Parse sklearn PolynomialFeatures name (e.g., "x0", "x0 x1", "x1^2") into readable format.
        
        Parameters:
        -----------
        poly_feat_name : str
            Feature name from PolynomialFeatures (e.g., "x0", "x0 x1", "x1^2", "x0^2 x1")
        feature_names : list
            List of readable feature names (e.g., ["x(t-1)", "xdot(t-1)", "xddot(t-1)"])
            
        Returns:
        --------
        str
            Readable feature name (e.g., "x(t-1)", "x(t-1)*xdot(t-1)", "xdot(t-1)^2")
        """
        try:
            # Handle bias term
            if poly_feat_name == "1" or poly_feat_name == "":
                return None  # Will be handled as intercept
            
            # Split by spaces to get individual features (e.g., "x0 x1" -> ["x0", "x1"])
            parts = poly_feat_name.split()
            readable_parts = []
            
            for part in parts:
                # Parse format like "x0", "x1^2", etc.
                if '^' in part:
                    var_part, power = part.split('^')
                    try:
                        power = int(power)
                    except ValueError:
                        power = 1
                else:
                    var_part = part
                    power = 1
                
                # Extract feature index (x0 -> 0, x1 -> 1, etc.)
                if var_part.startswith('x'):
                    try:
                        idx = int(var_part[1:])
                        if 0 <= idx < len(feature_names):
                            feature = feature_names[idx]
                            if power == 1:
                                readable_parts.append(feature)
                            else:
                                readable_parts.append(f"{feature}^{power}")
                    except (ValueError, IndexError):
                        continue
            
            if not readable_parts:
                return None
            
            # Join with * for multiplication
            return "*".join(readable_parts)
        except Exception as e:
            # If parsing fails, return None (term will be skipped)
            return None
