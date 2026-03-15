import math
import os.path
import time
import random
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
    visualize_rmse, visualize_rmse_vs_suggsted, visualize_starting_point_offset, full_trajectory_comparison, \
    debug_is_hit_last_frames, debug_all_events_full_trajectory, visualize_ground_collision_detection, \
    visualize_post_collision_trajectory, visualize_post_collision_trajectory_v2, plot_loo_cv_comparison
from agents.utility import GroundTruthType
import subprocess
from agents.utility.vision.relations import *
from agents.pddl.pddl_files.pddl_parser import write_problem_file, parse_solution_to_actions, inject_domain_file
from src.client.agent_client import GameState
from agents.pddl.metrics import calculate_rmse, compare_truncation_methods, get_robust_launch_angle

from numpy.polynomial import Polynomial


class PDDLAgent(BaselineAgent):
    """Birds in boots (server/client version)"""

    def __init__(self, agent_ind, agent_configs, min_deg: int = -4, max_deg: int = 78, deg_step: float = 1,
                 learn: bool = False, start_counting_from_game: int = 0, 
                 override_angle: float = None, debug_collision: bool = False):
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
        
        # Debug/Override options
        self.override_angle = random.randint(40,70)  # Set to a value (e.g., 45) to override PDDL planner angle
        self.debug_collision = True  # Set to True to visualize collision detection
        self.world_model = WorldModel({
            Params.gravity: 90,
            Params.velocity: 200
        })
        self.start_counting_from_game=8
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
                },
                "learning_history": []  # Track learning progress over games
            },
            "trajectories": []  # Store all past trajectories (first segment only, unlimited)
        }
        self.x =0

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
        
        # Win/loss tracking per level
        self.start_counting_from_game = start_counting_from_game  # Skip first X games before counting
        self.games_played = 0  # Total games played counter
        self.game_results = []  # Array of (level, "win"/"loss")

    def solve(self):
        """
        * Solve a particular level by shooting birds directly to pigs
        * @return GameState: the game state after shots.
        """
        ground_truth_type = GroundTruthType.ground_truth_screenshot

        vision = self._update_reader(ground_truth_type.value, self.if_check_gt)


        sling = vision.find_slingshot_mbr()[0]
        sling.width, sling.height = sling.height, sling.width
        # actions = self.get_action_to_perform(self.world_model)[0]
        # task, angle = actions
        angle =0

        # Override angle if specified (for debugging/testing)
        if self.override_angle is not None:
            self.override_angle =  random.choice([
                    random.randint(25, 35),   # shallow
                    random.randint(45, 55),   # medium
                    random.randint(70, 80),   # steep
                ])
            print(f"[DEBUG] OVERRIDING PDDL angle {angle} with {self.override_angle}")
            angle = self.override_angle

        print(f"Shooting Angle - {angle}")

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

        # Debug is_hit detection - shows last 20 frames with bird/pig positions and collision info
        # debug_is_hit_last_frames(objects_features, groundtruth_objects, n_frames=100)
        
        # Debug ALL events (ground collision, hit, platform collision) for entire trajectory
        # debug_all_events_full_trajectory(objects_features, groundtruth_objects)

        bird_observed_trajectory = groundtruth_trajectories["redBird_0"]

        bird_observed_features = objects_features["redBird_0"]

        event_indexes = sorted([val for values in event_indexes_by_event.values() for val in values])

        parts = np.split(bird_observed_trajectory, event_indexes)

        # DEBUG: Visualize collision detection BEFORE learning
        if self.debug_collision:
            print("\n[DEBUG] Visualizing ground collision detection...")
            visualize_ground_collision_detection(
                bird_observed_trajectory, 
                bird_observed_features, 
                event_indexes_by_event,
                world_model=self.world_model,
                angle=angle
            )

        # LEARN EVENT

        collisions = event_indexes_by_event["ground_collision"]
        
        FRAME_RATE = 0.02  # 50 fps
        
        # Use multiple frames for velocity calculation to avoid quantization
        # With 1-frame difference: v = Δposition / 0.02 → only multiples of 50
        # With N-frame difference: v = Δposition / (N * 0.02) → finer granularity
        VELOCITY_FRAMES = 3  # Use 3 frames for velocity calculation
        POST_OFFSET = 2  # Skip frames where bird is still at ground level

        # Minimum velocity threshold for a "real" bounce (not rolling/settling)
        MIN_BOUNCE_VELOCITY = 60  # pixels/second - adjusted for multi-frame calculation
        
        for collision_index in collisions:
            # Need enough frames before collision for velocity calculation
            if collision_index < VELOCITY_FRAMES:
                continue
            
            # Need enough frames after collision for velocity calculation
            if collision_index + POST_OFFSET + VELOCITY_FRAMES >= len(bird_observed_features):
                continue
                
            # Get states from features
            pre_features = bird_observed_features[collision_index]
            prev_features = bird_observed_features[collision_index - VELOCITY_FRAMES]
            
            # POST-COLLISION: Skip first POST_OFFSET frames (bird at ground), then measure velocity
            post_start = collision_index + POST_OFFSET
            post_end = post_start + VELOCITY_FRAMES
            post_features_start = bird_observed_features[post_start]
            post_features_end = bird_observed_features[post_end]
            
            # PRE-COLLISION state: velocity over VELOCITY_FRAMES frames before collision
            # v_pre = (position_at_collision - position_N_frames_before) / (N * dt)
            pre_state = pre_features.copy()
            pre_dt = VELOCITY_FRAMES * FRAME_RATE
            pre_state['v_x'] = (pre_features['x'] - prev_features['x']) / pre_dt
            pre_state['v_y'] = (pre_features['y'] - prev_features['y']) / pre_dt
            
            # POST-COLLISION state: velocity over VELOCITY_FRAMES frames after bounce starts
            post_state = post_features_start.copy()
            post_dt = VELOCITY_FRAMES * FRAME_RATE
            post_state['v_x'] = (post_features_end['x'] - post_features_start['x']) / post_dt
            post_state['v_y'] = (post_features_end['y'] - post_features_start['y']) / post_dt
            
            # FILTER: Only learn from significant bounces, not rolling/settling
            pre_speed = abs(pre_state['v_y'])
            
            if self.debug_collision:
                print(f"\n[Collision at frame {collision_index}]")
                print(f"  Pre-collision v_y: {pre_state['v_y']:.2f} (from frames {collision_index-VELOCITY_FRAMES} to {collision_index})")
                print(f"  Post-collision v_y: {post_state['v_y']:.2f} (from frames {post_start} to {post_end})")
                if abs(pre_state['v_y']) > 0.1:
                    print(f"  Ratio: {post_state['v_y']/pre_state['v_y']:.3f}")
                else:
                    print(f"  Ratio: N/A (v_y too small)")
            
            if pre_speed < MIN_BOUNCE_VELOCITY:
                if self.debug_collision:
                    print(f"  ⏭️  SKIPPED: |v_y|={pre_speed:.1f} < {MIN_BOUNCE_VELOCITY} (secondary bounce/rolling)")
                continue
            
            if self.debug_collision:
                print(f"  ✅ LEARNED: Significant bounce (|v_y|={pre_speed:.1f} >= {MIN_BOUNCE_VELOCITY})")
            
            # Train models and compare general vs domain-specific (debug output handled by manager)
            update_model_effects("collision", self.kb, pre_state, post_state, debug=False)
        
        # EVALUATE AND TRACK LEARNING PROGRESS
        self._evaluate_collision_learning(collisions, bird_observed_features, FRAME_RATE)
        
        # DEBUG: Print collision learning status with model comparison
        if self.debug_collision and "collision" in self.kb:
            self._print_collision_learning_status()
            # Plot LOO-CV comparison graph (needs at least 2 samples)
            n_samples = len(self.kb["collision"]["states"])
            if n_samples >= 2:
                # Save to file to avoid threading issues with matplotlib
                self.plot_model_comparison(save_path="loo_cv_comparison.png")
        
        # DEBUG: Visualize post-collision trajectory comparison AFTER learning
        if self.debug_collision and len(collisions) > 0:
            print("\n[DEBUG] Visualizing post-collision trajectory comparison (V2 - Direct Velocity)...")
            visualize_post_collision_trajectory_v2(
                bird_observed_trajectory,
                bird_observed_features,
                event_indexes_by_event,
                self.world_model,
                kb=self.kb
            )

        # LEARN PROCESS
        bird_observed_trajectory = parts[0] # override everything else, only learn on part 1

        # Store trajectory in KB (unlimited storage)
        if "trajectories" not in self.kb:
            self.kb["trajectories"] = []
        self.kb["trajectories"].append(bird_observed_trajectory.copy())

        # Only learn if we haven't reached the freeze threshold (8 games)
        if self.games_played < 8:
            # Learn physics parameters (gravity, velocity) - existing method
            # new_world_model = self.learn_process(bird_observed_trajectory)
            
            # Learn state transition functions - new method
            # This learns how to predict next state from previous state
            # Uses multiple trajectories from KB
            self.learn_process_transitions()
            
            # Create a new world model from learned transitions
            self.learned_transition_world_model = self._create_learned_transition_world_model()
            
            # Print both world models
            print(f"\nOriginal World Model: {self.learned_transition_world_model.hyperparams_values}")
            print(f"Learned Transition World Model: {self.learned_transition_world_model.hyperparams_values}")
        else:
            print(f"\nModel learning FROZEN (game {self.games_played} >= 8)")

        # VISUALIZE
        limit = np.max( bird_observed_trajectory,axis=0)[0]

        # Use RK4 integration for better accuracy (can also try 'midpoint' or 'euler')
        estimated_trajectory = construct_trajectory(
            bird_observed_trajectory[0], 
            angle, 
            self.world_model, 
            limit, 
            prt=False,
            integration_method='rk4'  # Options: 'euler', 'midpoint', 'rk4'
        )
        suggested_trajectoty = construct_trajectory(
            bird_observed_trajectory[0], 
            angle, 
            self.learned_transition_world_model, 
            limit, 
            prt=False,
            integration_method='rk4'  # Options: 'euler', 'midpoint', 'rk4'
        )

        self.rmse.append(calculate_rmse(bird_observed_trajectory, estimated_trajectory,trim_start_percent=0,trim_end_percent=0,apply_bias_correction=False))
        self.suggested_rmse.append(calculate_rmse(bird_observed_trajectory,suggested_trajectoty))

        # Visualize starting point offset to diagnose alignment issues
        # Get PDDL bird position (reference point from slingshot)
        ref_point = self.tp.get_reference_point(sling)
        pddl_ref_pos = (ref_point.X, 640 - ref_point.Y)  # PDDL uses inverted Y coordinate
        
        # Calculate PDDL bird position AFTER pa-twang action
        # pa-twang decreases x_bird by (* 22 (cosine)) and y_bird by (* 12 (sinus))
        # Check which domain file is used (domain.pddl uses 22, base_domain.pddl uses 16)
        import math
        angle_rad = angle * math.pi / 180
        cosine = math.cos(angle_rad)
        sinus = math.sin(angle_rad)
        
        # Use 22 for domain.pddl, 16 for base_domain.pddl (check which is active)
        # Default to 22 (domain.pddl) but can be adjusted
        patwang_x_offset = 22  # domain.pddl uses 22, base_domain.pddl uses 16
        patwang_y_offset = 12  # Both use 12
        
        pddl_bird_pos_after_patwang = (
            ref_point.X - patwang_x_offset * cosine,
            640 - ref_point.Y - patwang_y_offset * sinus
        )
        
        # visualize_starting_point_offset(
        #     bird_observed_trajectory,
        #     estimated_trajectory,
        #     show_first_n_points=15,  # Show first 15 points in detail
        #     pddl_bird_pos=pddl_bird_pos_after_patwang,  # PDDL bird position AFTER pa-twang
        #     pddl_ref_pos=pddl_ref_pos,  # PDDL reference point (before pa-twang)
        #     angle=angle  # Angle for display
        # )
        
        # visualize_compare( bird_observed_trajectory, estimated_trajectory, suggested_trajectoty)
        # visualize_rmse(self.rmse)
        # visualize_rmse_vs_suggsted(self.rmse,self.suggested_rmse)

        from agents.pddl.metrics import analyze_launch_angle

        from agents.pddl.metrics import compare_rmse_with_and_without_bias

        # Show visualizations starting from the 9th game
        if self.games_played >= 9:
            full_trajectory_comparison(bird_observed_trajectory, estimated_trajectory, frame_rate=0.02, n_frames=20)

        # get_robust_launch_angle(bird_observed_trajectory, num_points=20, commanded_angle_deg=-angle)

        game_result = self.ar.get_game_state() == GameState.WON
        self.wins.append(game_result)
        
        # Track wins/losses per level
        self.games_played += 1
        if self.games_played > self.start_counting_from_game:
            self.game_results.append((self.current_level, "win" if game_result else "loss"))
        
        # visuallize_wins_percentage(self.wins)

        from agents.pddl.metrics import compare_interpolation_methods

        # result = compare_interpolation_methods(bird_observed_trajectory, estimated_trajectory)
        # Compare truncation: Option 1 (observed range) vs Option 3 (minimum overlap)
        # result1 = compare_truncation_methods(bird_observed_trajectory, estimated_trajectory)

        from agents.pddl.metrics import analyze_error_by_position

        # Default: 10 segments
        # result = analyze_error_by_position(bird_observed_trajectory, estimated_trajectory)

        # Or specify more/fewer segments
        # result = analyze_error_by_position(bird_observed_trajectory, estimated_trajectory, num_segments=20)

        print(f"Old values- {self.world_model.hyperparams_values} ")
        print(f"New values- gravity: {self.learned_transition_world_model.hyperparams_values} ")
            # Update world model
        self.world_model = self.learned_transition_world_model
        self.world_model.kb = self.kb

        if len(self.aggravate_score) == 0:
            self.aggravate_score.append(self.check_current_level_score())
        else:
            self.aggravate_score.append(self.aggravate_score[-1] + self.check_current_level_score())

        # plot_score(self.aggravate_score)
        print(self.rmse)
        print(self.game_results)

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

    def learn_process_transitions(self):
        """
        Learn state transition functions using a per-trajectory aggregation approach:
        1. Process each trajectory independently - fit polynomials and compute derivatives
        2. Extract state transition pairs (prev_state -> curr_state) from each trajectory
        3. Aggregate all transition pairs from all trajectories
        4. Fit polynomial transition functions on the aggregated data
        
        This method learns how to predict the next state from the previous state,
        rather than learning global physics parameters. Each trajectory is processed
        independently to avoid time-axis discontinuities, then transitions are aggregated.
            
        Returns:
        --------
        None
            Learned transition polynomials are stored in self.learned_transitions
        """

        # Get all trajectories from KB
        trajectories = self.kb.get("trajectories", [])
        
        if len(trajectories) == 0:
            print("Warning: No trajectories in KB for learn_process_transitions(). Skipping.")
            return
        
        # Collect state transition pairs from all trajectories
        all_x_prev = []
        all_x_curr = []
        all_y_prev = []
        all_y_curr = []
        all_xdot_prev = []
        all_xdot_curr = []
        all_ydot_prev = []
        all_ydot_curr = []
        all_xddot_values = []
        all_yddot_values = []
        all_yddot_constants = []  # Store gravity values from each trajectory
        
        # Process each trajectory independently
        for traj_idx, traj in enumerate(trajectories):
            if len(traj) < 2:
                continue
            
            # Create time axis for this trajectory (starting at 0)
            function_range = np.array(range(len(traj))) / 50.0
            
            # ========================================================================
            # STEP 1: Fit curves to this trajectory's x,y positions
            # ========================================================================
            rank_x, poly_x = get_poly_rank(function_range, traj[:, 0])
            rank_y, poly_y = get_poly_rank(function_range, traj[:, 1])
            
            # ========================================================================
            # STEP 2: Sample curves and compute derivatives for this trajectory
            # ========================================================================
            x_values = np.array([poly_x(t) for t in function_range])
            y_values = np.array([poly_y(t) for t in function_range])
            
            # Compute derivatives for this trajectory
            xdot, xddot, ydot, yddot = compute_derivatives(poly_x, poly_y, function_range,
                                                          x_values=x_values, y_values=y_values)
            
            # Extract gravity (yddot constant) from this trajectory
            try:
                yddot_constant = poly_y.deriv(2)(0)  # Second derivative at t=0
            except:
                yddot_constant = np.mean(yddot) if len(yddot) > 0 else 0.0
            all_yddot_constants.append(yddot_constant)
            
            # ========================================================================
            # STEP 3: Extract state transition pairs from this trajectory
            # ========================================================================
            # For each time step t from 1 to n-1, create (prev_state, curr_state) pairs
            if len(x_values) >= 2:
                # x transitions: [x(t-1), xdot(t-1), xddot(t-1)] -> x(t)
                all_x_prev.append(np.column_stack([x_values[:-1], xdot[:-1], xddot[:-1]]))
                all_x_curr.append(x_values[1:])
                
                # y transitions: [y(t-1), ydot(t-1), yddot(t-1)] -> y(t)
                all_y_prev.append(np.column_stack([y_values[:-1], ydot[:-1], yddot[:-1]]))
                all_y_curr.append(y_values[1:])
                
                # xdot transitions: [xdot(t-1), xddot(t-1)] -> xdot(t)
                all_xdot_prev.append(np.column_stack([xdot[:-1], xddot[:-1]]))
                all_xdot_curr.append(xdot[1:])
                
                # ydot transitions: [ydot(t-1), yddot(t-1)] -> ydot(t)
                all_ydot_prev.append(np.column_stack([ydot[:-1], yddot[:-1]]))
                all_ydot_curr.append(ydot[1:])
                
                # xddot values (should be constant 0)
                all_xddot_values.extend(xddot)
        
        # Check if we have enough data
        if len(all_x_prev) == 0:
            print("Warning: Not enough trajectory samples in KB for learning transitions. Skipping.")
            return
        
        # Aggregate all transition pairs from all trajectories
        x_prev_combined = np.vstack(all_x_prev) if all_x_prev else None
        x_curr_combined = np.concatenate(all_x_curr) if all_x_curr else None
        y_prev_combined = np.vstack(all_y_prev) if all_y_prev else None
        y_curr_combined = np.concatenate(all_y_curr) if all_y_curr else None
        xdot_prev_combined = np.vstack(all_xdot_prev) if all_xdot_prev else None
        xdot_curr_combined = np.concatenate(all_xdot_curr) if all_xdot_curr else None
        ydot_prev_combined = np.vstack(all_ydot_prev) if all_ydot_prev else None
        ydot_curr_combined = np.concatenate(all_ydot_curr) if all_ydot_curr else None
        
        print(f"\nProcessed {len(trajectories)} trajectories from KB")
        print(f"Total state transition pairs: {len(x_curr_combined) if x_curr_combined is not None else 0}")
        
        # ========================================================================
        # STEP 4: Fit polynomial transition functions on aggregated data
        # ========================================================================
        
        try:
            # x(t) = poly(x(t-1), xdot(t-1), xddot(t-1))
            # Previous state features: [x[t-1], xdot[t-1], xddot[t-1]]
            if x_prev_combined is not None and x_curr_combined is not None:
                self.learned_transitions["x"] = fit_state_transition(x_curr_combined, x_prev_combined)
            
            # y(t) = poly(y(t-1), ydot(t-1), yddot(t-1))
            # Previous state features: [y[t-1], ydot[t-1], yddot[t-1]]
            if y_prev_combined is not None and y_curr_combined is not None:
                self.learned_transitions["y"] = fit_state_transition(y_curr_combined, y_prev_combined)
            
            # xdot(t) = poly(xdot(t-1), xddot(t-1))
            # Previous state features: [xdot[t-1], xddot[t-1]]
            if xdot_prev_combined is not None and xdot_curr_combined is not None:
                self.learned_transitions["xdot"] = fit_state_transition(xdot_curr_combined, xdot_prev_combined)
            
            # ydot(t) = poly(ydot(t-1), yddot(t-1))
            # Previous state features: [ydot[t-1], yddot[t-1]]
            if ydot_prev_combined is not None and ydot_curr_combined is not None:
                self.learned_transitions["ydot"] = fit_state_transition(ydot_curr_combined, ydot_prev_combined)
            
            # xddot(t) = constant (0 - no horizontal acceleration)
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
            
            # yddot(t) = constant (gravity doesn't change)
            # Average gravity (yddot) from all trajectories
            if len(all_yddot_constants) > 0:
                yddot_constant = np.mean(all_yddot_constants)
                print(f"  Averaged gravity from {len(all_yddot_constants)} trajectories: {yddot_constant:.4f}")
            else:
                yddot_constant = 0.0
            
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
            
            # Extract initial values from the most recent trajectory
            last_traj = trajectories[-1] if len(trajectories) > 0 and len(trajectories[-1]) > 0 else None
            if last_traj is not None:
                last_function_range = np.array(range(len(last_traj))) / 50.0
                last_rank_x, last_poly_x = get_poly_rank(last_function_range, last_traj[:, 0])
                last_rank_y, last_poly_y = get_poly_rank(last_function_range, last_traj[:, 1])
                last_x_values = np.array([last_poly_x(t) for t in last_function_range])
                last_y_values = np.array([last_poly_y(t) for t in last_function_range])
                last_xdot, _, last_ydot, _ = compute_derivatives(last_poly_x, last_poly_y, last_function_range,
                                                                 x_values=last_x_values, y_values=last_y_values)
                
                initial_values = {
                    "x": round(last_x_values[0], 2) if len(last_x_values) > 0 else 0.0,
                    "y": round(last_y_values[0], 2) if len(last_y_values) > 0 else 0.0,
                    "xdot": round(last_xdot[0], 2) if len(last_xdot) > 0 else 0.0,
                    "ydot": round(last_ydot[0], 2) if len(last_ydot) > 0 else 0.0,
                    "xddot": xddot_constant,
                    "yddot": yddot_constant
                }
            else:
                initial_values = {
                    "x": 0.0, "y": 0.0, "xdot": 0.0, "ydot": 0.0,
                    "xddot": xddot_constant, "yddot": yddot_constant
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

    def _evaluate_collision_learning(self, collisions, bird_observed_features, frame_rate):
        """
        Evaluate the collision model's performance using Leave-One-Out Cross-Validation.
        This measures how well the model GENERALIZES to unseen collisions.
        
        Tracks:
        - Training error (how well it fits known data)
        - LOO-CV error (how well it predicts unseen data - TRUE generalization measure)
        """
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression
        import numpy as np
        
        n_samples = len(self.kb["collision"]["states"])
        
        if n_samples < 2:
            # Can't do cross-validation with less than 2 samples
            return
        
        # Get all states and targets
        states = np.array([[s['x'], s['y'], s['v_x'], s['v_y']] for s in self.kb["collision"]["states"]])
        targets = {
            'v_x': np.array(self.kb["collision"]["variables"]["v_x"]["value"]),
            'v_y': np.array(self.kb["collision"]["variables"]["v_y"]["value"]),
            'y': np.array(self.kb["collision"]["variables"]["y"]["value"])
        }
        
        # Calculate Leave-One-Out Cross-Validation error
        loo_errors = {'v_x': [], 'v_y': [], 'y': []}
        training_errors = {'v_x': [], 'v_y': [], 'y': []}
        
        for var_name in ['v_x', 'v_y', 'y']:
            y = targets[var_name]
            
            # Training error (using all data)
            model = self.kb["collision"]["variables"][var_name]["model"]
            if model is not None:
                poly = PolynomialFeatures(degree=1, include_bias=False)
                X_poly = poly.fit_transform(states)
                predictions = model.predict(X_poly)
                train_mse = np.mean((predictions - y) ** 2)
                training_errors[var_name] = np.sqrt(train_mse)
            
            # LOO-CV error
            loo_predictions = []
            for i in range(n_samples):
                # Train on all except sample i
                X_train = np.delete(states, i, axis=0)
                y_train = np.delete(y, i)
                X_test = states[i:i+1]
                y_test = y[i]
                
                # Fit model
                poly = PolynomialFeatures(degree=1, include_bias=False)
                X_train_poly = poly.fit_transform(X_train)
                X_test_poly = poly.transform(X_test)
                
                loo_model = LinearRegression()
                loo_model.fit(X_train_poly, y_train)
                
                pred = loo_model.predict(X_test_poly)[0]
                loo_predictions.append((pred - y_test) ** 2)
            
            loo_mse = np.mean(loo_predictions)
            loo_errors[var_name] = np.sqrt(loo_mse)
        
        # Store in learning history
        history_entry = {
            'game': self.games_played,
            'n_samples': n_samples,
            'training_rmse': {k: float(v) for k, v in training_errors.items()},
            'loo_cv_rmse': {k: float(v) for k, v in loo_errors.items()}
        }
        self.kb["collision"]["learning_history"].append(history_entry)
        
        # Print learning progress
        print("\n" + "="*70)
        print(f"COLLISION LEARNING PROGRESS (Game {self.games_played}, {n_samples} samples)")
        print("="*70)
        print(f"{'Variable':<10} {'Train RMSE':<15} {'LOO-CV RMSE':<15} {'Generalization':<20}")
        print("-"*70)
        
        for var_name in ['v_x', 'v_y', 'y']:
            train_err = training_errors[var_name]
            loo_err = loo_errors[var_name]
            
            # Generalization gap: if LOO >> Train, model is overfitting
            if train_err > 0.01:
                gap_ratio = loo_err / train_err
                if gap_ratio < 1.5:
                    status = "✓ Good"
                elif gap_ratio < 3.0:
                    status = "⚠ Moderate overfit"
                else:
                    status = "✗ Overfitting"
            else:
                status = "Perfect fit"
            
            print(f"{var_name:<10} {train_err:<15.4f} {loo_err:<15.4f} {status:<20}")
        
        print("-"*70)
        
        # Show improvement over time
        if len(self.kb["collision"]["learning_history"]) > 1:
            prev = self.kb["collision"]["learning_history"][-2]
            curr = self.kb["collision"]["learning_history"][-1]
            
            print("\nImprovement from previous game:")
            for var_name in ['v_x', 'v_y']:
                prev_loo = prev['loo_cv_rmse'].get(var_name, float('inf'))
                curr_loo = curr['loo_cv_rmse'].get(var_name, float('inf'))
                
                if prev_loo > 0:
                    improvement = (prev_loo - curr_loo) / prev_loo * 100
                    arrow = "↓" if improvement > 0 else "↑"
                    print(f"  {var_name}: {arrow} {abs(improvement):.1f}% {'better' if improvement > 0 else 'worse'}")
        
        print("="*70 + "\n")
    
    def _print_collision_learning_status(self):
        """
        Print comprehensive collision learning status with model comparison.
        
        Shows:
        - Sample count and diversity warnings
        - Model comparison table (General vs Domain-specific)
        - Selected model coefficients
        - Actual sample ratios
        """
        from agents.pddl.pddl_files.events.learn_events import PhysicsRatioModel, AngleDependentFrictionModel
        
        n_samples = len(self.kb["collision"]["states"])
        states = self.kb["collision"]["states"]
        
        # Header
        print("\n" + "=" * 70)
        print(f"COLLISION LEARNING STATUS: {n_samples} sample(s) in KB")
        print("=" * 70)
        
        # Warning for few samples
        if n_samples < 5:
            print("⚠️  WARNING: Need more samples for reliable learning!")
            print("   With few samples, the model just memorizes - no generalization.")
        
        # Sample diversity
        if n_samples > 0:
            v_y_values = [s['v_y'] for s in states]
            v_x_values = [s['v_x'] for s in states]
            print(f"\n📊 SAMPLE DIVERSITY (pre-collision velocities):")
            print(f"   v_y_pre range: [{min(v_y_values):.1f}, {max(v_y_values):.1f}]  (spread: {max(v_y_values)-min(v_y_values):.1f})")
            print(f"   v_x_pre range: [{min(v_x_values):.1f}, {max(v_x_values):.1f}]  (spread: {max(v_x_values)-min(v_x_values):.1f})")
            if max(v_y_values) - min(v_y_values) < 50:
                print("   ⚠️  Low v_y diversity! Try different shooting angles for better learning.")
        
        # Model comparison for each variable
        print("\n" + "=" * 70)
        print("MODEL COMPARISON (General vs Domain-Specific)")
        print("=" * 70)
        
        for var_name in ["v_x", "v_y", "y"]:
            comparison = self.kb["collision"]["variables"][var_name].get("model_comparison")
            if comparison:
                self._print_model_comparison(var_name, comparison)
        
        # Actual ratios from samples
        if n_samples > 0:
            self._print_sample_ratios(states, n_samples)
        
        print("=" * 70)
    
    def _print_model_comparison(self, var_name, comparison):
        """Print formatted model comparison table for a single variable."""
        n = comparison['n_samples']
        
        print(f"\n--- {var_name} ({n} samples) ---")
        print(f"{'Model':<28} {'Train RMSE':<12} {'LOO-CV':<12} {'R²':<10}")
        print("-" * 62)
        
        # General model row
        gen = comparison['general_stats']
        gen_label = "General (Poly deg=1)"
        loo_str = f"{gen['loo_cv']:.4f}" if np.isfinite(gen['loo_cv']) else "N/A"
        print(f"{gen_label:<28} {gen['train_rmse']:<12.4f} {loo_str:<12} {gen['r2']:<10.4f}")
        
        # Domain model row (if exists)
        if comparison['domain_stats']:
            dom = comparison['domain_stats']
            domain_label = f"Domain ({comparison['domain_name']})"
            dom_loo_str = f"{dom['loo_cv']:.4f}" if np.isfinite(dom['loo_cv']) else "N/A"
            print(f"{domain_label:<28} {dom['train_rmse']:<12.4f} {dom_loo_str:<12} {dom['r2']:<10.4f}")
        
        # Winner
        print("-" * 62)
        winner_str = f"SELECTED: {comparison['winner_name']}"
        if comparison['improvement_pct'] > 0:
            winner_str += f" ({comparison['improvement_pct']:.1f}% better LOO-CV)"
        print(winner_str)
        
        # Show coefficients of selected model
        self._print_selected_coefficients(var_name, comparison)
    
    def _print_selected_coefficients(self, var_name, comparison):
        """Print coefficients of the selected model in physics-interpretable format."""
        from agents.pddl.pddl_files.events.learn_events import PhysicsRatioModel, AngleDependentFrictionModel
        
        model = comparison['winner']
        
        if isinstance(model, PhysicsRatioModel):
            print(f"\n  {var_name}_after = {model.ratio:.4f} * {var_name}_before")
            stats = model.get_stats()
            if stats['n_samples'] > 0:
                print(f"  └─ Learned ratio: {model.ratio:.4f} ± {stats['std']:.4f} (from {stats['n_samples']} samples)")
            if var_name == 'v_y' and abs(model.ratio - (-0.33)) < 0.15:
                print(f"  └─ ✅ Restitution coefficient close to expected ~-0.33")
                
        elif isinstance(model, AngleDependentFrictionModel):
            print(f"\n  {var_name}_ratio = {model.base_ratio:.4f} + ({model.angle_coef:.4f}) * impact_angle_factor")
            print(f"  └─ impact_angle_factor = |v_y| / (|v_x| + |v_y|)  [0=horizontal, 1=vertical]")
            stats = model.get_stats()
            if stats['n_samples'] > 0:
                print(f"  └─ R² score: {stats['r_squared']:.3f}, samples: {stats['n_samples']}")
            if model.angle_coef < -0.1:
                print(f"  └─ ✅ Steeper impacts lose more v_x (physically correct)")
            elif model.angle_coef > 0.1:
                print(f"  └─ ⚠️ Steeper impacts retain more v_x (unusual)")
                
        elif hasattr(model, 'coef_') and hasattr(model, 'intercept_'):
            # General linear model
            coef_names = ["x", "y", "v_x", "v_y"]
            terms = [f"{model.intercept_:.4f}"]
            for coef, name in zip(model.coef_, coef_names):
                if abs(coef) > 0.0001:
                    terms.append(f"({coef:.4f})*{name}")
            print(f"\n  {var_name}_after = " + " + ".join(terms))
    
    def _print_sample_ratios(self, states, n_samples):
        """Print actual ratios from all samples."""
        post_v_y = self.kb["collision"]["variables"]["v_y"]["value"]
        post_v_x = self.kb["collision"]["variables"]["v_x"]["value"]
        
        print(f"\n📈 ACTUAL RATIOS FROM SAMPLES:")
        for i, (pre, vy_post, vx_post) in enumerate(zip(states, post_v_y, post_v_x)):
            vy_ratio = vy_post / pre['v_y'] if abs(pre['v_y']) > 0.1 else 0
            vx_ratio = vx_post / pre['v_x'] if abs(pre['v_x']) > 0.1 else 0
            print(f"   Sample {i+1}: v_y ratio = {vy_ratio:.3f}, v_x ratio = {vx_ratio:.3f}")
        
        # Average ratios
        vy_ratios = [post_v_y[i] / states[i]['v_y'] for i in range(n_samples) if abs(states[i]['v_y']) > 0.1]
        vx_ratios = [post_v_x[i] / states[i]['v_x'] for i in range(n_samples) if abs(states[i]['v_x']) > 0.1]
        if vy_ratios:
            print(f"   Average v_y ratio: {np.mean(vy_ratios):.3f} ± {np.std(vy_ratios):.3f}")
        if vx_ratios:
            print(f"   Average v_x ratio: {np.mean(vx_ratios):.3f} ± {np.std(vx_ratios):.3f}")
    
    def plot_model_comparison(self, event_name="collision", save_path=None):
        """
        Plot LOO-CV comparison between General (Ridge) and Domain-Specific models.
        
        LOO-CV = Leave-One-Out Cross-Validation RMSE
        Lower values = better generalization to unseen data
        
        Usage:
            agent.plot_model_comparison()  # Interactive plot
            agent.plot_model_comparison(save_path="comparison.png")  # Save to file
        """
        plot_loo_cv_comparison(self.kb, event_name=event_name, save_path=save_path)
