import math
import os.path
import time
import random
import pickle
import numpy as np
from agents import BaselineAgent
from agents.pddl.optimizer import grid_search, get_poly_rank, get_param_values, calculate_aggregative_erros, \
    get_params_sensitivity, compute_derivatives, fit_state_transition
from agents.pddl.pddl_files.events.learn_events import update_model_effects, update_model_effects_with_ablation
from agents.pddl.pddl_files.pddl_objects import get_birds, get_pigs, get_blocks, get_platforms
from agents.pddl.pddl_files.segments import getSegmentsPelt, getSegmentsEvents
from agents.pddl.pddl_files.world_model.params import Params
from agents.pddl.pddl_files.world_model.process import Process
from agents.pddl.pddl_files.world_model.world_model import WorldModel
from agents.pddl.trajectory_parser import extract_real_trajectory, construct_trajectory
from agents.pddl.visualiator import visualize_compare, plot_loo_cv_comparison, visualize_learning_dashboard
from agents.pddl.angle_protocol import AngleTrainingProtocol
from agents.utility import GroundTruthType
import subprocess
from agents.utility.vision.relations import *
from agents.pddl.pddl_files.pddl_parser import write_problem_file, parse_solution_to_actions, inject_domain_file
from src.client.agent_client import GameState
from agents.pddl.metrics import calculate_rmse, calculate_impact_rmse

from numpy.polynomial import Polynomial


class PDDLAgent(BaselineAgent):
    """Birds in boots (server/client version)"""

    def __init__(self, agent_ind, agent_configs, min_deg: int = -4, max_deg: int = 78, deg_step: float = 1,
                 learn: bool = False, start_counting_from_game: int = 0, 
                 override_angle: float = None, debug_collision: bool = False,
                 determinism_test_mode: bool = False,
                 use_angle_protocol: bool = True):
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
        self.override_angle = None  # Set to a value (e.g., 45) to override PDDL planner angle
        self.debug_collision = False  # Set to True to visualize collision detection
        
        # Angle selection mode (priority order):
        # 1. use_angle_protocol=True: Use train/val/test protocol
        # 2. determinism_test_mode=True: Random angles [20, 80]
        # 3. override_angle set: Use fixed angle
        # 4. PDDL planner: Compute optimal angle
        self.use_angle_protocol = use_angle_protocol  # NEW: Enable train/val/test protocol
        self.determinism_test_mode = False  # Disabled when using protocol
        
        # Initialize angle training protocol
        # Protocol: Train 4 levels -> Val 1 level -> repeat until 40 train shots -> 10 test shots
        if self.use_angle_protocol:
            self.angle_protocol = AngleTrainingProtocol(
                train_ratio=0.70,
                val_ratio=0.15,
                test_ratio=0.15,
                seed=42,
                val_every_n_levels=5,      # Validate every 5th level
                test_after_n_trains=40,    # Test phase after 40 train shots
                test_shots=10              # 10 test shots
            )
        else:
            self.angle_protocol = None

        self.world_model = WorldModel({
            Params.gravity: 90,
            Params.velocity: 200
        })
        
        # Initialize learned_transition_world_model to current world model
        # (will be updated after first training shot)
        self.learned_transition_world_model = self.world_model
        
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
        self.rmse = list()
        self.suggested_rmse = list()
        self.wins = []
        
        # Impact zone tracking for visualization
        self.impact_rmse = []  # RMSE around impact zone per attempt
        self.impact_trajectories = []  # List of (observed_impact, estimated_impact, impact_idx)
        self.full_trajectories = []  # List of (observed, estimated, event_indexes) per attempt
        
        # Win/loss tracking per level
        self.start_counting_from_game = start_counting_from_game  # Skip first X games before counting
        self.games_played = 0  # Total games played counter
        self.game_results = []  # Array of (level, "win"/"loss")

    def learn_collision_effects(self, collisions, bird_observed_features, phase: str = "train", should_learn: bool = True):
        """
        Learn how collisions affect the bird's velocity (bounce physics).
        
        For each ground collision, extracts pre/post collision states and trains
        a model to predict post-collision velocity from pre-collision state.
        
        Parameters:
        -----------
        collisions : list
            List of frame indices where ground collisions occurred
        bird_observed_features : list
            List of feature dictionaries for each frame (x, y, v_x, v_y)
        phase : str
            Current phase: "train", "validation", or "test"
        should_learn : bool
            If True, train models on this data. If False, only record samples.
        
        Updates:
        --------
        self.kb["collision"] with new collision samples and retrained models (if should_learn)
        self.angle_protocol collision samples (always, if protocol is active)
        """
        FRAME_RATE = 0.02  # 50 fps
        VELOCITY_FRAMES = 3  # Use 3 frames for velocity calculation
        POST_OFFSET = 2  # Skip frames where bird is still at ground level
        MIN_BOUNCE_VELOCITY = 60  # Minimum velocity for a "real" bounce
        
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
            
            # POST-COLLISION: Skip first POST_OFFSET frames, then measure velocity
            post_start = collision_index + POST_OFFSET
            post_end = post_start + VELOCITY_FRAMES
            post_features_start = bird_observed_features[post_start]
            post_features_end = bird_observed_features[post_end]
            
            # PRE-COLLISION state
            pre_state = pre_features.copy()
            pre_dt = VELOCITY_FRAMES * FRAME_RATE
            pre_state['v_x'] = (pre_features['x'] - prev_features['x']) / pre_dt
            pre_state['v_y'] = (pre_features['y'] - prev_features['y']) / pre_dt
            
            # POST-COLLISION state
            post_state = post_features_start.copy()
            post_dt = VELOCITY_FRAMES * FRAME_RATE
            post_state['v_x'] = (post_features_end['x'] - post_features_start['x']) / post_dt
            post_state['v_y'] = (post_features_end['y'] - post_features_start['y']) / post_dt
            
            # Filter: only learn from significant bounces
            if abs(pre_state['v_y']) < MIN_BOUNCE_VELOCITY:
                continue
            
            # Record collision sample for algorithm comparison (always, regardless of phase)
            # Only record first valid collision per trajectory for consistent sample counts
            if self.use_angle_protocol and self.angle_protocol is not None:
                self.angle_protocol.record_collision(phase, pre_state, post_state)
            
            # Train collision model only during training phase
            if should_learn:
                # Use ablation study to compare models with/without velocity_ratio feature
                update_model_effects_with_ablation("collision", self.kb, pre_state, post_state, debug=False)
            
            # Only use first collision per trajectory
            break
    
    def learn_flight_physics(self, trajectory):
        """
        Learn flight physics (gravity, velocity) from observed trajectory.
        
        Stores trajectory in KB and learns state transition functions
        that can predict the next state from the previous state.
        
        Parameters:
        -----------
        trajectory : np.ndarray
            The observed trajectory (first segment, before any collision)
        
        Updates:
        --------
        self.kb["trajectories"] with new trajectory
        self.learned_transitions with updated models
        self.learned_transition_world_model with new WorldModel
        """
        # Store trajectory in KB
        if "trajectories" not in self.kb:
            self.kb["trajectories"] = []
        self.kb["trajectories"].append(trajectory.copy())
        
        self.learn_process_transitions()
        self.learned_transition_world_model = self._create_learned_transition_world_model()
        print(f"\nLearned World Model: {self.learned_transition_world_model.hyperparams_values}")

    def solve(self):
        """
        Solve a particular level by shooting birds directly to pigs.
        
        Flow:
        1. Get angle from protocol/planner
        2. Execute shot and record trajectory
        3. Learn collision effects (bounce physics) - only during train phase
        4. Learn flight physics (gravity, velocity) - only during train phase
        5. Visualize and update world model
        """
        ground_truth_type = GroundTruthType.ground_truth_screenshot
        vision = self._update_reader(ground_truth_type.value, self.if_check_gt)

        sling = vision.find_slingshot_mbr()[0]
        sling.width, sling.height = sling.height, sling.width
        
        # 1. Get angle (priority: protocol > determinism_test > override_angle > PDDL planner)
        should_learn = True  # Default: learn from shot
        current_phase = "train"  # Default phase
        
        if self.use_angle_protocol and self.angle_protocol is not None:
            # Use train/val/test protocol
            angle, current_phase, should_learn = self.angle_protocol.get_next_shot()
            
            if current_phase == "complete":
                print("\n" + "="*60)
                print("TRAINING PROTOCOL COMPLETE!")
                print("="*60)
                self.angle_protocol.print_final_results(kb=self.kb)
                return  # Exit solve - protocol is done
            
            self.angle_protocol.print_status()
            print(f"[{current_phase.upper()}] Angle: {angle}°, Learning: {should_learn}")
        
        elif self.determinism_test_mode:
            angle = round(random.uniform(20.0, 80.0), 1)
            print(f"\n[DETERMINISM] Random angle: {angle}° (uniform [20, 80], 1 decimal)")
        
        elif self.override_angle is not None:
            angle = self.override_angle
            print(f"\n[OVERRIDE] Using fixed angle: {angle}°")
        
        else:
            # Use PDDL planner
            actions = self.get_action_to_perform(self.world_model)[0]
            _, angle = actions
            print(f"\n[PDDL] Planner selected angle: {angle}°")

        # 2. Execute shot and record trajectory (always use full power)
        release_point = self.tp.find_release_point_partial_power(sling, angle * np.pi / 180, v_portion=1.0)
        batch_gt = self.ar.shoot_and_record_ground_truth(release_point.X, release_point.Y, 0, 0, 1, 0)

        # Extract trajectory and events
        groundtruth_trajectories, groundtruth_objects = extract_real_trajectory(batch_gt, angle, self.model, self.target_class)
        event_indexes_by_event, objects_features = getSegmentsEvents(groundtruth_trajectories, groundtruth_objects)

        bird_observed_trajectory = groundtruth_trajectories["redBird_0"]
        bird_observed_features = objects_features["redBird_0"]
        event_indexes = sorted([val for values in event_indexes_by_event.values() for val in values])
        parts = np.split(bird_observed_trajectory, event_indexes)
        
        # Debug: Event detection info
        print(f"\n[EVENT DEBUG] Trajectory length: {len(bird_observed_trajectory)} frames")
        print(f"[EVENT DEBUG] Objects tracked: {list(groundtruth_objects.keys())}")
        print(f"[EVENT DEBUG] Events detected: {event_indexes_by_event}")
        if len(event_indexes) == 0:
            # Check why no events
            last_frame = bird_observed_features[-1] if bird_observed_features else {}
            first_frame = bird_observed_features[0] if bird_observed_features else {}
            print(f"[EVENT DEBUG] No events! First frame y={first_frame.get('y', 'N/A')}, Last frame y={last_frame.get('y', 'N/A')}")
            print(f"[EVENT DEBUG] Last frame v_y={last_frame.get('v_y', 'N/A')}")
            if last_frame.get('y', 100) > 3:
                print(f"[EVENT DEBUG] Bird still in air at end of tracking (y={last_frame.get('y', 'N/A')} > 3)")
        else:
            print(f"[EVENT DEBUG] First event at frame {event_indexes[0]}")
        
        # 3. Process collision effects (bounce physics)
        # Always extract and record collision samples for algorithm comparison
        # Only train models during train phase
        collisions = event_indexes_by_event["ground_collision"]
        self.learn_collision_effects(collisions, bird_observed_features, phase=current_phase, should_learn=should_learn)
        if not should_learn:
            print(f"[{current_phase.upper()}] Collision samples recorded (no training in evaluation mode)")
        
        # 3.5 Visualize General vs CART vs M5 comparison for collision learning
        # if len(collisions) > 0 and "collision" in self.kb:
        #     plot_loo_cv_comparison(self.kb, event_name="collision")

        # 4. Learn flight physics (gravity, velocity) - ONLY during train phase
        first_segment = parts[0]
        if should_learn:
            self.learn_flight_physics(first_segment)
        else:
            print(f"[{current_phase.upper()}] Skipping flight physics learning (evaluation mode)")

        # 5. Visualize trajectory comparison
        # Trim 2 frames from end of first_segment for cleaner RMSE (avoid noisy impact transition)
        first_segment_trimmed = first_segment[:-2] if len(first_segment) > 5 else first_segment
        
        limit = np.max(first_segment_trimmed, axis=0)[0]
        estimated_trajectory = construct_trajectory(
            first_segment_trimmed[0], angle, self.world_model, limit,
            prt=False, integration_method='rk4', stop_at_ground=True
        )
        
        # Use learned model if available, otherwise use current world model
        model_for_suggested = getattr(self, 'learned_transition_world_model', None) or self.world_model
        suggested_trajectory = construct_trajectory(
            first_segment_trimmed[0], angle, model_for_suggested, limit,
            prt=False, integration_method='rk4', stop_at_ground=True
        )

        current_rmse = calculate_rmse(first_segment_trimmed, estimated_trajectory, trim_start_percent=0, trim_end_percent=0, apply_bias_correction=False)
        self.rmse.append(current_rmse)
        self.suggested_rmse.append(calculate_rmse(first_segment_trimmed, suggested_trajectory))
        
        # Record result in angle protocol (if using)
        if self.use_angle_protocol and self.angle_protocol is not None:
            self.angle_protocol.record_result(
                angle=angle,
                phase=current_phase,
                rmse=current_rmse,
                gravity=self.world_model.hyperparams_values.get(Params.gravity),
                velocity=self.world_model.hyperparams_values.get(Params.velocity),
                n_collisions=len(collisions)
            )
        
        # Store full trajectory data for visualization
        self.full_trajectories.append({
            'observed': bird_observed_trajectory.copy(),
            'estimated': estimated_trajectory.copy(),
            'suggested': suggested_trajectory.copy(),
            'event_indexes': event_indexes.copy(),
            'event_indexes_by_event': {k: list(v) for k, v in event_indexes_by_event.items()},
            'angle': angle,
            'phase': current_phase,
            'should_learn': should_learn
        })
        
        # Track impact zone metrics (use first event as primary impact)
        if len(event_indexes) > 0:
            first_impact_idx = event_indexes[0]
            
            # Create extended estimated trajectory that reaches the impact zone
            # Use the full observed trajectory's x-range as limit
            bird_traj_array = np.array(bird_observed_trajectory)
            impact_limit = np.max(bird_traj_array[:first_impact_idx + 21, 0]) if first_impact_idx + 21 < len(bird_traj_array) else np.max(bird_traj_array[:, 0])
            extended_estimated = construct_trajectory(
                bird_traj_array[0], angle, self.world_model, impact_limit,
                prt=False, integration_method='rk4', stop_at_ground=False  # Don't stop at ground for impact comparison
            )
            
            impact_result = calculate_impact_rmse(
                bird_observed_trajectory, extended_estimated, 
                first_impact_idx, frames_before=10, frames_after=20
            )
            # Show both RMSE methods side by side
            rmse_x = impact_result.get('rmse_x_aligned', impact_result['rmse'])
            rmse_t = impact_result.get('rmse_time_aligned', impact_result['rmse'])
            method = impact_result.get('method_used', 'x_aligned')
            x_per_frame = impact_result.get('x_per_frame', 0)
            
            print(f"[IMPACT DEBUG] Impact at frame {first_impact_idx}")
            print(f"[IMPACT DEBUG] RMSE Comparison: x_aligned={rmse_x:.2f} | time_aligned={rmse_t:.2f} | selected={method} (x/frame={x_per_frame:.2f})")
            print(f"[IMPACT DEBUG] Observed window: {len(impact_result['observed_window'])} frames, Estimated window: {len(impact_result['estimated_window'])} frames")
            
            # Use the adaptively selected RMSE
            self.impact_rmse.append(impact_result['rmse'])
            self.impact_trajectories.append({
                'observed_window': impact_result['observed_window'].copy(),
                'estimated_window': impact_result['estimated_window'].copy(),
                'impact_idx': first_impact_idx,
                'impact_idx_in_window': impact_result['impact_idx_in_window'],
                'attempt': len(self.impact_rmse)
            })
        else:
            # No impact detected, use inf for RMSE
            print(f"[IMPACT DEBUG] No events detected, setting RMSE=inf")
            self.impact_rmse.append(float('inf'))
            self.impact_trajectories.append(None)
        
        # Show combined learning dashboard every 10 attempts
        if len(self.full_trajectories) % 10 == 0:
            visualize_learning_dashboard(
                self.full_trajectories, 
                self.rmse, 
                self.suggested_rmse,
                self.impact_rmse, 
                self.impact_trajectories
            )

        # 6. Update game state and world model
        game_result = self.ar.get_game_state() == GameState.WON
        self.wins.append(game_result)
        
        self.games_played += 1
        if self.games_played > self.start_counting_from_game:
            self.game_results.append((self.current_level, "win" if game_result else "loss"))

        # Only update world model during training phase
        if should_learn and hasattr(self, 'learned_transition_world_model') and self.learned_transition_world_model is not None:
            print(f"Old World Model: {self.world_model.hyperparams_values}")
            print(f"New World Model: {self.learned_transition_world_model.hyperparams_values}")
            
            # Update world model with learned parameters
            self.world_model = self.learned_transition_world_model
            self.world_model.kb = self.kb
        else:
            print(f"[{current_phase.upper()}] World model NOT updated (evaluation mode)")
            print(f"Current World Model: {self.world_model.hyperparams_values}")

        print(f"RMSE: {self.rmse[-1]:.2f} (current) | Mean: {np.mean(self.rmse):.2f}")
        print(f"Game results: {len([r for r in self.game_results if r[1] == 'win'])}/{len(self.game_results)} wins")

        # Print collision model comparison after each validation shot
        if self.use_angle_protocol and self.angle_protocol is not None and current_phase == "validation":
            print("\n[INTERIM COMPARISON] Printing current model comparison after validation shot...")
            self.angle_protocol.print_collision_comparison(self.kb)

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
            
            # Print learned transition summary
            print("\nLearned transitions:")
            for var_name in ["yddot", "xdot", "ydot"]:
                if self.learned_transitions[var_name] is not None and "string" in self.learned_transitions[var_name]:
                    print(f"  {self.learned_transitions[var_name]['string']}")
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

