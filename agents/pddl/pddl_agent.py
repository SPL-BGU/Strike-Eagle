import math
import os
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
from agents.pddl.trajectory_parser import (
    extract_real_trajectory,
    construct_trajectory,
    construct_trajectory_from_velocity,
    estimate_launch_from_trajectory,
)
from agents.pddl.visualiator import (visualize_compare, plot_loo_cv_comparison, visualize_learning_dashboard,
                                     visualize_level_setup, visualize_trajectory_segment0,
                                     visualize_starting_point_offset, log_direct_hit_analysis,
                                     debug_all_events_full_trajectory,
                                     visualize_expected_vs_actual_hit)
from agents.pddl.angle_protocol import AngleTrainingProtocol
from agents.pddl.phyq_metrics import PhyQMetrics, PhyQLevelMapper
from agents.pddl.comparison_csv import AgentComparisonCSV
from agents.pddl.phyq_generalization import (
    PhyQGeneralizationProtocol,
    GeneralizationType,
    create_local_generalization_protocol,
    create_broad_generalization_protocol,
    create_protocol_from_config
)
from agents.utility import GroundTruthType
import subprocess
from agents.utility.vision.relations import *
from agents.pddl.pddl_files.pddl_parser import (
    write_problem_file, parse_solution_to_actions, inject_domain_file,
    pddl_bird_position_before_pa_twang,
    pddl_bird_position_after_pa_twang,
    ANGLE_REPLAN_THRESHOLD_DEG,
    ANGLE_BIAS_DEGREES,
    simulate_pddl_shot_plan,
    ballistic_angle_to_target,
    FORCE_MIN, FORCE_MAX, FORCE_RATE, FORCE_V_SCALE,
)
from src.client.agent_client import GameState
from agents.pddl.metrics import calculate_rmse, calculate_impact_rmse

from numpy.polynomial import Polynomial


def _extract_force_angle(actions, default_force=1.0):
    """Extract (force, angle) from an actions list returned by get_action_to_perform.

    The list may contain ('set_force', force) and/or ('shoot', angle) tuples.
    Falls back to default_force when no set_force action is present.
    """
    force = default_force
    angle = None
    for action_type, value in actions:
        if action_type == 'set_force':
            force = value
        elif action_type == 'shoot':
            angle = value
    return force, angle


class PDDLAgent(BaselineAgent):
    """Birds in boots (server/client version)"""

    def __init__(self, agent_ind, agent_configs, min_deg: int = -20, max_deg: int = 89, deg_step: float = 0.5,
                 learn: bool = False, start_counting_from_game: int = 0, 
                 override_angle: float = None, debug_collision: bool = False,
                 determinism_test_mode: bool = True,
                 use_angle_protocol: bool = False,
                 validate_alpha_on_validation: bool = False,
                 phyq_config_path: str = "./config_phyq_sample.xml",  # Path to config_phyq_*.xml for Phy-Q benchmark
                 # Phy-Q Generalization Protocol options
                 use_generalization_protocol: bool = False,
                 use_config_metadata: bool = False,  # Auto-load settings from .meta.json file
                 generalization_type: str = "local",  # "local" or "broad"
                 generalization_train_ratio: float = 0.8,  # For local: 80/20 split within templates
                 generalization_train_templates: list = None,  # For broad: e.g., [1, 2, 3, 4]
                 generalization_test_templates: list = None,  # For broad: e.g., [5, 6]
                 generalization_seed: int = 42,
                 scenario_filter: str = None,  # Filter by scenario: e.g., "single_force"
                 levels_per_template: int = None,  # Limit levels per template
                 visualize_pddl_input: bool = True,  # Show PDDL visualization on level loss
                 # Agent comparison CSV options
                 comparison_csv_path: str = "agent_comparison_results.csv",
                 human_baseline_path: str = "external/phy-q/playdata/broad_generalization_all_agents.csv",
                 # Debug mag comparison mode
                 debug_mag_comparison: bool = True,
                 mag_comparison_angle: float = 60.0,  # Fixed angle in degrees
                 mag_comparison_start: float = 5.0,   # Starting mag value
                 mag_comparison_decrement: float = 0.1,
                 # Force -> velocity learning mode
                 force_learning_mode: bool = False,
                 force_learning_min_samples: int = 5):
        super().__init__(
            agent_ind=agent_ind,
            agent_configs=agent_configs)
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.deg_step = deg_step
        self._last_planned_angle = None
        self._last_plan_uses_ground = False

        # Override sim speed from 20
        self.sim_speed = 20
        self.visualize = False
        
        # Enable level visualization - shown only when a level is lost
        self.visualize_pddl_input = visualize_pddl_input
        self.pddl_viz_dir = "pddl_level_viz"
        self._pending_level_viz = None
        
        # Multi-bird tracking: which bird index we're currently shooting
        self.current_bird_shot = 0  # 0 = first bird, 1 = second bird, etc.
        
        if self.visualize_pddl_input:
            print("[PDDL VIZ] Visualization ENABLED - will display only when a level is LOST")
        self.ground_truth_type = GroundTruthType.ground_truth_screenshot
        self.learn = True
        
        # Debug/Override options
        self.override_angle = None  # Set to a value (e.g., 45) to override PDDL planner angle
        self.debug_collision = False  # Set to True to visualize collision detection
        
        # Debug mag comparison mode - tests how different mag values affect trajectory
        self.debug_mag_comparison = debug_mag_comparison
        self.mag_comparison_angle = mag_comparison_angle
        self.mag_comparison_start = mag_comparison_start
        self.mag_comparison_decrement = mag_comparison_decrement
        self.mag_comparison_results = []  # Store results
        self.mag_comparison_current_mag = mag_comparison_start
        self.mag_comparison_baseline_trajectory = None  # First trajectory (mag=5.0) for comparison
        self.mag_comparison_done = False
        
        # Force -> velocity learning state
        self.force_learning_mode = force_learning_mode
        self.force_learning_min_samples = force_learning_min_samples
        self.kb_force = []        # list of (force, v_meas) tuples
        self.force_lr_model = None
        self.force_lr_r2 = None
        
        # Angle selection mode (priority order):
        # 0. debug_mag_comparison=True: Fixed angle, varying mag
        # 1. use_angle_protocol=True: Use train/val/test protocol
        # 2. determinism_test_mode=True: Random angles [20, 80]
        # 3. override_angle set: Use fixed angle
        # 4. PDDL planner: Compute optimal angle
        self.use_angle_protocol = use_angle_protocol  # Set to False to use PDDL planner
        self.validate_alpha_on_validation = validate_alpha_on_validation  # Run alpha validation after each validation shot
        self.determinism_test_mode = False  # Disabled - using PDDL planner
        
        # Initialize angle training protocol
        # Protocol: 15 train shots -> 10 test shots (no validation)
        if self.use_angle_protocol:
            self.angle_protocol = AngleTrainingProtocol(
                train_ratio=0.80,
                val_ratio=0.0,
                test_ratio=0.2,
                seed=42,
                val_every_n_levels=0,      # Disabled - no validation
                test_after_n_trains=30,    # Test phase after 10 train shots
                test_shots=5               # 5 test shots
            )
        else:
            self.angle_protocol = None

        # Initialize Phy-Q benchmark metrics tracking
        self.phyq_metrics = PhyQMetrics()
        self.phyq_level_mapper = PhyQLevelMapper(phyq_config_path)
        self.phyq_config_path = phyq_config_path
        
        # Initialize Phy-Q Generalization Protocol
        # This controls the train/test split based on the Phy-Q paper's evaluation protocols:
        # - Local: 80/20 split within each template (tests within-task generalization)
        # - Broad: Train on some templates, test on others (tests cross-task generalization)
        self.use_generalization_protocol = use_generalization_protocol
        self.generalization_protocol = None
        
        if use_generalization_protocol:
            print("\n" + "=" * 70)
            print("INITIALIZING PHY-Q GENERALIZATION PROTOCOL")
            print("=" * 70)
            print(f"Config path: {phyq_config_path}")
            print(f"Use config metadata: {use_config_metadata}")
            
            # Try to load from metadata file first if enabled
            if use_config_metadata:
                print("\n[MODE] Auto-loading settings from config metadata (default behavior)")
                print("[MODE] Use --no-config-metadata to disable and use command-line parameters instead")
                self.generalization_protocol = create_protocol_from_config(phyq_config_path)
                
                if self.generalization_protocol is not None:
                    print("\n[MODE] SUCCESS - Protocol loaded from config metadata!")
                    print("[MODE] Agent settings are synchronized with config generation settings.")
                else:
                    print("\n[MODE] Metadata not found - falling back to command-line parameters")
                    print("[MODE] To generate metadata, run: python scripts/generate_phyq_configs.py --generalization <type> ...")
            else:
                print("\n[MODE] Config metadata disabled - using command-line parameters")
            
            # Fall back to manual parameters if metadata loading failed or not enabled
            if self.generalization_protocol is None:
                print(f"\n[MANUAL CONFIG] Creating protocol from command-line parameters:")
                print(f"  - Generalization type: {generalization_type}")
                print(f"  - Train ratio: {generalization_train_ratio}")
                print(f"  - Seed: {generalization_seed}")
                print(f"  - Levels per template: {levels_per_template or 'unlimited'}")
                if scenario_filter:
                    print(f"  - Scenario filter: {scenario_filter}")
                if generalization_train_templates:
                    print(f"  - Train templates: {generalization_train_templates}")
                if generalization_test_templates:
                    print(f"  - Test templates: {generalization_test_templates}")
                
                if generalization_type == "local":
                    self.generalization_protocol = create_local_generalization_protocol(
                        config_path=phyq_config_path,
                        train_ratio=generalization_train_ratio,
                        seed=generalization_seed,
                        scenario_filter=scenario_filter,
                        levels_per_template=levels_per_template
                    )
                else:  # "broad"
                    self.generalization_protocol = create_broad_generalization_protocol(
                        config_path=phyq_config_path,
                        train_templates=generalization_train_templates,
                        test_templates=generalization_test_templates,
                        seed=generalization_seed,
                        scenario_filter=scenario_filter,
                        levels_per_template=levels_per_template
                    )
            
            self.generalization_protocol.print_summary()
            
            # Update level mapper to use generalization protocol's level order
            self._generalization_level_order = self.generalization_protocol.get_all_levels_ordered()
            print(f"\n[GENERALIZATION] Level order set: {len(self._generalization_level_order)} levels")
            print(f"[GENERALIZATION] Train: {len(self.generalization_protocol.get_train_levels())} levels")
            print(f"[GENERALIZATION] Test: {len(self.generalization_protocol.get_test_levels())} levels")

        self.world_model = WorldModel({
            Params.gravity: 85,
            Params.velocity: 180
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
        # Always use base_domain_modified.pddl (not domain.pddl) for ENHSP
        self.world_model.kb = self.kb
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
        self._last_problem_data = None  # PDDL objects from last get_action_to_perform()

        # Win/loss tracking per level
        self.start_counting_from_game = start_counting_from_game  # Skip first X games before counting
        self.games_played = 0  # Total games played counter
        self.game_results = []  # Array of (level, "win"/"loss")
        
        # Per-shot planner status tracking (for CSV reporting)
        self._last_planner_unsolvable = False  # True if planner reported "unsolvable"
        self._last_plan_source = "planner"  # "planner" or "fallback"
        
        # Initialize agent comparison CSV for tracking results across agents
        self.comparison_csv = AgentComparisonCSV(
            output_path=comparison_csv_path,
            human_baseline_path=human_baseline_path
        )

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
                # Learn collision model (ablation disabled)
                update_model_effects("collision", self.kb, pre_state, post_state, debug=False)
            
            # Only use first collision per trajectory
            break
    
    def learn_flight_physics(self, trajectory, force_scale: float = 1.0):
        """
        Learn flight physics (gravity, velocity) from observed trajectory.
        
        Stores trajectory in KB and learns state transition functions
        that can predict the next state from the previous state.
        
        Parameters:
        -----------
        trajectory : np.ndarray
            The observed trajectory (first segment, before any collision)
        force_scale : float
            The planned_force fraction (0.2–1.0) used for the shot.
            In the linear regime v_meas ≈ v_full * force, so dividing by force_scale
            recovers the force=1.0 baseline velocity stored in the world model.
        
        Updates:
        --------
        self.kb["trajectories"] with new trajectory
        self.learned_transitions with updated models
        self.learned_transition_world_model with new WorldModel
        """
        print("\n" + "=" * 60)
        print("[FLIGHT PHYSICS LEARNING] Starting...")
        print("=" * 60)
        
        # Log trajectory details
        print(f"\n[TRAJECTORY INPUT]")
        print(f"  Length: {len(trajectory)} frames")
        if len(trajectory) > 0:
            print(f"  Start position: ({trajectory[0][0]:.1f}, {trajectory[0][1]:.1f})")
            print(f"  End position: ({trajectory[-1][0]:.1f}, {trajectory[-1][1]:.1f})")
            
            # Calculate observed physics from trajectory
            if len(trajectory) > 2:
                dt = 1/50  # 50 fps
                dx = np.diff(trajectory[:, 0])
                dy = np.diff(trajectory[:, 1])
                vx = dx / dt
                vy = dy / dt
                
                print(f"\n[OBSERVED PHYSICS FROM TRAJECTORY]")
                print(f"  Initial Vx: {vx[0]:.1f}")
                print(f"  Initial Vy: {vy[0]:.1f}")
                print(f"  Initial |V|: {np.sqrt(vx[0]**2 + vy[0]**2):.1f}")
                
                if len(vy) > 1:
                    ay = np.diff(vy) / dt
                    print(f"  Observed avg acceleration (gravity): {np.mean(ay):.1f}")
        
        # Store trajectory in KB
        if "trajectories" not in self.kb:
            self.kb["trajectories"] = []
        self.kb["trajectories"].append(trajectory.copy())
        print(f"\n[KB UPDATE] Now have {len(self.kb['trajectories'])} trajectories in knowledge base")
        
        self.learn_process_transitions()
        current_v_bird = self.world_model.hyperparams_values.get(Params.velocity)
        self.learned_transition_world_model = self._create_learned_transition_world_model(
            force_scale=force_scale, current_v_bird=current_v_bird
        )
        
        print("\n" + "-" * 60)
        print("[LEARNED WORLD MODEL]")
        print(f"  Gravity: {self.learned_transition_world_model.hyperparams_values.get(Params.gravity, 'N/A'):.2f}")
        print(f"  Velocity: {self.learned_transition_world_model.hyperparams_values.get(Params.velocity, 'N/A'):.2f}")
        print("=" * 60 + "\n")

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
        print("\n" + "="*60)
        print("[DEBUG] solve() STARTED")
        print("="*60)
        
        print("[DEBUG] Step 1: Getting ground truth type...")
        ground_truth_type = GroundTruthType.ground_truth_screenshot
        
        print("[DEBUG] Step 2: Updating vision reader...")
        vision = self._update_reader(ground_truth_type.value, self.if_check_gt)
        print("[DEBUG] Vision reader updated successfully")

        print("[DEBUG] Step 3: Finding slingshot...")
        sling = vision.find_slingshot_mbr()[0]
        sling.width, sling.height = sling.height, sling.width
        print(f"[DEBUG] Slingshot found at: ({sling.X}, {sling.Y}), size: {sling.width}x{sling.height}")
        
        # 1. Get angle (priority: generalization_protocol > angle_protocol > determinism_test > override_angle > PDDL planner)
        should_learn = True  # Default: learn from shot
        current_phase = "train"  # Default phase
        
        print(f"[DEBUG] Step 4: Selecting angle method...")
        print(f"[DEBUG]   debug_mag_comparison={self.debug_mag_comparison}")
        print(f"[DEBUG]   force_learning_mode={self.force_learning_mode}")
        print(f"[DEBUG]   use_generalization_protocol={self.use_generalization_protocol}")
        print(f"[DEBUG]   use_angle_protocol={self.use_angle_protocol}")
        print(f"[DEBUG]   determinism_test_mode={self.determinism_test_mode}")
        print(f"[DEBUG]   override_angle={self.override_angle}")
        
        # Force learning mode (highest priority)
        if self.force_learning_mode:
            return self._solve_force_learning(sling)
        
        # Debug mag comparison mode (second priority)
        if self.debug_mag_comparison:
            return self._solve_mag_comparison(sling)
        
        # Default force for non-PDDL branches (determinism test, override angle, etc.)
        planned_force = FORCE_MAX

        # Phy-Q Generalization Protocol (takes priority if enabled)
        if self.use_generalization_protocol and self.generalization_protocol is not None:
            # Get phase based on current level index
            current_phase, should_learn = self.generalization_protocol.get_phase_for_level(self.current_level)
            
            if current_phase == "complete":
                print("\n" + "="*60)
                print("GENERALIZATION PROTOCOL COMPLETE!")
                print("="*60)
                
                # Print final results summary
                split = self.generalization_protocol.get_split()
                train_wins = sum(1 for r in self.phyq_metrics.results 
                                if r.won and r.level_path in split.train_levels)
                train_total = len([r for r in self.phyq_metrics.results 
                                  if r.level_path in split.train_levels])
                test_wins = sum(1 for r in self.phyq_metrics.results 
                               if r.won and r.level_path in split.test_levels)
                test_total = len([r for r in self.phyq_metrics.results 
                                 if r.level_path in split.test_levels])
                
                print(f"\n{'='*70}")
                print(f"GENERALIZATION RESULTS ({split.generalization_type.value.upper()})")
                print(f"{'='*70}")
                print(f"Train Performance: {train_wins}/{train_total} "
                      f"({100*train_wins/train_total:.1f}%)" if train_total > 0 else "N/A")
                print(f"Test Performance:  {test_wins}/{test_total} "
                      f"({100*test_wins/test_total:.1f}%)" if test_total > 0 else "N/A")
                
                if split.generalization_type.value == "broad":
                    print(f"\nTrain templates: {sorted(split.train_templates)}")
                    print(f"Test templates:  {sorted(split.test_templates)}")
                
                # Full Phy-Q report
                self.phyq_metrics.print_report()
                
                # Save results
                self.phyq_metrics.save("phyq_generalization_results.json")
                print(f"\n[PHY-Q] Results saved to phyq_generalization_results.json")
                
                print("\n[GENERALIZATION] Exiting - protocol finished.")
                import sys
                sys.exit(0)
            
            # Print status
            split = self.generalization_protocol.get_split()
            n_train = len(split.train_levels)
            n_test = len(split.test_levels)
            level_in_phase = self.current_level if current_phase == "train" else self.current_level - n_train
            phase_total = n_train if current_phase == "train" else n_test
            
            print(f"\n[GENERALIZATION] Phase: {current_phase.upper()} | "
                  f"Level: {level_in_phase}/{phase_total} | Learning: {should_learn}")
            
            # Use PDDL planner for angle selection
            print(f"[{current_phase.upper()}] Using PDDL planner")
            print("[DEBUG] Calling get_action_to_perform()...")
            actions = self.get_action_to_perform(self.world_model)
            planned_force, angle = _extract_force_angle(actions)
            print(f"[{current_phase.upper()}] PDDL selected angle: {angle}°, force: {planned_force}")
        
        elif self.use_angle_protocol and self.angle_protocol is not None:
            # Use train/val/test protocol for phase tracking, but PDDL for angle selection
            _, current_phase, should_learn = self.angle_protocol.get_next_shot()
            
            if current_phase == "complete":
                print("\n" + "="*60)
                print("TRAINING PROTOCOL COMPLETE!")
                print("="*60)
                self.angle_protocol.print_final_results(kb=self.kb, phyq_metrics=self.phyq_metrics)
                
                # Save Phy-Q results to file
                self.phyq_metrics.save("phyq_results.json")
                print(f"\n[PHY-Q] Results saved to phyq_results.json")
                
                print("\n[PROTOCOL] Exiting - training protocol finished.")
                import sys
                sys.exit(0)  # Exit program - protocol is done
            
            self.angle_protocol.print_status()
            
            # Use PDDL planner for angle selection in all phases
            print(f"[{current_phase.upper()}] Using PDDL planner, Learning: {should_learn}")
            print("[DEBUG] Calling get_action_to_perform()...")
            actions = self.get_action_to_perform(self.world_model)
            planned_force, angle = _extract_force_angle(actions)
            print(f"[{current_phase.upper()}] PDDL selected angle: {angle}°, force: {planned_force}")
        
        elif self.determinism_test_mode:
            angle = round(random.uniform(20.0, 80.0), 1)
            print(f"\n[DETERMINISM] Random angle: {angle}° (uniform [20, 80], 1 decimal)")
        
        elif self.override_angle is not None:
            angle = self.override_angle
            print(f"\n[OVERRIDE] Using fixed angle: {angle}°")
        
        else:
            # Use PDDL planner
            print("[DEBUG] Using PDDL planner to select angle...")
            print("[DEBUG] Calling get_action_to_perform()...")
            actions = self.get_action_to_perform(self.world_model)
            planned_force, angle = _extract_force_angle(actions)
            print(f"\n[PDDL] Planner selected angle: {angle}°, force: {planned_force}")

        # 2. Execute shot using planned force (mag = height * 0.5 * force, linear regime)
        # PDDL flight physics use (angle - angle_bias); match that in the slingshot pull.
        flight_angle_deg = angle - ANGLE_BIAS_DEGREES
        print(f"[DEBUG] Step 5: Executing shot — PDDL dial={angle:.1f}°, "
              f"flight θ={flight_angle_deg:.1f}° (bias={ANGLE_BIAS_DEGREES}°), force={planned_force:.3f}...")
        release_point = self.tp.find_release_point_partial_power(
            sling, math.radians(flight_angle_deg), v_portion=planned_force * FORCE_V_SCALE
        )
        print(f"[DEBUG] Release point: ({release_point.X}, {release_point.Y})")
        print("[DEBUG] Calling shoot_and_record_ground_truth()...")
        batch_gt = self.ar.shoot_and_record_ground_truth(release_point.X, release_point.Y, 0, 0, 1, 0)
        print(f"[DEBUG] Shot executed, got {len(batch_gt) if batch_gt else 0} ground truth frames")

        # Extract trajectory and events
        groundtruth_trajectories, groundtruth_objects = extract_real_trajectory(batch_gt, angle, self.model, self.target_class)
        event_indexes_by_event, objects_features = getSegmentsEvents(groundtruth_trajectories, groundtruth_objects)

        # === MULTI-BIRD SUPPORT ===
        # Find the active bird - could be redBird_0, redBird_1, etc.
        # The active bird is typically the one with the longest trajectory (actively flying)
        bird_keys = [k for k in groundtruth_trajectories.keys() if k.startswith("redBird")]
        
        print(f"\n[MULTI-BIRD DEBUG] Birds detected in trajectory: {bird_keys}")
        print(f"[MULTI-BIRD DEBUG] All objects tracked: {list(groundtruth_objects.keys())}")
        
        # Find the bird with the longest/most dynamic trajectory (the one we just shot)
        active_bird_key = None
        max_trajectory_length = 0
        
        for bird_key in bird_keys:
            traj = groundtruth_trajectories.get(bird_key, [])
            traj_len = len(traj) if isinstance(traj, (list, np.ndarray)) else 0
            print(f"[MULTI-BIRD DEBUG] {bird_key}: {traj_len} frames")
            
            # Also check if the bird moved significantly (active bird)
            if traj_len > 0:
                traj_array = np.array(traj)
                if len(traj_array) >= 2:
                    # Calculate total displacement
                    displacement = np.sqrt((traj_array[-1, 0] - traj_array[0, 0])**2 + 
                                         (traj_array[-1, 1] - traj_array[0, 1])**2)
                    print(f"[MULTI-BIRD DEBUG] {bird_key}: displacement = {displacement:.1f} pixels")
                    
                    # Prefer bird with significant movement AND longest trajectory
                    if traj_len > max_trajectory_length and displacement > 10:
                        max_trajectory_length = traj_len
                        active_bird_key = bird_key
        
        # Fallback to redBird_0 if no moving bird found
        if active_bird_key is None:
            active_bird_key = "redBird_0" if "redBird_0" in groundtruth_trajectories else (bird_keys[0] if bird_keys else None)
            print(f"[MULTI-BIRD DEBUG] No moving bird found, using fallback: {active_bird_key}")
        else:
            print(f"[MULTI-BIRD DEBUG] Active bird detected: {active_bird_key}")
        
        if active_bird_key is None or active_bird_key not in groundtruth_trajectories:
            print(f"[MULTI-BIRD ERROR] No bird trajectory found! Keys available: {list(groundtruth_trajectories.keys())}")
            # Create empty trajectory to avoid crash
            bird_observed_trajectory = np.array([[0, 0]])
            bird_observed_features = [{'x': 0, 'y': 0, 'v_x': 0, 'v_y': 0, 'a_x': 0, 'a_y': 0}]
        else:
            bird_observed_trajectory = groundtruth_trajectories[active_bird_key]
            bird_observed_features = objects_features[active_bird_key]
        
        event_indexes = sorted([val for values in event_indexes_by_event.values() for val in values])
        parts = np.split(bird_observed_trajectory, event_indexes)
        
        # Debug: Event detection info
        print(f"\n[EVENT DEBUG] Active bird: {active_bird_key}")
        print(f"[EVENT DEBUG] Trajectory length: {len(bird_observed_trajectory)} frames")
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
        # DISABLED FOR DIRECT HIT MODE - collision learning from ground bounces
        # corrupts the physics model. Only enable if you need bounce predictions.
        ENABLE_COLLISION_LEARNING = True  # Set to True to re-enable bounce learning
        
        collisions = event_indexes_by_event["ground_collision"]
        hits = event_indexes_by_event.get("hit", [])
        block_collisions = event_indexes_by_event.get("block_collision", [])
        
        platform_collisions = event_indexes_by_event.get("platform_collision", [])
        
        print(f"\n[COLLISION DEBUG] Ground: {len(collisions)}, Hits: {len(hits)}, Blocks: {len(block_collisions)}, Platforms: {len(platform_collisions)}")
        if block_collisions:
            print(f"[COLLISION DEBUG] First block collision at frame {block_collisions[0]}")
        if platform_collisions:
            print(f"[COLLISION DEBUG] Platform collision detected at frames: {platform_collisions[:5]}{'...' if len(platform_collisions) > 5 else ''}")
        
        # Debug platform collision detection - visualize trajectory with all events
        DEBUG_PLATFORM_COLLISION = False  # Set to True to enable visualization
        if DEBUG_PLATFORM_COLLISION:
            # Check for platforms in the scene
            platform_names = [name for name in groundtruth_objects.keys() if "hill" in name.lower()]
            if platform_names:
                print(f"\n[PLATFORM DEBUG] Platforms in scene: {platform_names}")
                for pname in platform_names:
                    dims = groundtruth_objects.get(pname, [])
                    print(f"[PLATFORM DEBUG] {pname} dimensions: {dims}")
                
                # Debug: manually check collision with detailed output
                from agents.pddl.pddl_files.events.event_conditions import is_platform_collision
                
                # Build frames for collision check (same as in check_events)
                max_time = max(len(traj) for traj in objects_features.values())
                features_copy = {k: list(v) for k, v in objects_features.items()}
                for obj, traj in features_copy.items():
                    while len(traj) < max_time:
                        traj.append(traj[-1])
                frames = []
                for frame_values in zip(*features_copy.values()):
                    frame_dict = dict(zip(features_copy.keys(), frame_values))
                    frames.append(frame_dict)
                
                # Check every 10th frame with debug output
                print(f"\n[PLATFORM DEBUG] Checking {len(frames)} frames for collision...")
                collision_found = False
                for i in range(0, len(frames), 10):
                    if is_platform_collision(frames, groundtruth_objects, i, debug=True):
                        collision_found = True
                        print(f"[PLATFORM DEBUG] Collision detected at frame {i}!")
                
                if not collision_found:
                    # Check frame 0 to see bird/platform setup
                    bird = frames[0].get("redBird_0", {})
                    print(f"\n[PLATFORM DEBUG] Frame 0: Bird at ({bird.get('x', 'N/A')}, {bird.get('y', 'N/A')})")
                    for pname in platform_names:
                        if pname in frames[0]:
                            plat = frames[0][pname]
                            print(f"[PLATFORM DEBUG] Frame 0: {pname} at ({plat.get('x', 'N/A')}, {plat.get('y', 'N/A')})")
                
                # Show visualization
                debug_all_events_full_trajectory(objects_features, groundtruth_objects)
        
        if ENABLE_COLLISION_LEARNING:
            # Always record collision samples (for alpha validation), but only train during train phase
            self.learn_collision_effects(collisions, bird_observed_features, phase=current_phase, should_learn=should_learn)
        else:
            print(f"[DIRECT HIT MODE] Collision learning DISABLED (prevents physics contamination)")
            if not getattr(self, "_last_plan_uses_ground", False):
                print(
                    "[DIRECT HIT MODE] Skipping shot outcome log "
                    "(PDDL plan does not use ground collision)"
                )
            elif len(hits) > 0:
                print(f"[DIRECT HIT MODE] SUCCESS - Bird hit something! Frames: {hits}")
            elif len(collisions) > 0:
                print(
                    f"[DIRECT HIT MODE] MISS - Bird hit ground without hitting target. "
                    f"First collision at frame {collisions[0]}"
                )

        # 4. Learn flight physics (gravity, velocity) - ONLY during train phase
        first_segment = parts[0]
        
        print(f"\n[SEGMENT 0 DEBUG] Length: {len(first_segment)} frames")
        if len(first_segment) > 0:
            print(f"[SEGMENT 0 DEBUG] Start: ({first_segment[0][0]:.1f}, {first_segment[0][1]:.1f})")
            print(f"[SEGMENT 0 DEBUG] End: ({first_segment[-1][0]:.1f}, {first_segment[-1][1]:.1f})")
        
        if should_learn:
            self.learn_flight_physics(first_segment, force_scale=planned_force)
        else:
            print(f"[{current_phase.upper()}] Skipping flight physics learning (evaluation mode)")

        # 5. Visualize trajectory comparison
        # Trim 2 frames from end of first_segment for cleaner RMSE (avoid noisy impact transition)
        first_segment_trimmed = first_segment[:-2] if len(first_segment) > 5 else first_segment

        gravity = self.world_model.hyperparams_values[Params.gravity]
        try:
            n_vel = min(5, max(1, len(first_segment_trimmed) - 2))
            n_pos = min(3, max(1, len(first_segment_trimmed)))
            launch = estimate_launch_from_trajectory(
                first_segment_trimmed, n_vel=n_vel, n_pos=n_pos,
            )
        except ValueError:
            dt = 0.02
            release = np.asarray(first_segment_trimmed[0], dtype=float)
            if len(first_segment_trimmed) > 1:
                launch = {
                    "release": release,
                    "vx": (first_segment_trimmed[1, 0] - first_segment_trimmed[0, 0]) / dt,
                    "vy": (first_segment_trimmed[1, 1] - first_segment_trimmed[0, 1]) / dt,
                    "v_meas": self.world_model.hyperparams_values[Params.velocity],
                    "theta_deg": angle,
                }
                launch["v_meas"] = float(np.hypot(launch["vx"], launch["vy"]))
                launch["theta_deg"] = float(np.degrees(np.arctan2(launch["vy"], launch["vx"])))
            else:
                v = self.world_model.hyperparams_values[Params.velocity]
                rad = math.radians(angle)
                launch = {
                    "release": release,
                    "vx": v * math.cos(rad),
                    "vy": v * math.sin(rad),
                    "v_meas": v,
                    "theta_deg": angle,
                }
            print("[LAUNCH ESTIMATE] Short segment — using fallback velocity estimate")

        release = launch["release"]

        print(f"\n[LAUNCH ESTIMATE] release=({release[0]:.2f}, {release[1]:.2f}) "
              f"v=({launch['vx']:.1f}, {launch['vy']:.1f}) |v|={launch['v_meas']:.1f} "
              f"θ_meas={launch['theta_deg']:.1f}° (planner dial={angle:.1f}°, flight θ={angle - ANGLE_BIAS_DEGREES:.1f}°)")

        if should_learn:
            v_old = self.world_model.hyperparams_values[Params.velocity]
            # Only update v_bird EMA from near-full-force shots.
            # Partial-force shots cannot reliably back-calculate v_full because the
            # game's force→velocity curve has a non-zero intercept (v ≈ a*f + b),
            # so v_meas/force overshoots and corrupts the calibration.
            if planned_force >= 0.95:
                v_new = 0.85 * v_old + 0.15 * launch["v_meas"]
                self.world_model.hyperparams_values[Params.velocity] = v_new
                print(f"[LAUNCH ESTIMATE] v_bird: {v_old:.2f} -> {v_new:.2f} "
                      f"(EMA updated, v_meas={launch['v_meas']:.2f}, force={planned_force:.3f})")
            else:
                print(f"[LAUNCH ESTIMATE] v_bird: {v_old:.2f} -> {v_old:.2f} "
                      f"(EMA skipped, partial force={planned_force:.3f}, v_meas={launch['v_meas']:.2f})")

        limit = np.max(first_segment_trimmed, axis=0)[0]
        estimated_trajectory = construct_trajectory_from_velocity(
            release, launch["vx"], launch["vy"], gravity, limit,
            prt=False, integration_method='rk4', stop_at_ground=True,
        )

        model_for_suggested = getattr(self, 'learned_transition_world_model', None) or self.world_model
        suggested_gravity = model_for_suggested.hyperparams_values[Params.gravity]
        suggested_trajectory = construct_trajectory_from_velocity(
            release, launch["vx"], launch["vy"], suggested_gravity, limit,
            prt=False, integration_method='rk4', stop_at_ground=True,
        )

        current_rmse = calculate_rmse(first_segment_trimmed, estimated_trajectory, trim_start_percent=0, trim_end_percent=0, apply_bias_correction=False)
        self.rmse.append(current_rmse)
        self.suggested_rmse.append(calculate_rmse(first_segment_trimmed, suggested_trajectory))
        
        # === SEGMENT 0 TRAJECTORY VISUALIZATION ===
        # Visualize observed vs estimated trajectory for segment 0 (flight physics)
        world_model_params = {
            'gravity': self.world_model.hyperparams_values.get(Params.gravity, 85),
            'velocity': self.world_model.hyperparams_values.get(Params.velocity, 180)
        }
        
        print(f"\n[SEGMENT 0 PHYSICS] Current World Model:")
        print(f"  Gravity: {world_model_params['gravity']:.2f}")
        print(f"  Velocity: {world_model_params['velocity']:.2f}")
        print(f"  RMSE: {current_rmse:.2f}")
        
        # Display expected-vs-actual trajectory comparison after every level (blocks until window closed)
        # try:
        #     level_idx = getattr(self, 'current_level', 0)
        #     attempt_num = len(self.rmse)
        #     visualize_trajectory_segment0(
        #         first_segment_trimmed, estimated_trajectory,
        #         angle=angle, world_model_params=world_model_params,
        #         title=f"Level {level_idx} Attempt {attempt_num} - Expected vs Actual (angle={angle:.1f}°, RMSE={current_rmse:.2f})"
        #     )
        # except Exception as e:
        #     print(f"[SEGMENT 0 VIZ] Visualization error (non-fatal): {e}")

        # VISUALIZATION DISABLED - uncomment to enable
        # Starting-point diagnostic: GT release vs PDDL ref / post-pa-twang (aligned) vs estimated
        # try:
        #     if len(first_segment_trimmed) > 0:
        #         ref_pos = pddl_bird_position_before_pa_twang(
        #             float(release[0]), float(release[1]), angle
        #         )
        #         pddl_after = (float(release[0]), float(release[1]))
        #         print("\n[START OFFSET VIZ] Opening starting-point comparison (close window to continue)...")
        #         visualize_starting_point_offset(
        #             first_segment_trimmed,
        #             estimated_trajectory,
        #             pddl_ref_pos=ref_pos,
        #             pddl_bird_pos=pddl_after,
        #             angle=angle,
        #             show_first_n_points=10,
        #         )
        #     else:
        #         print("[START OFFSET VIZ] Skipped (no PDDL problem data or empty segment)")
        # except Exception as e:
        #     print(f"[START OFFSET VIZ] Visualization error (non-fatal): {e}")
        
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
            extended_estimated = construct_trajectory_from_velocity(
                release, launch["vx"], launch["vy"], gravity, impact_limit,
                prt=False, integration_method='rk4', stop_at_ground=False,
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

        # Show combined learning dashboard every 10 attempts (DISABLED)
        # if len(self.full_trajectories) % 10 == 0:
        #     visualize_learning_dashboard(
        #         self.full_trajectories, 
        #         self.rmse, 
        #         self.suggested_rmse,
        #         self.impact_rmse, 
        #         self.impact_trajectories
        #     )

        # 6. Update game state and world model
        # Only record game results when level is FINISHED (WON or LOST)
        # Not while still PLAYING (more birds available)
        game_state = self.ar.get_game_state()
        level_finished = game_state == GameState.WON or game_state == GameState.LOST
        game_result = game_state == GameState.WON
        
        # Track attempts per level (for multi-bird scenarios)
        if not hasattr(self, '_current_level_attempts'):
            self._current_level_attempts = 0
        self._current_level_attempts += 1
        
        if level_finished:
            # Level is complete - record the final result
            self.wins.append(game_result)
            self.games_played += 1
            
            if self.games_played > self.start_counting_from_game:
                self.game_results.append((self.current_level, "win" if game_result else "loss"))

            # Record Phy-Q benchmark result
            level_path = self.get_current_level_path()
            score = self.ar.get_current_score() if game_result else 0
            self.phyq_metrics.record(
                level_path=level_path,
                won=game_result,
                attempts=self._current_level_attempts,
                score=score
            )
            scenario = self.phyq_metrics.extract_scenario_from_path(level_path)
            level_name = level_path.split('/')[-1] if '/' in level_path else level_path
            scenario_str = scenario if scenario else "unknown"
            print(f"[PHY-Q] Level: {level_name} | Scenario: {scenario_str} | "
                  f"Result: {'WIN' if game_result else 'LOSS'} | Score: {score} | "
                  f"Birds used: {self._current_level_attempts}")
            
            # Record to agent comparison CSV
            if self.comparison_csv:
                self.comparison_csv.write_result(
                    level_path=level_path,
                    agent="PDDLAgent",
                    won=game_result,
                    mode=current_phase,
                    score=score,
                    plan_source=self._last_plan_source,
                    unsolvable=self._last_planner_unsolvable
                )
            
            # Reset attempt counter for next level
            self._current_level_attempts = 0

            if game_state == GameState.LOST:
                self._show_pddl_visualizations_on_loss(
                    angle, bird_observed_trajectory, event_indexes, world_model_params
                )
        else:
            # Level still in progress (more birds available)
            print(f"[MULTI-BIRD] Shot {self._current_level_attempts} complete, game still PLAYING - more birds available")

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
            
            # Run alpha validation if enabled
            if self.validate_alpha_on_validation:
                print("\n[ALPHA VALIDATION] Running alpha hyperparameter validation...")
                self.angle_protocol.validate_alpha()

        time.sleep(3)

    def _solve_force_learning(self, sling):
        """
        Force -> velocity learning mode.

        Each call:
          1. Pick a random angle in [20, 80] degrees
          2. Pick a random force in [0.2, 1.0]
          3. Shoot and record trajectory
          4. Estimate initial velocity from trajectory
          5. Append (force, v_meas) to kb_force
          6. If enough samples, fit LinearRegression v = a*force + b and print R²
        """
        from math import radians
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score
        import numpy as np

        n = len(self.kb_force) + 1
        angle = round(random.uniform(20.0, 80.0), 1)
        force = round(random.uniform(0.2, 1.0), 3)

        print("\n" + "="*60)
        print(f"[FORCE LEARN] Shot {n} | angle={angle}° | force={force:.3f}")
        print("="*60)

        # Build release point using multiplier=1 (height * 1 * force)
        # This gives pullbacks of 7.6-38px (spans the velocity threshold zone)
        from src.utils.point2D import Point2D
        theta = radians(angle)
        ref = self.tp.get_reference_point(sling)
        mag = sling.height * 0.5 * force
        release_point = Point2D(
            int(ref.X - mag * np.cos(theta)),
            int(ref.Y + mag * np.sin(theta))
        )

        pullback = ((release_point.X - ref.X)**2 + (release_point.Y - ref.Y)**2) ** 0.5
        print(f"  Release point: ({release_point.X}, {release_point.Y})")
        print(f"  Effective pullback: {pullback:.1f} px (height×1×force = {sling.height}×1×{force:.3f} = {mag:.1f})")

        # Shoot
        batch_gt = self.ar.shoot_and_record_ground_truth(release_point.X, release_point.Y, 0, 0, 1, 0)
        print(f"  Got {len(batch_gt) if batch_gt else 0} ground truth frames")

        # Extract trajectory
        groundtruth_trajectories, _ = extract_real_trajectory(batch_gt, angle, self.model, self.target_class)

        # Find active bird trajectory
        bird_keys = [k for k in groundtruth_trajectories.keys() if k.startswith("redBird")]
        first_segment = np.array([])
        for bk in bird_keys:
            traj = groundtruth_trajectories.get(bk, [])
            if len(traj) > len(first_segment):
                first_segment = np.array(traj)

        if len(first_segment) < 3:
            print(f"  [FORCE LEARN] WARNING: trajectory too short ({len(first_segment)} pts), skipping")
            time.sleep(1)
            return self.ar.get_game_state()

        # Estimate initial velocity from trajectory
        try:
            n_vel = min(5, max(1, len(first_segment) - 2))
            n_pos = min(3, max(1, len(first_segment)))
            launch = estimate_launch_from_trajectory(first_segment, n_vel=n_vel, n_pos=n_pos)
            v_meas = launch["v_meas"]
        except Exception as e:
            print(f"  [FORCE LEARN] WARNING: velocity estimation failed ({e}), skipping")
            time.sleep(1)
            return self.ar.get_game_state()

        print(f"  v_meas={v_meas:.2f}")

        # Record into KB
        self.kb_force.append((force, v_meas))

        # Fit model if enough samples
        if len(self.kb_force) >= self.force_learning_min_samples:
            from agents.pddl.optimizer import get_poly_rank
            forces_1d = np.array([f for f, _ in self.kb_force])
            forces_2d = forces_1d.reshape(-1, 1)
            velocities = np.array([v for _, v in self.kb_force])

            # --- Model A: Linear Regression ---
            model_lr = LinearRegression()
            model_lr.fit(forces_2d, velocities)
            v_pred_lr = model_lr.predict(forces_2d)
            r2_lr = r2_score(velocities, v_pred_lr)
            self.force_lr_model = model_lr
            self.force_lr_r2 = r2_lr

            a = model_lr.coef_[0]
            b = model_lr.intercept_
            sign = "+" if b >= 0 else "-"

            # --- Model B: Best-degree Polynomial (get_poly_rank) ---
            rank, poly = get_poly_rank(forces_1d, velocities, max_rank=5, threshold=1.0)
            v_pred_poly = poly(forces_1d)
            r2_poly = r2_score(velocities, v_pred_poly)

            # --- Side-by-side comparison ---
            n = len(self.kb_force)
            print(f"\n  {'='*50}")
            print(f"  [FORCE LEARN] Model comparison (n={n})")
            print(f"  {'='*50}")
            print(f"  [A] Linear:  v = {a:.2f} * force {sign} {abs(b):.2f}  |  R² = {r2_lr:.4f}")
            print(f"  [B] Poly d={rank}: {poly}  |  R² = {r2_poly:.4f}")
            winner = "A (Linear)" if r2_lr >= r2_poly else f"B (Poly d={rank})"
            print(f"  --> Better fit: {winner}")
            print(f"  {'='*50}")

            # Show all data points
            print(f"  [FORCE LEARN] KB: ", end="")
            print(", ".join(f"({f:.2f},{v:.1f})" for f, v in self.kb_force))
        else:
            print(f"  [FORCE LEARN] Collecting samples ({len(self.kb_force)}/{self.force_learning_min_samples} needed for model)")

        time.sleep(1)
        return self.ar.get_game_state()

    def _solve_mag_comparison(self, sling):
        """
        Debug mode: Test how different mag values affect trajectory.
        
        Shoots at a fixed angle with decreasing mag values to find the
        threshold where trajectory changes.
        """
        import json
        
        if self.mag_comparison_done:
            print("\n" + "="*60)
            print("[MAG COMPARISON] Test complete! Exiting...")
            print("="*60)
            
            # Save results
            with open('mag_comparison_results.json', 'w') as f:
                json.dump(self.mag_comparison_results, f, indent=2)
            print(f"Results saved to mag_comparison_results.json")
            
            import sys
            sys.exit(0)
        
        angle = self.mag_comparison_angle
        current_mag = self.mag_comparison_current_mag
        
        print("\n" + "="*60)
        print(f"[MAG COMPARISON] Testing mag = {current_mag:.2f}")
        print(f"[MAG COMPARISON] Fixed angle = {angle}°")
        print("="*60)
        
        # Calculate release point with current mag
        # Using the same formula as find_release_point but with variable mag
        theta = angle * np.pi / 180
        ref = self.tp.get_reference_point(sling)
        
        # The release point formula: ref - mag * direction
        # find_release_point uses: mag = sling.height * 5
        # We use: mag = sling.height * current_mag (to test different values)
        effective_mag = sling.height * current_mag
        
        from src.utils.point2D import Point2D
        release_x = int(ref.X - effective_mag * np.cos(theta))
        release_y = int(ref.Y + effective_mag * np.sin(theta))
        release_point = Point2D(release_x, release_y)
        
        print(f"  Sling height: {sling.height}")
        print(f"  Reference point: ({ref.X}, {ref.Y})")
        print(f"  Release point: ({release_point.X}, {release_point.Y})")
        print(f"  Effective pullback: {effective_mag:.1f} pixels")
        print(f"  mag_multiplier * height = {current_mag:.2f} * {sling.height} = {effective_mag:.1f}")
        
        # Shoot and record trajectory
        batch_gt = self.ar.shoot_and_record_ground_truth(release_point.X, release_point.Y, 0, 0, 1, 0)
        print(f"  Got {len(batch_gt) if batch_gt else 0} ground truth frames")
        
        # Extract trajectory
        groundtruth_trajectories, groundtruth_objects = extract_real_trajectory(
            batch_gt, angle, self.model, self.target_class
        )
        
        # Find the active bird trajectory
        bird_keys = [k for k in groundtruth_trajectories.keys() if k.startswith("redBird")]
        trajectory = []
        
        for bird_key in bird_keys:
            traj = groundtruth_trajectories.get(bird_key, [])
            if len(traj) > len(trajectory):
                trajectory = traj
        
        trajectory = np.array(trajectory) if len(trajectory) > 0 else np.array([])
        
        print(f"  Trajectory points: {len(trajectory)}")
        if len(trajectory) > 0:
            print(f"  Start: ({trajectory[0][0]:.1f}, {trajectory[0][1]:.1f})")
            if len(trajectory) > 5:
                print(f"  Point 5: ({trajectory[5][0]:.1f}, {trajectory[5][1]:.1f})")
            if len(trajectory) > 10:
                print(f"  Point 10: ({trajectory[10][0]:.1f}, {trajectory[10][1]:.1f})")
        
        # Store result
        result = {
            'iteration': len(self.mag_comparison_results) + 1,
            'mag_multiplier': current_mag,
            'effective_mag': effective_mag,
            'sling_height': sling.height,
            'release_point': (release_point.X, release_point.Y),
            'trajectory_points': len(trajectory),
            'first_10_points': trajectory[:10].tolist() if len(trajectory) > 0 else []
        }
        self.mag_comparison_results.append(result)
        
        # Compare with BASELINE trajectory (first shot at mag=5.0)
        if self.mag_comparison_baseline_trajectory is None and len(trajectory) > 0:
            # First iteration - save as baseline
            self.mag_comparison_baseline_trajectory = trajectory.copy()
            print(f"\n  [BASELINE] Saved as reference trajectory")
        elif self.mag_comparison_baseline_trajectory is not None and len(trajectory) > 0:
            baseline_traj = self.mag_comparison_baseline_trajectory
            
            # Compare first N points
            compare_n = min(20, len(trajectory), len(baseline_traj))
            
            if compare_n >= 5:
                total_diff = 0
                for i in range(compare_n):
                    diff = np.sqrt(
                        (trajectory[i][0] - baseline_traj[i][0])**2 + 
                        (trajectory[i][1] - baseline_traj[i][1])**2
                    )
                    total_diff += diff
                
                avg_diff = total_diff / compare_n
                result['avg_diff_from_baseline'] = avg_diff
                
                print(f"\n  Comparison with BASELINE (mag={self.mag_comparison_start:.2f}):")
                print(f"    Average point difference: {avg_diff:.2f} pixels")
                
                # Trajectory changed significantly (threshold: 5 pixels average)
                if avg_diff > 5.0:
                    print(f"\n" + "*"*60)
                    print(f"*** TRAJECTORY CHANGED at mag = {current_mag:.2f} ***")
                    print(f"*** Baseline mag = {self.mag_comparison_start:.2f} ***")
                    print(f"*** Avg difference from baseline = {avg_diff:.2f} pixels ***")
                    print(f"*"*60)
                    
                    result['trajectory_changed'] = True
                    self.mag_comparison_done = True
                else:
                    print(f"    Trajectory SIMILAR to baseline (threshold: 5.0 pixels)")
                    result['trajectory_changed'] = False
        
        # Decrement mag for next iteration
        self.mag_comparison_current_mag -= self.mag_comparison_decrement
        
        # Check if we've gone too low
        if self.mag_comparison_current_mag < 0.1:
            print("\n[MAG COMPARISON] Reached minimum mag (0.1), stopping test")
            self.mag_comparison_done = True
        
        # Print summary so far
        print(f"\n[MAG COMPARISON] Results so far ({len(self.mag_comparison_results)} iterations):")
        print(f"  Comparing against BASELINE (mag={self.mag_comparison_start:.2f})")
        for r in self.mag_comparison_results:
            changed = "CHANGED" if r.get('trajectory_changed', False) else "same"
            diff = f", diff_from_baseline={r.get('avg_diff_from_baseline', 0):.2f}px" if 'avg_diff_from_baseline' in r else " [BASELINE]"
            print(f"  mag={r['mag_multiplier']:.2f} | pullback={r['effective_mag']:.1f}px | {changed}{diff}")
        
        # Return to let the game state handler restart the level
        time.sleep(2)
        return self.ar.get_game_state()

    def get_current_level_path(self) -> str:
        """
        Get the current level path for Phy-Q metrics tracking.
        
        Uses the generalization protocol's level order if enabled,
        otherwise uses the PhyQLevelMapper to convert current_level index to level path.
        Falls back to a generic path if neither is configured.
        
        Returns:
            Level path string (e.g., "./Levels/phy_q/scenario_03_rolling/train/rolling_t01_00001.xml")
        """
        # Generalization protocol takes priority
        if self.use_generalization_protocol and self.generalization_protocol is not None:
            level_path = self.generalization_protocol.get_level_path_for_index(self.current_level)
            if level_path:
                return level_path
        
        # Fall back to level mapper
        if self.phyq_level_mapper and len(self.phyq_level_mapper) > 0:
            level_path = self.phyq_level_mapper.get_level_path(self.current_level)
            if level_path:
                return level_path
        
        return f"level_{self.current_level}"

    def _world_model_params(self, agent_world_model: WorldModel) -> dict:
        return {
            'gravity': agent_world_model.hyperparams_values.get(Params.gravity, 85),
            'velocity': agent_world_model.hyperparams_values.get(Params.velocity, 180),
        }

    def _gather_problem_data(self, vision, sling, agent_world_model: WorldModel, ref_angle_guess: float):
        # Debug: dump all object types detected by vision
        if hasattr(vision, 'allObj') and vision.allObj:
            print("\n[VISION DEBUG] All detected object types:")
            if isinstance(vision.allObj, dict):
                for key, val in vision.allObj.items():
                    count = len(val) if val else 0
                    print(f"  '{key}': {count} object(s)")
            else:
                print(f"  allObj type: {type(vision.allObj)}")
        
        print("[PDDL DEBUG] Getting birds...")
        bird_objects = get_birds(
            vision, sling, self.tp, agent_world_model, ref_angle_guess=ref_angle_guess
        )
        print(f"[PDDL DEBUG] Birds found: {len(bird_objects) if bird_objects else 0}")
        print(f"[PDDL DEBUG] Bird ref computed with pa-twang guess angle: {ref_angle_guess:.1f}°")

        print("[PDDL DEBUG] Getting pigs...")
        pigs_objects = get_pigs(vision, sling, self.tp)
        print(f"[PDDL DEBUG] Pigs found: {len(pigs_objects) if pigs_objects else 0}")

        print("[PDDL DEBUG] Getting blocks...")
        block_objects = get_blocks(vision, sling, self.tp)
        print(f"[PDDL DEBUG] Blocks found: {len(block_objects) if block_objects else 0}")

        print("[PDDL DEBUG] Getting platforms...")
        platform_objects = get_platforms(vision, sling, self.tp)
        print(f"[PDDL DEBUG] Platforms found: {len(platform_objects) if platform_objects else 0}")

        problem_data = bird_objects | pigs_objects | block_objects | platform_objects
        return problem_data, bird_objects, pigs_objects

    def _calculate_fallback_angle(
        self,
        pigs_objects,
        bird_objects,
        world_model_params: dict,
        ref_angle_guess: float,
        problem_data: dict = None,
    ) -> float:
        if not pigs_objects:
            print("[PDDL DEBUG] No pig found, using default: 45.0°")
            return 45.0
        pig = list(pigs_objects.values())[0]
        bird = list(bird_objects.values())[0]
        v = bird.get('v_bird', 180)
        g = world_model_params.get('gravity', 85)
        launch_x, launch_y = pddl_bird_position_after_pa_twang(
            bird['x_bird'], bird['y_bird'], ref_angle_guess
        )
        try:
            term = v ** 4 - g * (g * (pig['x_pig'] - launch_x) ** 2 + 2 * (pig['y_pig'] - launch_y) * v ** 2)
            angle_low = angle_high = None
            if term >= 0 and pig['x_pig'] > launch_x:
                dx = pig['x_pig'] - launch_x
                angle_low = np.degrees(np.arctan((v ** 2 - np.sqrt(term)) / (g * dx)))
                angle_high = np.degrees(np.arctan((v ** 2 + np.sqrt(term)) / (g * dx)))
            
            if angle_low is not None and angle_high is not None:
                print(f"[PDDL DEBUG] Calculated angles: low={angle_low:.1f}°, high={angle_high:.1f}°")
                
                # Check both angles for platform collision if we have problem_data
                if problem_data is not None:
                    # Test low arc
                    sim_low = simulate_pddl_shot_plan(problem_data, angle_low, gravity=g)
                    low_hits_platform = sim_low.get('platform_collision', False)
                    low_kills_pig = sim_low.get('pig_killed_in_sim', False)
                    
                    # Test high arc
                    sim_high = simulate_pddl_shot_plan(problem_data, angle_high, gravity=g)
                    high_hits_platform = sim_high.get('platform_collision', False)
                    high_kills_pig = sim_high.get('pig_killed_in_sim', False)
                    
                    print(f"[PDDL DEBUG] Low arc ({angle_low:.1f}°): platform_hit={low_hits_platform}, pig_killed={low_kills_pig}")
                    print(f"[PDDL DEBUG] High arc ({angle_high:.1f}°): platform_hit={high_hits_platform}, pig_killed={high_kills_pig}")
                    
                    # Prefer angle that kills pig
                    if low_kills_pig and not low_hits_platform:
                        print(f"[PDDL DEBUG] Using LOW arc (kills pig): {angle_low:.1f}°")
                        return angle_low
                    if high_kills_pig and not high_hits_platform:
                        print(f"[PDDL DEBUG] Using HIGH arc (kills pig): {angle_high:.1f}°")
                        return angle_high
                    
                    # If neither kills pig cleanly, prefer one that doesn't hit platform
                    if not low_hits_platform:
                        print(f"[PDDL DEBUG] Using LOW arc (no platform hit): {angle_low:.1f}°")
                        return angle_low
                    if not high_hits_platform:
                        print(f"[PDDL DEBUG] Using HIGH arc (no platform hit): {angle_high:.1f}°")
                        return angle_high
                    
                    # Both hit platform - try high arc (better chance of clearing)
                    print(f"[PDDL DEBUG] Both arcs hit platform, trying HIGH arc: {angle_high:.1f}°")
                    return angle_high
            
            # Fallback to default ballistic (low arc)
            fallback = ballistic_angle_to_target(
                launch_x, launch_y, pig['x_pig'], pig['y_pig'], v, g,
                min_angle=self.min_deg, max_angle=self.max_deg,
            )
            if fallback is not None:
                print(f"[PDDL DEBUG] Launch→pig ballistic angle: {fallback:.1f}°")
                return fallback
            direct_angle = np.degrees(np.arctan2(pig['y_pig'] - launch_y, pig['x_pig'] - launch_x))
            fallback_angle = max(self.min_deg, min(self.max_deg, direct_angle))
            print(f"[PDDL DEBUG] Pig may be unreachable, using direct angle: {fallback_angle:.1f}°")
            return fallback_angle
        except Exception as e:
            print(f"[PDDL DEBUG] Angle calculation failed ({e}), using default: 45.0°")
            return 45.0

    def _run_enhsp_planner(self, problem_data: dict, agent_world_model: WorldModel):
        """Write problem/domain, run ENHSP. Returns (actions or None, planner_output)."""
        domain_path = 'base_domain_modified.pddl'

        print("[PDDL DEBUG] Writing problem file...")
        print(
            f"[PDDL DEBUG] Physics: gravity={agent_world_model.hyperparams_values[Params.gravity]:.2f}, "
            f"velocity={agent_world_model.hyperparams_values[Params.velocity]:.2f}"
        )
        print(f"[PDDL DEBUG] Using angle range: min={self.min_deg}°, max={self.max_deg}° (start={self.max_deg}°)")
        write_problem_file(
            'agents/pddl/pddl_files/problem.pddl',
            problem_data,
            self.max_deg,
            self.deg_step,
            agent_world_model,
            min_angle=self.min_deg,
            max_angle=self.max_deg,
        )
        print("[PDDL DEBUG] Problem file written")

        print("[PDDL DEBUG] Injecting base_domain.pddl (collision + learned physics)...")
        inject_domain_file('agents/pddl/pddl_files/base_domain.pddl', agent_world_model)
        print("[PDDL DEBUG] Domain file injected")
        print(f"[PDDL DEBUG] Using domain: {domain_path}")

        root_cwd = os.getcwd()
        pddl_dir = os.path.join(root_cwd, 'agents', 'pddl', 'pddl_files')

        print(f"[PDDL DEBUG] Current working directory: {root_cwd}")
        os.chdir(pddl_dir)
        print(f"[PDDL DEBUG] Changed to: {os.getcwd()}")

        try:
            print("[PDDL DEBUG] Running ENHSP planner (timeout=200s)...")
            print(
                f"[PDDL DEBUG] Command: java -jar enhsp-20.jar -o {domain_path} "
                f"-f problem.pddl -sp solution.pddl -planner sat-pt"
            )
            if os.path.exists('solution.pddl'):
                os.remove('solution.pddl')
                print("[PDDL DEBUG] Deleted old solution file")

            result = subprocess.run(
                [
                    'java', '-jar', 'enhsp-20.jar', '-o', domain_path,
                    '-f', 'problem.pddl', '-sp', 'solution.pddl', '-planner', 'sat-pt',
                ],
                timeout=200,
                capture_output=True,
                text=True,
            )
            print("[PDDL DEBUG] ENHSP planner finished")
            planner_output = (result.stdout or '') + (result.stderr or '')

            if "unsolvable" in planner_output.lower():
                print("[PDDL DEBUG] *** PLANNER REPORTED: Problem unsolvable ***")
                print(f"[PDDL DEBUG] Planner output: {planner_output[:300]}")
                return None, planner_output

            if not os.path.exists('solution.pddl'):
                print("[PDDL DEBUG] *** No solution file generated ***")
                print(f"[PDDL DEBUG] Planner stdout: {result.stdout[:500] if result.stdout else 'empty'}")
                print(f"[PDDL DEBUG] Planner stderr: {result.stderr[:500] if result.stderr else 'empty'}")
                return None, planner_output

            print("[PDDL DEBUG] Parsing solution...")
            actions = parse_solution_to_actions(
                'solution.pddl', self.max_deg, self.deg_step,
                force_min=FORCE_MIN, force_rate=FORCE_RATE,
            )
            print(f"[PDDL DEBUG] Parsed actions: {actions}")
            return actions, planner_output
        except Exception as e:
            print(f"[PDDL DEBUG] EXCEPTION: {type(e).__name__}: {e}")
            return None, str(e)
        finally:
            os.chdir(root_cwd)
            print(f"[PDDL DEBUG] Changed back to: {os.getcwd()}")

    def _show_pddl_visualizations_on_loss(
            self, angle, bird_observed_trajectory, event_indexes, world_model_params):
        """Display PDDL debug plots only when the level was lost."""
        if not getattr(self, 'visualize_pddl_input', False):
            return

        viz_dir = getattr(self, 'pddl_viz_dir', 'pddl_level_viz')
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)

        pending = getattr(self, '_pending_level_viz', None)
        if pending:
            try:
                visualize_level_setup(
                    pending['problem_data'],
                    pending['world_model_params'],
                    title=pending['title'],
                    save_path=pending['save_path'],
                    show_plot=True,
                    trajectory=pending.get('trajectory'),
                    angle=pending.get('angle'),
                )
                print(f"[PDDL VIZ] Saved level visualization to: {pending['save_path']}")
            except Exception as e:
                import traceback
                print(f"[PDDL VIZ] Visualization error (non-fatal): {e}")
                traceback.print_exc()
            self._pending_level_viz = None

        try:
            planned_sim_traj = getattr(self, '_last_sim_trajectory', [])
            last_problem = getattr(self, '_last_problem_data', {})
            impact_idx_for_viz = event_indexes[0] if len(event_indexes) > 0 else None
            level_idx = getattr(self, 'current_level', 0)
            attempt_num = len(self.impact_rmse)
            hit_save = os.path.join(
                viz_dir, f"level_{level_idx:03d}_attempt_{attempt_num:02d}_hit.png"
            )
            hit_title = (
                f"Level {level_idx} Attempt {attempt_num} — Expected vs Actual Hit\n"
                f"Angle={angle:.1f}°  |  Impact frame: {impact_idx_for_viz}"
            )
            visualize_expected_vs_actual_hit(
                problem_data=last_problem,
                planned_trajectory=planned_sim_traj,
                actual_trajectory=bird_observed_trajectory,
                actual_impact_idx=impact_idx_for_viz,
                angle=angle,
                world_model_params=world_model_params,
                title=hit_title,
                save_path=hit_save,
                show_plot=True,
            )
        except Exception as e:
            import traceback
            print(f"[HIT VIZ] Visualization error (non-fatal): {e}")
            traceback.print_exc()

    def _finalize_plan_metadata(self, problem_data: dict, angle: float, world_model_params: dict,
                               force: float = 1.0):
        """Store plan flags and optionally run direct-hit analysis for ground-bounce plans."""
        g = world_model_params['gravity']
        # Run simulation with debug=True to see detailed collision checks
        sim = simulate_pddl_shot_plan(problem_data, angle, gravity=g, force=force, debug=True)
        self._last_planned_angle = angle
        self._last_plan_uses_ground = sim['uses_ground_collision']
        self._last_problem_data = problem_data
        self._last_sim_trajectory = sim.get('trajectory', [])

        print(
            f"[PDDL DEBUG] Plan sim: ground_touches={sim['ground_touches']}, "
            f"pig_killed_in_sim={sim['pig_killed_in_sim']}, "
            f"uses_ground_collision={sim['uses_ground_collision']}, "
            f"platform_collision={sim.get('platform_collision', False)}"
        )
        if sim.get('platform_collision'):
            print(f"[PDDL DEBUG] Platform hit: {sim.get('platform_hit_name')} at {sim.get('platform_hit_pos')}")

        # Defer level visualization until we know the level was lost
        if getattr(self, 'visualize_pddl_input', False):
            level_idx = getattr(self, 'current_level', 0)
            attempt_num = len(self.rmse) + 1
            viz_dir = getattr(self, 'pddl_viz_dir', 'pddl_level_viz')
            sim_result = (
                "HIT PIG" if sim['pig_killed_in_sim']
                else ("HIT PLATFORM" if sim['platform_collision'] else "MISS")
            )
            self._pending_level_viz = {
                'problem_data': problem_data,
                'world_model_params': world_model_params,
                'title': (
                    f"Level {level_idx} - PDDL Input View\n"
                    f"Angle={angle:.1f}° | G={world_model_params['gravity']:.1f} | "
                    f"V={world_model_params['velocity']:.1f} | Sim: {sim_result}"
                ),
                'save_path': os.path.join(
                    viz_dir, f"level_{level_idx:03d}_attempt_{attempt_num:02d}.png"
                ),
                'trajectory': sim.get('trajectory'),
                'angle': angle,
            }

    def get_action_to_perform(self, agent_world_model: WorldModel):
        """
        Formulate_image
        """
        print("\n[PDDL DEBUG] ========== get_action_to_perform() STARTED ==========")

        ground_truth_type = GroundTruthType.ground_truth_screenshot

        print("[PDDL DEBUG] Sleeping 1 second...")
        time.sleep(1)

        print("[PDDL DEBUG] Updating vision reader...")
        vision = self._update_reader(ground_truth_type.value, self.if_check_gt)

        print("[PDDL DEBUG] Finding slingshot...")
        sling = vision.find_slingshot_mbr()[0]
        sling.width, sling.height = sling.height, sling.width
        print(f"[PDDL DEBUG] Slingshot: ({sling.X}, {sling.Y})")

        ref_guess = self._last_planned_angle if self._last_planned_angle is not None else self.max_deg
        problem_data, bird_objects, pigs_objects = self._gather_problem_data(
            vision, sling, agent_world_model, ref_angle_guess=ref_guess
        )
        print(f"[PDDL DEBUG] Total problem data keys: {list(problem_data.keys())}")

        world_model_params = self._world_model_params(agent_world_model)

        actions, planner_output = self._run_enhsp_planner(problem_data, agent_world_model)
        
        # Track planner status for CSV reporting
        self._last_planner_unsolvable = "unsolvable" in planner_output.lower() if planner_output else False
        
        if not actions:
            print("[PDDL DEBUG] Falling back to calculated trajectory angle...")
            self._last_plan_source = "fallback"
            fallback_angle = self._calculate_fallback_angle(
                pigs_objects, bird_objects, world_model_params, ref_guess, problem_data
            )
            problem_data, bird_objects, pigs_objects = self._gather_problem_data(
                vision, sling, agent_world_model, ref_angle_guess=fallback_angle
            )
            fallback_angle = self._calculate_fallback_angle(
                pigs_objects, bird_objects, world_model_params, fallback_angle, problem_data
            )
            actions = [("shoot", fallback_angle)]
            self._finalize_plan_metadata(problem_data, fallback_angle, world_model_params)
            print(f"[PDDL DEBUG] ========== get_action_to_perform() RETURNING: {actions} ==========\n")
            return actions
        
        self._last_plan_source = "planner"

        planned_force, angle = _extract_force_angle(actions)

        # if abs(angle - ref_guess) > ANGLE_REPLAN_THRESHOLD_DEG:
        #     print(
        #         f"[PDDL DEBUG] Replanning: angle {angle:.1f}° differs from "
        #         f"pa-twang guess {ref_guess:.1f}° — refreshing bird ref"
        #     )
        #     problem_data, bird_objects, pigs_objects = self._gather_problem_data(
        #         vision, sling, agent_world_model, ref_angle_guess=angle
        #     )
        #     actions, _ = self._run_enhsp_planner(problem_data, agent_world_model)
        #     if not actions:
        #         print("[PDDL DEBUG] Replan failed; refreshing bird ref at first-pass angle")
        #         problem_data, bird_objects, pigs_objects = self._gather_problem_data(
        #             vision, sling, agent_world_model, ref_angle_guess=angle
        #         )
        #         actions = [("shoot", angle)]
        #     else:
        #         _, angle = actions[0]

        self._finalize_plan_metadata(problem_data, angle, world_model_params, force=planned_force)
        print(f"[PDDL DEBUG] ========== get_action_to_perform() RETURNING: {actions} ==========\n")
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
            # Quality gate: skip trajectories that are too short or nearly flat (unreliable 2nd derivative)
            MIN_TRAJ_FRAMES = 40
            MIN_Y_EXCURSION_PX = 40  # require meaningful vertical motion for reliable gravity estimate
            y_excursion = float(np.max(traj[:, 1]) - np.min(traj[:, 1]))
            traj_quality_ok = len(traj) >= MIN_TRAJ_FRAMES and y_excursion >= MIN_Y_EXCURSION_PX
            try:
                if rank_y <= 2:
                    # Degree ≤ 2: 2nd derivative is a constant everywhere
                    yddot_traj = poly_y.deriv(2)(0)
                else:
                    # Degree 3+: evaluate 2nd derivative at EVERY time step and take the mean.
                    # poly_y.deriv(2)(0) only gives 2*c2 which is unreliable when the cubic
                    # coefficient is large (noise-driven on short/flat trajectories).
                    yddot_traj = float(np.mean([poly_y.deriv(2)(t) for t in function_range]))
            except Exception:
                yddot_traj = float(np.mean(yddot)) if len(yddot) > 0 else 0.0
            if traj_quality_ok:
                all_yddot_constants.append(yddot_traj)
            else:
                print(f"  [GRAVITY] Traj {traj_idx}: SKIP (frames={len(traj)}, y_excursion={y_excursion:.1f}px < thresholds {MIN_TRAJ_FRAMES}/{MIN_Y_EXCURSION_PX})")
            
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
            # Filter outliers: expected gravity is around -90, filter values outside [-150, -30]
            print(f"\n[GRAVITY LEARNING] Raw gravity values from {len(all_yddot_constants)} trajectories:")
            for i, g in enumerate(all_yddot_constants):
                print(f"  Trajectory {i}: gravity = {g:.2f}")
            
            if len(all_yddot_constants) > 0:
                # Filter outliers: expected gravity is ~-86 to -90 for this engine.
                # Use a tighter range that still allows natural variation but excludes
                # wildly wrong estimates from noisy/degenerate polynomial fits.
                GRAVITY_MIN, GRAVITY_MAX = -112, -65
                filtered_gravity = [g for g in all_yddot_constants if GRAVITY_MIN <= g <= GRAVITY_MAX]
                
                if len(filtered_gravity) > 0:
                    # Use median for robustness against remaining outliers
                    yddot_constant = float(np.median(filtered_gravity))
                    print(f"[GRAVITY LEARNING] Filtered to {len(filtered_gravity)} values in range [{GRAVITY_MIN}, {GRAVITY_MAX}]")
                    print(f"[GRAVITY LEARNING] Final median gravity: {yddot_constant:.4f}")
                else:
                    # All values are outliers, use default
                    yddot_constant = -90.0
                    print(f"[GRAVITY LEARNING] WARNING: All gravity values were outliers! Using default: {yddot_constant}")
            else:
                yddot_constant = -90.0
                print(f"[GRAVITY LEARNING] No qualifying trajectories, using default gravity: {yddot_constant}")
            
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
    
    def _create_learned_transition_world_model(self, force_scale: float = 1.0, current_v_bird: float = None):
        """
        Create a new WorldModel from learned transitions.
        Updates gravity from every shot (force-independent), but only updates
        velocity from near-full-force shots (force_scale >= 0.95) because the
        game's force→velocity curve has a non-zero intercept.  Dividing partial-
        force v_meas by force overshoots and corrupts the calibration.

        Parameters:
        -----------
        force_scale : float
            planned_force fraction (0.2–1.0) used for the shot.
        current_v_bird : float, optional
            The current world-model velocity to use as fallback when force_scale < 0.95.
        
        Returns:
        --------
        WorldModel
            A new WorldModel instance with updated gravity and (conditionally) velocity.
        """
        initial_values = {}
        
        # Gravity is force-independent — always update it
        if self.learned_transitions.get("yddot") is not None:
            yddot_model = self.learned_transitions["yddot"].get("model")
            if hasattr(yddot_model, 'constant_value'):
                initial_values[Params.gravity] = abs(yddot_model.constant_value)
        
        # Velocity: only trust the measurement from near-full-force shots.
        # At force < 0.95 the v_meas/force ratio overshoots due to the game's
        # non-proportional (offset) force→velocity relationship.
        vx = None
        vy = None
        if self.learned_transitions.get("xdot") is not None and "initial_value" in self.learned_transitions["xdot"]:
            vx = self.learned_transitions["xdot"]["initial_value"]
        if self.learned_transitions.get("ydot") is not None and "initial_value" in self.learned_transitions["ydot"]:
            vy = self.learned_transitions["ydot"]["initial_value"]
        
        if vx is not None and vy is not None:
            v_partial = math.sqrt(vx**2 + vy**2)
            if force_scale >= 0.95:
                # At full (or near-full) force the measured speed IS v_full — no division needed
                initial_values[Params.velocity] = v_partial
                print(f"[LEARNED MODEL] v_partial={v_partial:.2f}, planned_force={force_scale:.4f}, v_full={v_partial:.2f} (full-force, direct)")
            else:
                # Keep the existing v_bird; only gravity was reliably learned this shot
                if current_v_bird is not None:
                    initial_values[Params.velocity] = current_v_bird
                    print(f"[LEARNED MODEL] v_partial={v_partial:.2f}, planned_force={force_scale:.4f}, v_full=kept {current_v_bird:.2f} (partial-force, no update)")
                else:
                    initial_values[Params.velocity] = v_partial
                    print(f"[LEARNED MODEL] v_partial={v_partial:.2f}, planned_force={force_scale:.4f}, v_full={v_partial:.2f} (partial-force fallback)")
        
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

