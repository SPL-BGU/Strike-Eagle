import copy
import math
import os
import time
import random
import pickle
from typing import Optional, Tuple
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
    AngleCalibrator,
    simulate_pddl_shot_plan,
    ballistic_angle_to_target,
    align_dial_to_planner_grid,
    extend_platforms_in_problem_data,
    FORCE_MIN, FORCE_MAX, FORCE_RATE, FORCE_V_SCALE,
    pddl_force_to_v_portion, launch_speed_at_force, pullback_pixels,
    ANGLE_BIAS_DEGREES, MAG_MULT,
    iter_force_grid,
    ANGLE_EXECUTION_SLACK_DEG, GAP_CLEARANCE_ANGLE_NUDGE_DEG,
    PLATFORM_EXECUTION_SLACK_DEG, PLANNER_FLIGHT_SLACK_DEG,
    pddl_flight_angle_deg, pddl_planned_launch_state, pddl_shot_to_release_point,
)
from src.client.agent_client import GameState
from agents.pddl.metrics import calculate_rmse, calculate_impact_rmse

from numpy.polynomial import Polynomial

# Coarse step for planner-failure fallback grid (ENHSP dial grid stays at deg_step).
FALLBACK_GRID_DEG_STEP = 1.0
# Local angle band when prefer_sim_plan compares ENHSP vs nearby sim-scored shots.
PLAN_PICK_LOCAL_HALF_WIDTH_DEG = 4.0
PLAN_PICK_LOCAL_STEP_DEG = 0.25
# Fast plan-pick: coarser local refine + skip full grid when local kill exists.
PLAN_PICK_FAST_LOCAL_STEP_DEG = 0.5
PLAN_PICK_FULL_SEARCH_DEG_STEP = 1.0
PLAN_PICK_NEAR_SEARCH_HALF_WIDTH_DEG = 15.0
# Wall-clock cap for plan-pick forward sim only (ENHSP keeps its own timeout). 0 = no cap.
PLAN_PICK_SEARCH_TIMEOUT_SEC = 60.0


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

    def __init__(self, agent_ind, agent_configs, min_deg: int = 5, max_deg: int = 89, deg_step: float = 0.5,
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
                 visualize_pddl_input: bool = False,  # Show PDDL visualization on level loss
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
                 force_learning_min_samples: int = 5,
                 disable_sim_override: bool = False,
                 planner_only: bool = True,
                 disable_forward_sim: bool = False,
                 prefer_sim_plan: bool = False,
                 plan_pick_fast: bool = True,
                 plan_pick_timeout_sec: float = PLAN_PICK_SEARCH_TIMEOUT_SEC,
                 sim_gate_test_only: bool = True,
                 max_train_attempts: int = 5):
        super().__init__(
            agent_ind=agent_ind,
            agent_configs=agent_configs)

        # Training-level retry state:
        # On a LOST training level: dismiss LOST via load_next, then load_level(target)
        # on the next PLAYING tick (requires attempt_limit_per_level >= max_train_attempts).
        # Default 5 (down from 8): run_20260810_064920 abandoned levels 1/7/9/15
        # after 8 identical LOSSes each (all pre-flagged by sim-gate), consuming
        # ~90 min of the 200-min budget on hopeless retries and preventing every
        # test level from being played. 5 attempts still gives ENHSP room to
        # self-correct via updated calibration while leaving budget for test.
        self.max_train_attempts = max(1, int(max_train_attempts))
        self._current_level_retry_index = 0  # number of prior LOSSes on the current level
        # Plan source of the most recently FINISHED attempt on the current level
        # (updated when we bump _current_level_retry_index on LOSS; reset at level end).
        # Used by _diversify_plan_for_retry so that the first retry after a
        # FALLBACK attempt (ENHSP unsolvable → ballistic / sim_search / etc.) fires
        # the raw ENHSP plan un-perturbed instead of diversifying around a plan
        # that was never actually tried. See run_20260812_114238 train-17
        # (t02_00023): attempt 1 hit ENHSP unsolvable and used ballistic 27.5°;
        # attempt 2's ENHSP produced dial 68.0° (the historical winner), but
        # diversification perturbed it to 74.0° and we lost 5 straight.
        self._prev_attempt_plan_source = "planner"
        # Class A fix: ballistic-fallback (angle, force) history for the current
        # level. Used by _calculate_fallback_shot to diversify across retries
        # instead of firing near-identical shots (see run_20260821_065726 level 1
        # (t04_00033): 5 attempts fired at 40.0/41.0/40.5/40.5/40.5° f=1.0).
        # Reset at level end / start alongside _current_level_retry_index.
        self._current_level_fallback_history: list[tuple[float, float]] = []
        self._retry_current_level = False    # flag consumed by BaselineAgent.run() on LOST
        self._reload_in_progress = False     # suppresses transient menu-state handlers
        self._reload_target_level = None     # level number to load when reload_in_progress
        self._reload_attempts = 0            # safety cap for jump-back attempts
        self._last_seen_level_for_retry = None  # detect level changes and reset the counter

        # Science Birds "training" mode replays a level up to attempt_limit_per_level
        # times even after a WIN. Track which level_paths we have already recorded so
        # each unique level appears exactly once in CSV / phyq_metrics / game_results
        # (WIN on first solve, or ABANDONED after max_train_attempts LOSSes).
        self._finalized_level_paths = set()
        self.scenario_filter = scenario_filter

        # Mode B pixel-nudge retry: remember winning release pixels and probe ±1px
        # around near-misses before large ENHSP dial diversification (run_20260826:
        # 00023/24/16 regressed when first-shot release drifted by 1–3 px).
        self._winning_release_by_level: dict = {}
        self._release_nudge_delta: Optional[Tuple[int, int]] = None
        self._skip_retry_diversify: bool = False
        self._retry_anchor_release: Optional[Tuple[int, int]] = None
        self._near_miss_active: bool = False
        self._RELEASE_PIXEL_NUDGE_OFFSETS = (
            (1, 1), (-1, -1),
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, -1), (-1, 1),
        )
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.deg_step = deg_step
        self.angle_calibrator = AngleCalibrator()
        self._last_planned_angle = None
        self._last_planned_force = 1.0
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
        self.planner_only = planner_only
        self.disable_sim_override = disable_sim_override
        self.disable_forward_sim = disable_forward_sim
        self.sim_gate_test_only = bool(sim_gate_test_only)
        self.prefer_sim_plan = prefer_sim_plan
        self.plan_pick_fast = plan_pick_fast
        self.plan_pick_timeout_sec = max(0.0, float(plan_pick_timeout_sec))
        self._plan_pick_deadline = None
        self._plan_pick_timeout_logged = False
        if self.prefer_sim_plan and self.disable_forward_sim:
            print(
                "[PDDL] WARNING: prefer_sim_plan requires forward sim — enabling forward sim"
            )
            self.disable_forward_sim = False
        if self.disable_forward_sim:
            print(
                "[PDDL] Forward sim disabled — no fallback grid, refine, or plan simulation"
            )
        if self.prefer_sim_plan:
            mode = "fast (local refine; full grid only if needed)" if self.plan_pick_fast else "thorough (full sim grid)"
            timeout_note = (
                f", sim search cap={self.plan_pick_timeout_sec:.0f}s"
                if self.plan_pick_timeout_sec > 0 else ", sim search uncapped"
            )
            print(
                f"[PDDL] Prefer-sim-plan [{mode}{timeout_note}] — after ENHSP, pick best "
                "forward-sim score among planner / local refine / sim search"
            )
        elif self.planner_only:
            if self.disable_forward_sim:
                print(
                    "[PDDL] Planner-only mode — ENHSP plans used as-is; "
                    "forward sim disabled; fallback grid disabled"
                )
            elif self.sim_gate_test_only:
                print(
                    "[PDDL] Planner-only mode (narrow gate) — TRAIN and TEST both keep "
                    "ENHSP plan; forward sim only rejects platform-hit-short (bird stops "
                    "in front of pig) via sim search replacement"
                )
            else:
                print(
                    "[PDDL] Planner-only mode (legacy broad TEST gate) — TRAIN keeps "
                    "ENHSP plan (narrow gate); TEST falls through to sim_search/fallback"
                )
        elif self.disable_sim_override:
            print(
                "[PDDL] Sim override DISABLED — ENHSP plans kept when sim-acceptable; "
                "sim gate replaces rejected plans"
            )
        else:
            print("[PDDL] Sim override ENABLED — forward sim may reject ENHSP plans and run sim-search refinement")
        self.kb_force = []        # list of (force, v_meas, flight_angle_deg) tuples
        self.force_lr_model = None
        self.force_lr_r2 = None
        self._force_kb_max_samples = 40
        # ── BamBirds angle-aware velocity model (Fix #2) ──────────────────
        # Predicts launch |v| given (flight_angle, force) via
        # ``bambirds_shot_helper.angle_to_velocity(θ) × bambirds_scale × force``.
        # ``bambirds_scale`` is the px/s per (BamBirds-unit × force) mapping;
        # learned online from clean shots via ``recalculate_scaling_factor``.
        # Starts at None → falls back to the linear LR model until we have
        # at least one clean (angle, force, v) sample. See analysis in
        # `run_20260822_190314.log`: linear v = a·force + b hit R²≈0.74
        # because it ignored angle; L12 lost with perfect angle but |v|
        # short by 10 % — this fixes that failure mode.
        self.bambirds_scale = None
        self.bambirds_scale_n = 0

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

            if self.scenario_filter is None and getattr(self.generalization_protocol, 'scenario_filter', None):
                self.scenario_filter = self.generalization_protocol.scenario_filter
                print(f"[GENERALIZATION] Scenario filter from protocol: {self.scenario_filter}")
            
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
            "platform_collision": {
                "states": [],
                "variables": {
                    "v_x": {"value": [], "model": None},
                    "v_y": {"value": [], "model": None},
                    "y": {"value": [], "model": None},
                },
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

    # Minimum segment quality for trusting flight-physics learning.
    # Gravity needs the 2nd derivative of y, which requires visible arc curvature
    # (both a lower bound on frames AND on y-excursion). Initial launch velocity
    # is a 1st-derivative quantity that is well-estimated from any trajectory
    # with enough frames, even on nearly-flat shots.
    MIN_FLIGHT_SEGMENT_FRAMES = 40
    MIN_FLIGHT_Y_EXCURSION_PX = 40
    MIN_VELOCITY_SEGMENT_FRAMES = 15
    VELOCITY_SANITY_MIN = 120.0
    VELOCITY_SANITY_MAX = 250.0
    # Cap on |v_partial - current_v_bird| accepted from the flight-fit each shot.
    # Historic per-shot drift on stable physics is ±1–3; the old 35 threshold let a
    # spurious +14.6 jump through on train-23 high-arc shot (t02_00017, dial 80.5°)
    # in run_20260811_233043 — flight-fit gave |v|=196.4 while the direct launch
    # measurement was 179.9, and the corrupted v=196.4 then broke every test level
    # (25 and 30 flipped from wins to losses). 8 keeps 2.5× headroom over clean
    # drift while catching multi-sigma fit artifacts.
    VELOCITY_MAX_DELTA = 8.0
    # Cross-check: reject the flight-fit velocity when it disagrees with the same
    # shot's direct launch measurement by more than this many px/s. Long segments
    # (>150 frames) that reach into drag/near-collision regions produce biased
    # initial-velocity fits even when the launch itself was measured accurately
    # in the first 3–5 frames. 6 px/s is ~3× typical launch-measurement noise.
    VELOCITY_MAX_MEAS_DELTA = 6.0

    @classmethod
    def _flight_segment_quality_ok(cls, trajectory, min_frames=None, min_y_excursion=None):
        """
        Strict gate for GRAVITY learning: needs both enough frames and enough
        y-excursion so the 2nd derivative of y is reliable. Do NOT use this to
        gate initial-velocity updates — see `_flight_velocity_segment_quality_ok`.
        """
        min_frames = min_frames if min_frames is not None else cls.MIN_FLIGHT_SEGMENT_FRAMES
        min_y_excursion = min_y_excursion if min_y_excursion is not None else cls.MIN_FLIGHT_Y_EXCURSION_PX
        if trajectory is None or len(trajectory) < min_frames:
            n = len(trajectory) if trajectory is not None else 0
            return False, f"frames={n} < {min_frames}"
        traj = np.asarray(trajectory)
        y_excursion = float(np.max(traj[:, 1]) - np.min(traj[:, 1]))
        if y_excursion < min_y_excursion:
            return False, f"y_excursion={y_excursion:.1f}px < {min_y_excursion}px"
        return True, None

    @classmethod
    def _flight_velocity_segment_quality_ok(cls, trajectory, min_frames=None):
        """
        Looser gate for INITIAL-VELOCITY learning: only requires enough frames
        to fit a low-order polynomial and read off the first derivative at t=0.
        Nearly-flat trajectories (low y-excursion) are fine here because vx/vy
        at launch do not depend on later curvature.
        """
        min_frames = min_frames if min_frames is not None else cls.MIN_VELOCITY_SEGMENT_FRAMES
        if trajectory is None or len(trajectory) < min_frames:
            n = len(trajectory) if trajectory is not None else 0
            return False, f"frames={n} < {min_frames}"
        return True, None

    @classmethod
    def _velocity_update_acceptable(cls, v_candidate, current_v_bird, v_measured=None):
        """Reject velocity estimates that are physically implausible or jump too far.

        Three gates (all must pass):
          1. Sanity: within [VELOCITY_SANITY_MIN, VELOCITY_SANITY_MAX].
          2. Drift:  |v_candidate - current_v_bird| <= VELOCITY_MAX_DELTA.
          3. Cross-check (if ``v_measured`` provided): the flight-fit and the
             direct launch measurement from the same shot must agree within
             VELOCITY_MAX_MEAS_DELTA. Long-segment fits that reach into drag or
             pre-collision regions tend to overshoot vy; the launch measurement
             is a 3–5 frame direct read of the actual bird speed leaving the
             sling and is a much more reliable ground-truth pin.
        """
        if v_candidate < cls.VELOCITY_SANITY_MIN or v_candidate > cls.VELOCITY_SANITY_MAX:
            return False, f"|v|={v_candidate:.1f} outside [{cls.VELOCITY_SANITY_MIN}, {cls.VELOCITY_SANITY_MAX}]"
        if current_v_bird is not None and abs(v_candidate - current_v_bird) > cls.VELOCITY_MAX_DELTA:
            return False, f"delta={v_candidate - current_v_bird:+.1f} exceeds ±{cls.VELOCITY_MAX_DELTA}"
        if v_measured is not None and abs(v_candidate - v_measured) > cls.VELOCITY_MAX_MEAS_DELTA:
            return False, (
                f"fit vs measured mismatch: v_fit={v_candidate:.1f} vs "
                f"v_meas={v_measured:.1f} (Δ={v_candidate - v_measured:+.1f}, "
                f"cap ±{cls.VELOCITY_MAX_MEAS_DELTA})"
            )
        return True, None

    def _speed_at_force(
        self, v_full: float, force: float, angle_deg: float = None,
    ) -> float:
        """Launch speed hook shared by forward sim.

        Prefers the BamBirds angle-aware model
        (``angle_to_velocity(θ) × bambirds_scale × force``) once we have a
        calibrated ``bambirds_scale`` from at least one clean shot.
        Falls back to the linear LR model, and finally to the pa-twang
        linear formula when nothing is learned yet. ``angle_deg`` is the
        flight angle (``pddl_flight_angle_deg`` of the dial angle).
        """
        if self.bambirds_scale is not None and angle_deg is not None:
            from agents.pddl.pddl_files.bambirds_shot_helper import angle_to_velocity
            theta_rad = math.radians(float(angle_deg))
            speed_units = angle_to_velocity(theta_rad, self.bambirds_scale)
            v = speed_units * float(force)
            if 20.0 <= v <= 400.0:
                return v
        return launch_speed_at_force(v_full, force, force_lr_model=self.force_lr_model)

    def _log_execution_mapping(
        self,
        planned_angle: float,
        exec_angle: float,
        planned_force: float,
        sling,
        planned_launch: dict = None,
    ) -> None:
        """Log PDDL angle → release point contract (direct flight-angle execution)."""
        v_bird = self.world_model.hyperparams_values.get(Params.velocity, 180)
        pddl_flight = pddl_flight_angle_deg(exec_angle)
        v_portion = pddl_force_to_v_portion(planned_force)
        expected_v = self._speed_at_force(v_bird, planned_force, angle_deg=pddl_flight)
        domain_v = float(v_bird) * float(planned_force)
        sling_h = float(getattr(sling, "height", 0) or 0)
        print(
            f"\n[EXEC MAP] === PDDL angle → game execution (direct) ===\n"
            f"[EXEC MAP] PDDL angle: planned={planned_angle:.1f}° exec={exec_angle:.1f}° "
            f"(flight θ = angle − {ANGLE_BIAS_DEGREES:.0f}° → {pddl_flight:.1f}°)\n"
            f"[EXEC MAP] Release: flight θ={pddl_flight:.1f}° "
            f"v_portion={v_portion:.3f} pullback≈{pullback_pixels(sling_h, planned_force):.1f}px\n"
            f"[EXEC MAP] Speed: domain v_bird×force={domain_v:.1f} | "
            f"expected |v|={expected_v:.1f}"
            + (
                f" (learned LR, n={len(self.kb_force)})"
                if self.force_lr_model is not None else " (linear, no LR yet)"
            )
        )
        if planned_launch is not None:
            print(
                f"[EXEC MAP] Launch position (PDDL after pa-twang): "
                f"({planned_launch['launch_x']:.1f}, {planned_launch['launch_y']:.1f}) "
                f"v=({planned_launch['vx']:.1f}, {planned_launch['vy']:.1f})"
            )

    def _simulate_shot(
        self,
        problem_data: dict,
        angle: float,
        world_model_params: dict,
        force: float = 1.0,
        debug: bool = False,
    ) -> dict:
        return simulate_pddl_shot_plan(
            problem_data,
            angle,
            gravity=world_model_params["gravity"],
            force=force,
            debug=debug,
            platform_kb=self.kb.get("platform_collision"),
            speed_at_force=self._speed_at_force,
            force_lr_model=self.force_lr_model,
        )

    def _execution_slack_deg(self, problem_data: dict) -> float:
        """Launch-angle slack for robustness checks (wider on platform levels)."""
        if self._level_has_platforms(problem_data):
            return max(ANGLE_EXECUTION_SLACK_DEG, PLATFORM_EXECUTION_SLACK_DEG)
        return ANGLE_EXECUTION_SLACK_DEG

    def _simulate_planned_shot(
        self,
        problem_data: dict,
        planned_angle: float,
        world_model_params: dict,
        force: float = 1.0,
        debug: bool = False,
        probe_sim: dict = None,
    ):
        """
        Forward sim matching in-game execution: probe at planned dial, then sim at
        exec dial (planned + gap nudge when clearing platforms).

        _simulate_shot maps each dial through AngleCalibrator (pull clamp → measured θ).

        Returns (exec_sim, planned_angle, exec_angle, gap_nudge, probe_sim).
        """
        planned_angle = self._clamp_planner_dial(planned_angle, problem_data)
        force = round(float(force), 4)
        if probe_sim is None:
            probe_sim = self._simulate_shot(
                problem_data, planned_angle, world_model_params, force=force, debug=False,
            )
        gap_nudge = self._gap_clearance_execution_nudge(problem_data, probe_sim)
        exec_angle = max(self.min_deg, min(self.max_deg, planned_angle + gap_nudge))
        if gap_nudge != 0.0:
            nudge_sim = self._simulate_shot(
                problem_data, exec_angle, world_model_params, force=force, debug=False,
            )
            if nudge_sim.get("platform_collision") and not nudge_sim.get("pig_killed_in_sim"):
                gap_nudge = 0.0
                exec_angle = planned_angle
        if abs(exec_angle - planned_angle) < 1e-6 and not debug:
            exec_sim = probe_sim
        else:
            exec_sim = self._simulate_shot(
                problem_data, exec_angle, world_model_params, force=force, debug=debug,
            )
        return exec_sim, planned_angle, exec_angle, gap_nudge, probe_sim

    def _record_force_sample(
        self,
        force: float,
        v_meas: float,
        raw_pull_deg: float = None,
        game_pull_deg: float = None,
        flight_err_deg: float = None,
        flight_slack_deg: float = None,
        segment_ok: bool = None,
        flight_target_deg: float = None,
    ) -> None:
        """Append a (force, v_meas) pair and refit LR model when enough samples exist.

        Quality gates (all optional — skipped when signal is not provided, to keep
        callers that don't yet plumb this info working):
          - Clamped pull: shot did not execute at requested (force, angle) contract.
          - Angle execution error > slack: the trajectory doesn't reflect the
            intended launch — |v_meas| is contaminated by a bad launch angle.
          - Bad flight segment: too few frames to trust the velocity estimate.

        These mirror the gates already used by ``AngleCalibrator.record_shot`` and
        the v_bird EMA update path (which are `force >= 0.95` + unclamped + drift-safe).
        Without these, one clamped partial-force shot (e.g. force=0.85, pull=15.5°,
        angle-err=-11°) can wreck the LR fit — this was observed in the field:
        a great `v ≈ 190·force − 6, R²=0.93` (n=5) collapsed to
        `v ≈ 135·force + 48, R²=0.76` (n=6) after a single such sample, and the
        model never recovered over 40 shots.
        """
        force = round(float(force), 3)
        v_meas = float(v_meas)
        if force < FORCE_MIN or force > FORCE_MAX or v_meas <= 0:
            return
        if flight_err_deg is not None and flight_slack_deg is not None:
            if abs(float(flight_err_deg)) > float(flight_slack_deg):
                print(
                    f"[FORCE MODEL] skip sample force={force:.2f}: angle-err "
                    f"{flight_err_deg:+.1f}° > slack ±{flight_slack_deg:.0f}° "
                    f"(v_meas={v_meas:.1f} contaminated by mis-executed launch)"
                )
                return
        if segment_ok is False:
            print(
                f"[FORCE MODEL] skip sample force={force:.2f}: bad flight segment "
                f"(v_meas={v_meas:.1f} not trustworthy)"
            )
            return
        self.kb_force.append((force, v_meas, flight_target_deg))
        if len(self.kb_force) > self._force_kb_max_samples:
            self.kb_force = self.kb_force[-self._force_kb_max_samples:]
        self._fit_force_model()
        # BamBirds angle-aware scale calibration (Fix #2). Only clean shots
        # reach this point (angle-err and segment gates above). We track a
        # single px-per-(BamBirds-unit × force) scalar
        # ``bambirds_scale`` such that
        #     v_pred = angle_to_velocity(θ, scale=1) * bambirds_scale * force
        # Blended with EMA to smooth jitter. Extreme angles skipped because
        # the BamBirds polynomial is only calibrated in [~5°, ~83°].
        if (
            flight_target_deg is not None and force > 0
            and 5.0 <= float(flight_target_deg) <= 83.0
        ):
            from agents.pddl.pddl_files.bambirds_shot_helper import angle_to_velocity
            theta_rad = math.radians(float(flight_target_deg))
            base = angle_to_velocity(theta_rad, 1.0)  # unscaled polynomial output
            if base > 0:
                observed_scale = float(v_meas) / (base * float(force))
                prior = self.bambirds_scale
                if prior is None:
                    self.bambirds_scale = observed_scale
                else:
                    # Reject wild outliers (>25 % away from prior) to
                    # protect from bad segment fits that slipped past the
                    # angle-err gate. Comparable to BamBirds'
                    # ``recalculateScalingFactor`` guard rails.
                    if abs(observed_scale - prior) / prior <= 0.25:
                        alpha = 0.4 if self.bambirds_scale_n < 5 else 0.2
                        self.bambirds_scale = (
                            alpha * observed_scale + (1.0 - alpha) * prior
                        )
                self.bambirds_scale_n += 1
                v_check = angle_to_velocity(theta_rad, self.bambirds_scale) * float(force)
                print(
                    f"[BAMBIRDS SCALE] θ={float(flight_target_deg):.1f}° "
                    f"force={force:.3f} v_meas={v_meas:.1f} "
                    f"(obs_scale={observed_scale:.2f}) → "
                    f"scale={self.bambirds_scale:.2f} "
                    f"(n={self.bambirds_scale_n}, predicts |v|={v_check:.1f})"
                )

    def _fit_force_model(self) -> None:
        if len(self.kb_force) < self.force_learning_min_samples:
            return
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score
        import numpy as np

        forces_1d = np.array([row[0] for row in self.kb_force])
        forces_2d = forces_1d.reshape(-1, 1)
        velocities = np.array([row[1] for row in self.kb_force])
        model_lr = LinearRegression()
        model_lr.fit(forces_2d, velocities)
        self.force_lr_model = model_lr
        self.force_lr_r2 = r2_score(velocities, model_lr.predict(forces_2d))
        a = model_lr.coef_[0]
        b = model_lr.intercept_
        sign = "+" if b >= 0 else "-"
        print(
            f"[FORCE MODEL] v = {a:.2f}*force {sign} {abs(b):.2f} "
            f"(n={len(self.kb_force)}, R²={self.force_lr_r2:.3f})"
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

    def learn_platform_effects(
        self,
        platform_collisions,
        bird_observed_features,
        hit_frames=None,
        phase: str = "train",
        should_learn: bool = True,
    ):
        """
        Learn post-contact bird state after platform (hill) collisions.

        Mirrors learn_collision_effects but uses platform-specific filters suited to
        sliding hill contact rather than ground bounces.
        """
        FRAME_RATE = 0.02
        VELOCITY_FRAMES = 3
        POST_OFFSET = 2
        MIN_PLATFORM_SPEED = 25.0

        first_hit = hit_frames[0] if hit_frames else None

        for collision_index in platform_collisions:
            if first_hit is not None and collision_index >= first_hit:
                continue
            if collision_index < VELOCITY_FRAMES:
                continue
            if collision_index + POST_OFFSET + VELOCITY_FRAMES >= len(bird_observed_features):
                continue

            pre_features = bird_observed_features[collision_index]
            prev_features = bird_observed_features[collision_index - VELOCITY_FRAMES]
            post_start = collision_index + POST_OFFSET
            post_end = post_start + VELOCITY_FRAMES
            post_features_start = bird_observed_features[post_start]
            post_features_end = bird_observed_features[post_end]

            pre_state = pre_features.copy()
            pre_dt = VELOCITY_FRAMES * FRAME_RATE
            pre_state["v_x"] = (pre_features["x"] - prev_features["x"]) / pre_dt
            pre_state["v_y"] = (pre_features["y"] - prev_features["y"]) / pre_dt

            post_state = post_features_start.copy()
            post_dt = VELOCITY_FRAMES * FRAME_RATE
            post_state["v_x"] = (post_features_end["x"] - post_features_start["x"]) / post_dt
            post_state["v_y"] = (post_features_end["y"] - post_features_start["y"]) / post_dt

            pre_speed = math.hypot(pre_state["v_x"], pre_state["v_y"])
            if pre_speed < MIN_PLATFORM_SPEED:
                continue

            post_speed = math.hypot(post_state["v_x"], post_state["v_y"])
            print(
                f"[PLATFORM LEARN] frame={collision_index} pre_speed={pre_speed:.1f} "
                f"post_speed={post_speed:.1f} pre=({pre_state['x']:.1f},{pre_state['y']:.1f}) "
                f"post_v=({post_state['v_x']:.1f},{post_state['v_y']:.1f})"
            )

            if should_learn:
                update_model_effects(
                    "platform_collision", self.kb, pre_state, post_state, debug=False
                )
                n = len(self.kb["platform_collision"]["states"])
                print(f"[PLATFORM LEARN] KB samples: {n}")

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
        # Velocity learning uses the loose (frames-only) gate: launch |v| is a
        # 1st-derivative quantity and does not require a curved arc. Gravity
        # updates still use the strict gate inside learn_process_transitions.
        segment_ok, segment_reason = self._flight_velocity_segment_quality_ok(trajectory)
        if not segment_ok:
            print(f"  [VELOCITY] Segment quality FAIL: {segment_reason} — will not update v_bird from this shot")
        # Direct ground-truth launch speed from the FIRST 3 recorded frames.
        # Used as the "measured" pin in _velocity_update_acceptable so long-
        # segment flight-fits (e.g. 252-frame high-arc shots) can't overwrite
        # v_bird with a drag/pre-collision-biased value. On the train-23 win in
        # run_20260811_233043 the fit said 196.4 while these first frames said
        # 179.9 — the mismatch flipped test-25 and test-30 from wins to losses.
        v_measured_launch = None
        try:
            if trajectory is not None and len(trajectory) >= 3:
                dt_frame = 1.0 / 50.0  # SB ground-truth is 50 fps
                dx0 = float(trajectory[1][0]) - float(trajectory[0][0])
                dy0 = float(trajectory[1][1]) - float(trajectory[0][1])
                vx0 = dx0 / dt_frame
                vy0 = dy0 / dt_frame
                v_measured_launch = math.sqrt(vx0 * vx0 + vy0 * vy0)
        except Exception:
            v_measured_launch = None
        self.learned_transition_world_model = self._create_learned_transition_world_model(
            force_scale=force_scale,
            current_v_bird=current_v_bird,
            segment_quality_ok=segment_ok,
            v_measured=v_measured_launch,
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

        # Early exit: Science Birds "training" mode replays every level
        # attempt_limit_per_level times regardless of outcome. Once we have
        # recorded a level (WIN on attempt N, or ABANDONED after 8 LOSSes),
        # additional replays are pure overhead. Skip PDDL planning + GT-shot +
        # physics learning; just fire a cheap throwaway shot so SB burns the
        # attempt and eventually auto-advances to the next unique level. The
        # dedup guard in the finalize branch below is the second line of
        # defence in case we ever fall through.
        if self.use_generalization_protocol and self.generalization_protocol is not None:
            _early_level_path = self.get_current_level_path()
            if _early_level_path in self._finalized_level_paths:
                self._burn_finalized_attempt(sling, _early_level_path)
                return

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
            self._allow_gap_clearance_nudge = (current_phase != "test")
            
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
            self._allow_gap_clearance_nudge = (current_phase not in ("test", "validation"))
            
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
            self._allow_gap_clearance_nudge = True
            # Use PDDL planner
            print("[DEBUG] Using PDDL planner to select angle...")
            print("[DEBUG] Calling get_action_to_perform()...")
            actions = self.get_action_to_perform(self.world_model)
            planned_force, angle = _extract_force_angle(actions)
            print(f"\n[PDDL] Planner selected angle: {angle}°, force: {planned_force}")

        # 2. Execute shot — use PDDL flight angle and launch position directly.
        exec_angle = angle
        gap_nudge = 0.0
        last_planned = getattr(self, "_last_planned_angle", None)
        if (
            last_planned is not None
            and abs(float(last_planned) - float(angle)) < 1e-6
            and getattr(self, "_last_exec_angle", None) is not None
        ):
            exec_angle = float(self._last_exec_angle)
            gap_nudge = float(getattr(self, "_last_gap_nudge", 0.0) or 0.0)
        elif not self.disable_sim_override and not self.disable_forward_sim:
            last_sim = getattr(self, "_last_probe_sim", None) or getattr(self, "_last_sim_result", None) or {}
            gap_nudge = self._gap_clearance_execution_nudge(
                getattr(self, "_last_problem_data", {}), last_sim
            )
            if gap_nudge != 0.0:
                exec_angle = max(self.min_deg, min(self.max_deg, angle + gap_nudge))
        if gap_nudge != 0.0:
            print(
                f"[PDDL] Gap-clearance nudge: angle {angle:.1f}° → {exec_angle:.1f}° "
                f"({gap_nudge:+.1f}°) to avoid top-platform hits from launch overshoot"
            )
        ref = self.tp.get_reference_point(sling)
        ref_x, ref_y_pddl = float(ref.X), float(640 - ref.Y)
        v_bird = self.world_model.hyperparams_values.get(Params.velocity, 180)
        planned_launch = pddl_planned_launch_state(
            exec_angle, ref_x, ref_y_pddl, planned_force, v_bird,
            force_lr_model=self.force_lr_model,
            speed_at_force=self._speed_at_force,
        )
        pddl_flight_deg = planned_launch["flight_angle_deg"]
        release_point = pddl_shot_to_release_point(
            self.tp, sling, exec_angle, planned_force,
        )
        override = getattr(self, "_release_nudge_delta", None)
        if override is not None:
            from src.utils.point2D import Point2D
            dx, dy = override
            ox = int(release_point.X) + int(dx)
            oy = int(release_point.Y) + int(dy)
            print(
                f"[PIXEL-NUDGE] Nudging ENHSP release "
                f"({release_point.X}, {release_point.Y}) + ({dx:+d}, {dy:+d}) "
                f"→ ({ox}, {oy})"
            )
            release_point = Point2D(ox, oy)
            self._release_nudge_delta = None
        # Log the BamBirds correction being applied so we can see the actual
        # game-request angle vs the domain's flight angle.
        from agents.pddl.pddl_files.bambirds_shot_helper import actual_to_launch_deg
        _bam_request_deg = actual_to_launch_deg(pddl_flight_deg)
        _bam_correction_deg = _bam_request_deg - pddl_flight_deg
        print(
            f"[BAMBIRDS] flight_target={pddl_flight_deg:.2f}° "
            f"→ actualToLaunch → request from game={_bam_request_deg:.2f}° "
            f"(correction=+{_bam_correction_deg:.2f}°)"
        )
        # All three points expressed in PDDL Y-up world frame (H=640, origin
        # bottom-left) — the same frame extract_real_trajectory / GT segments
        # already use (see trajectory_parser.py:26 ``640 - entity.Y``). Comparing
        # across frames was masking a coord-system flip in earlier logs.
        sim_launch_x = float(planned_launch["launch_x"])
        sim_launch_y = float(planned_launch["launch_y"])
        release_x_pddl = float(release_point.X)
        release_y_pddl = 640.0 - float(release_point.Y)
        delta_release_vs_sim = math.hypot(
            release_x_pddl - sim_launch_x, release_y_pddl - sim_launch_y,
        )
        self._last_shot_diag = {
            "planned_angle": float(angle),
            "exec_angle": float(exec_angle),
            "flight_target": float(pddl_flight_deg),
            "launch_x": sim_launch_x,
            "launch_y": sim_launch_y,
            "release_x_pddl": release_x_pddl,
            "release_y_pddl": release_y_pddl,
            "release_x_screen": float(release_point.X),
            "release_y_screen": float(release_point.Y),
            "expected_speed": float(planned_launch["speed"]),
            "force": float(planned_force),
        }
        self._log_execution_mapping(angle, exec_angle, planned_force, sling, planned_launch)
        print(f"[DEBUG] Step 5: Executing shot — PDDL angle={exec_angle:.1f}° "
              f"(planned {angle:.1f}°), "
              f"flight θ={pddl_flight_deg:.1f}°, "
              f"launch=({sim_launch_x:.1f}, {sim_launch_y:.1f}), "
              f"force={planned_force:.3f}...")
        print(
            f"[SHOT DIAG PRE] angle={exec_angle:.2f}° (planned {angle:.2f}°)  "
            f"flight_target={pddl_flight_deg:.2f}°  force={planned_force:.3f}  "
            f"|v|={planned_launch['speed']:.1f}\n"
            f"[SHOT DIAG PRE]   [all coords in PDDL Y-up frame, H=640]\n"
            f"[SHOT DIAG PRE]   sim_launch (pa-twang)=({sim_launch_x:.1f}, {sim_launch_y:.1f})\n"
            f"[SHOT DIAG PRE]   game_release (SimpleTrajPlanner)=({release_x_pddl:.1f}, {release_y_pddl:.1f})\n"
            f"[SHOT DIAG PRE]   Δ(release, sim_launch)={delta_release_vs_sim:.1f}px  "
            f"[intra-PDDL disagreement — Day 2 will unify]\n"
            f"[SHOT DIAG PRE]   game_release (raw screen, sent to game)="
            f"({release_point.X}, {release_point.Y})"
        )
        print(f"[DEBUG] Release point (screen): ({release_point.X}, {release_point.Y})")
        print("[DEBUG] Calling shoot_and_record_ground_truth()...")
        batch_gt = self.ar.shoot_and_record_ground_truth(release_point.X, release_point.Y, 0, 0, 1, 0)
        gt_frame_count = len(batch_gt) if batch_gt else 0
        print(f"[DEBUG] Shot executed, got {gt_frame_count} ground truth frames")

        # Extract trajectory and events. Science Birds occasionally returns
        # zero ground-truth frames (transient game/network glitch, focus loss,
        # extremely short shot, etc.) — see the run_20260814_082859 crash at
        # train-21. In that case the downstream fallback (see the
        # "MULTI-BIRD ERROR" branch below) synthesizes a placeholder
        # trajectory so this shot is counted as a loss and normal retry logic
        # takes over. Log the glitch prominently so it's easy to spot in the
        # timeline and so we can distinguish it from ordinary miss shots.
        if gt_frame_count == 0:
            print(
                "[SHOT WARNING] Science Birds returned 0 ground-truth frames — "
                "treating this shot as a miss and falling through to the "
                "empty-trajectory fallback so the level can retry."
            )
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

        self._log_sim_vs_game_comparison(
            event_indexes_by_event,
            planned_angle=angle,
            planned_force=planned_force,
            phase="events",
        )
        
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
            if platform_collisions and hits:
                self.learn_platform_effects(
                    platform_collisions,
                    bird_observed_features,
                    hit_frames=hits,
                    phase=current_phase,
                    should_learn=should_learn,
                )
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
            grav_ok, grav_reason = self._flight_segment_quality_ok(first_segment)
            vel_ok, vel_reason = self._flight_velocity_segment_quality_ok(first_segment)
            y_exc = float(np.max(first_segment[:, 1]) - np.min(first_segment[:, 1]))
            grav_tag = 'OK' if grav_ok else f'BAD ({grav_reason})'
            vel_tag = 'OK' if vel_ok else f'BAD ({vel_reason})'
            print(
                f"[SEGMENT 0 DEBUG] y_excursion={y_exc:.1f}px, "
                f"gravity_quality={grav_tag}, velocity_quality={vel_tag}"
            )

        # Post-shot diagnostic: compare GT launch vs PDDL planned angle and position.
        diag = getattr(self, "_last_shot_diag", None)
        if diag is not None and len(first_segment) >= 3:
            try:
                n_vel = min(5, max(1, len(first_segment) - 2))
                n_pos = min(3, max(1, len(first_segment)))
                launch_est = estimate_launch_from_trajectory(
                    first_segment, n_vel=n_vel, n_pos=n_pos,
                )
                actual_deg = float(launch_est.get("theta_deg", float("nan")))
                actual_x = float(launch_est["release"][0])
                actual_y = float(launch_est["release"][1])
            except Exception:
                dt = 0.02
                dx = float(first_segment[1][0] - first_segment[0][0])
                dy = -float(first_segment[1][1] - first_segment[0][1])
                actual_deg = math.degrees(math.atan2(dy, dx)) if dx != 0.0 else float("nan")
                actual_x = float(first_segment[0][0])
                actual_y = float(first_segment[0][1])
            # All coords now consistently in PDDL Y-up frame (H=640, origin
            # bottom-left). GT (actual_x/y from estimate_launch_from_trajectory)
            # is already Y-up per trajectory_parser.py:26.
            angle_err = actual_deg - diag["flight_target"]
            sim_launch_x = diag["launch_x"]
            sim_launch_y = diag["launch_y"]
            rel_x = diag.get("release_x_pddl")
            rel_y = diag.get("release_y_pddl")
            pos_err_sim = math.hypot(
                actual_x - sim_launch_x, actual_y - sim_launch_y,
            )
            pos_err_release = (
                math.hypot(actual_x - rel_x, actual_y - rel_y)
                if rel_x is not None else float("nan")
            )
            # Intra-PDDL disagreement: sim-launch vs release. Day 2 target.
            pos_err_release_vs_sim = (
                math.hypot(rel_x - sim_launch_x, rel_y - sim_launch_y)
                if rel_x is not None else float("nan")
            )
            if abs(angle_err) > 3.0 and pos_err_sim > 8.0:
                verdict = "BOTH (angle + launch position)"
            elif abs(angle_err) > 3.0:
                verdict = "ANGLE (actual θ != PDDL flight θ)"
            elif pos_err_sim > 8.0:
                verdict = "POSITION (launch point != pa-twang position)"
            else:
                verdict = "OK (angle and launch position match PDDL)"
            print(
                f"[SHOT DIAG POST] [Y-up PDDL frame, H=640]\n"
                f"[SHOT DIAG POST]   actual_launch=({actual_x:.1f}, {actual_y:.1f}) "
                f"θ={actual_deg:.2f}°\n"
                f"[SHOT DIAG POST]   pddl_target  =({sim_launch_x:.1f}, {sim_launch_y:.1f}) "
                f"θ={diag['flight_target']:.2f}°\n"
                f"[SHOT DIAG POST]   Δθ={angle_err:+.2f}°  Δpos(sim→actual)={pos_err_sim:.1f}px  "
                f"force={diag['force']:.3f}"
            )
            if rel_x is not None:
                print(
                    f"[SHOT DIAG POST]   3-way: game_release=({rel_x:.1f}, {rel_y:.1f}) "
                    f"→ sim_launch=({sim_launch_x:.1f}, {sim_launch_y:.1f}) "
                    f"→ actual=({actual_x:.1f}, {actual_y:.1f})\n"
                    f"[SHOT DIAG POST]   Δ(release→sim)={pos_err_release_vs_sim:.1f}px "
                    f"[intra-PDDL, want→0]  "
                    f"Δ(release→actual)={pos_err_release:.1f}px "
                    f"[game snap-back, expected 5-15px]  "
                    f"Δ(sim→actual)={pos_err_sim:.1f}px "
                    f"[sim vs GT — the number that matters]"
                )
            print(f"[SHOT DIAG POST] verdict: {verdict}")

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
              f"θ_meas={launch['theta_deg']:.1f}° (exec angle={exec_angle:.1f}°, planned={angle:.1f}°, "
              f"pddl flight θ={pddl_flight_deg:.1f}°, "
              f"θ_err={launch['theta_deg'] - pddl_flight_deg:+.1f}°)")

        self._log_sim_vs_game_comparison(
            event_indexes_by_event,
            exec_angle=exec_angle,
            planned_angle=angle,
            planned_force=planned_force,
            pddl_flight_deg=pddl_flight_deg,
            launch=launch,
            phase="execution",
        )

        flight_err = launch['theta_deg'] - pddl_flight_deg
        slack = self._execution_slack_deg(getattr(self, '_last_problem_data', {}))
        if abs(flight_err) > slack:
            print(
                f"[EXEC MAP] ANGLE MISMATCH: measured flight={launch['theta_deg']:.1f}° vs "
                f"ENHSP target={pddl_flight_deg:.1f}° (err={flight_err:+.1f}°, slack=±{slack:.0f}°)"
            )
        v_expected = self._speed_at_force(
            self.world_model.hyperparams_values.get(Params.velocity, 180),
            planned_force, angle_deg=pddl_flight_deg,
        )
        v_err = launch['v_meas'] - v_expected
        if abs(v_err) > 8.0:
            print(
                f"[EXEC MAP] SPEED MISMATCH: measured |v|={launch['v_meas']:.1f} vs "
                f"expected={v_expected:.1f} (err={v_err:+.1f})"
            )

        force_sample_seg_ok, _ = self._flight_velocity_segment_quality_ok(first_segment)
        self._record_force_sample(
            planned_force,
            launch["v_meas"],
            flight_err_deg=flight_err,
            flight_slack_deg=slack,
            segment_ok=force_sample_seg_ok,
            flight_target_deg=pddl_flight_deg,
        )

        if should_learn:
            v_old = self.world_model.hyperparams_values[Params.velocity]
            seg_ok, seg_reason = self._flight_velocity_segment_quality_ok(first_segment)
            v_sane, v_sane_reason = self._velocity_update_acceptable(launch["v_meas"], v_old)
            # Update v_bird EMA from any near-full-force shot with enough frames
            # to give a stable launch-velocity read. Nearly-flat trajectories are
            # accepted because launch |v| is a 1st-derivative quantity — only the
            # sanity band + ±35 EMA delta guard against outliers.
            if planned_force >= 0.95 and seg_ok and v_sane:
                v_new = 0.85 * v_old + 0.15 * launch["v_meas"]
                self.world_model.hyperparams_values[Params.velocity] = v_new
                print(f"[LAUNCH ESTIMATE] v_bird: {v_old:.2f} -> {v_new:.2f} "
                      f"(EMA updated, v_meas={launch['v_meas']:.2f}, force={planned_force:.3f})")
            elif planned_force >= 0.95 and not seg_ok:
                print(f"[LAUNCH ESTIMATE] v_bird: {v_old:.2f} -> {v_old:.2f} "
                      f"(EMA skipped, bad segment: {seg_reason})")
            elif planned_force >= 0.95 and not v_sane:
                print(f"[LAUNCH ESTIMATE] v_bird: {v_old:.2f} -> {v_old:.2f} "
                      f"(EMA skipped, v_meas={launch['v_meas']:.2f} rejected: {v_sane_reason})")
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
        
        should_retry_level = False
        if level_finished:
            # Common metadata for both retry and finalize paths
            level_path = self.get_current_level_path()
            score = self.ar.get_current_score()
            scenario = self.phyq_metrics.extract_scenario_from_path(level_path)
            level_name = level_path.split('/')[-1] if '/' in level_path else level_path
            scenario_str = scenario if scenario else "unknown"

            # Determine phase: only training levels get retried.
            is_train_level = False
            if self.use_generalization_protocol and self.generalization_protocol is not None:
                phase_for_level, _ = self.generalization_protocol.get_phase_for_level(self.current_level)
                is_train_level = (phase_for_level == "train")

            # Compute the attempt number for THIS finished play of the level.
            # _current_level_retry_index counts prior LOSSes on this level.
            this_attempt_number = self._current_level_retry_index + 1

            # If the game is replaying a level we already recorded (WIN or ABANDONED),
            # never retry — we want to fall through to finalize so the loop resets
            # cleanly and lets the main loop advance. The record itself is skipped
            # below (see already_finalized guard).
            already_finalized = level_path in self._finalized_level_paths

            # Decide: retry same level, or finalize (record + advance)?
            should_retry_level = (
                (not game_result)
                and is_train_level
                and this_attempt_number < self.max_train_attempts
                and not already_finalized
            )
            abandoned = (
                (not game_result)
                and is_train_level
                and this_attempt_number >= self.max_train_attempts
                and not already_finalized
            )

            # Per-attempt sim-vs-game telemetry runs regardless of retry decision.
            self._log_sim_vs_game_comparison(
                event_indexes_by_event,
                exec_angle=exec_angle,
                planned_angle=angle,
                planned_force=planned_force,
                pddl_flight_deg=pddl_flight_deg,
                launch=launch,
                game_won=game_result,
                phase="outcome",
            )

            if should_retry_level:
                self._prepare_release_override_for_retry(
                    level_path, score, this_attempt_number,
                )
                # Bump retry counter and signal main loop to restart the same level.
                self._current_level_retry_index = this_attempt_number
                # Remember what THIS attempt shot so the next attempt's
                # diversifier can tell whether the raw ENHSP plan has already
                # been fired (see _prev_attempt_plan_source docstring).
                self._prev_attempt_plan_source = getattr(
                    self, "_last_plan_source", "planner"
                ) or "planner"
                self._retry_current_level = True

                print(f"[TRAIN-RETRY] Level {self.current_level} ({level_name}) LOST on "
                      f"attempt {this_attempt_number}/{self.max_train_attempts} - "
                      f"restarting same level.")
                print(f"[PHY-Q] Level: {level_name} | Scenario: {scenario_str} | "
                      f"Result: LOSS (retry {this_attempt_number}/{self.max_train_attempts}) | "
                      f"Birds used: {self._current_level_attempts}")
                time.sleep(0.5)
            else:
                # Finalize: record wins / game_results / phyq_metrics / CSV once per level.
                attempts_used = this_attempt_number  # win-attempt count, or max on abandon

                if already_finalized:
                    # Science Birds replayed a level we already recorded. Do NOT
                    # write another CSV row / bump game_results / re-record phyq_metrics.
                    # Just let the main loop advance by resetting the retry state.
                    print(f"[LOGGER] Level {level_name} (idx={self.current_level}) already "
                          f"recorded this run — skipping duplicate log "
                          f"(game replayed the level after finalize).")
                else:
                    self._finalized_level_paths.add(level_path)
                    if game_result:
                        self._record_winning_release(level_path)

                    self.wins.append(game_result)
                    self.games_played += 1

                    if self.games_played > self.start_counting_from_game:
                        self.game_results.append(
                            (self.current_level, "win" if game_result else "loss")
                        )

                    self.phyq_metrics.record(
                        level_path=level_path,
                        won=game_result,
                        attempts=self._current_level_attempts,
                        score=score,
                    )

                    if abandoned:
                        result_label = "ABANDONED"
                    elif game_result:
                        result_label = "WIN"
                    else:
                        result_label = "LOSS"

                    print(f"[PHY-Q] Level: {level_name} | Scenario: {scenario_str} | "
                          f"Result: {result_label} | Score: {score} | "
                          f"Birds used: {self._current_level_attempts} | "
                          f"Attempts used: {attempts_used}")

                    if self.comparison_csv:
                        self.comparison_csv.write_result(
                            level_path=level_path,
                            agent="PDDLAgent",
                            won=game_result,
                            mode=current_phase,
                            score=score,
                            plan_source=self._last_plan_source,
                            unsolvable=self._last_planner_unsolvable,
                            attempts_used=attempts_used,
                            abandoned=abandoned,
                        )

                # Reset per-level retry state for the next level.
                self._current_level_retry_index = 0
                self._prev_attempt_plan_source = "planner"
                self._current_level_fallback_history = []
                self._retry_current_level = False
                self._release_nudge_delta = None
                self._skip_retry_diversify = False
                self._retry_anchor_release = None
                self._near_miss_active = False

            # Reset per-attempt shot counter regardless of retry vs. finalize.
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
        print(f"Game results: {len([r for r in self.game_results if r[1] == 'win'])}/{len(self.game_results)} unique levels won "
              f"(one record per level; up to {self.max_train_attempts} retries collapse into a single row)")

        # Print collision model comparison after each validation shot
        if self.use_angle_protocol and self.angle_protocol is not None and current_phase == "validation":
            print("\n[INTERIM COMPARISON] Printing current model comparison after validation shot...")
            self.angle_protocol.print_collision_comparison(self.kb)
            
            # Run alpha validation if enabled
            if self.validate_alpha_on_validation:
                print("\n[ALPHA VALIDATION] Running alpha hyperparameter validation...")
                self.angle_protocol.validate_alpha()

        if not should_retry_level:
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
        self._record_force_sample(force, v_meas)

        # Fit model if enough samples
        if len(self.kb_force) >= self.force_learning_min_samples:
            from agents.pddl.optimizer import get_poly_rank
            forces_1d = np.array([row[0] for row in self.kb_force])
            velocities = np.array([row[1] for row in self.kb_force])

            # --- Model B: Best-degree Polynomial (get_poly_rank) ---
            rank, poly = get_poly_rank(forces_1d, velocities, max_rank=5, threshold=1.0)
            v_pred_poly = poly(forces_1d)
            r2_poly = r2_score(velocities, v_pred_poly)

            # --- Side-by-side comparison ---
            n = len(self.kb_force)
            a = self.force_lr_model.coef_[0]
            b = self.force_lr_model.intercept_
            sign = "+" if b >= 0 else "-"
            r2_lr = self.force_lr_r2
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
            print(", ".join(f"({row[0]:.2f},{row[1]:.1f})" for row in self.kb_force))
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

    def _burn_finalized_attempt(self, sling, level_path: str) -> None:
        """Fire a quick throwaway shot to consume SB's per-level attempt budget.

        Science Birds "training" mode with attempt_limit_per_level>1 replays
        every level (WIN or LOSS) until the budget is exhausted, then advances
        to the next unique level. Once we have recorded a level in
        self._finalized_level_paths (WIN on some attempt N, or ABANDONED after
        max_train_attempts LOSSes), further replays are pure overhead — but SB
        still requires a shot per attempt before it will advance.

        This method bypasses:
          * PDDL problem construction and ENHSP planning (~5-15 s),
          * shoot_and_record_ground_truth frame capture (~2-5 s),
          * post-shot trajectory / physics learning (~1 s),
        and just launches a weak, low-angle shot via fast_shoot so SB
        registers the attempt as done. If the shot happens to WIN, that's
        fine — the finalize dedup guard skips the duplicate record and the
        main-loop WON handler advances via load_next_available_level().
        """
        print(f"[BURN-ATTEMPT] Level {level_path} already finalized — firing "
              f"quick throwaway shot to burn SB attempt (skipping PDDL/GT/learning).")
        try:
            # Low angle + weak pull: bird typically lands short of any pigs,
            # producing a fast LOST outcome without accidental physics
            # perturbation. Values chosen to be safely inside the sling's
            # pull range so find_release_point_partial_power never returns
            # off-screen coordinates.
            release_point = self.tp.find_release_point_partial_power(
                sling,
                math.radians(15.0),
                v_portion=0.35,
            )
            self.ar.fast_shoot(release_point.X, release_point.Y, 0, 0, False)
        except Exception as e:
            # Fallback: click centre to at least dismiss any transient popup.
            # SB's own attempt counter will still tick eventually.
            print(f"[BURN-ATTEMPT] fast_shoot failed ({e}); falling back to click_in_centre.")
            try:
                self.ar.click_in_centre()
            except Exception as e2:
                print(f"[BURN-ATTEMPT] click_in_centre also failed ({e2}); sleeping 1s.")
                time.sleep(1.0)
        # Keep the per-attempt counter zeroed so it does not leak into the
        # next unique level's shot count. Retry index is left untouched
        # (this level is finalized; retries are meaningless).
        self._current_level_attempts = 0

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
        extend_platforms_in_problem_data(problem_data, debug=True)
        return problem_data, bird_objects, pigs_objects

    @staticmethod
    def _build_shot_actions(angle: float, force: float = 1.0) -> list:
        force = round(float(force), 4)
        if force < 1.0 - 1e-6:
            return [("set_force", force), ("shoot", angle)]
        return [("set_force", 1.0), ("shoot", angle)]

    # Retry-diversification schedule: (dial_delta_deg, force_delta).
    # Applied on retry attempts N>=1 to explore trajectory space around the
    # ENHSP baseline plan. Analysis of run_20260811_184720 losing levels:
    #   - platform-slide-into-blocks (7, 15) needs HIGHER dial (+6 to +12°)
    #   - block-wall stops bird (18, 20) needs MUCH HIGHER dial or lower force
    #   - high-arc miss (22, 23, 24) needs LOWER dial (-6 to -12°) or force<1
    # Schedule alternates direction with widening amplitude so all clusters
    # get a shot within 4 retries.
    _RETRY_NEAR_MISS_SCHEDULE = (
        (+1.0, 0.0),
        (-1.0, 0.0),
        (+2.0, 0.0),
        (-2.0, 0.0),
    )
    _RETRY_DIVERSIFY_SCHEDULE = (
        (+6.0, 0.0),
        (-6.0, 0.0),
        (+12.0, -0.15),
        (-12.0, -0.15),
    )

    # Plan sources that count as "the raw ENHSP plan has been fired" — retry
    # diversification only makes sense once such a shot has already been tried
    # (and lost). Anything else (ballistic_fallback, sim_search, sim_gate,
    # fallback, ...) means the base plan hasn't actually been played yet.
    _PLANNER_PLAN_SOURCES = frozenset({"planner", "planner_diversified"})

    @staticmethod
    def _level_template_key(level_path: str) -> str:
        """e.g. single_force_t05_00023 → single_force_t05 (shared geometry family)."""
        name = level_path.rstrip("/").split("/")[-1].replace(".xml", "")
        parts = name.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            return parts[0]
        return name

    def _last_release_screen(self) -> Optional[Tuple[int, int]]:
        diag = getattr(self, "_last_shot_diag", None)
        if not diag:
            return None
        return (
            int(round(diag["release_x_screen"])),
            int(round(diag["release_y_screen"])),
        )

    def _record_winning_release(self, level_path: str) -> None:
        rp = self._last_release_screen()
        if rp is None:
            return
        self._winning_release_by_level[level_path] = rp
        print(
            f"[PIXEL-NUDGE] Stored winning release {rp} for {level_path} "
            f"(template {self._level_template_key(level_path)})"
        )

    def _prepare_release_override_for_retry(
        self,
        level_path: str,
        score: int,
        failed_attempt_number: int,
    ) -> None:
        """
        Before large dial diversification, probe ±1px release neighbours.

        Only fires on partial-score near-misses (score > 0). Nudge is applied
        relative to the *current* ENHSP release (not a stale absolute anchor),
        so calibration drift between attempts does not pull the shot to the
        wrong pixel (run_20260829: 00015 nudged toward (90,341) while ENHSP
        release was (82,341)).
        """
        self._release_nudge_delta = None
        self._skip_retry_diversify = False
        self._near_miss_active = False

        if int(score) <= 0:
            return

        self._near_miss_active = True
        slot = max(0, int(failed_attempt_number) - 1)
        offsets = self._RELEASE_PIXEL_NUDGE_OFFSETS
        dx, dy = offsets[slot % len(offsets)]

        last_rp = self._last_release_screen()
        if last_rp is not None and self._retry_anchor_release is None:
            self._retry_anchor_release = last_rp

        if self._retry_anchor_release is None and last_rp is None:
            return

        anchor = self._retry_anchor_release or last_rp
        self._release_nudge_delta = (dx, dy)
        self._skip_retry_diversify = True
        print(
            f"[PIXEL-NUDGE] Retry attempt {failed_attempt_number + 1}: "
            f"partial score={score} near-miss (anchor {anchor}) — "
            f"will nudge ENHSP release by ({dx:+d}, {dy:+d}) "
            f"(skipping dial diversification this attempt)"
        )

    def _diversify_plan_for_retry(
        self,
        planned_angle: float,
        planned_force: float,
        problem_data: dict,
    ) -> tuple:
        """Perturb ENHSP plan on retries to explore alternative trajectories.

        Attempt 0 (first play) → unchanged. Attempts 1..N cycle through the
        ``_RETRY_DIVERSIFY_SCHEDULE`` so 5 attempts on a level fire 5 distinct
        (angle, force) pairs instead of repeating the same failing ENHSP shot.
        Perturbed dial is clamped to the executable range and force to
        [0.5, 1.0]. Test levels never retry so this is naturally train-only.

        Special case: if the previous attempt did NOT run the raw ENHSP plan
        (e.g. ENHSP returned "unsolvable" → ballistic_fallback fired), we skip
        diversification for one attempt so the freshly-produced ENHSP plan
        gets to fire un-perturbed. Otherwise we would burn the retry on a
        perturbed version of a plan we've never even seen shoot cleanly. This
        recovers run_20260812_114238 train-17 (t02_00023): attempt 1 was
        ballistic_fallback 27.5° (ENHSP unsolvable that attempt), attempt 2's
        ENHSP produced dial 68.0° (the historical winner), and the diversifier
        pushed it to 74.0° for a fifth-straight loss — with the "fresh plan"
        exemption attempt 2 would have fired 68.0° raw.
        """
        retry_index = int(getattr(self, "_current_level_retry_index", 0) or 0)
        if retry_index <= 0:
            return planned_angle, planned_force
        if getattr(self, "_release_nudge_delta", None) is not None:
            return planned_angle, planned_force
        if getattr(self, "_skip_retry_diversify", False):
            self._skip_retry_diversify = False
            return planned_angle, planned_force
        prev_source = str(getattr(self, "_prev_attempt_plan_source", "planner") or "planner")
        if prev_source not in self._PLANNER_PLAN_SOURCES:
            print(
                f"[RETRY-DIVERSIFY] Attempt {retry_index + 1}: previous attempt used "
                f"'{prev_source}' — firing fresh ENHSP plan un-perturbed "
                f"(dial {planned_angle:.1f}°, force {planned_force:.2f})."
            )
            return planned_angle, planned_force

        schedule = self._RETRY_DIVERSIFY_SCHEDULE
        slot = retry_index - 1
        label = "standard"
        if getattr(self, "_near_miss_active", False) and slot < len(self._RETRY_NEAR_MISS_SCHEDULE):
            schedule = self._RETRY_NEAR_MISS_SCHEDULE
            label = "near-miss micro"
        elif getattr(self, "_near_miss_active", False):
            slot = slot - len(self._RETRY_NEAR_MISS_SCHEDULE)

        dial_delta, force_delta = schedule[slot % len(schedule)]
        exec_min, exec_max = self._planner_executable_dial_bounds()
        dial_min = max(exec_min, float(self.min_deg))
        dial_max = min(exec_max, float(self.max_deg))
        new_angle = max(dial_min, min(dial_max, float(planned_angle) + dial_delta))
        new_force = max(0.5, min(1.0, float(planned_force) + force_delta))
        # No-op if clamping ate the entire perturbation.
        if abs(new_angle - planned_angle) < 1e-3 and abs(new_force - planned_force) < 1e-3:
            return planned_angle, planned_force
        print(
            f"[RETRY-DIVERSIFY] Attempt {retry_index + 1}: {label} perturbation "
            f"(dial {planned_angle:.1f}° → {new_angle:.1f}°, "
            f"force {planned_force:.2f} → {new_force:.2f}; "
            f"slot {slot + 1} = dial{dial_delta:+.1f}°, force{force_delta:+.2f})"
        )
        return new_angle, new_force

    def _try_retry_relaxed_sim_search(
        self,
        problem_data: dict,
        world_model_params: dict,
        hint_force: float = 1.0,
    ):
        """
        On train retries after near-miss or repeated diversified failures, run a
        relaxed angle×force grid that accepts platform-slide pig kills.
        """
        search_angle, search_force = self._search_fallback_by_sim(
            problem_data, world_model_params, hint_force=hint_force, quiet=True,
        )
        if search_angle is None:
            return None, None, None
        exec_sim = self._simulate_shot(
            problem_data, search_angle, world_model_params, force=search_force,
        )
        if not exec_sim.get("pig_killed_in_sim"):
            return None, None, None
        actions = self._build_shot_actions(search_angle, search_force)
        return actions, search_angle, search_force

    def _search_fallback_by_sim(
        self,
        problem_data: dict,
        world_model_params: dict,
        hint_force: float = 1.0,
        quiet: bool = False,
        angle_step: float = None,
    ):
        """
        Relaxed angle×force grid for ballistic fallback.
        Uses _sim_shot_score (allows platform-slide kills); no _sim_plan_is_acceptable gate.
        Coarse angle_step with early exit at the highest force that yields a pig kill.
        """
        angle_step = float(self.deg_step if angle_step is None else angle_step)
        best_angle = None
        best_force = None
        best_score = (-1, -1, -1, -1, -1.0, float("inf"))
        planner_max = self._planner_max_angle(problem_data)
        planner_min = self._planner_min_angle(problem_data)

        forces = list(iter_force_grid())
        hint_force = round(float(hint_force), 4)
        if hint_force in forces:
            forces.remove(hint_force)
            forces.insert(0, hint_force)

        for planned_force in forces:
            if self._plan_pick_timed_out():
                self._log_plan_pick_timeout_once("Fallback sim grid")
                break
            force_best_angle = None
            force_best_score = (-1, -1, -1, -1, -1.0, float("inf"))
            angle = planner_min
            while angle <= planner_max + 1e-6:
                if self._plan_pick_timed_out():
                    self._log_plan_pick_timeout_once("Fallback sim grid")
                    break
                sim = self._simulate_planned_shot(
                    problem_data, angle, world_model_params, force=planned_force,
                )[0]
                score = self._sim_shot_score(sim, angle, planned_force)
                if score > force_best_score:
                    force_best_score = score
                    force_best_angle = angle
                if score > best_score:
                    best_score = score
                    best_angle = angle
                    best_force = planned_force
                angle += angle_step

            if force_best_score[0] == 1 and force_best_angle is not None:
                refined = self._refine_angle_by_sim(
                    problem_data, world_model_params, force_best_angle, planned_force=planned_force,
                )
                if not quiet:
                    print(
                        f"[PDDL DEBUG] Fallback sim grid: pig-kill force={planned_force:.2f}, "
                        f"angle={refined:.1f}° (coarse {force_best_angle:.1f}°)"
                    )
                return refined, planned_force

        if best_angle is not None:
            if not quiet:
                print(
                    f"[PDDL DEBUG] Fallback sim grid: no kill; best score force={best_force:.2f}, "
                    f"angle={best_angle:.1f}°"
                )
            return best_angle, best_force
        if not quiet:
            print("[PDDL DEBUG] Fallback sim grid: no candidate (angle, force) pair")
        return None, None

    def _calculate_fallback_ballistic_angle(
        self,
        pigs_objects,
        bird_objects,
        world_model_params: dict,
        ref_angle_guess: float,
        problem_data: dict = None,
    ) -> float:
        """Legacy two-arc ballistic estimate at full force (used when sim grid finds nothing)."""
        problem_data = problem_data or {}
        planner_min = self._planner_min_angle(problem_data)
        planner_max = self._planner_max_angle(problem_data)

        def _clamp(angle):
            return self._clamp_planner_dial(angle, problem_data)

        if not pigs_objects:
            print("[PDDL DEBUG] No pig found, using default clamped dial")
            return _clamp(45.0)
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

                if problem_data and not self.disable_forward_sim:
                    best_arc_angle = None
                    best_arc_score = (-1, -1, -1, -1, -1.0, float("inf"))
                    for label, arc_angle in (("LOW", angle_low), ("HIGH", angle_high)):
                        exec_sim, planned, _, _, _ = self._simulate_planned_shot(
                            problem_data, arc_angle, world_model_params,
                        )
                        score = self._sim_shot_score(exec_sim, planned, 1.0)
                        print(
                            f"[PDDL DEBUG] {label} arc ({planned:.1f}°): "
                            f"platform_hit={exec_sim.get('platform_collision', False)}, "
                            f"pig_killed={exec_sim.get('pig_killed_in_sim', False)}"
                        )
                        if score > best_arc_score:
                            best_arc_score = score
                            best_arc_angle = arc_angle
                    if best_arc_angle is not None:
                        print(f"[PDDL DEBUG] Using best ballistic arc: {best_arc_angle:.1f}°")
                        return _clamp(best_arc_angle)
                else:
                    print(f"[PDDL DEBUG] Using low ballistic arc: {angle_low:.1f}°")
                    return _clamp(angle_low)

            fallback = ballistic_angle_to_target(
                launch_x, launch_y, pig['x_pig'], pig['y_pig'], v, g,
                min_angle=planner_min, max_angle=planner_max,
            )
            if fallback is not None:
                print(f"[PDDL DEBUG] Launch→pig ballistic angle: {fallback:.1f}°")
                return _clamp(fallback)
            direct_angle = np.degrees(np.arctan2(pig['y_pig'] - launch_y, pig['x_pig'] - launch_x))
            fallback_angle = _clamp(direct_angle)
            print(f"[PDDL DEBUG] Pig may be unreachable, using clamped direct angle: {fallback_angle:.1f}°")
            return fallback_angle
        except Exception as e:
            print(f"[PDDL DEBUG] Angle calculation failed ({e}), using clamped default")
            return _clamp(45.0)

    def _calculate_fallback_shot(
        self,
        pigs_objects,
        bird_objects,
        world_model_params: dict,
        ref_angle_guess: float,
        problem_data: dict = None,
    ) -> tuple[float, float]:
        """Return (angle, force) for planner failure fallback.

        Class A fix: when the current level has already burned a fallback
        attempt, override the default (sim-search / low-arc) result with a
        deterministic per-retry diversification schedule so we don't fire
        the same ~40° full-force shot 5x in a row on UNSOLVABLE levels
        (see run_20260821_065726 level 1 / t04_00033).
        """
        problem_data = problem_data or {}
        retry_idx = int(getattr(self, "_current_level_retry_index", 0) or 0)
        history: list[tuple[float, float]] = list(
            getattr(self, "_current_level_fallback_history", []) or []
        )

        diversified = None
        if retry_idx >= 1 and history:
            diversified = self._diversified_fallback_shot(
                pigs_objects, bird_objects, world_model_params,
                ref_angle_guess, problem_data, retry_idx, history,
            )

        if diversified is not None:
            angle, force = diversified
            print(
                f"[FALLBACK-DIVERSIFY] retry {retry_idx} — angle={angle:.1f}°, "
                f"force={force:.2f} (prior fallbacks: "
                f"{[(round(a,1), round(f,2)) for a,f in history]})"
            )
        elif problem_data and not self.disable_forward_sim:
            angle, force = self._search_fallback_by_sim(
                problem_data, world_model_params, quiet=False,
                angle_step=FALLBACK_GRID_DEG_STEP,
            )
            if angle is not None:
                angle = self._clamp_planner_dial(angle, problem_data)
                force = force if force is not None else 1.0
            else:
                angle = self._calculate_fallback_ballistic_angle(
                    pigs_objects, bird_objects, world_model_params,
                    ref_angle_guess, problem_data,
                )
                force = 1.0
        else:
            angle = self._calculate_fallback_ballistic_angle(
                pigs_objects, bird_objects, world_model_params,
                ref_angle_guess, problem_data,
            )
            force = 1.0

        # Record for future retries on this level.
        if not hasattr(self, "_current_level_fallback_history"):
            self._current_level_fallback_history = []
        self._current_level_fallback_history.append((float(angle), float(force)))
        return angle, force

    def _diversified_fallback_shot(
        self,
        pigs_objects,
        bird_objects,
        world_model_params: dict,
        ref_angle_guess: float,
        problem_data: dict,
        retry_idx: int,
        history: list[tuple[float, float]],
    ):
        """Deterministic diversification schedule for repeated fallbacks.

        Uses the ballistic low/high arc pair as the two "bands", then varies
        force. Returns (angle, force) or None if we can't compute the arcs.
        """
        if not pigs_objects or not bird_objects:
            return None
        pig = list(pigs_objects.values())[0]
        bird = list(bird_objects.values())[0]
        v = bird.get("v_bird", 180)
        g = world_model_params.get("gravity", 85)
        launch_x, launch_y = pddl_bird_position_after_pa_twang(
            bird["x_bird"], bird["y_bird"], ref_angle_guess
        )
        try:
            dx = pig["x_pig"] - launch_x
            if dx <= 0:
                return None
            term = v ** 4 - g * (g * dx ** 2 + 2 * (pig["y_pig"] - launch_y) * v ** 2)
            if term < 0:
                return None
            angle_low = float(np.degrees(np.arctan((v ** 2 - np.sqrt(term)) / (g * dx))))
            angle_high = float(np.degrees(np.arctan((v ** 2 + np.sqrt(term)) / (g * dx))))
        except Exception:
            return None

        prior_angles = [a for a, _ in history]
        # Was the last shot in the "low" band (below ~45°)?
        last_was_low = prior_angles[-1] < 45.0
        low_tried = any(a < 45.0 for a in prior_angles)
        high_tried = any(a >= 45.0 for a in prior_angles)

        # retry_idx meaning: 0 = first attempt (never diversified),
        # 1 = 2nd attempt (first retry), … 4 = 5th attempt.
        if retry_idx == 1:
            pick = ("HIGH" if last_was_low else "LOW", 1.00)
        elif retry_idx == 2:
            pick = ("LOW" if last_was_low else "HIGH", 0.85)
        elif retry_idx == 3:
            pick = ("HIGH" if last_was_low else "LOW", 0.85)
        else:  # retry 4 (final): weak rainbow
            pick = ("HIGH", 0.70)

        band, force = pick
        raw_angle = angle_high if band == "HIGH" else angle_low
        clamped = self._clamp_planner_dial(raw_angle, problem_data)

        # Guard against picking a near-duplicate of any prior fallback shot.
        # If we do, try the other band at the same force before giving up.
        def _too_close(a, f):
            return any(
                abs(a - pa) < 2.0 and abs(f - pf) < 0.05
                for pa, pf in history
            )

        if _too_close(clamped, force):
            other_raw = angle_low if band == "HIGH" else angle_high
            other = self._clamp_planner_dial(other_raw, problem_data)
            if not _too_close(other, force):
                return (float(other), float(force))
            # Fall back to reducing force further
            for f_try in (0.70, 0.60, 0.50):
                if not _too_close(clamped, f_try):
                    return (float(clamped), float(f_try))
            return None
        return (float(clamped), float(force))

    def _calculate_fallback_angle(
        self,
        pigs_objects,
        bird_objects,
        world_model_params: dict,
        ref_angle_guess: float,
        problem_data: dict = None,
    ) -> float:
        angle, _force = self._calculate_fallback_shot(
            pigs_objects, bird_objects, world_model_params, ref_angle_guess, problem_data,
        )
        return angle

    def _resolve_fallback_plan(
        self,
        vision,
        sling,
        agent_world_model,
        world_model_params: dict,
        pigs_objects,
        bird_objects,
        ref_guess: float,
        problem_data: dict,
    ) -> list:
        """Fallback: coarse angle×force grid, re-gather bird ref, refine only (no second grid)."""
        fallback_angle, fallback_force = self._calculate_fallback_shot(
            pigs_objects, bird_objects, world_model_params, ref_guess, problem_data,
        )
        problem_data, bird_objects, pigs_objects = self._gather_problem_data(
            vision, sling, agent_world_model, ref_angle_guess=fallback_angle,
        )
        if not self.disable_forward_sim:
            fallback_angle = self._clamp_planner_dial(
                self._refine_angle_by_sim(
                    problem_data, world_model_params, fallback_angle, planned_force=fallback_force,
                ),
                problem_data,
            )
        else:
            fallback_angle = self._clamp_planner_dial(fallback_angle, problem_data)
        self._last_plan_source = "fallback"
        actions = self._build_shot_actions(fallback_angle, fallback_force)
        self._finalize_plan_metadata(
            problem_data, fallback_angle, world_model_params, force=fallback_force,
        )
        return actions

    def _sim_shot_score(self, sim: dict, angle: float, force: float = 1.0):
        """Higher is better. Tie-break: prefer higher force, then lower dial angle."""
        return (
            1 if sim.get("pig_killed_in_sim") else 0,
            0 if sim.get("block_collision") else 1,
            0 if sim.get("platform_collision") and not sim.get("platform_slide_continued") else 1,
            0 if sim.get("uses_ground_collision") else 1,
            force,
            -angle,
        )

    def _sim_angle_score(self, sim: dict, angle: float, force: float = 1.0):
        return self._sim_shot_score(sim, angle, force)

    def _sim_plan_is_acceptable(self, sim: dict) -> bool:
        """Forward sim must predict a pig kill without blocking obstacles."""
        if not sim.get("pig_killed_in_sim"):
            return False
        if sim.get("block_collision"):
            return False
        if sim.get("platform_collision") and not sim.get("platform_slide_continued"):
            return False
        return True

    # Horizontal window around a pig within which a sim-predicted platform-hit
    # (with no pig kill) is treated as "bird got stuck on pig's hill" and
    # rejected by the narrow gate. 30 px covers:
    #   - #27 (t02_00055): sim hit x=239.8, pig x=256.5 → dist 16.7 (short)
    #   - #25 (t02_00070): sim hit x=240.5, pig x=228.0 → dist 12.5 (past)
    # Anything farther is a genuine miss into empty scenery; sim_search is
    # unlikely to salvage it, so keep the ENHSP plan and don't burn the search.
    PLATFORM_HIT_PIG_NEIGHBORHOOD_PX = 30.0

    @staticmethod
    def _plan_is_platform_hit_short(sim: dict, problem_data: dict) -> bool:
        """
        Sim predicts a platform-hit failure in the pig's neighborhood.

        Fires when:
          - sim reports platform_collision=True, AND
          - sim reports pig_killed_in_sim=False, AND
          - the platform_hit_pos is within
            ±``PLATFORM_HIT_PIG_NEIGHBORHOOD_PX`` of some pig horizontally
            (before, at, or slightly past that pig).

        Historically the gate also required ``platform_slide_continued=False``,
        but the sim's post-platform slide model is fictional on hill front-face
        hits — see run_20260812_102041 test-25 (t02_00070) and test-27
        (t02_00055): both lost with sim reporting `slide=True + no pig kill`
        while in the game the bird simply stopped on the hill. Removing that
        short-circuit recovers both losses without burning sim_search on
        genuine miss-into-empty-space plans (they fail the neighborhood check).

        Used as the reject filter in planner-only mode for BOTH train and
        test: all four current test losses in run_20260812_102041 land in the
        "sim predicts non-lethal platform contact on the pig's hill" class,
        and the sim's simulation of what happens *after* that contact is
        unreliable enough that it is safer to try sim_search than to trust
        the ENHSP plan.
        """
        if not sim.get("platform_collision"):
            return False
        if sim.get("pig_killed_in_sim"):
            return False
        pos = sim.get("platform_hit_pos")
        if not pos or pos[0] is None:
            return False
        pig_xs = [
            float(v.get("x_pig")) for k, v in problem_data.items()
            if k.startswith("pig_") and v.get("x_pig") is not None
        ]
        if not pig_xs:
            return False
        hit_x = float(pos[0])
        return any(
            abs(hit_x - pig_x) <= PDDLAgent.PLATFORM_HIT_PIG_NEIGHBORHOOD_PX
            for pig_x in pig_xs
        )

    # Class B fix: shelter/block-hit miss predicate.
    # Fires when the forward sim clearly predicts the shot fails because a
    # block absorbs the impact short of the pig — a class of "clean shot,
    # sim knows it misses, we fired anyway" losses (see run_20260821_065726
    # levels 00015/00036/00045/00046: every attempt logged pig_killed=False
    # + block_collision=True in the sim, planner-only kept the plan).
    @staticmethod
    def _plan_is_block_shelter_miss(sim: dict, problem_data: dict) -> bool:
        if sim.get("pig_killed_in_sim"):
            return False
        if not sim.get("block_collision"):
            return False
        # If there's a platform slide continuing toward the pig, the platform
        # branch (narrow gate) already handles / trusts it — don't double-fire.
        if sim.get("platform_slide_continued"):
            return False
        # Require a block was actually the trajectory terminator: the sim's
        # last event should be a block hit, not just an incidental brush.
        block_pos = sim.get("block_hit_pos") or sim.get("block_collision_pos")
        pig_xs = [
            float(v.get("x_pig")) for k, v in problem_data.items()
            if k.startswith("pig_") and v.get("x_pig") is not None
        ]
        if not pig_xs or not block_pos or block_pos[0] is None:
            # Fall back to structural check: block hit, no pig kill, no slide
            # → still a miss worth retrying via sim_search.
            return True
        hit_x = float(block_pos[0])
        # Only fire when the block hit is short of (or at) the pig — a hit
        # past the pig means the bird already flew over. Use the same
        # neighborhood window as the platform-hit-short predicate.
        return any(
            hit_x <= pig_x + PDDLAgent.PLATFORM_HIT_PIG_NEIGHBORHOOD_PX
            for pig_x in pig_xs
        )

    @staticmethod
    def _level_has_platforms(problem_data: dict) -> bool:
        return any(k.startswith("platform_") for k in problem_data)

    def _sim_plan_robust_to_execution_error(
        self,
        problem_data: dict,
        planned_angle: float,
        world_model_params: dict,
        planned_force: float,
        exec_sim: dict,
        exec_angle: float = None,
    ) -> bool:
        """
        Reject plans that only work at the exact executed dial.

        Validates pig kill (and platform clearance on gap levels) under measured
        launch-angle error at the dial actually sent to the slingshot.
        """
        if exec_angle is None:
            exec_angle = planned_angle
        slack = self._execution_slack_deg(problem_data)

        def _bad_platform_stop(sim: dict) -> bool:
            return sim.get("platform_collision") and not sim.get("platform_slide_continued")

        def _acceptable_at_dial(test_angle: float) -> bool:
            test_angle = max(self.min_deg, min(self.max_deg, test_angle))
            sim = self._simulate_shot(
                problem_data, test_angle, world_model_params, force=planned_force,
            )
            return self._sim_plan_is_acceptable(sim)

        intentional_slide = (
            exec_sim.get("platform_collision") and exec_sim.get("platform_slide_continued")
        )

        if intentional_slide:
            expected_plat = exec_sim.get("platform_hit_name")
            for delta in (-slack, slack):
                test_angle = max(self.min_deg, min(self.max_deg, exec_angle + delta))
                sim = self._simulate_shot(
                    problem_data, test_angle, world_model_params, force=planned_force,
                )
                if _bad_platform_stop(sim) and sim.get("platform_hit_name") != expected_plat:
                    return False
                if not sim.get("pig_killed_in_sim"):
                    return False
            return True

        test_dials = {exec_angle}
        for delta in (-slack, slack):
            test_dials.add(max(self.min_deg, min(self.max_deg, exec_angle + delta)))

        if self._level_has_platforms(problem_data):
            for delta in (-slack, slack):
                planned_test = max(self.min_deg, min(self.max_deg, planned_angle + delta))
                probe = self._simulate_shot(
                    problem_data, planned_test, world_model_params, force=planned_force,
                )
                nudge = self._gap_clearance_execution_nudge(problem_data, probe)
                test_dials.add(max(self.min_deg, min(self.max_deg, planned_test + nudge)))
            for test_angle in test_dials:
                sim = self._simulate_shot(
                    problem_data, test_angle, world_model_params, force=planned_force,
                )
                if _bad_platform_stop(sim):
                    return False
                if not self._sim_plan_is_acceptable(sim):
                    return False
            return True

        for test_angle in test_dials:
            if not _acceptable_at_dial(test_angle):
                return False
        return True

    def _gap_clearance_execution_nudge(self, problem_data: dict, sim: dict) -> float:
        """Dial-angle nudge for gap/over-flight plans on platform levels."""
        if not getattr(self, "_allow_gap_clearance_nudge", True):
            return 0.0
        if not self._level_has_platforms(problem_data):
            return 0.0
        if sim.get("platform_collision"):
            return 0.0
        return -GAP_CLEARANCE_ANGLE_NUDGE_DEG

    def _begin_plan_pick_search_budget(self) -> None:
        """Start wall-clock budget for plan-pick forward sim (not ENHSP)."""
        self._plan_pick_timeout_logged = False
        if self.plan_pick_timeout_sec > 0:
            self._plan_pick_deadline = time.monotonic() + self.plan_pick_timeout_sec
        else:
            self._plan_pick_deadline = None

    def _clear_plan_pick_search_budget(self) -> None:
        self._plan_pick_deadline = None
        self._plan_pick_timeout_logged = False

    def _plan_pick_timed_out(self) -> bool:
        if self._plan_pick_deadline is None:
            return False
        return time.monotonic() >= self._plan_pick_deadline

    def _log_plan_pick_timeout_once(self, context: str = "sim search") -> None:
        if self._plan_pick_timeout_logged:
            return
        self._plan_pick_timeout_logged = True
        print(
            f"[PLAN-PICK] {context} capped at {self.plan_pick_timeout_sec:.0f}s "
            f"(ENHSP not affected) — using best result so far"
        )

    def _refine_angle_by_sim(
        self,
        problem_data: dict,
        world_model_params: dict,
        center_angle: float,
        planned_force: float = 1.0,
        half_width: float = 1.0,
        step: float = 0.1,
    ):
        """Fine search around a coarse best angle at fixed force."""
        best_angle = center_angle
        best_score = (-1, -1, -1, -1, -1.0, float("inf"))

        angle = center_angle - half_width
        while angle <= center_angle + half_width + 1e-6:
            if self._plan_pick_timed_out():
                self._log_plan_pick_timeout_once("Local refine")
                break
            angle = max(self.min_deg, min(self.max_deg, angle))
            exec_sim, _, _, _, _ = self._simulate_planned_shot(
                problem_data, angle, world_model_params, force=planned_force,
            )
            score = self._sim_shot_score(exec_sim, angle, planned_force)
            if score > best_score:
                best_score = score
                best_angle = angle
            angle += step

        return best_angle

    def _search_shot_by_sim(
        self,
        problem_data: dict,
        world_model_params: dict,
        hint_force: float = 1.0,
        quiet: bool = False,
        angle_step: float = None,
        angle_min: float = None,
        angle_max: float = None,
        max_forces: int = None,
    ):
        """Grid-search dial angles × force; prefer pig kill with higher force."""
        best_angle = None
        best_force = None
        best_score = (-1, -1, -1, -1, -1.0, float("inf"))
        step = float(self.deg_step if angle_step is None else angle_step)
        planner_max = (
            float(angle_max) if angle_max is not None
            else self._planner_max_angle(problem_data)
        )
        planner_min = (
            float(angle_min) if angle_min is not None
            else self._planner_min_angle(problem_data)
        )

        forces = list(iter_force_grid())
        hint_force = round(float(hint_force), 4)
        if hint_force in forces:
            forces.remove(hint_force)
            forces.insert(0, hint_force)
        if max_forces is not None and max_forces > 0:
            forces = forces[:max_forces]

        for planned_force in forces:
            if self._plan_pick_timed_out():
                self._log_plan_pick_timeout_once("Sim grid search")
                break
            angle = planner_min
            while angle <= planner_max + 1e-6:
                if self._plan_pick_timed_out():
                    self._log_plan_pick_timeout_once("Sim grid search")
                    break
                exec_sim, planned, exec_angle, _, _ = self._simulate_planned_shot(
                    problem_data, angle, world_model_params, force=planned_force,
                )
                if not self._sim_plan_is_acceptable(exec_sim):
                    angle += step
                    continue
                if not self._sim_plan_robust_to_execution_error(
                    problem_data, planned, world_model_params, planned_force,
                    exec_sim, exec_angle,
                ):
                    angle += step
                    continue
                score = self._sim_shot_score(exec_sim, planned, planned_force)
                if score > best_score:
                    best_score = score
                    best_angle = angle
                    best_force = planned_force
                angle += step

        refine_step = 0.1 if step <= 0.5 else 0.25
        if best_score[0] == 1:
            refined = self._refine_angle_by_sim(
                problem_data, world_model_params, best_angle, planned_force=best_force,
                step=refine_step,
            )
            if not quiet:
                print(
                    f"[PDDL DEBUG] Sim search: pig-kill force={best_force:.2f}, "
                    f"angle={refined:.1f}° (coarse {best_angle:.1f}°)"
                )
            return refined, best_force
        if best_score[2] == 1 and best_angle is not None:
            if not quiet:
                print(
                    f"[PDDL DEBUG] Sim search: no kill; best no-platform "
                    f"force={best_force:.2f}, angle={best_angle:.1f}°"
                )
            return best_angle, best_force
        if not quiet:
            print("[PDDL DEBUG] Sim search: no acceptable (angle, force) pair")
        return None, None

    def _search_angle_by_sim(
        self,
        problem_data: dict,
        world_model_params: dict,
        planned_force: float = 1.0,
        quiet: bool = False,
    ):
        """Backward-compatible wrapper: angle-only search at fixed force."""
        angle, _ = self._search_shot_by_sim(
            problem_data, world_model_params, hint_force=planned_force, quiet=quiet
        )
        return angle

    def _log_sim_shadow_advisory(
        self,
        problem_data: dict,
        world_model_params: dict,
        planned_force: float,
        planner_angle: float = None,
        planner_sim: dict = None,
    ):
        """Log what sim search would pick vs the planner shot."""
        print("[SIM SHADOW] Evaluating sim search alternative...")

        if planner_angle is not None:
            if planner_sim is None:
                exec_sim, planned, _, _, probe = self._simulate_planned_shot(
                    problem_data, planner_angle, world_model_params, force=planned_force,
                )
            else:
                probe = getattr(self, "_last_probe_sim", None) or planner_sim
                exec_sim, planned, _, _, _ = self._simulate_planned_shot(
                    problem_data, planner_angle, world_model_params,
                    force=planned_force, probe_sim=probe,
                )
            planner_ok = self._sim_plan_is_acceptable(exec_sim)
            print(
                f"[SIM SHADOW] Planner shot: force={planned_force:.3f}, angle={planned:.1f}°, "
                f"acceptable={planner_ok}, pig_killed={exec_sim.get('pig_killed_in_sim')}, "
                f"platform={exec_sim.get('platform_collision')}, "
                f"block={exec_sim.get('block_collision')}, "
                f"ground={exec_sim.get('uses_ground_collision')}"
            )
        else:
            print("[SIM SHADOW] Planner: no solution (unsolvable / failed)")

        sim_angle, sim_force = self._search_shot_by_sim(
            problem_data, world_model_params, hint_force=planned_force, quiet=True
        )
        if sim_angle is None:
            print("[SIM SHADOW] Sim search: no acceptable (angle, force) pair")
            return

        exec_sim, _, _, _, _ = self._simulate_planned_shot(
            problem_data, sim_angle, world_model_params, force=sim_force,
        )
        sim_ok = self._sim_plan_is_acceptable(exec_sim)
        if planner_angle is not None:
            delta = sim_angle - planner_angle
            force_delta = sim_force - planned_force
            print(
                f"[SIM SHADOW] Sim search would use: force={sim_force:.2f}, angle={sim_angle:.1f}° "
                f"(Δforce={force_delta:+.2f}, Δangle={delta:+.1f}° vs planner), acceptable={sim_ok}, "
                f"pig_killed={exec_sim.get('pig_killed_in_sim')}, "
                f"platform={exec_sim.get('platform_collision')}, "
                f"block={exec_sim.get('block_collision')}, "
                f"ground={exec_sim.get('uses_ground_collision')}"
            )
            if abs(delta) < 0.05 and abs(force_delta) < 0.05 and planner_ok == sim_ok:
                print("[SIM SHADOW] Sim agrees with planner on this shot")
            elif not planner_ok and sim_ok:
                print("[SIM SHADOW] Sim would OVERRIDE planner (planner rejected, sim has kill)")
            elif planner_ok and not sim_ok:
                print("[SIM SHADOW] Planner passes sim gate; sim search found no better acceptable shot")
        else:
            print(
                f"[SIM SHADOW] Sim search would use: force={sim_force:.2f}, angle={sim_angle:.1f}°, "
                f"acceptable={sim_ok}, pig_killed={exec_sim.get('pig_killed_in_sim')}"
            )

    def _candidate_from_sim(
        self,
        problem_data: dict,
        world_model_params: dict,
        angle: float,
        force: float,
        source: str,
        sim: dict = None,
    ) -> dict:
        """Build a plan-pick candidate scored at the executed dial (planned + gap nudge)."""
        exec_sim, planned, exec_angle, gap_nudge, probe_sim = self._simulate_planned_shot(
            problem_data, angle, world_model_params, force=force, probe_sim=sim,
        )
        force = round(float(force), 4)
        score = self._sim_shot_score(exec_sim, planned, force)
        acceptable = self._sim_plan_is_acceptable(exec_sim)
        robust = self._sim_plan_robust_to_execution_error(
            problem_data, planned, world_model_params, force, exec_sim, exec_angle,
        )
        return {
            "source": source,
            "angle": planned,
            "exec_angle": exec_angle,
            "gap_nudge": gap_nudge,
            "force": force,
            "sim": exec_sim,
            "probe_sim": probe_sim,
            "score": score,
            "acceptable": acceptable,
            "robust": robust,
        }

    def _evaluate_shot_candidate(
        self,
        problem_data: dict,
        world_model_params: dict,
        angle: float,
        force: float,
        source: str,
    ) -> dict:
        """Score one (angle, force) pair with execution-aligned forward sim."""
        return self._candidate_from_sim(
            problem_data, world_model_params, angle, force, source, sim=None,
        )

    @staticmethod
    def _format_sim_shot_score(score: tuple) -> str:
        pig, no_block, plat_ok, no_ground, force, neg_angle = score
        return (
            f"pig={pig}, no_block={no_block}, plat_ok={plat_ok}, "
            f"no_ground={no_ground}, force={force:.2f}, angle={-neg_angle:.1f}°"
        )

    @staticmethod
    def _strong_plan_pick_candidates(candidates: list) -> list:
        """Acceptable, execution-robust sim shots that kill the pig."""
        return [
            c for c in candidates
            if c["acceptable"] and c["robust"] and c["score"][0] == 1
        ]

    def _run_plan_pick_full_search(
        self,
        problem_data: dict,
        world_model_params: dict,
        planner_angle: float,
        planner_force: float,
    ):
        """Escalating sim grid: near planner -> full coarse -> all forces."""
        if self._plan_pick_timed_out():
            return None, None
        if self.plan_pick_fast:
            near_min = max(
                self._planner_min_angle(problem_data),
                planner_angle - PLAN_PICK_NEAR_SEARCH_HALF_WIDTH_DEG,
            )
            near_max = min(
                self._planner_max_angle(problem_data),
                planner_angle + PLAN_PICK_NEAR_SEARCH_HALF_WIDTH_DEG,
            )
            search_angle, search_force = self._search_shot_by_sim(
                problem_data,
                world_model_params,
                hint_force=planner_force,
                quiet=True,
                angle_step=PLAN_PICK_FULL_SEARCH_DEG_STEP,
                angle_min=near_min,
                angle_max=near_max,
                max_forces=1,
            )
            if search_angle is not None:
                return search_angle, search_force

            search_angle, search_force = self._search_shot_by_sim(
                problem_data,
                world_model_params,
                hint_force=planner_force,
                quiet=True,
                angle_step=PLAN_PICK_FULL_SEARCH_DEG_STEP,
                max_forces=3,
            )
            if search_angle is not None:
                return search_angle, search_force

        return self._search_shot_by_sim(
            problem_data,
            world_model_params,
            hint_force=planner_force,
            quiet=True,
            angle_step=PLAN_PICK_FULL_SEARCH_DEG_STEP if self.plan_pick_fast else None,
        )

    def _select_best_sim_shot_plan(
        self,
        problem_data: dict,
        world_model_params: dict,
        planner_angle: float,
        planner_force: float,
        *,
        include_full_search: bool = True,
        planner_sim: dict = None,
    ):
        """
        Compare ENHSP plan against locally refined and full sim-search candidates.
        Returns (actions, angle, force, plan_source, sim) for the highest sim score.
        """
        self._begin_plan_pick_search_budget()
        try:
            return self._select_best_sim_shot_plan_inner(
                problem_data,
                world_model_params,
                planner_angle,
                planner_force,
                include_full_search=include_full_search,
                planner_sim=planner_sim,
            )
        finally:
            self._clear_plan_pick_search_budget()

    def _select_best_sim_shot_plan_inner(
        self,
        problem_data: dict,
        world_model_params: dict,
        planner_angle: float,
        planner_force: float,
        *,
        include_full_search: bool = True,
        planner_sim: dict = None,
    ):
        local_step = (
            PLAN_PICK_FAST_LOCAL_STEP_DEG if self.plan_pick_fast else PLAN_PICK_LOCAL_STEP_DEG
        )
        if planner_sim is not None:
            probe = getattr(self, "_last_probe_sim", None) or planner_sim
            candidates = [
                self._candidate_from_sim(
                    problem_data, world_model_params,
                    planner_angle, planner_force, "planner", probe,
                )
            ]
        else:
            candidates = [
                self._evaluate_shot_candidate(
                    problem_data, world_model_params, planner_angle, planner_force, "planner"
                )
            ]

        refined = self._refine_angle_by_sim(
            problem_data,
            world_model_params,
            planner_angle,
            planned_force=planner_force,
            half_width=PLAN_PICK_LOCAL_HALF_WIDTH_DEG,
            step=local_step,
        )
        if abs(refined - candidates[0]["angle"]) > 0.05:
            candidates.append(
                self._evaluate_shot_candidate(
                    problem_data, world_model_params, refined, planner_force, "sim_refine"
                )
            )

        need_full_search = include_full_search
        if need_full_search and self.plan_pick_fast and self._strong_plan_pick_candidates(candidates):
            need_full_search = False
            print(
                "[PLAN-PICK] Fast path: acceptable local candidate — skipping full sim grid"
            )

        if need_full_search and not self._plan_pick_timed_out():
            search_angle, search_force = self._run_plan_pick_full_search(
                problem_data, world_model_params, planner_angle, planner_force,
            )
            if search_angle is not None:
                candidates.append(
                    self._evaluate_shot_candidate(
                        problem_data,
                        world_model_params,
                        search_angle, search_force, "sim_search"
                    )
                )

        valid = [c for c in candidates if c["acceptable"] and c["robust"]]
        pool = valid if valid else [c for c in candidates if c["acceptable"]]
        if not pool:
            pool = candidates

        best = max(pool, key=lambda c: c["score"])

        if best["score"][0] == 0 and not self._plan_pick_timed_out():
            print(
                "[PLAN-PICK] No sim pig-kill among planner/refine/search — "
                "running relaxed sim grid..."
            )
            fb_angle, fb_force = self._search_fallback_by_sim(
                problem_data,
                world_model_params,
                hint_force=planner_force,
                quiet=True,
                angle_step=(
                    PLAN_PICK_FULL_SEARCH_DEG_STEP if self.plan_pick_fast else FALLBACK_GRID_DEG_STEP
                ),
            )
            if fb_angle is not None:
                fb_sim, _, fb_exec, _, _ = self._simulate_planned_shot(
                    problem_data, fb_angle, world_model_params, force=fb_force,
                )
                candidates.append(
                    self._candidate_from_sim(
                        problem_data, world_model_params,
                        fb_angle, fb_force, "sim_relaxed", fb_sim,
                    )
                )
                relaxed = candidates[-1]
                if relaxed["score"][0] == 1:
                    best = relaxed
                    pool = [relaxed]
                elif not relaxed["acceptable"]:
                    print(
                        "[PLAN-PICK] Relaxed grid found pig kill but shot blocked by "
                        "sim gate (block/platform) — keeping best scored candidate"
                    )
            else:
                print(
                    "[PLAN-PICK] Relaxed sim grid found no pig-kill angle — "
                    "keeping highest-scored candidate (may still miss in game)"
                )

        planner_c = candidates[0]

        print("\n[PLAN-PICK] === Sim plan comparison ===")
        for c in candidates:
            tag = " <-- chosen" if c is best else ""
            exec_note = ""
            if c.get("gap_nudge", 0) != 0:
                exec_note = f", exec={c.get('exec_angle', c['angle']):.1f}°"
            print(
                f"[PLAN-PICK] {c['source']:11s} angle={c['angle']:5.1f}°{exec_note} force={c['force']:.2f} "
                f"pig={c['sim'].get('pig_killed_in_sim')} "
                f"acceptable={c['acceptable']} robust={c['robust']} "
                f"score=({self._format_sim_shot_score(c['score'])}){tag}"
            )

        if best["source"] != "planner":
            print(
                f"[PLAN-PICK] Prefer {best['source']} over ENHSP "
                f"({planner_c['angle']:.1f}° -> {best['angle']:.1f}°, "
                f"force {planner_c['force']:.2f} -> {best['force']:.2f})"
            )
        else:
            print("[PLAN-PICK] Keeping ENHSP plan (best sim score among candidates)")

        actions = self._build_shot_actions(best["angle"], best["force"])
        source_map = {"sim_refine": "sim_refine", "sim_relaxed": "sim_relaxed"}
        plan_source = source_map.get(best["source"], best["source"])
        return actions, best["angle"], best["force"], plan_source, best["sim"]

    def _try_sim_search_plan(
        self,
        problem_data: dict,
        world_model_params: dict,
        planned_force: float = 1.0,
    ):
        """Run joint (angle, force) sim search; return (actions, angle, force) or (None, None, None)."""
        search_angle, search_force = self._search_shot_by_sim(
            problem_data, world_model_params, hint_force=planned_force
        )
        if search_angle is None:
            return None, None, None
        exec_sim, _, _, _, _ = self._simulate_planned_shot(
            problem_data, search_angle, world_model_params, force=search_force,
        )
        if not self._sim_plan_is_acceptable(exec_sim):
            return None, None, None
        if search_force < 1.0:
            actions = [("set_force", search_force), ("shoot", search_angle)]
        else:
            actions = [("set_force", 1.0), ("shoot", search_angle)]
        return actions, search_angle, search_force

    def _executable_dial_bounds(self, flight_slack: float = None) -> tuple:
        """PDDL angle range used for planning (direct execution, no pull table)."""
        return float(self.min_deg), float(self.max_deg)

    def _planner_executable_dial_bounds(self) -> tuple:
        """Executable dial range for ENHSP using tighter planning flight slack."""
        return self._executable_dial_bounds(flight_slack=PLANNER_FLIGHT_SLACK_DEG)

    def _planner_min_angle(self, problem_data: dict) -> float:
        """ENHSP dial floor.

        Historically clamped by the calibrator's executable-dial floor, which
        pushed min_angle up to ~24.5° on t04-style scenes and made pigs
        sheltered by a roof platform UNSAT (the solving arcs sit at 10-20°
        and rely on collision_platform → bounce_count=3 → explode_pig_from_bird
        firing when the bird passes within 50 px of the pig). Trusting the
        configured self.min_deg keeps those shelter-buster arcs available;
        the calibrator can still snap the executed dial post-planning.
        """
        return float(self.min_deg)

    def _current_level_phase(self):
        """Return current level phase ("train"/"test") if generalization is active, else None."""
        if not self.use_generalization_protocol or self.generalization_protocol is None:
            return None
        try:
            phase, _ = self.generalization_protocol.get_phase_for_level(self.current_level)
            return phase
        except Exception:
            return None

    def _planner_only_effective(self) -> bool:
        """
        Per-level planner-only gate.

        Both TRAIN and TEST trust ENHSP by default and only reject the narrow
        "platform-hit-short" failure inside the planner-only branch (see
        ``_plan_is_platform_hit_short``). run_20260810_193858 demonstrated that
        the previous broad TEST gate destroyed both baseline test wins (levels
        25/30): it rejected valid ENHSP plans, sim_search found nothing, and the
        fallback grid produced plans the calibrator couldn't execute accurately
        (+11° angle residual at low dials → hill-face impact on level 25). The
        narrow gate catches the pathological "stops in front of pig" pattern
        (level 1 style) without discarding otherwise-viable plans.

        Opt back into the old broad TEST-mode gate by passing
        ``sim_gate_test_only=False`` (CLI: ``--no-sim-gate-test-only``): TEST
        levels then fall through to the sim_search / fallback path.
        """
        if not self.planner_only:
            return False
        if self.disable_forward_sim:
            return True
        # Default (sim_gate_test_only=True): narrow gate on both train and test.
        # sim_gate_test_only=False: broad gate on test only (legacy behavior).
        if not getattr(self, "sim_gate_test_only", True):
            return self._current_level_phase() != "test"
        return True

    def _planner_max_angle(self, problem_data: dict) -> float:
        """ENHSP dial ceiling: executable dials only (calibrator + planning slack)."""
        _, exec_max = self._planner_executable_dial_bounds()
        return min(exec_max, float(self.max_deg))

    def _clamp_planner_dial(self, angle: float, problem_data: dict) -> float:
        """Snap angle to ENHSP grid within executable planner bounds."""
        planner_min = self._planner_min_angle(problem_data)
        planner_max = self._planner_max_angle(problem_data)
        return align_dial_to_planner_grid(
            angle, planner_max, self.deg_step, planner_min, planner_max,
        )

    @staticmethod
    def _enforce_planner_angle_on_actions(actions, clamped_angle: float):
        """Replace shoot angle in parsed actions with grid-clamped dial."""
        return [
            (action_type, clamped_angle if action_type == 'shoot' else value)
            for action_type, value in actions
        ]

    def _snapshot_advisory_case(self, planned_angle, exec_angle, planned_force,
                                exec_sim, problem_data):
        """
        Save problem.pddl / base_domain_modified.pddl / solution.pddl plus a
        JSON summary whenever the planner produces a plan the forward sim
        rejects (planner-only mode keeps it, so we normally lose evidence).
        Snapshots enable offline replay via
        ``agents.pddl.tools.enhsp_vs_sim_diff``.
        """
        try:
            import shutil as _sh
            import time as _tm
            import json as _json
            pddl_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 'pddl_files',
            )
            snap_dir = os.path.join(pddl_dir, '_advisory_snapshots')
            os.makedirs(snap_dir, exist_ok=True)
            ts = _tm.strftime('%Y%m%d_%H%M%S')
            for name in ('problem.pddl', 'base_domain_modified.pddl', 'solution.pddl'):
                src = os.path.join(pddl_dir, name)
                if os.path.exists(src):
                    _sh.copyfile(src, os.path.join(snap_dir, f'{ts}_{name}'))
            meta = {
                'timestamp': ts,
                'planned_dial': float(planned_angle),
                'exec_dial': float(exec_angle) if exec_angle is not None else None,
                'planned_force': float(planned_force),
                'sim_result': {
                    'pig_killed_in_sim': bool(exec_sim.get('pig_killed_in_sim')),
                    'platform_collision': bool(exec_sim.get('platform_collision')),
                    'block_collision': bool(exec_sim.get('block_collision')),
                    'ground_touches': int(exec_sim.get('ground_touches', 0) or 0),
                    'uses_ground_collision': bool(exec_sim.get('uses_ground_collision')),
                    'platform_hit_point': exec_sim.get('platform_hit_point'),
                },
                'problem_data_summary': {
                    k: {kk: (float(vv) if isinstance(vv, (int, float)) else vv)
                        for kk, vv in v.items()
                        if not kk.startswith('_') and isinstance(vv, (int, float, str, bool))}
                    for k, v in problem_data.items() if isinstance(v, dict)
                },
            }
            meta_path = os.path.join(snap_dir, f'{ts}_meta.json')
            with open(meta_path, 'w', encoding='utf-8') as fh:
                _json.dump(meta, fh, indent=2, default=str)
            print(f"[ADVISORY SNAPSHOT] Saved to {snap_dir} (prefix {ts})")
        except Exception as _e:
            print(f"[ADVISORY SNAPSHOT] failed: {_e}")

    def _run_enhsp_planner(self, problem_data: dict, agent_world_model: WorldModel):
        """Write problem/domain, run ENHSP. Returns (actions or None, planner_output).

        Two-pass strategy:
          A. Pessimistic platform hard-stop + raw block geometry (t01/t02 sweet
             spot; also historically solved t04 platform-slide levels like
             00008/00025).
          B. Shelter mode — slide/M5 platform + tightened block AABB / weakened
             block_life so ENHSP prefers collapse plans (needed for t04
             block-collapse levels like 00030/00068). Only entered when Pass A
             returns unsolvable.
        """
        from agents.pddl.pddl_files.pddl_parser import (
            apply_block_shelter_adjustments,
            block_sheltered_layout,
        )

        has_platforms = any(k.startswith("platform_") for k in problem_data)
        shelter_applicable = has_platforms and block_sheltered_layout(problem_data)

        actions, planner_output = self._run_enhsp_planner_single(
            problem_data,
            agent_world_model,
            pessimistic_platform=has_platforms,
            pass_label="A: pessimistic platform, raw blocks",
        )
        if actions is not None:
            return actions, planner_output

        unsolvable = "unsolvable" in (planner_output or "").lower()
        if not (unsolvable and shelter_applicable):
            return actions, planner_output

        print(
            "[PDDL DEBUG] Pass A unsolvable — retrying with shelter adjustments "
            "(slide/M5 platform + tightened block AABB)"
        )
        shelter_data = copy.deepcopy(problem_data)
        apply_block_shelter_adjustments(shelter_data, debug=True)
        return self._run_enhsp_planner_single(
            shelter_data,
            agent_world_model,
            pessimistic_platform=False,
            pass_label="B: slide/M5 platform, sheltered blocks",
        )

    def _run_enhsp_planner_single(
        self,
        problem_data: dict,
        agent_world_model: WorldModel,
        *,
        pessimistic_platform: bool,
        pass_label: str,
    ):
        """Single ENHSP invocation. Called by _run_enhsp_planner for each pass."""
        domain_path = 'base_domain_modified.pddl'
        planner_min = self._planner_min_angle(problem_data)
        planner_max = self._planner_max_angle(problem_data)
        planner_min = align_dial_to_planner_grid(
            planner_min, planner_max, self.deg_step, planner_min, planner_max,
        )
        exec_min, exec_max = self._planner_executable_dial_bounds()

        print(f"[PDDL DEBUG] ENHSP pass {pass_label}")
        print("[PDDL DEBUG] Writing problem file...")
        print(
            f"[PDDL DEBUG] Physics: gravity={agent_world_model.hyperparams_values[Params.gravity]:.2f}, "
            f"velocity={agent_world_model.hyperparams_values[Params.velocity]:.2f}"
        )
        print(
            f"[PDDL DEBUG] Executable dial range (calibrator, ±{PLANNER_FLIGHT_SLACK_DEG:.1f}° flight): "
            f"[{exec_min:.1f}°, {exec_max:.1f}°]"
        )
        print(
            f"[PDDL DEBUG] Using angle range: min={planner_min}°, max={planner_max}° "
            f"(start={planner_max}°)"
        )
        write_problem_file(
            'agents/pddl/pddl_files/problem.pddl',
            problem_data,
            planner_max,
            self.deg_step,
            agent_world_model,
            min_angle=planner_min,
            max_angle=planner_max,
        )
        print("[PDDL DEBUG] Problem file written")

        print("[PDDL DEBUG] Injecting base_domain.pddl (collision + learned physics)...")
        has_platforms = any(k.startswith("platform_") for k in problem_data)
        if has_platforms:
            mode = "pessimistic hard-stop" if pessimistic_platform else "slide/M5"
            print(f"[PDDL DEBUG] ENHSP platform model: {mode}")
        inject_domain_file(
            'agents/pddl/pddl_files/base_domain.pddl',
            agent_world_model,
            defer_ground_m5=has_platforms,
            planning_pessimistic_platform=pessimistic_platform,
        )
        print("[PDDL DEBUG] Domain file injected")
        print(f"[PDDL DEBUG] Using domain: {domain_path}")

        root_cwd = os.getcwd()
        pddl_dir = os.path.join(root_cwd, 'agents', 'pddl', 'pddl_files')

        print(f"[PDDL DEBUG] Current working directory: {root_cwd}")
        os.chdir(pddl_dir)
        print(f"[PDDL DEBUG] Changed to: {os.getcwd()}")

        try:
            # ENHSP invocation notes:
            #   -planner sat-hmrp: metric relaxed-plan heuristic (informative on
            #     numeric/process domains). Previous `sat-pt` was silently
            #     unrecognized and fell back to gbfs+hadd (blind on this domain).
            #   -delta: restored to ENHSP default (1.0) per git baseline. The
            #     -delta 2.0 override was skipping too much of the ballistic
            #     trajectory per step and contributing to false collisions in
            #     the AABB collision events.
            #   -Xmx4g + cleared _JAVA_OPTIONS: default 512M was thrashing GC.
            enhsp_cmd = [
                'java', '-Xmx4g', '-jar', 'enhsp-20.jar',
                '-o', domain_path,
                '-f', 'problem.pddl',
                '-sp', 'solution.pddl',
                '-planner', 'sat-hmrp',
            ]
            enhsp_env = os.environ.copy()
            enhsp_env['_JAVA_OPTIONS'] = ''
            print("[PDDL DEBUG] Running ENHSP planner (timeout=200s)...")
            print(f"[PDDL DEBUG] Command: {' '.join(enhsp_cmd)}")
            try:
                import hashlib as _hl
                for _p in (domain_path, 'problem.pddl'):
                    if os.path.exists(_p):
                        with open(_p, 'rb') as _fh:
                            _b = _fh.read()
                        _h = _hl.sha1(_b).hexdigest()[:12]
                        _lines = _b.count(b'\n') + 1
                        print(f"[ENHSP INPUT] {_p}: sha1={_h} bytes={len(_b)} lines={_lines}")
                    else:
                        print(f"[ENHSP INPUT] {_p}: MISSING")
            except Exception as _e:
                print(f"[ENHSP INPUT] hash failed: {_e}")
            if os.path.exists('solution.pddl'):
                os.remove('solution.pddl')
                print("[PDDL DEBUG] Deleted old solution file")

            result = subprocess.run(
                enhsp_cmd,
                env=enhsp_env,
                timeout=200,
                capture_output=True,
                text=True,
            )
            print("[PDDL DEBUG] ENHSP planner finished")
            planner_output = (result.stdout or '') + (result.stderr or '')

            if "unsolvable" in planner_output.lower():
                print("[PDDL DEBUG] *** PLANNER REPORTED: Problem unsolvable ***")
                print(f"[PDDL DEBUG] Planner output: {planner_output[:300]}")
                try:
                    import shutil as _sh
                    import time as _tm
                    _dbg_dir = os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        'pddl_files', '_unsolvable_snapshots',
                    )
                    os.makedirs(_dbg_dir, exist_ok=True)
                    _ts = _tm.strftime('%Y%m%d_%H%M%S')
                    for _p in (domain_path, 'problem.pddl'):
                        if os.path.exists(_p):
                            _dst = os.path.join(
                                _dbg_dir,
                                f"{_ts}_{os.path.basename(_p)}",
                            )
                            _sh.copyfile(_p, _dst)
                    print(f"[ENHSP INPUT] Snapshot saved to {_dbg_dir} (prefix {_ts})")
                except Exception as _e:
                    print(f"[ENHSP INPUT] snapshot failed: {_e}")
                return None, planner_output

            if not os.path.exists('solution.pddl'):
                print("[PDDL DEBUG] *** No solution file generated ***")
                print(f"[PDDL DEBUG] Planner stdout: {result.stdout[:500] if result.stdout else 'empty'}")
                print(f"[PDDL DEBUG] Planner stderr: {result.stderr[:500] if result.stderr else 'empty'}")
                return None, planner_output

            print("[PDDL DEBUG] Parsing solution...")
            actions = parse_solution_to_actions(
                'solution.pddl', planner_max, self.deg_step,
                force_min=FORCE_MIN, force_rate=FORCE_RATE,
            )
            planned_force_ext, planned_angle = _extract_force_angle(actions)
            planned_force_check = float(planned_force_ext) if planned_force_ext is not None else 1.0
            if planned_angle is not None:
                clamped = align_dial_to_planner_grid(
                    planned_angle, planner_max, self.deg_step, planner_min, planner_max,
                )
                if abs(clamped - planned_angle) > 1e-6:
                    print(
                        f"[PDDL DEBUG] Clamped planner angle "
                        f"{planned_angle:.1f}° → {clamped:.1f}° "
                        f"(grid [{planner_min:.1f}°, {planner_max:.1f}°])"
                    )
                    actions = self._enforce_planner_angle_on_actions(actions, clamped)
                    planned_angle = clamped
                if not self.disable_forward_sim:
                    planned_force, _ = _extract_force_angle(actions)
                    world_model_params = self._world_model_params(agent_world_model)
                    exec_sim, _, exec_angle, _, _ = self._simulate_planned_shot(
                        problem_data,
                        planned_angle,
                        world_model_params,
                        force=planned_force if planned_force is not None else 1.0,
                    )
                    if not self._sim_plan_is_acceptable(exec_sim):
                        msg = (
                            f"planned dial {planned_angle:.1f}°, exec dial {exec_angle:.1f}°, "
                            f"pig={exec_sim.get('pig_killed_in_sim')}, "
                            f"block={exec_sim.get('block_collision')}, "
                            f"ground={exec_sim.get('uses_ground_collision')}"
                        )
                        if self._planner_only_effective():
                            print(
                                f"[PDDL DEBUG] ADVISORY: ENHSP plan fails execution-aware sim — "
                                f"{msg} (planner-only keeps plan)"
                            )
                            self._snapshot_advisory_case(
                                planned_angle=planned_angle,
                                exec_angle=exec_angle,
                                planned_force=planned_force if planned_force is not None else 1.0,
                                exec_sim=exec_sim,
                                problem_data=problem_data,
                            )
                        else:
                            print(
                                f"[PDDL DEBUG] REJECTED: ENHSP plan fails execution-aware sim — "
                                f"{msg}"
                            )
                            return None, planner_output
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

    def _log_sim_vs_game_comparison(
        self,
        event_indexes_by_event: dict,
        *,
        exec_angle=None,
        planned_angle=None,
        planned_force=None,
        pddl_flight_deg=None,
        launch=None,
        game_won=None,
        phase="events",
    ) -> None:
        """Log where forward-sim predictions diverge from observed game behavior."""
        sim = getattr(self, "_last_sim_result", None) or {}
        if not sim and phase != "outcome":
            return

        hits = event_indexes_by_event.get("hit") or []
        platform_frames = event_indexes_by_event.get("platform_collision") or []
        block_frames = event_indexes_by_event.get("block_collision") or []
        ground_frames = event_indexes_by_event.get("ground_collision") or []

        game_hit = len(hits) > 0
        game_platform = len(platform_frames) > 0
        game_block = len(block_frames) > 0
        game_ground = len(ground_frames) > 0

        first_event_frame = None
        first_event_type = None
        for etype, frames in event_indexes_by_event.items():
            if frames:
                f0 = min(frames)
                if first_event_frame is None or f0 < first_event_frame:
                    first_event_frame = f0
                    first_event_type = etype

        sim_pig = bool(sim.get("pig_killed_in_sim"))
        sim_platform = bool(sim.get("platform_collision"))
        sim_block = bool(sim.get("block_collision"))
        sim_ground = bool(sim.get("uses_ground_collision"))
        sim_acceptable = self._sim_plan_is_acceptable(sim) if sim else None
        planner_only = getattr(self, "planner_only", True)

        mismatches = []

        if phase in ("events", "outcome"):
            if sim_pig != game_hit:
                mismatches.append(
                    f"pig: sim={'KILL' if sim_pig else 'MISS'} vs game={'HIT' if game_hit else 'NO_HIT'}"
                )
            if sim_platform != game_platform:
                mismatches.append(
                    f"platform: sim={'YES' if sim_platform else 'NO'} vs game={'YES' if game_platform else 'NO'}"
                    + (f" (first game frame {platform_frames[0]})" if platform_frames else "")
                )
            if sim_block != game_block:
                mismatches.append(
                    f"block: sim={'YES' if sim_block else 'NO'} vs game={'YES' if game_block else 'NO'}"
                    + (f" (first game frame {block_frames[0]})" if block_frames else "")
                )
            if sim_ground != game_ground:
                mismatches.append(
                    f"ground: sim={'YES' if sim_ground else 'NO'} vs game={'YES' if game_ground else 'NO'}"
                    + (f" (first game frame {ground_frames[0]})" if ground_frames else "")
                )
            if sim_pig and game_platform and not game_hit:
                mismatches.append(
                    f"sim direct-kill path but game platform first at frame {platform_frames[0]}"
                )
            if sim_platform and sim.get("platform_slide_continued") and game_hit and not sim_pig:
                mismatches.append("game pig hit via path sim did not predict as direct kill")

        if phase == "execution" and launch is not None and pddl_flight_deg is not None:
            theta_err = launch["theta_deg"] - pddl_flight_deg
            if abs(theta_err) > 2.0:
                mismatches.append(
                    f"launch angle: target flight={pddl_flight_deg:.1f}° vs measured={launch['theta_deg']:.1f}° "
                    f"(err={theta_err:+.1f}°)"
                )
            v_model = self.world_model.hyperparams_values.get(Params.velocity)
            if v_model is not None and planned_force is not None and planned_force >= 0.95:
                v_err = launch["v_meas"] - v_model
                if abs(v_err) > 8.0:
                    mismatches.append(
                        f"launch speed: model v={v_model:.1f} vs measured={launch['v_meas']:.1f} "
                        f"(err={v_err:+.1f})"
                    )

        if phase == "outcome" and game_won is not None:
            if sim_pig and not game_won:
                mismatches.append("sim predicted pig kill but level LOST")
            elif not sim_pig and game_won:
                mismatches.append("sim predicted miss but level WON (indirect/collapse path)")

        print(f"\n[SIM/REAL] === Comparison ({phase}) ===")
        print(
            f"[SIM/REAL] Plan: source={getattr(self, '_last_plan_source', '?')}, "
            f"angle={planned_angle if planned_angle is not None else getattr(self, '_last_planned_angle', '?')}°, "
            f"force={planned_force if planned_force is not None else getattr(self, '_last_planned_force', 1.0):.3f}, "
            f"planner_only={planner_only}, sim_gate={'SKIP' if planner_only else 'ON'}"
        )
        if sim:
            print(
                f"[SIM/REAL] Sim: pig={'KILL' if sim_pig else 'miss'}, "
                f"platform={'YES' if sim_platform else 'no'}"
                f"{'' if not sim_platform else (' slide_ok' if sim.get('platform_slide_continued') else ' STOP')}, "
                f"block={'YES' if sim_block else 'no'}, ground={'YES' if sim_ground else 'no'}, "
                f"acceptable={sim_acceptable}"
            )
            if sim.get("platform_hit_name"):
                print(f"[SIM/REAL] Sim platform target: {sim.get('platform_hit_name')}")
            if sim.get("block_hit_name"):
                print(f"[SIM/REAL] Sim block target: {sim.get('block_hit_name')}")
        if phase != "outcome":
            print(
                f"[SIM/REAL] Game events: hit={len(hits)}, platform={len(platform_frames)}, "
                f"block={len(block_frames)}, ground={len(ground_frames)}"
            )
            if first_event_frame is not None:
                print(f"[SIM/REAL] Game first event: {first_event_type} @ frame {first_event_frame}")
        if game_won is not None:
            print(f"[SIM/REAL] Level outcome: {'WIN' if game_won else 'LOSS'}")
        if launch is not None and phase == "execution":
            print(
                f"[SIM/REAL] Execution: dial={exec_angle:.1f}°, "
                f"θ_meas={launch['theta_deg']:.1f}°, |v|={launch['v_meas']:.1f}"
            )
        if mismatches:
            for msg in mismatches:
                print(f"[SIM/REAL] MISMATCH: {msg}")
        else:
            print("[SIM/REAL] No major sim/game divergence detected in this phase")

    def _empty_sim_result(self) -> dict:
        return {
            "uses_ground_collision": False,
            "pig_killed_in_sim": False,
            "ground_touches": 0,
            "platform_collision": False,
            "block_collision": False,
            "block_hit_name": None,
            "platform_slide_continued": False,
            "platform_hit_name": None,
            "platform_hit_pos": None,
            "trajectory": [],
        }

    def _finalize_plan_metadata(self, problem_data: dict, angle: float, world_model_params: dict,
                               force: float = 1.0) -> dict:
        """Store plan flags and optionally run direct-hit analysis for ground-bounce plans."""
        if self.disable_forward_sim:
            sim = self._empty_sim_result()
            probe_sim = sim
            exec_angle = angle
            gap_nudge = 0.0
            print("[PDDL DEBUG] Forward sim disabled — skipping plan simulation")
        else:
            sim, angle, exec_angle, gap_nudge, probe_sim = self._simulate_planned_shot(
                problem_data, angle, world_model_params, force=force, debug=True,
            )
        self._last_sim_result = sim
        self._last_probe_sim = probe_sim
        self._last_exec_angle = exec_angle
        self._last_gap_nudge = gap_nudge
        self._last_planned_angle = angle
        self._last_planned_force = force
        self._last_plan_uses_ground = sim['uses_ground_collision']
        self._last_problem_data = problem_data
        self._last_sim_trajectory = sim.get('trajectory', [])

        if not self.disable_forward_sim:
            exec_note = ""
            if gap_nudge != 0.0:
                exec_note = f", exec_dial={exec_angle:.1f}° (nudge {gap_nudge:+.1f}°)"
            print(
                f"[PDDL DEBUG] Plan sim: ground_touches={sim['ground_touches']}, "
                f"pig_killed_in_sim={sim['pig_killed_in_sim']}, "
                f"uses_ground_collision={sim['uses_ground_collision']}, "
                f"platform_collision={sim.get('platform_collision', False)}, "
                f"block_collision={sim.get('block_collision', False)}"
                f"{exec_note}"
            )
            if sim.get('block_collision'):
                print(f"[PDDL DEBUG] Block hit: {sim.get('block_hit_name')}")
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
        return sim

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
            if self._planner_only_effective():
                # Template 4 (and some t02 train levels): ENHSP often reports
                # "unsolvable" on block-heavy layouts even when forward sim finds
                # a kill. On train only, try sim_search before weak ballistic arc.
                phase = self._current_level_phase()
                if phase != "test" and not self.disable_forward_sim:
                    print(
                        "[PDDL DEBUG] Planner failed on train — trying sim search "
                        "before ballistic fallback..."
                    )
                    sim_actions, sim_angle, sim_force = self._try_sim_search_plan(
                        problem_data, world_model_params, planned_force=1.0,
                    )
                    if sim_actions is not None:
                        self._last_plan_source = "sim_search"
                        self._finalize_plan_metadata(
                            problem_data, sim_angle, world_model_params, force=sim_force,
                        )
                        print(
                            f"[PDDL DEBUG] ========== get_action_to_perform() "
                            f"RETURNING: {sim_actions} ==========\n"
                        )
                        return sim_actions
                print(
                    "[PDDL DEBUG] Planner failed — fallback grid disabled; "
                    "using ballistic angle"
                )
                fallback_angle = self._calculate_fallback_ballistic_angle(
                    pigs_objects, bird_objects, world_model_params, ref_guess, problem_data,
                )
                fallback_angle = self._clamp_planner_dial(fallback_angle, problem_data)
                fallback_force = 1.0
                self._last_plan_source = "ballistic_fallback"
                actions = self._build_shot_actions(fallback_angle, fallback_force)
                self._finalize_plan_metadata(
                    problem_data, fallback_angle, world_model_params, force=fallback_force,
                )
                print(f"[PDDL DEBUG] ========== get_action_to_perform() RETURNING: {actions} ==========\n")
                return actions
            if self.disable_sim_override:
                print("[PDDL DEBUG] Planner failed — running fallback grid...")
                actions = self._resolve_fallback_plan(
                    vision, sling, agent_world_model, world_model_params,
                    pigs_objects, bird_objects, ref_guess, problem_data,
                )
                print(f"[PDDL DEBUG] ========== get_action_to_perform() RETURNING: {actions} ==========\n")
                return actions
            print("[PDDL DEBUG] Planner failed — trying sim search...")

            sim_actions, sim_angle, sim_force = self._try_sim_search_plan(
                problem_data, world_model_params, planned_force=1.0
            )
            if sim_actions is not None:
                self._last_plan_source = "sim_search"
                self._finalize_plan_metadata(
                    problem_data, sim_angle, world_model_params, force=sim_force
                )
                print(f"[PDDL DEBUG] ========== get_action_to_perform() RETURNING: {sim_actions} ==========\n")
                return sim_actions

            print("[PDDL DEBUG] Sim search failed — falling back to angle×force grid...")
            actions = self._resolve_fallback_plan(
                vision, sling, agent_world_model, world_model_params,
                pigs_objects, bird_objects, ref_guess, problem_data,
            )
            print(f"[PDDL DEBUG] ========== get_action_to_perform() RETURNING: {actions} ==========\n")
            return actions
        
        self._last_plan_source = "planner"

        planned_force, angle = _extract_force_angle(actions)

        # Retry diversification: on retries, perturb the ENHSP plan so the 5
        # attempts explore different trajectories instead of firing the same
        # doomed shot 5 times. No-op on first attempt (retry_index == 0).
        new_angle, new_force = self._diversify_plan_for_retry(
            angle, planned_force, problem_data
        )
        if new_angle != angle or new_force != planned_force:
            actions = self._build_shot_actions(new_angle, new_force)
            angle = new_angle
            planned_force = new_force
            self._last_plan_source = "planner_diversified"

        retry_index = int(getattr(self, "_current_level_retry_index", 0) or 0)
        if (
            retry_index >= 2
            and not self.disable_forward_sim
            and self._current_level_phase() == "train"
            and getattr(self, "_release_nudge_delta", None) is None
        ):
            try_relaxed = (
                getattr(self, "_near_miss_active", False)
                or retry_index >= 3
            )
            if try_relaxed:
                relaxed = self._try_retry_relaxed_sim_search(
                    problem_data, world_model_params, hint_force=planned_force,
                )
                if relaxed[0] is not None:
                    actions, angle, planned_force = relaxed
                    self._last_plan_source = "sim_search_retry"
                    print(
                        f"[RETRY-SIM] Attempt {retry_index + 1}: using relaxed "
                        f"sim-search plan (dial {angle:.1f}°, force {planned_force:.2f})"
                    )

        sim = self._finalize_plan_metadata(problem_data, angle, world_model_params, force=planned_force)

        if not self.disable_forward_sim and self.prefer_sim_plan:
            actions, angle, planned_force, picked_source, sim = self._select_best_sim_shot_plan(
                problem_data, world_model_params, angle, planned_force,
                planner_sim=sim,
            )
            self._last_plan_source = picked_source
            self._finalize_plan_metadata(
                problem_data, angle, world_model_params, force=planned_force
            )
            print(f"[PDDL DEBUG] ========== get_action_to_perform() RETURNING: {actions} ==========\n")
            return actions

        if self._planner_only_effective():
            replaced_by_sim = False
            if not self.disable_forward_sim:
                sim_acceptable = self._sim_plan_is_acceptable(sim)
                if not sim_acceptable:
                    print(
                        "[SIM-GATE] Plan would be REJECTED if sim gate enabled: "
                        f"pig_killed={sim.get('pig_killed_in_sim')}, "
                        f"platform={sim.get('platform_collision')}, "
                        f"platform_slide={sim.get('platform_slide_continued')}, "
                        f"block={sim.get('block_collision')}, "
                        f"ground={sim.get('uses_ground_collision')}"
                    )
                # Narrow train-mode gate: only reject the "bird stops on hill face
                # before reaching any pig" failure (see run_20260810_064920 level 1:
                # 8 identical losses all pre-flagged as platform-hit-short). Every
                # other train plan still fires so the physics learner keeps growing.
                short_gate = self._plan_is_platform_hit_short(sim, problem_data)
                shelter_gate = (
                    not short_gate
                    and self._plan_is_block_shelter_miss(sim, problem_data)
                )
                # Broadened planner-only gate (added after run_20260822_104751):
                # any planner shot whose sim predicts platform_collision with no
                # pig kill is worth trying to replace via sim_search. The old
                # 30 px pig-neighborhood restriction let through the "planner
                # shoots at wrong platform / into hillside" failures we saw
                # (t04_00015/00008 lost with platform_hit far from pig, no
                # kill). If sim_search finds nothing we still keep the ENHSP
                # plan, so the downside vs. the narrow gate is only extra
                # sim_search cost.
                platform_miss_gate = (
                    not short_gate
                    and not shelter_gate
                    and bool(sim.get("platform_collision"))
                    and not bool(sim.get("pig_killed_in_sim"))
                    and not bool(sim.get("platform_slide_continued"))
                )
                if short_gate or shelter_gate or platform_miss_gate:
                    if short_gate:
                        reason = "platform-hit-short (bird stops in front of pig)"
                    elif shelter_gate:
                        reason = "block-shelter miss (block absorbs shot short of pig)"
                    else:
                        reason = "platform-miss (sim hits platform, no pig kill)"
                    print(
                        "[SIM-GATE] Train-mode narrow gate: sim predicts "
                        f"{reason} — invoking sim search for a viable alternative"
                    )
                    sim_actions, sim_angle, sim_force = self._try_sim_search_plan(
                        problem_data, world_model_params, planned_force=planned_force
                    )
                    if sim_actions is not None:
                        if short_gate:
                            self._last_plan_source = "sim_gate_short"
                        elif shelter_gate:
                            self._last_plan_source = "sim_gate_shelter"
                        else:
                            self._last_plan_source = "sim_gate_platform_miss"
                        actions = sim_actions
                        planned_force = sim_force
                        self._finalize_plan_metadata(
                            problem_data, sim_angle, world_model_params,
                            force=sim_force,
                        )
                        replaced_by_sim = True
                        print(
                            f"[PDDL DEBUG] Train-mode narrow-gate ({reason}): "
                            f"planner plan replaced by sim search "
                            f"(force={sim_force:.2f}, angle={sim_angle:.1f}°)"
                        )
                    else:
                        print(
                            f"[PDDL DEBUG] Train-mode narrow-gate ({reason}): "
                            "no viable sim alternative — keeping ENHSP plan"
                        )
            phase = self._current_level_phase()
            phase_tag = f" [{phase}]" if phase else ""
            if replaced_by_sim:
                print(
                    f"[PDDL DEBUG] Planner-only mode{phase_tag} — "
                    "replaced ENHSP plan via train narrow gate"
                )
            else:
                print(
                    f"[PDDL DEBUG] Planner-only mode{phase_tag} — "
                    "keeping ENHSP plan (sim gate skipped)"
                )
            print(f"[PDDL DEBUG] ========== get_action_to_perform() RETURNING: {actions} ==========\n")
            return actions
        phase = self._current_level_phase()
        if phase == "test" and self.planner_only:
            print(
                "[SIM-GATE] TEST-mode sim gate ACTIVE — pre-flagged bad plans will be "
                "rejected and replaced by sim search"
            )

        robust = self._sim_plan_robust_to_execution_error(
            problem_data,
            angle,
            world_model_params,
            planned_force,
            sim,
            getattr(self, "_last_exec_angle", angle),
        )
        if not robust:
            print(
                f"[PDDL DEBUG] Planner shot fragile under ±{self._execution_slack_deg(problem_data):.0f}° "
                f"execution error (gap/platform clearance) — sim search..."
            )
        if not self._sim_plan_is_acceptable(sim) or not robust:
            print(
                "[PDDL DEBUG] Planner shot rejected by forward sim "
                f"(pig_killed={sim.get('pig_killed_in_sim')}, "
                f"platform={sim.get('platform_collision')}, "
                f"block={sim.get('block_collision')}, robust={robust}) — sim search..."
            )
            self._log_sim_shadow_advisory(
                problem_data,
                world_model_params,
                planned_force,
                planner_angle=angle,
                planner_sim=sim,
            )

            sim_actions, sim_angle, sim_force = self._try_sim_search_plan(
                problem_data, world_model_params, planned_force=planned_force
            )
            if sim_actions is not None:
                self._last_plan_source = "sim_gate" if self.disable_sim_override else "sim_search"
                actions = sim_actions
                planned_force = sim_force
                self._finalize_plan_metadata(
                    problem_data, sim_angle, world_model_params, force=sim_force
                )
                if self.disable_sim_override:
                    print(
                        f"[PDDL DEBUG] Sim gate: rejected planner replaced by "
                        f"sim search (force={sim_force:.2f}, angle={sim_angle:.1f}°)"
                    )
            elif self.disable_sim_override:
                print(
                    f"[PDDL DEBUG] Sim gate: no acceptable sim alternative — "
                    f"keeping planner plan (force={planned_force:.3f}, angle={angle:.1f}°)"
                )
            else:
                print("[PDDL DEBUG] Sim search failed; relaxed fallback grid...")
                fallback_angle, fallback_force = self._calculate_fallback_shot(
                    pigs_objects, bird_objects, world_model_params, angle, problem_data
                )
                angle = fallback_angle
                planned_force = fallback_force
                self._last_plan_source = "fallback"
                if planned_force < 1.0 - 1e-6:
                    actions = [("set_force", planned_force), ("shoot", angle)]
                elif len(actions) > 1 and actions[0][0] == "set_force":
                    actions = [("set_force", planned_force), ("shoot", angle)]
                else:
                    actions = [("shoot", angle)]
                self._finalize_plan_metadata(
                    problem_data, angle, world_model_params, force=planned_force
                )

        if self.disable_sim_override and self._sim_plan_is_acceptable(sim):
            self._log_sim_shadow_advisory(
                problem_data,
                world_model_params,
                planned_force,
                planner_angle=angle,
                planner_sim=sim,
            )

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
            y_excursion = float(np.max(traj[:, 1]) - np.min(traj[:, 1]))
            traj_quality_ok, _ = self._flight_segment_quality_ok(traj)
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
                print(f"  [GRAVITY] Traj {traj_idx}: SKIP (frames={len(traj)}, y_excursion={y_excursion:.1f}px < thresholds {self.MIN_FLIGHT_SEGMENT_FRAMES}/{self.MIN_FLIGHT_Y_EXCURSION_PX})")
            
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
    
    def _create_learned_transition_world_model(
        self,
        force_scale: float = 1.0,
        current_v_bird: float = None,
        segment_quality_ok: bool = True,
        v_measured: float = None,
    ):
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
        v_measured : float, optional
            The direct launch |v| measured from the first few trajectory frames of
            THIS shot (before any drag or collision). When provided it acts as a
            ground-truth pin against long-segment flight-fit bias (see
            ``_velocity_update_acceptable`` gate 3).

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
                if not segment_quality_ok:
                    if current_v_bird is not None:
                        initial_values[Params.velocity] = current_v_bird
                        print(
                            f"[LEARNED MODEL] v_partial={v_partial:.2f}, planned_force={force_scale:.4f}, "
                            f"v_full=kept {current_v_bird:.2f} (bad segment quality)"
                        )
                    else:
                        initial_values[Params.velocity] = v_partial
                        print(
                            f"[LEARNED MODEL] v_partial={v_partial:.2f}, planned_force={force_scale:.4f}, "
                            f"v_full={v_partial:.2f} (bad segment, no prior v_bird)"
                        )
                else:
                    ok, reason = self._velocity_update_acceptable(
                        v_partial, current_v_bird, v_measured=v_measured
                    )
                    if ok:
                        initial_values[Params.velocity] = v_partial
                        print(
                            f"[LEARNED MODEL] v_partial={v_partial:.2f}, planned_force={force_scale:.4f}, "
                            f"v_full={v_partial:.2f} (full-force, direct)"
                        )
                    elif current_v_bird is not None:
                        initial_values[Params.velocity] = current_v_bird
                        print(
                            f"[LEARNED MODEL] v_partial={v_partial:.2f}, planned_force={force_scale:.4f}, "
                            f"v_full=kept {current_v_bird:.2f} (rejected: {reason})"
                        )
                    else:
                        initial_values[Params.velocity] = v_partial
                        print(
                            f"[LEARNED MODEL] v_partial={v_partial:.2f}, planned_force={force_scale:.4f}, "
                            f"v_full={v_partial:.2f} (full-force fallback, no prior v_bird)"
                        )
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

