import threading
import time
import src.demo.naive_agent_groundtruth as na
from agents import BirdsInBoots
from agents.quatzel.quatzel_agent import QuatzelAgent
from agents.owler.owler_agent import OwlerAgent
from agents.pddl.pddl_agent import PDDLAgent
from src.utils.tee_logger import start_tee_logging

import argparse
import json


class AgentThread(threading.Thread):
    def __init__(self, agent_configs, agent_ind: str, 
                 use_generalization_protocol: bool = False,
                 use_config_metadata: bool = False,
                 generalization_type: str = "local",
                 generalization_train_ratio: float = 0.8,
                 generalization_train_templates: list = None,
                 generalization_test_templates: list = None,
                 scenario_filter: str = None,
                 phyq_config_path: str = "ScienceBirds/win6.6/win/config_phyq_sample.xml",
                 levels_per_template: int = None,
                 visualize_pddl: bool = False,
                 visualize_pddl_board: bool = False,
                 debug_mag_comparison: bool = False,  # Disabled - use ENHSP planner
                 mag_comparison_angle: float = 60.0,
                 mag_comparison_start: float = 5.0,
                 mag_comparison_decrement: float = 0.1,
                 force_learning_mode: bool = False,
                 force_learning_min_samples: int = 5,
                 platform_aim_override: bool = False,
                 block_aim_override: bool = False,
                 planner_roll_block_only: bool = False,  # Disabled - use full problem with pig
                 disable_sim_override: bool = True,
                 planner_only: bool = True,
                 disable_forward_sim: bool = False,
                 prefer_sim_plan: bool = False,
                 plan_pick_fast: bool = True,
                 plan_pick_timeout_sec: float = 60.0,
                 sim_gate_test_only: bool = True,
                 disable_narrow_sim_gate: bool = False,
                 max_train_attempts: int = 5,
                 world_model_load: str = None,
                 world_model_save: str = None,
                 enhsp_ground_placeholder: bool = False,
                 enhsp_timeout_sec: float = 120.0,
                 event_determinism_mode: bool = False,
                 event_determinism_repeats: int = 5,
                 event_determinism_save: str = "event_determinism_results.json",
                 angle_residual_correction: bool = True,
                 angle_residual_calibration_path: str = "angle_residual_calibration.json",
                 angle_residual_bootstrap_path: str = "event_determinism_results.json",
                 angle_residual_live_update: bool = True,
                 test_force_clamp_to_full: bool = True,
                 test_force_clamp_min: float = 1.0,
                 override_angle: float = None,
                 override_force: float = None,
                 ):
        """
        Constructor function
        Parameters
        ----------
        agent_configs :
        agent_ind : agents ID , typically a number
        use_generalization_protocol : bool
            Enable Phy-Q generalization evaluation protocol
        use_config_metadata : bool
            Auto-load settings from config's .meta.json file
        generalization_type : str
            "local" for 80/20 within-template split, "broad" for across-template split
        generalization_train_ratio : float
            For local: train/test ratio (default 0.8 = 80% train, 20% test)
        generalization_train_templates : list
            For broad: list of template numbers to train on (e.g., [1,2,3,4])
        generalization_test_templates : list
            For broad: list of template numbers to test on (e.g., [5,6])
        scenario_filter : str
            Only include levels from this scenario (e.g., "single_force")
        phyq_config_path : str
            Path to Phy-Q config XML file
        levels_per_template : int, optional
            Limit the number of levels per template
        visualize_pddl : bool
            Extra PDDL plots when a level is lost (default: disabled)
        visualize_pddl_board : bool
            Show/save board geometry on each ENHSP problem write (default: disabled)
        debug_mag_comparison : bool
            Enable mag comparison debug mode
        mag_comparison_angle : float
            Fixed angle for mag comparison test (degrees)
        mag_comparison_start : float
            Starting mag multiplier value
        mag_comparison_decrement : float
            Mag decrement per iteration
        """
        self.agent_ind = agent_ind
        self.agent_configs = agent_configs
        self.use_generalization_protocol = use_generalization_protocol
        self.use_config_metadata = use_config_metadata
        self.generalization_type = generalization_type
        self.generalization_train_ratio = generalization_train_ratio
        self.generalization_train_templates = generalization_train_templates
        self.generalization_test_templates = generalization_test_templates
        self.scenario_filter = scenario_filter
        self.phyq_config_path = phyq_config_path
        self.levels_per_template = levels_per_template
        self.visualize_pddl = visualize_pddl
        self.visualize_pddl_board = visualize_pddl_board
        self.debug_mag_comparison = debug_mag_comparison
        self.mag_comparison_angle = mag_comparison_angle
        self.mag_comparison_start = mag_comparison_start
        self.mag_comparison_decrement = mag_comparison_decrement
        self.force_learning_mode = force_learning_mode
        self.force_learning_min_samples = force_learning_min_samples
        self.platform_aim_override = platform_aim_override
        self.block_aim_override = block_aim_override
        self.planner_roll_block_only = planner_roll_block_only
        self.disable_sim_override = disable_sim_override
        self.planner_only = planner_only
        self.disable_forward_sim = disable_forward_sim
        self.prefer_sim_plan = prefer_sim_plan
        self.plan_pick_fast = plan_pick_fast
        self.plan_pick_timeout_sec = plan_pick_timeout_sec
        self.sim_gate_test_only = sim_gate_test_only
        self.disable_narrow_sim_gate = disable_narrow_sim_gate
        self.max_train_attempts = max_train_attempts
        self.world_model_load = world_model_load
        self.world_model_save = world_model_save
        self.enhsp_ground_placeholder = enhsp_ground_placeholder
        self.enhsp_timeout_sec = enhsp_timeout_sec
        self.event_determinism_mode = event_determinism_mode
        self.event_determinism_repeats = event_determinism_repeats
        self.event_determinism_save = event_determinism_save
        self.angle_residual_correction = angle_residual_correction
        self.angle_residual_calibration_path = angle_residual_calibration_path
        self.angle_residual_bootstrap_path = angle_residual_bootstrap_path
        self.angle_residual_live_update = angle_residual_live_update
        self.test_force_clamp_to_full = test_force_clamp_to_full
        self.test_force_clamp_min = test_force_clamp_min
        self.override_angle = override_angle
        self.override_force = override_force
        threading.Thread.__init__(self)

    def run(self):
        """
        Runs a single agents
        Returns
        -------

        """
        # agent = na.ClientNaiveAgent(self.agent_ind,self.agent_configs)
        # agent = QuatzelAgent(self.agent_ind, self.agent_configs)
        # agent = OwlerAgent(self.agent_ind, self.agent_configs)
        agent = PDDLAgent(
            self.agent_ind, 
            self.agent_configs,
            use_angle_protocol=not self.use_generalization_protocol and not self.debug_mag_comparison,  # Disable if using generalization or mag debug
            validate_alpha_on_validation=False,
            debug_collision=False,
            phyq_config_path=self.phyq_config_path,
            # Phy-Q Generalization Protocol
            use_generalization_protocol=self.use_generalization_protocol,
            use_config_metadata=self.use_config_metadata,
            generalization_type=self.generalization_type,
            generalization_train_ratio=self.generalization_train_ratio,
            generalization_train_templates=self.generalization_train_templates,
            generalization_test_templates=self.generalization_test_templates,
            scenario_filter=self.scenario_filter,
            levels_per_template=self.levels_per_template,
            visualize_pddl_input=self.visualize_pddl,
            visualize_pddl_board=self.visualize_pddl_board,
            # Debug mag comparison mode
            debug_mag_comparison=self.debug_mag_comparison,
            mag_comparison_angle=self.mag_comparison_angle,
            mag_comparison_start=self.mag_comparison_start,
            mag_comparison_decrement=self.mag_comparison_decrement,
            # Force -> velocity learning mode
            force_learning_mode=self.force_learning_mode,
            force_learning_min_samples=self.force_learning_min_samples,
            platform_aim_override=self.platform_aim_override,
            block_aim_override=self.block_aim_override,
            planner_roll_block_only=self.planner_roll_block_only,
            disable_sim_override=self.disable_sim_override,
            planner_only=self.planner_only,
            disable_forward_sim=self.disable_forward_sim,
            prefer_sim_plan=self.prefer_sim_plan,
            plan_pick_fast=self.plan_pick_fast,
            plan_pick_timeout_sec=self.plan_pick_timeout_sec,
            sim_gate_test_only=self.sim_gate_test_only,
            disable_narrow_sim_gate=self.disable_narrow_sim_gate,
            max_train_attempts=self.max_train_attempts,
            world_model_load=self.world_model_load,
            world_model_save=self.world_model_save,
            enhsp_ground_placeholder=self.enhsp_ground_placeholder,
            enhsp_timeout_sec=self.enhsp_timeout_sec,
            event_determinism_mode=self.event_determinism_mode,
            event_determinism_repeats=self.event_determinism_repeats,
            event_determinism_save=self.event_determinism_save,
            angle_residual_correction=self.angle_residual_correction,
            angle_residual_calibration_path=self.angle_residual_calibration_path,
            angle_residual_bootstrap_path=self.angle_residual_bootstrap_path,
            angle_residual_live_update=self.angle_residual_live_update,
            test_force_clamp_to_full=self.test_force_clamp_to_full,
            test_force_clamp_min=self.test_force_clamp_min,
            override_angle=self.override_angle,
            override_force=self.override_force,
        )
        agent.run()


def main(agent_configs, 
         use_generalization_protocol: bool = False,
         use_config_metadata: bool = False,
         generalization_type: str = "local",
         generalization_train_ratio: float = 0.8,
         generalization_train_templates: list = None,
         generalization_test_templates: list = None,
         scenario_filter: str = None,
         phyq_config_path: str = "ScienceBirds/win6.6/win/config_phyq_sample.xml",
         levels_per_template: int = None,
         visualize_pddl: bool = False,
         visualize_pddl_board: bool = False,
         debug_mag_comparison: bool = False,
         mag_comparison_angle: float = 60.0,
         mag_comparison_start: float = 5.0,
         mag_comparison_decrement: float = 0.1,
         force_learning_mode: bool = False,
         force_learning_min_samples: int = 5,
         platform_aim_override: bool = False,
         block_aim_override: bool = False,
         planner_roll_block_only: bool = False,  # Disabled - use full problem with pig
         disable_sim_override: bool = True,
         planner_only: bool = True,
         disable_forward_sim: bool = False,
         prefer_sim_plan: bool = False,
         plan_pick_fast: bool = True,
         plan_pick_timeout_sec: float = 60.0,
         sim_gate_test_only: bool = True,
         disable_narrow_sim_gate: bool = False,
         max_train_attempts: int = 5,
         world_model_load: str = None,
         world_model_save: str = None,
         enhsp_ground_placeholder: bool = False,
         enhsp_timeout_sec: float = 120.0,
         event_determinism_mode: bool = False,
         event_determinism_repeats: int = 5,
         event_determinism_save: str = "event_determinism_results.json",
         angle_residual_correction: bool = True,
         angle_residual_calibration_path: str = "angle_residual_calibration.json",
         angle_residual_bootstrap_path: str = "event_determinism_results.json",
         angle_residual_live_update: bool = True,
         test_force_clamp_to_full: bool = True,
         test_force_clamp_min: float = 1.0,
         override_angle: float = None,
         override_force: float = None,
):
    """
    Main function to start the agent.
    
    Parameters:
    -----------
    agent_configs : argparse.Namespace
        Command-line arguments for agent configuration
    use_generalization_protocol : bool
        Enable Phy-Q generalization evaluation protocol
    use_config_metadata : bool
        Auto-load settings from config's .meta.json file
    generalization_type : str
        "local" for 80/20 within-template split, "broad" for across-template split
    generalization_train_ratio : float
        For local: train/test ratio (default 0.8)
    generalization_train_templates : list
        For broad: template numbers to train on
    generalization_test_templates : list
        For broad: template numbers to test on
    scenario_filter : str
        Only include levels from this scenario (e.g., "single_force")
    phyq_config_path : str
        Path to Phy-Q config XML file
    levels_per_template : int, optional
        Limit the number of levels per template
    visualize_pddl : bool
        Extra PDDL plots when a level is lost (default: disabled)
    visualize_pddl_board : bool
        Board plot on each ENHSP problem write (default: disabled)
    debug_mag_comparison : bool
        Enable mag comparison debug mode
    mag_comparison_angle : float
        Fixed angle for mag comparison test (degrees)
    mag_comparison_start : float
        Starting mag multiplier value
    mag_comparison_decrement : float
        Mag decrement per iteration
    """
    for x in range(1):
        print('naive agents %s running' % str(x))
        agent = AgentThread(
            agent_configs, 
            agent_ind=str(x),
            use_generalization_protocol=use_generalization_protocol,
            use_config_metadata=use_config_metadata,
            generalization_type=generalization_type,
            generalization_train_ratio=generalization_train_ratio,
            generalization_train_templates=generalization_train_templates,
            generalization_test_templates=generalization_test_templates,
            scenario_filter=scenario_filter,
            phyq_config_path=phyq_config_path,
            levels_per_template=levels_per_template,
            visualize_pddl=visualize_pddl,
            visualize_pddl_board=visualize_pddl_board,
            debug_mag_comparison=debug_mag_comparison,
            mag_comparison_angle=mag_comparison_angle,
            mag_comparison_start=mag_comparison_start,
            mag_comparison_decrement=mag_comparison_decrement,
            force_learning_mode=force_learning_mode,
            force_learning_min_samples=force_learning_min_samples,
            platform_aim_override=platform_aim_override,
            block_aim_override=block_aim_override,
            planner_roll_block_only=planner_roll_block_only,
            disable_sim_override=disable_sim_override,
            planner_only=planner_only,
            disable_forward_sim=disable_forward_sim,
            prefer_sim_plan=prefer_sim_plan,
            plan_pick_fast=plan_pick_fast,
            plan_pick_timeout_sec=plan_pick_timeout_sec,
            sim_gate_test_only=sim_gate_test_only,
            disable_narrow_sim_gate=disable_narrow_sim_gate,
            max_train_attempts=max_train_attempts,
            world_model_load=world_model_load,
            world_model_save=world_model_save,
            enhsp_ground_placeholder=enhsp_ground_placeholder,
            enhsp_timeout_sec=enhsp_timeout_sec,
            event_determinism_mode=event_determinism_mode,
            event_determinism_repeats=event_determinism_repeats,
            event_determinism_save=event_determinism_save,
            angle_residual_correction=angle_residual_correction,
            angle_residual_calibration_path=angle_residual_calibration_path,
            angle_residual_bootstrap_path=angle_residual_bootstrap_path,
            angle_residual_live_update=angle_residual_live_update,
            test_force_clamp_to_full=test_force_clamp_to_full,
            test_force_clamp_min=test_force_clamp_min,
            override_angle=override_angle,
            override_force=override_force,
        )
        agent.start()
        time.sleep(5)


def str2bool(v):
    """
    Given an str returns its boolean equivalent
    Parameters
    ----------
    v :

    Returns
    -------

    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":

    # Mirror every print (and stderr) to a timestamped log file under ./log/.
    # Installed FIRST so subsequent startup prints are captured too. Override
    # location / disable via --log-dir, --log-file, or --no-log.
    _pre_parser = argparse.ArgumentParser(add_help=False)
    _pre_parser.add_argument("--log-dir", default="log")
    _pre_parser.add_argument("--log-file", default=None,
                             help="Explicit log filename inside --log-dir (default: run_<timestamp>.log)")
    _pre_parser.add_argument("--no-log", action="store_true",
                             help="Do not mirror stdout/stderr to a log file")
    _pre_args, _ = _pre_parser.parse_known_args()
    if not _pre_args.no_log:
        start_tee_logging(log_dir=_pre_args.log_dir, filename=_pre_args.log_file)

    default_agent_host = "127.0.0.1"
    default_agent_port = 2004
    default_observer_host = "127.0.0.1"
    default_observer_port = 2006

    try:
        # Wrapper of the communicating messages
        with open('./src/client/server_client_config.json', 'r') as config:
            sc_json_config = json.load(config)
            default_agent_host = sc_json_config[0]["host"]
            default_agent_port = sc_json_config[0]["port"]

    except EnvironmentError:  # parent of IOError, OSError *and* WindowsError where available
        print("server_client_config.json not found")

    try:
        with open('./src/client/server_observer_client_config.json', 'r') as observer_config:
            observer_sc_json_config = json.load(observer_config)
            default_observer_host = observer_sc_json_config[0]["host"]
            default_observer_port = observer_sc_json_config[0]["port"]
    except EnvironmentError:  # parent of IOError, OSError *and* WindowsError where available
        print("server_observer_client_config.json not found")

    parser = argparse.ArgumentParser(
        description="Phy-Q Benchmark Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Phy-Q Generalization Protocols:
  LOCAL:  80/20 split within each template (tests within-task generalization)
  BROAD:  Train on some templates, test on others (tests cross-task generalization)

Config Metadata (enabled by default):
  Settings are auto-loaded from .meta.json files generated by generate_phyq_configs.py.
  This keeps the agent in sync with the config generation parameters automatically.

Examples:
  # Recommended: Run with auto-loaded metadata (default behavior)
  python main.py --generalization local --config ScienceBirds/win6.6/win/config_phyq_single_force_local.xml
  
  # Disable metadata auto-loading, use manual parameters instead
  python main.py --generalization local --config ScienceBirds/win6.6/win/config_phyq_single_force_full.xml --no-config-metadata --scenario single_force
  
  # Run single_force with broad generalization (manual parameters)
  python main.py --generalization broad --config ScienceBirds/win6.6/win/config_phyq_single_force_full.xml --no-config-metadata --train-templates 1 2 3 4 --test-templates 5 6
  
  # Run with local generalization, 70/30 split (manual)
  python main.py --generalization local --scenario single_force --train-ratio 0.7 --no-config-metadata
  
  # Run with local generalization, limiting to 30 levels per template (manual)
  python main.py --generalization local --scenario single_force --levels-per-template 30 --no-config-metadata
"""
    )

    # Logging (parsed once above too; declared here so --help lists them and
    # argparse doesn't complain about unknown args)
    parser.add_argument("--log-dir", default="log",
                        help="Directory for the mirrored stdout/stderr log file (default: log/)")
    parser.add_argument("--log-file", default=None,
                        help="Explicit log filename inside --log-dir (default: run_<timestamp>.log)")
    parser.add_argument("--no-log", action="store_true",
                        help="Do not mirror stdout/stderr to a log file")

    # Connection settings
    parser.add_argument("-s", "--save_logs", type=str2bool, nargs='?',
                        const=True, default=True, required=False,
                        help="if you want to save the logs for all agents")
    parser.add_argument("-a", "--agent_host", default=default_agent_host, required=False,
                        help="host ip address")
    parser.add_argument("-p", "--agent_port", type=int, default=default_agent_port, required=False,
                        help="host port for action robot")
    parser.add_argument("-b", "--observer_host", default=default_observer_host, required=False,
                        help="host ip address")
    parser.add_argument("-o", "--observer_port", type=int, default=default_observer_port, required=False,
                        help="host port for observer agents")
    
    # Phy-Q Generalization Protocol options
    parser.add_argument("--generalization", type=str, choices=["local", "broad", "none"], default="none",
                        help="Generalization protocol: 'local' (80/20 within template), "
                             "'broad' (train/test on different templates), or 'none' (use angle protocol)")
    parser.add_argument("--scenario", type=str, default=None,
                        choices=["single_force", "multiple_forces", "rolling", "falling", "sliding",
                                 "bouncing", "relative_weight", "relative_height", "relative_width",
                                 "shape_difference", "non_greedy", "structural_analysis", 
                                 "clearing_paths", "adequate_timing", "manoeuvring"],
                        help="Filter levels to only include this scenario (e.g., 'single_force')")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                        help="For local generalization: train/test split ratio (default: 0.8 = 80%% train)")
    parser.add_argument("--train-templates", type=int, nargs='+', default=None,
                        help="For broad generalization: template numbers to train on (e.g., --train-templates 1 2 3 4)")
    parser.add_argument("--test-templates", type=int, nargs='+', default=None,
                        help="For broad generalization: template numbers to test on (e.g., --test-templates 5 6)")
    parser.add_argument("--config", type=str, default="ScienceBirds/win6.6/win/config_phyq_sample.xml",
                        help="Path to Phy-Q config XML file")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible splits (default: 42)")
    parser.add_argument("--levels-per-template", type=int, default=None,
                        help="Limit the number of levels per template (e.g., --levels-per-template 30)")
    parser.add_argument("--use-config-metadata", action="store_true", default=True,
                        help="Auto-load generalization settings from the config's .meta.json file "
                             "(generated by generate_phyq_configs.py). Enabled by default.")
    parser.add_argument("--no-config-metadata", action="store_true",
                        help="Disable auto-loading from .meta.json file, use command-line parameters only.")
    parser.add_argument("--visualize", action="store_true",
                        help="PDDL loss plots on level failure + board on each ENHSP write")
    parser.add_argument("--visualize-pddl-board", action="store_true",
                        help="Show/save visualize_level_setup after problem.pddl is written "
                             "(pddl_level_viz/*_board_*.png); does not require a loss")
    parser.add_argument("--no-visualize", action="store_true",
                        help="Disable PDDL visualization (default)")
    
    # Debug mag comparison mode
    parser.add_argument("--debug-mag", action="store_true", default=False,
                        help="Enable mag comparison debug mode: test how different mag values affect trajectory")
    parser.add_argument("--no-debug-mag", action="store_true",
                        help="Disable mag comparison debug mode")
    parser.add_argument("--mag-angle", type=float, default=60.0,
                        help="Fixed angle for mag comparison test (default: 60 degrees)")
    parser.add_argument("--mag-start", type=float, default=5.0,
                        help="Starting mag multiplier value (default: 5.0)")
    parser.add_argument("--mag-decrement", type=float, default=0.1,
                        help="Mag decrement per iteration (default: 0.1)")
    
    # Force -> velocity learning mode
    parser.add_argument("--force-learning", action="store_true", default=False,
                        help="Enable force->velocity learning mode: random angle+force, fit v=f(force) model")
    parser.add_argument("--force-min-samples", type=int, default=5,
                        help="Min samples before fitting force model (default: 5)")

    # Platform-aim override (default OFF) — optional ballistic aim at platform top for rolling learning
    parser.set_defaults(platform_aim_override=False)
    parser.add_argument("--platform-aim-override", dest="platform_aim_override",
                        action="store_true",
                        help="Override ENHSP; aim at platform top to collect rolling samples")
    parser.add_argument("--no-platform-aim-override", dest="platform_aim_override",
                        action="store_false",
                        help="(default) Use normal ENHSP planner for angle selection")

    parser.set_defaults(block_aim_override=False)
    parser.add_argument("--block-aim-override", dest="block_aim_override",
                        action="store_true",
                        help="FF: skip ENHSP; ballistic aim at on-platform block (rolling indirect)")
    parser.add_argument("--no-block-aim-override", dest="block_aim_override",
                        action="store_false",
                        help="(default) Use ENHSP planner instead of block-aim FF")

    parser.set_defaults(planner_roll_block_only=False)  # Disabled - use full problem with pig
    parser.add_argument("--planner-roll-block-only", dest="planner_roll_block_only",
                        action="store_true",
                        help="(default) ENHSP problem keeps only the chosen on-platform roll block")
    parser.add_argument("--no-planner-roll-block-only", dest="planner_roll_block_only",
                        action="store_false",
                        help="Include all vision blocks in the planner problem")

    parser.set_defaults(disable_sim_override=True)
    parser.add_argument("--enable-sim-override", dest="disable_sim_override", action="store_false",
                        help="Allow angle×force sim search when ENHSP fails or sim gate rejects")
    parser.add_argument("--disable-sim-override", dest="disable_sim_override", action="store_true",
                        help="(default) Skip sim search; ENHSP then block/platform/ballistic fallback only")
    parser.set_defaults(planner_only=True)
    parser.add_argument("--enable-sim-gate", dest="planner_only", action="store_false",
                        help="Run sim search when ENHSP fails; sim gate may reject bad plans")
    parser.add_argument("--planner-only", dest="planner_only", action="store_true",
                        help="(default) Use ENHSP plans as-is; ballistic fallback only if planner fails")
    parser.set_defaults(disable_forward_sim=False)
    parser.add_argument("--enable-forward-sim", dest="disable_forward_sim", action="store_false",
                        help="(default) Forward sim for validation, fallback grid, and plan metadata")
    parser.add_argument("--disable-forward-sim", dest="disable_forward_sim", action="store_true",
                        help="Skip forward sim; use ENHSP plan and ballistic fallback only")
    parser.set_defaults(prefer_sim_plan=False)
    parser.add_argument("--prefer-sim-plan", dest="prefer_sim_plan", action="store_true",
                        help="After ENHSP, pick best forward-sim shot among planner / "
                             "local refine / sim search (implies --enable-forward-sim)")
    parser.add_argument("--no-prefer-sim-plan", dest="prefer_sim_plan", action="store_false",
                        help="(default) Keep ENHSP plan; run forward sim for validation only")
    parser.set_defaults(plan_pick_fast=True)
    parser.add_argument("--plan-pick-thorough", dest="plan_pick_fast", action="store_false",
                        help="Always run full angle×force sim grid during plan-pick (slower)")
    parser.add_argument("--plan-pick-fast", dest="plan_pick_fast", action="store_true",
                        help="(default) Skip full grid when local planner/refine already finds a kill")
    parser.add_argument(
        "--plan-pick-timeout",
        type=float,
        default=60.0,
        metavar="SEC",
        help="Wall-clock cap for plan-pick forward sim search only (default: 60; 0 = no cap; ENHSP unchanged)",
    )
    parser.set_defaults(sim_gate_test_only=True)
    parser.add_argument("--sim-gate-test-only", dest="sim_gate_test_only", action="store_true",
                        help="(default) With --planner-only and --narrow-sim-gate: TRAIN/TEST "
                             "narrow gate may replace ENHSP (platform-short, ground-bounce, etc.)")
    parser.add_argument("--no-sim-gate-test-only", dest="sim_gate_test_only", action="store_false",
                        help="Legacy: enable the broad sim gate on TEST levels "
                             "(rejects any unacceptable plan and runs sim_search/fallback)")
    parser.set_defaults(disable_narrow_sim_gate=False)
    parser.add_argument("--no-narrow-sim-gate", dest="disable_narrow_sim_gate", action="store_true",
                        help="Forward sim for logging/advisory only; keep ENHSP plan")
    parser.add_argument("--narrow-sim-gate", dest="disable_narrow_sim_gate", action="store_false",
                        help="(default) Replace bad ENHSP plans via sim search when forward sim rejects "
                             "them (ground-bounce miss, platform-short, etc.)")
    parser.add_argument("--max-train-attempts", type=int, default=5, metavar="N",
                        help="Cap on retries per TRAIN level before it is ABANDONED "
                             "(default: 5; was 8 — reduced to leave SB's 12000s time "
                             "budget for TEST levels)")
    parser.add_argument("--world-model-load", type=str, default=None, metavar="PATH",
                        help="Load pre-trained physics + KB from JSON at startup "
                             "(M5/LR/sklearn models embedded in _pickle_b64)")
    parser.add_argument(
        "--enhsp-ground-placeholder",
        action="store_true",
        help="A/B: ENHSP planning uses conservative ground-roll placeholder instead "
             "of learned ground LR/M5 inject (forward sim + KB learning unchanged)",
    )
    parser.add_argument(
        "--enhsp-timeout",
        type=float,
        default=120.0,
        metavar="SEC",
        help="Wall-clock cap for ENHSP Java subprocess (default: 120; "
             "was 300 — lowered to fit 200-min trial budget on hard templates "
             "like t06 where ENHSP frequently exhausts its 300s pass without "
             "finding a plan; see log/run_20260927_190840.log)",
    )
    parser.set_defaults(save_world_model=True)
    parser.add_argument("--world-model-save", type=str, default=None, metavar="PATH",
                        help="World model save path (default: saved_models/world_model_YYYYMMDD_HHMMSS.json)")
    parser.add_argument("--no-world-model-save", dest="save_world_model", action="store_false",
                        help="Disable automatic world model KB persistence")

    parser.set_defaults(event_determinism_mode=False)
    parser.add_argument("--event-determinism", dest="event_determinism_mode",
                        action="store_true",
                        help="Test event determinism — 20 BamBirds high-deviation dials, "
                             "5 identical shots each on the first level (no ENHSP)")
    parser.add_argument("--no-event-determinism", dest="event_determinism_mode",
                        action="store_false",
                        help="(default) Normal PDDL / generalization; angle residual correction applies")
    parser.add_argument("--event-determinism-repeats", type=int, default=5, metavar="N",
                        help="Repeats per dial angle in event determinism mode (default: 5)")
    parser.add_argument("--event-determinism-save", type=str,
                        default="event_determinism_results.json",
                        help="JSON report path for event determinism mode")

    parser.set_defaults(angle_residual_correction=True)
    parser.add_argument("--angle-residual-correction", dest="angle_residual_correction",
                        action="store_true",
                        help="(default) Apply ε(dial) from calibration JSON: exec ← planned − ε")
    parser.add_argument("--no-angle-residual-correction", dest="angle_residual_correction",
                        action="store_false",
                        help="Disable dial residual correction")
    parser.add_argument("--angle-residual-calibration", type=str,
                        default="angle_residual_calibration.json",
                        help="Live ε(dial) table path (updated during play)")
    parser.add_argument("--angle-residual-bootstrap", type=str,
                        default="event_determinism_results.json",
                        help="One-time seed JSON if live calibration file is missing")
    parser.set_defaults(angle_residual_live_update=True)
    parser.add_argument("--angle-residual-live-update", dest="angle_residual_live_update",
                        action="store_true",
                        help="(default) Update calibration from [SHOT DIAG POST] after each shot")
    parser.add_argument("--no-angle-residual-live-update", dest="angle_residual_live_update",
                        action="store_false",
                        help="Do not write live ε updates during normal play")

    # TEST-phase force clamp (Mode-B actuator floor mitigation).
    # See run_20260927_102019: SB's release mapping floors scaled_mag at ~72 px,
    # so force<1.0 still delivers |v|≈176. Planner picks force=0.6/0.8 for TEST,
    # sim expects short trajectory, actual overshoots. 0/8 TEST wins that run.
    parser.set_defaults(test_force_clamp_to_full=True)
    parser.add_argument("--test-force-clamp-to-full", dest="test_force_clamp_to_full",
                        action="store_true",
                        help="(default) Clamp planner force to --test-force-clamp-min in TEST "
                             "phase (Mode-B actuator floor mitigation; TRAIN unaffected)")
    parser.add_argument("--allow-partial-force-in-test", dest="test_force_clamp_to_full",
                        action="store_false",
                        help="Let ENHSP pick force<1.0 in TEST too (A/B: measure Mode-B overshoot)")
    parser.add_argument("--test-force-clamp-min", type=float, default=1.0, metavar="F",
                        help="Force floor applied in TEST when clamp is active (default: 1.0)")

    parser.add_argument(
        "--override-angle",
        type=float,
        default=None,
        metavar="DEG",
        help="Skip ENHSP; fire fixed PDDL dial each shot (e.g. 50 for t01_00033 left shelf). "
             "No gap-clearance nudge.",
    )
    parser.add_argument(
        "--override-force",
        type=float,
        default=None,
        metavar="F",
        help="Force with --override-angle (default: 1.0)",
    )

    args = parser.parse_args()
    
    if args.prefer_sim_plan:
        args.disable_forward_sim = False
        args.disable_sim_override = False
    if not args.planner_only:
        args.disable_sim_override = False
    
    # Determine if generalization protocol should be used
    use_generalization = args.generalization != "none"
    
    # Handle config metadata flag (--no-config-metadata overrides default)
    use_config_metadata = args.use_config_metadata and not args.no_config_metadata
    
    # Handle visualization flags (--no-visualize overrides default)
    visualize_pddl = args.visualize and not args.no_visualize
    visualize_pddl_board = (
        (args.visualize_pddl_board or args.visualize) and not args.no_visualize
    )
    
    # Handle debug mag flag (--no-debug-mag overrides default)
    debug_mag = args.debug_mag and not args.no_debug_mag
    if args.event_determinism_mode:
        debug_mag = False
        use_generalization = False

    from agents.pddl.world_model_persistence import default_save_path
    world_model_save = None
    if args.save_world_model:
        world_model_save = args.world_model_save or default_save_path()
    
    main(
        args,
        use_generalization_protocol=use_generalization,
        use_config_metadata=use_config_metadata,
        generalization_type=args.generalization if use_generalization else "local",
        generalization_train_ratio=args.train_ratio,
        generalization_train_templates=args.train_templates,
        generalization_test_templates=args.test_templates,
        scenario_filter=args.scenario,
        phyq_config_path=args.config,
        levels_per_template=args.levels_per_template,
        visualize_pddl=visualize_pddl,
        visualize_pddl_board=visualize_pddl_board,
        debug_mag_comparison=debug_mag,
        mag_comparison_angle=args.mag_angle,
        mag_comparison_start=args.mag_start,
        mag_comparison_decrement=args.mag_decrement,
        force_learning_mode=args.force_learning,
        force_learning_min_samples=args.force_min_samples,
        platform_aim_override=args.platform_aim_override,
        block_aim_override=args.block_aim_override,
        planner_roll_block_only=args.planner_roll_block_only,
        disable_sim_override=args.disable_sim_override,
        planner_only=args.planner_only,
        disable_forward_sim=args.disable_forward_sim,
        prefer_sim_plan=args.prefer_sim_plan,
        plan_pick_fast=args.plan_pick_fast,
        plan_pick_timeout_sec=args.plan_pick_timeout,
        sim_gate_test_only=args.sim_gate_test_only,
        disable_narrow_sim_gate=args.disable_narrow_sim_gate,
        max_train_attempts=args.max_train_attempts,
        world_model_load=args.world_model_load,
        world_model_save=world_model_save,
        enhsp_ground_placeholder=args.enhsp_ground_placeholder,
        enhsp_timeout_sec=args.enhsp_timeout,
        event_determinism_mode=args.event_determinism_mode,
        event_determinism_repeats=args.event_determinism_repeats,
        event_determinism_save=args.event_determinism_save,
        angle_residual_correction=args.angle_residual_correction,
        angle_residual_calibration_path=args.angle_residual_calibration,
        angle_residual_bootstrap_path=args.angle_residual_bootstrap,
        angle_residual_live_update=args.angle_residual_live_update,
        test_force_clamp_to_full=args.test_force_clamp_to_full,
        test_force_clamp_min=args.test_force_clamp_min,
        override_angle=args.override_angle,
        override_force=args.override_force,
    )
