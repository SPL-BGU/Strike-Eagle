import threading
import time
import src.demo.naive_agent_groundtruth as na
from agents import BirdsInBoots
from agents.quatzel.quatzel_agent import QuatzelAgent
from agents.owler.owler_agent import OwlerAgent
from agents.pddl.pddl_agent import PDDLAgent

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
                 visualize_pddl: bool = True,
                 debug_mag_comparison: bool = True,
                 mag_comparison_angle: float = 60.0,
                 mag_comparison_start: float = 5.0,
                 mag_comparison_decrement: float = 0.1,
                 force_learning_mode: bool = False,
                 force_learning_min_samples: int = 5):
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
            Show PDDL visualization when a level is lost (default: enabled)
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
        self.debug_mag_comparison = debug_mag_comparison
        self.mag_comparison_angle = mag_comparison_angle
        self.mag_comparison_start = mag_comparison_start
        self.mag_comparison_decrement = mag_comparison_decrement
        self.force_learning_mode = force_learning_mode
        self.force_learning_min_samples = force_learning_min_samples
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
            # Debug mag comparison mode
            debug_mag_comparison=self.debug_mag_comparison,
            mag_comparison_angle=self.mag_comparison_angle,
            mag_comparison_start=self.mag_comparison_start,
            mag_comparison_decrement=self.mag_comparison_decrement,
            # Force -> velocity learning mode
            force_learning_mode=self.force_learning_mode,
            force_learning_min_samples=self.force_learning_min_samples
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
         visualize_pddl: bool = True,
         debug_mag_comparison: bool = False,
         mag_comparison_angle: float = 60.0,
         mag_comparison_start: float = 5.0,
         mag_comparison_decrement: float = 0.1,
         force_learning_mode: bool = False,
         force_learning_min_samples: int = 5):
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
        Show PDDL visualization when a level is lost (default: enabled)
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
            debug_mag_comparison=debug_mag_comparison,
            mag_comparison_angle=mag_comparison_angle,
            mag_comparison_start=mag_comparison_start,
            mag_comparison_decrement=mag_comparison_decrement,
            force_learning_mode=force_learning_mode,
            force_learning_min_samples=force_learning_min_samples
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
    parser.add_argument("--visualize", action="store_true", default=True,
                        help="Show PDDL visualization when a level is lost (default: enabled)")
    parser.add_argument("--no-visualize", action="store_true",
                        help="Disable PDDL visualization")
    
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
    
    args = parser.parse_args()
    
    # Determine if generalization protocol should be used
    use_generalization = args.generalization != "none"
    
    # Handle config metadata flag (--no-config-metadata overrides default)
    use_config_metadata = args.use_config_metadata and not args.no_config_metadata
    
    # Handle visualization flag (--no-visualize overrides default)
    visualize_pddl = args.visualize and not args.no_visualize
    
    # Handle debug mag flag (--no-debug-mag overrides default)
    debug_mag = args.debug_mag and not args.no_debug_mag
    
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
        debug_mag_comparison=debug_mag,
        mag_comparison_angle=args.mag_angle,
        mag_comparison_start=args.mag_start,
        mag_comparison_decrement=args.mag_decrement,
        force_learning_mode=args.force_learning,
        force_learning_min_samples=args.force_min_samples
    )
