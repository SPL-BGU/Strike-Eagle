"""
Agent Comparison CSV Writer

This module provides CSV-based tracking for comparing agent performance
across different agents (e.g., PDDL agent vs human benchmarks).

The output CSV contains per-level results with the following columns:
- level_name: The XML filename (e.g., rolling_t01_00041.xml)
- variation: The variation number extracted from the filename
- scenario: The physical reasoning scenario (e.g., rolling, bouncing)
- template: The template number within the scenario
- agent: The agent name (e.g., PDDLAgent, Human)
- mode: train or test
- result: Win or Lose (or pass rate for Human)
"""

import csv
import os
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


SCENARIO_MAP = {
    "single_force": {"id": 1, "category": "1_01"},
    "multiple_forces": {"id": 2, "category": "1_02"},
    "rolling": {"id": 3, "category": "2_01"},
    "falling": {"id": 4, "category": "2_02"},
    "sliding": {"id": 5, "category": "2_03"},
    "bouncing": {"id": 6, "category": "2_04"},
    "relative_weight": {"id": 7, "category": "3_01"},
    "relative_height": {"id": 8, "category": "3_02"},
    "relative_width": {"id": 9, "category": "3_03"},
    "shape_difference": {"id": 10, "category": "3_04"},
    "non_greedy": {"id": 11, "category": "3_05"},
    "structural_analysis": {"id": 12, "category": "3_06"},
    "clearing_paths": {"id": 13, "category": "3_07"},
    "adequate_timing": {"id": 14, "category": "3_08"},
    "manoeuvring": {"id": 15, "category": "3_09"},
}

SCENARIO_ID_TO_NAME = {
    1: "single_force",
    2: "multiple_forces",
    3: "rolling",
    4: "falling",
    5: "sliding",
    6: "bouncing",
    7: "relative_weight",
    8: "relative_height",
    9: "relative_width",
    10: "shape_difference",
    11: "non_greedy",
    12: "structural_analysis",
    13: "clearing_paths",
    14: "adequate_timing",
    15: "manoeuvring",
}


@dataclass
class LevelComparison:
    """Data class for a single level comparison result."""
    level_name: str
    variation: int
    scenario: str
    template: int
    agent: str
    mode: str
    result: str


class AgentComparisonCSV:
    """
    Manages CSV output for agent comparison results.
    
    Writes to TWO separate files for easier analysis:
    1. pddl_agent_results.csv - Per-level PDDLAgent results (level_index, name, result, score)
    2. baseline_comparison.csv - Baseline agent pass rates by scenario/template (unique entries)
    
    Usage:
        csv_writer = AgentComparisonCSV(
            output_dir="results",
            human_baseline_path="external/phy-q/playdata/broad_generalization_all_agents.csv"
        )
        
        # After each level completion:
        csv_writer.write_result(
            level_path="./Levels/phy_q/scenario_03_rolling/train/rolling_t01_00041.xml",
            agent="PDDLAgent",
            won=True,
            score=10000
        )
    """
    
    PDDL_HEADERS = ["level_index", "level_name", "scenario", "template", "variation", "mode", "result", "score", "plan_source", "unsolvable"]
    BASELINE_HEADERS = ["scenario", "template", "agent", "pass_rate"]
    
    def __init__(
        self, 
        output_path: str = "agent_comparison_results.csv",  # Legacy - now used as output_dir base
        human_baseline_path: Optional[str] = None,
        baseline_agents_dir: Optional[str] = "data/baseline_agent_data"
    ):
        """
        Initialize the CSV writer.
        
        Args:
            output_path: Base path (directory extracted, or used directly for legacy compatibility)
            human_baseline_path: Path to the human baseline CSV (optional)
            baseline_agents_dir: Directory containing baseline agent CSVs (optional)
        """
        # Determine output directory from output_path
        output_dir = os.path.dirname(output_path) if os.path.dirname(output_path) else "."
        
        self.pddl_results_path = os.path.join(output_dir, "pddl_agent_results.csv")
        self.baseline_path = os.path.join(output_dir, "baseline_comparison.csv")
        self.output_path = output_path  # Keep for backward compatibility
        
        self.agent_baselines: Dict[str, Dict[str, float]] = {}  # {agent_name: {scenario: pass_rate}}
        self.baseline_agents: Dict[str, Dict[Tuple[int, int, int], str]] = {}
        self.results: List[LevelComparison] = []
        
        # Track level index for PDDLAgent
        self.level_index = 0
        
        # Track which baseline entries we've already written (to avoid duplicates)
        self.written_baselines: set = set()  # (scenario, template, agent)
        
        if human_baseline_path and os.path.exists(human_baseline_path):
            self.load_all_agent_baselines(human_baseline_path)
        
        self._init_csv_files()
    
    def _init_csv_files(self) -> None:
        """Initialize both CSV files with headers (creates fresh files each run)."""
        # PDDLAgent results file - always create fresh
        with open(self.pddl_results_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(self.PDDL_HEADERS)
        self.level_index = 0
        print(f"[COMPARISON CSV] Created: {self.pddl_results_path}")
        
        # Baseline comparison file
        if not os.path.exists(self.baseline_path):
            with open(self.baseline_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(self.BASELINE_HEADERS)
            print(f"[COMPARISON CSV] Created: {self.baseline_path}")
        else:
            # Load existing baseline entries to avoid duplicates
            with open(self.baseline_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    key = (row.get('scenario', ''), row.get('template', ''), row.get('agent', ''))
                    self.written_baselines.add(key)
            print(f"[COMPARISON CSV] Appending to: {self.baseline_path} ({len(self.written_baselines)} existing entries)")
    
    def load_all_agent_baselines(self, csv_path: str) -> None:
        """
        Load pass rates for all agents from the Phy-Q benchmark CSV.
        
        The CSV has format: ,category,pass_rate,name,event,description
        We extract all agents and map category to scenario.
        
        Args:
            csv_path: Path to broad_generalization_all_agents.csv
        """
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    agent_name = row.get('name', '')
                    category = row.get('category', '')
                    pass_rate = float(row.get('pass_rate', 0))
                    
                    if not agent_name:
                        continue
                    
                    if agent_name not in self.agent_baselines:
                        self.agent_baselines[agent_name] = {}
                    
                    for scenario_name, info in SCENARIO_MAP.items():
                        if info['category'] == category:
                            self.agent_baselines[agent_name][scenario_name] = pass_rate
                            break
            
            agent_names = list(self.agent_baselines.keys())
            print(f"[COMPARISON CSV] Loaded baselines for {len(agent_names)} agents: {', '.join(agent_names)}")
            
        except FileNotFoundError:
            print(f"[COMPARISON CSV] Warning: Baseline file not found: {csv_path}")
        except Exception as e:
            print(f"[COMPARISON CSV] Warning: Failed to load baselines: {e}")
    
    def load_baseline_agents(self, agents_dir: str) -> None:
        """
        Load baseline agent results from CSV files.
        
        Each CSV has columns: LevelStatus (Pass/Fail), levelName (path with level file)
        Level files are named like: X_SS_TT_VVVVV.xml (prefix_scenario_template_variation)
        
        Args:
            agents_dir: Directory containing agent CSV files (Bambirds.csv, DataLab.csv, etc.)
        """
        agent_files = {
            "Bambirds": os.path.join(agents_dir, "Bambirds.csv"),
            "DataLab": os.path.join(agents_dir, "DataLab.csv"),
            "EagleWings": os.path.join(agents_dir, "EagleWings.csv"),
            "Naive": os.path.join(agents_dir, "Naive.csv"),
        }
        
        for agent_name, csv_path in agent_files.items():
            if not os.path.exists(csv_path):
                continue
            
            self.baseline_agents[agent_name] = {}
            
            try:
                with open(csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        level_name = row.get('levelName', '')
                        level_status = row.get('LevelStatus', '')
                        
                        level_key = self._parse_baseline_level_key(level_name)
                        if level_key and level_status in ('Pass', 'Fail'):
                            existing = self.baseline_agents[agent_name].get(level_key)
                            if existing is None or existing == 'Fail':
                                self.baseline_agents[agent_name][level_key] = 'Win' if level_status == 'Pass' else 'Lose'
                
                print(f"[COMPARISON CSV] Loaded {agent_name}: {len(self.baseline_agents[agent_name])} levels")
                
            except Exception as e:
                print(f"[COMPARISON CSV] Warning: Failed to load {agent_name}: {e}")
    
    def _parse_baseline_level_key(self, level_path: str) -> Optional[Tuple[int, int, int]]:
        """
        Parse baseline agent level path to extract (scenario, template, variation).
        
        Level names are like: X_SS_TT_VVVVV.xml where SS=scenario, TT=template, VVVVV=variation
        
        Returns:
            Tuple of (scenario_id, template, variation) or None if parsing fails
        """
        filename = os.path.basename(level_path)
        
        match = re.search(r'(\d+)_(\d+)_(\d+)_(\d+)\.xml$', filename)
        if match:
            scenario_id = int(match.group(2))
            template = int(match.group(3))
            variation = int(match.group(4))
            return (scenario_id, template, variation)
        
        return None
    
    def _get_level_key_from_pddl_path(self, level_path: str) -> Optional[Tuple[int, int, int]]:
        """
        Convert PDDL level path to baseline lookup key.
        
        PDDL paths are like: single_force_t01_00001.xml
        Returns: (scenario_id, template, variation)
        """
        scenario = self.extract_scenario(level_path)
        template = self.extract_template(level_path)
        variation = self.extract_variation(level_path)
        
        if scenario and template is not None and variation is not None:
            scenario_id = SCENARIO_MAP.get(scenario, {}).get('id')
            if scenario_id:
                return (scenario_id, template, variation)
        
        return None
    
    def get_baseline_result(self, level_path: str, agent_name: str) -> Optional[str]:
        """
        Look up baseline agent result for a level.
        
        Args:
            level_path: PDDL-style level path
            agent_name: Name of baseline agent (Bambirds, DataLab, etc.)
            
        Returns:
            "Win" or "Lose" if found, None otherwise
        """
        if agent_name not in self.baseline_agents:
            return None
        
        level_key = self._get_level_key_from_pddl_path(level_path)
        if level_key:
            return self.baseline_agents[agent_name].get(level_key)
        
        return None
    
    def extract_level_name(self, level_path: str) -> str:
        """
        Extract the level filename from a full path.
        
        Args:
            level_path: Full or relative path like ./Levels/phy_q/.../rolling_t01_00041.xml
            
        Returns:
            Just the filename: rolling_t01_00041.xml
        """
        return os.path.basename(level_path)
    
    def extract_scenario(self, level_path: str) -> Optional[str]:
        """
        Extract scenario name from level path.
        
        Handles paths like:
        - ./Levels/phy_q/scenario_03_rolling/train/rolling_t01_00001.xml
        - rolling_t01_00001.xml (filename only)
        
        Args:
            level_path: Path to the level file
            
        Returns:
            Scenario name (e.g., "rolling") or None
        """
        path_str = str(level_path).replace("\\", "/")
        
        scenario_match = re.search(r'scenario_\d+_(\w+)', path_str)
        if scenario_match:
            return scenario_match.group(1)
        
        for scenario_name in SCENARIO_MAP.keys():
            if scenario_name in path_str.lower():
                return scenario_name
        
        return None
    
    def extract_template(self, level_path: str) -> Optional[int]:
        """
        Extract template number from level path.
        
        Args:
            level_path: Path containing pattern like _t01_ or _t02_
            
        Returns:
            Template number (e.g., 1, 2) or None
        """
        template_match = re.search(r'_t(\d+)_', str(level_path))
        if template_match:
            return int(template_match.group(1))
        return None
    
    def extract_variation(self, level_path: str) -> Optional[int]:
        """
        Extract variation number from level path.
        
        Parses filenames like:
        - rolling_t01_00041.xml -> variation 41
        - single_force_t02_00001.xml -> variation 1
        
        Args:
            level_path: Path to the level file
            
        Returns:
            Variation number or None
        """
        filename = self.extract_level_name(level_path)
        
        variation_match = re.search(r'_(\d{5})\.xml$', filename)
        if variation_match:
            return int(variation_match.group(1))
        
        variation_match = re.search(r'_t\d+_(\d+)', filename)
        if variation_match:
            return int(variation_match.group(1))
        
        return None
    
    def write_result(
        self, 
        level_path: str, 
        agent: str, 
        won: bool,
        mode: str = "unknown",
        score: int = 0,
        plan_source: str = "planner",
        unsolvable: bool = False
    ) -> None:
        """
        Write level results to TWO CSV files:
        1. pddl_agent_results.csv - PDDLAgent per-level results
        2. baseline_comparison.csv - Baseline agent pass rates (unique per scenario/template)
        
        Args:
            level_path: Full path to the level file
            agent: Name of the agent (e.g., "PDDLAgent")
            won: Whether the agent won the level
            mode: Current mode - "train" or "test"
            score: Score achieved (only meaningful if won)
            plan_source: "planner" if PDDL planner succeeded, "fallback" if fallback was used
            unsolvable: True if the PDDL planner reported the problem as unsolvable
        """
        level_name = self.extract_level_name(level_path)
        scenario = self.extract_scenario(level_path) or "unknown"
        template = self.extract_template(level_path) or 0
        variation = self.extract_variation(level_path) or 0
        result = "Win" if won else "Lose"
        
        # 1. Write PDDLAgent result to pddl_agent_results.csv
        self.level_index += 1
        
        agent_comparison = LevelComparison(
            level_name=level_name,
            variation=variation,
            scenario=scenario,
            template=template,
            agent=agent,
            mode=mode,
            result=result
        )
        self.results.append(agent_comparison)
        
        with open(self.pddl_results_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.level_index,
                level_name,
                scenario,
                template,
                variation,
                mode,
                result,
                score if won else 0,
                plan_source,
                unsolvable
            ])
        
        # 2. Write baseline comparisons to baseline_comparison.csv (unique entries only)
        baseline_rows = []
        for baseline_agent, baselines in self.agent_baselines.items():
            baseline_key = (scenario, str(template), baseline_agent)
            
            # Only write if we haven't written this combination before
            if baseline_key not in self.written_baselines:
                baseline_rate = baselines.get(scenario)
                if baseline_rate is not None:
                    baseline_rows.append([scenario, template, baseline_agent, f"{baseline_rate:.3f}"])
                    self.written_baselines.add(baseline_key)
        
        if baseline_rows:
            with open(self.baseline_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                for row in baseline_rows:
                    writer.writerow(row)
        
        human_rate = self.agent_baselines.get("Human", {}).get(scenario)
        human_str = f"{human_rate:.3f}" if human_rate else "N/A"
        unsolvable_str = "UNSOLVABLE" if unsolvable else ""
        print(f"[CSV] #{self.level_index} {level_name} | {result} | Score: {score} | Plan: {plan_source} | {unsolvable_str} | Mode: {mode}")
    
    def get_summary(self) -> Dict:
        """
        Get a summary of recorded results.
        
        Returns:
            Dictionary with win/loss counts per agent and scenario
        """
        summary = {
            "total_results": len(self.results),
            "by_agent": {},
            "by_scenario": {}
        }
        
        for result in self.results:
            if result.agent not in summary["by_agent"]:
                summary["by_agent"][result.agent] = {"wins": 0, "losses": 0}
            
            if result.result == "Win":
                summary["by_agent"][result.agent]["wins"] += 1
            else:
                summary["by_agent"][result.agent]["losses"] += 1
            
            if result.scenario not in summary["by_scenario"]:
                summary["by_scenario"][result.scenario] = {"wins": 0, "losses": 0}
            
            if result.result == "Win":
                summary["by_scenario"][result.scenario]["wins"] += 1
            else:
                summary["by_scenario"][result.scenario]["losses"] += 1
        
        return summary
    
    def print_summary(self) -> None:
        """Print a formatted summary of results."""
        summary = self.get_summary()
        
        print("\n" + "=" * 60)
        print("AGENT COMPARISON SUMMARY")
        print("=" * 60)
        print(f"Total results recorded: {summary['total_results']}")
        print()
        
        print("By Agent:")
        for agent, stats in summary["by_agent"].items():
            total = stats["wins"] + stats["losses"]
            rate = stats["wins"] / total * 100 if total > 0 else 0
            print(f"  {agent}: {stats['wins']}/{total} wins ({rate:.1f}%)")
        
        print()
        print("By Scenario:")
        for scenario, stats in summary["by_scenario"].items():
            total = stats["wins"] + stats["losses"]
            rate = stats["wins"] / total * 100 if total > 0 else 0
            human_rate = self.human_baselines.get(scenario, 0) * 100
            print(f"  {scenario}: {stats['wins']}/{total} wins ({rate:.1f}%) | Human: {human_rate:.1f}%")
        
        print("=" * 60 + "\n")


def create_comparison_csv(
    output_path: str = "agent_comparison_results.csv",
    human_baseline_path: Optional[str] = None
) -> AgentComparisonCSV:
    """
    Factory function to create an AgentComparisonCSV instance.
    
    Args:
        output_path: Where to write the CSV
        human_baseline_path: Path to human baseline data
        
    Returns:
        Configured AgentComparisonCSV instance
    """
    return AgentComparisonCSV(
        output_path=output_path,
        human_baseline_path=human_baseline_path
    )
