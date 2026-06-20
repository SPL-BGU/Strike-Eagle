"""
Phy-Q Benchmark Metrics Tracking

This module provides metrics tracking for evaluating agent performance
on the Phy-Q physical reasoning benchmark.

The Phy-Q benchmark tests 15 physical reasoning scenarios:
1. Single force - Direct destruction with one hit
2. Multiple forces - Requires multiple hits to destroy
3. Rolling - Roll circular objects to reach targets
4. Falling - Drop objects onto targets
5. Sliding - Slide non-circular objects to targets
6. Bouncing - Bounce off surfaces to reach targets
7. Relative weight - Select objects by weight
8. Relative height - Select objects by height
9. Relative width - Select openings/objects by width
10. Shape difference - Select objects by shape
11. Non-greedy actions - Strategic action ordering
12. Structural analysis - Find weak points in structures
13. Clearing paths - Create paths to reach targets
14. Adequate timing - Time-constrained actions
15. Manoeuvring - Activate object powers correctly

Reference:
    Xue et al. "Phy-Q as a measure for physical reasoning intelligence"
    Nature Machine Intelligence, 2022
    https://www.nature.com/articles/s42256-022-00583-4
"""

import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


SCENARIO_INFO = {
    "single_force": {
        "id": 1,
        "category": 1,
        "description": "Some target objects can be destroyed with a single force",
        "skill": "Force application"
    },
    "multiple_forces": {
        "id": 2,
        "category": 1,
        "description": "Some target objects need multiple forces to destroy",
        "skill": "Force application"
    },
    "rolling": {
        "id": 3,
        "category": 2,
        "description": "Circular objects can be rolled along a surface to a target",
        "skill": "Motion prediction"
    },
    "falling": {
        "id": 4,
        "category": 2,
        "description": "Objects can be fallen on to a target",
        "skill": "Motion prediction"
    },
    "sliding": {
        "id": 5,
        "category": 2,
        "description": "Non-circular objects can be slid along a surface to a target",
        "skill": "Motion prediction"
    },
    "bouncing": {
        "id": 6,
        "category": 2,
        "description": "Objects can be bounced off a surface to reach a target",
        "skill": "Motion prediction"
    },
    "relative_weight": {
        "id": 7,
        "category": 3,
        "description": "Objects with correct weight need to be moved to reach a target",
        "skill": "Property reasoning"
    },
    "relative_height": {
        "id": 8,
        "category": 3,
        "description": "Objects with correct height need to be moved to reach a target",
        "skill": "Property reasoning"
    },
    "relative_width": {
        "id": 9,
        "category": 3,
        "description": "Objects with correct width or opening should be selected",
        "skill": "Property reasoning"
    },
    "shape_difference": {
        "id": 10,
        "category": 3,
        "description": "Objects with correct shape need to be moved/destroyed",
        "skill": "Property reasoning"
    },
    "non_greedy": {
        "id": 11,
        "category": 3,
        "description": "Actions need to be selected in the correct order based on physical consequences",
        "skill": "Strategic planning"
    },
    "structural_analysis": {
        "id": 12,
        "category": 3,
        "description": "The correct target needs to be chosen to break the stability of a structure",
        "skill": "Structural reasoning"
    },
    "clearing_paths": {
        "id": 13,
        "category": 3,
        "description": "A path needs to be created before the target can be reached",
        "skill": "Sequential planning"
    },
    "adequate_timing": {
        "id": 14,
        "category": 3,
        "description": "Correct actions need to be performed within time constraints",
        "skill": "Temporal reasoning"
    },
    "manoeuvring": {
        "id": 15,
        "category": 3,
        "description": "Powers of objects need to be activated correctly to reach a target",
        "skill": "Ability utilization"
    },
}

CATEGORY_NAMES = {
    1: "Force Scenarios",
    2: "Motion Scenarios", 
    3: "Complex Reasoning"
}


@dataclass
class LevelResult:
    """Result for a single level attempt."""
    level_path: str
    scenario: str
    template: Optional[int]
    won: bool
    attempts: int
    score: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass 
class ScenarioStats:
    """Statistics for a single scenario."""
    wins: int = 0
    losses: int = 0
    total_attempts: int = 0
    total_score: int = 0
    levels_played: int = 0
    
    @property
    def total(self) -> int:
        return self.wins + self.losses
    
    @property
    def pass_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.wins / self.total
    
    @property
    def avg_attempts(self) -> float:
        if self.levels_played == 0:
            return 0.0
        return self.total_attempts / self.levels_played
    
    @property
    def avg_score(self) -> float:
        if self.wins == 0:
            return 0.0
        return self.total_score / self.wins


class PhyQMetrics:
    """
    Track performance per physical reasoning scenario for Phy-Q benchmark.
    
    Usage:
        metrics = PhyQMetrics()
        
        # Record results
        metrics.record("./Levels/phy_q/scenario_03_rolling/train/rolling_t01_00001.xml", won=True, attempts=2, score=45000)
        
        # Get statistics
        print(metrics.get_scenario_stats("rolling"))
        print(f"Phy-Q Score: {metrics.compute_phyq_score():.2f}")
        
        # Save results
        metrics.save("phyq_results.json")
    """
    
    def __init__(self):
        self.results: List[LevelResult] = []
        self.scenario_stats: Dict[str, ScenarioStats] = {
            name: ScenarioStats() for name in SCENARIO_INFO.keys()
        }
    
    def extract_scenario_from_path(self, level_path: str) -> Optional[str]:
        """
        Extract scenario name from level path.
        
        Handles paths like:
        - ./Levels/phy_q/scenario_03_rolling/train/rolling_t01_00001.xml
        - Levels/phy_q/scenario_03_rolling/test/rolling_t01_00001.xml
        - rolling_t01_00001.xml (filename only)
        """
        path_str = str(level_path).replace("\\", "/")
        
        scenario_match = re.search(r'scenario_\d+_(\w+)', path_str)
        if scenario_match:
            return scenario_match.group(1)
        
        for scenario_name in SCENARIO_INFO.keys():
            if scenario_name in path_str:
                return scenario_name
        
        return None
    
    def extract_template_from_path(self, level_path: str) -> Optional[int]:
        """Extract template number from level path."""
        path_str = str(level_path)
        
        template_match = re.search(r'_t(\d+)_', path_str)
        if template_match:
            return int(template_match.group(1))
        
        return None
    
    def record(
        self, 
        level_path: str, 
        won: bool, 
        attempts: int = 1, 
        score: int = 0
    ) -> None:
        """
        Record a level result.
        
        Args:
            level_path: Path to the level file (can be full or relative)
            won: Whether the level was completed successfully
            attempts: Number of attempts taken
            score: Final score achieved (only meaningful if won)
        """
        scenario = self.extract_scenario_from_path(level_path)
        template = self.extract_template_from_path(level_path)
        
        result = LevelResult(
            level_path=level_path,
            scenario=scenario or "unknown",
            template=template,
            won=won,
            attempts=attempts,
            score=score if won else 0
        )
        self.results.append(result)
        
        if scenario and scenario in self.scenario_stats:
            stats = self.scenario_stats[scenario]
            stats.levels_played += 1
            stats.total_attempts += attempts
            if won:
                stats.wins += 1
                stats.total_score += score
            else:
                stats.losses += 1
    
    def get_scenario_stats(self, scenario: str) -> Dict:
        """Get detailed statistics for a scenario."""
        if scenario not in self.scenario_stats:
            return {}
        
        stats = self.scenario_stats[scenario]
        info = SCENARIO_INFO.get(scenario, {})
        
        return {
            "scenario": scenario,
            "id": info.get("id"),
            "category": info.get("category"),
            "category_name": CATEGORY_NAMES.get(info.get("category", 0), "Unknown"),
            "description": info.get("description"),
            "skill": info.get("skill"),
            "wins": stats.wins,
            "losses": stats.losses,
            "total": stats.total,
            "pass_rate": stats.pass_rate,
            "avg_attempts": stats.avg_attempts,
            "avg_score": stats.avg_score,
        }
    
    def get_category_stats(self, category: int) -> Dict:
        """Get aggregated statistics for a category."""
        scenarios_in_category = [
            name for name, info in SCENARIO_INFO.items()
            if info.get("category") == category
        ]
        
        total_wins = sum(self.scenario_stats[s].wins for s in scenarios_in_category)
        total_losses = sum(self.scenario_stats[s].losses for s in scenarios_in_category)
        total = total_wins + total_losses
        
        return {
            "category": category,
            "name": CATEGORY_NAMES.get(category, "Unknown"),
            "scenarios": scenarios_in_category,
            "wins": total_wins,
            "losses": total_losses,
            "total": total,
            "pass_rate": total_wins / total if total > 0 else 0.0
        }
    
    def compute_phyq_score(self) -> float:
        """
        Compute the Phy-Q score as per the benchmark paper.
        
        The Phy-Q score is the average pass rate across all 15 scenarios,
        scaled to 0-100. This provides a single metric for overall
        physical reasoning ability.
        
        Returns:
            Phy-Q score (0-100)
        """
        pass_rates = []
        for scenario in SCENARIO_INFO.keys():
            stats = self.scenario_stats[scenario]
            if stats.total > 0:
                pass_rates.append(stats.pass_rate)
        
        if not pass_rates:
            return 0.0
        
        return (sum(pass_rates) / len(pass_rates)) * 100
    
    def compute_local_generalization_score(self) -> Dict[str, float]:
        """
        Compute local generalization scores per scenario.
        
        Local generalization measures how well the agent performs
        on variations within the same task template.
        
        Returns:
            Dictionary mapping scenario names to pass rates
        """
        return {
            scenario: stats.pass_rate
            for scenario, stats in self.scenario_stats.items()
            if stats.total > 0
        }
    
    def compute_broad_generalization_score(self) -> Dict[str, float]:
        """
        Compute broad generalization scores per category.
        
        Broad generalization measures how well the agent generalizes
        to different scenarios within the same category.
        
        Returns:
            Dictionary mapping category names to pass rates
        """
        scores = {}
        for category in [1, 2, 3]:
            cat_stats = self.get_category_stats(category)
            if cat_stats["total"] > 0:
                scores[cat_stats["name"]] = cat_stats["pass_rate"]
        return scores
    
    def get_summary(self) -> Dict:
        """Get a complete summary of all metrics."""
        return {
            "phyq_score": self.compute_phyq_score(),
            "total_levels": sum(s.total for s in self.scenario_stats.values()),
            "total_wins": sum(s.wins for s in self.scenario_stats.values()),
            "total_losses": sum(s.losses for s in self.scenario_stats.values()),
            "overall_pass_rate": (
                sum(s.wins for s in self.scenario_stats.values()) /
                max(1, sum(s.total for s in self.scenario_stats.values()))
            ),
            "local_generalization": self.compute_local_generalization_score(),
            "broad_generalization": self.compute_broad_generalization_score(),
            "scenarios": {
                name: self.get_scenario_stats(name)
                for name in SCENARIO_INFO.keys()
            },
            "categories": {
                CATEGORY_NAMES[cat]: self.get_category_stats(cat)
                for cat in [1, 2, 3]
            }
        }
    
    def save(self, filepath: str) -> None:
        """Save results to a JSON file."""
        data = {
            "summary": self.get_summary(),
            "results": [asdict(r) for r in self.results],
            "timestamp": datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, filepath: str) -> None:
        """Load results from a JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.results = []
        self.scenario_stats = {
            name: ScenarioStats() for name in SCENARIO_INFO.keys()
        }
        
        for r in data.get("results", []):
            self.record(
                level_path=r["level_path"],
                won=r["won"],
                attempts=r["attempts"],
                score=r["score"]
            )
    
    def print_report(self) -> None:
        """Print a formatted performance report."""
        summary = self.get_summary()
        
        print("=" * 70)
        print("PHY-Q BENCHMARK PERFORMANCE REPORT")
        print("=" * 70)
        print()
        
        print(f"Overall Phy-Q Score: {summary['phyq_score']:.2f} / 100")
        print(f"Total Levels: {summary['total_levels']}")
        print(f"Wins: {summary['total_wins']} | Losses: {summary['total_losses']}")
        print(f"Overall Pass Rate: {summary['overall_pass_rate']*100:.1f}%")
        print()
        
        print("-" * 70)
        print("CATEGORY PERFORMANCE (Broad Generalization)")
        print("-" * 70)
        for cat_name, cat_stats in summary["categories"].items():
            if cat_stats["total"] > 0:
                print(f"  {cat_name}: {cat_stats['pass_rate']*100:.1f}% "
                      f"({cat_stats['wins']}/{cat_stats['total']})")
        print()
        
        print("-" * 70)
        print("SCENARIO PERFORMANCE (Local Generalization)")
        print("-" * 70)
        for scenario, stats in sorted(summary["scenarios"].items(), 
                                       key=lambda x: x[1].get("id", 99)):
            if stats["total"] > 0:
                print(f"  {stats['id']:2d}. {scenario:20s}: {stats['pass_rate']*100:5.1f}% "
                      f"({stats['wins']:3d}/{stats['total']:3d}) "
                      f"- {stats['skill']}")
        print()
        print("=" * 70)


def create_phyq_metrics() -> PhyQMetrics:
    """Factory function to create a PhyQMetrics instance."""
    return PhyQMetrics()


def parse_config_level_paths(config_path: str) -> List[str]:
    """
    Parse a Science Birds config.xml file and extract all level paths.
    
    The config file structure contains <game_levels level_path="..."/> elements
    that define the level sequence.
    
    Args:
        config_path: Path to the config XML file (e.g., config_phyq_sample.xml)
        
    Returns:
        List of level paths in order (1-indexed when used with current_level)
    """
    from xml.etree import ElementTree as ET
    
    level_paths = []
    
    try:
        tree = ET.parse(config_path)
        root = tree.getroot()
        
        for game_level in root.iter('game_levels'):
            level_path = game_level.get('level_path')
            if level_path:
                level_paths.append(level_path)
                
        print(f"[PHY-Q CONFIG] Loaded {len(level_paths)} level paths from {config_path}")
        
    except FileNotFoundError:
        print(f"[PHY-Q CONFIG] Warning: Config file not found: {config_path}")
    except ET.ParseError as e:
        print(f"[PHY-Q CONFIG] Warning: Failed to parse config: {e}")
    
    return level_paths


class PhyQLevelMapper:
    """
    Maps level indices to level paths for Phy-Q benchmark tracking.
    
    Usage:
        mapper = PhyQLevelMapper("config_phyq_sample.xml")
        level_path = mapper.get_level_path(current_level)  # 1-indexed
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the level mapper.
        
        Args:
            config_path: Path to config XML. If None, will try to auto-detect.
        """
        self.level_paths: List[str] = []
        self.config_path = config_path
        
        if config_path:
            self.level_paths = parse_config_level_paths(config_path)
    
    def get_level_path(self, level_index: int) -> Optional[str]:
        """
        Get the level path for a given level index.
        
        Args:
            level_index: 1-indexed level number (as returned by game)
            
        Returns:
            Level path string, or None if index is out of range
        """
        if not self.level_paths:
            return None
            
        idx = level_index - 1
        if 0 <= idx < len(self.level_paths):
            return self.level_paths[idx]
        
        return None
    
    def __len__(self) -> int:
        return len(self.level_paths)
