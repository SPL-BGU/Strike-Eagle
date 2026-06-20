"""
Compute Phy-Q score from agent_comparison_results.csv

Usage:
    python agents/pddl/compute_phyq_from_csv.py
    python agents/pddl/compute_phyq_from_csv.py path/to/results.csv
"""

import csv
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple


SCENARIO_ORDER = [
    "single_force", "multiple_forces", "rolling", "falling", "sliding",
    "bouncing", "relative_weight", "relative_height", "relative_width",
    "shape_difference", "non_greedy", "structural_analysis", "clearing_paths",
    "adequate_timing", "manoeuvring"
]

CATEGORY_MAP = {
    "single_force": "Force",
    "multiple_forces": "Force",
    "rolling": "Motion",
    "falling": "Motion",
    "sliding": "Motion",
    "bouncing": "Motion",
    "relative_weight": "Complex Reasoning",
    "relative_height": "Complex Reasoning",
    "relative_width": "Complex Reasoning",
    "shape_difference": "Complex Reasoning",
    "non_greedy": "Complex Reasoning",
    "structural_analysis": "Complex Reasoning",
    "clearing_paths": "Complex Reasoning",
    "adequate_timing": "Complex Reasoning",
    "manoeuvring": "Complex Reasoning",
}


def load_csv(filepath: str) -> List[Dict]:
    """Load results from CSV file."""
    results = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)
    return results


def compute_phyq_score(results: List[Dict], agent: str = "PDDLAgent") -> Dict:
    """
    Compute Phy-Q score for a specific agent.
    
    For agents with Win/Lose results: calculates pass rate from wins/losses.
    For baseline agents with numeric pass rates: uses the pass rate directly.
    
    Returns dict with:
        - phyq_score: Overall score (0-100)
        - scenario_pass_rates: Pass rate per scenario
        - category_pass_rates: Pass rate per category
        - total_wins, total_losses
    """
    scenario_stats = defaultdict(lambda: {"wins": 0, "losses": 0})
    scenario_pass_rates_direct = {}
    
    for row in results:
        if row.get("agent") != agent:
            continue
        
        scenario = row.get("scenario", "unknown")
        result = row.get("result", "")
        
        if result == "Win":
            scenario_stats[scenario]["wins"] += 1
        elif result == "Lose":
            scenario_stats[scenario]["losses"] += 1
        else:
            try:
                pass_rate = float(result)
                if scenario not in scenario_pass_rates_direct:
                    scenario_pass_rates_direct[scenario] = pass_rate
            except ValueError:
                pass
    
    scenario_pass_rates = {}
    for scenario in SCENARIO_ORDER:
        if scenario in scenario_pass_rates_direct:
            scenario_pass_rates[scenario] = scenario_pass_rates_direct[scenario]
        else:
            stats = scenario_stats.get(scenario, {"wins": 0, "losses": 0})
            total = stats["wins"] + stats["losses"]
            if total > 0:
                scenario_pass_rates[scenario] = stats["wins"] / total
    
    category_stats = defaultdict(lambda: {"wins": 0, "losses": 0})
    for scenario, stats in scenario_stats.items():
        category = CATEGORY_MAP.get(scenario, "Unknown")
        category_stats[category]["wins"] += stats["wins"]
        category_stats[category]["losses"] += stats["losses"]
    
    category_pass_rates = {}
    for category, stats in category_stats.items():
        total = stats["wins"] + stats["losses"]
        if total > 0:
            category_pass_rates[category] = stats["wins"] / total
    
    if scenario_pass_rates:
        phyq_score = (sum(scenario_pass_rates.values()) / len(scenario_pass_rates)) * 100
    else:
        phyq_score = 0.0
    
    total_wins = sum(s["wins"] for s in scenario_stats.values())
    total_losses = sum(s["losses"] for s in scenario_stats.values())
    
    return {
        "phyq_score": phyq_score,
        "scenario_pass_rates": scenario_pass_rates,
        "category_pass_rates": category_pass_rates,
        "scenario_stats": dict(scenario_stats),
        "total_wins": total_wins,
        "total_losses": total_losses,
        "total_levels": total_wins + total_losses
    }


def get_human_baseline(results: List[Dict]) -> Dict[str, float]:
    """Extract human pass rates from the CSV (they're stored as the result value)."""
    human_rates = {}
    for row in results:
        if row.get("agent") == "Human":
            scenario = row.get("scenario", "unknown")
            try:
                rate = float(row.get("result", "0"))
                human_rates[scenario] = rate
            except ValueError:
                pass
    return human_rates


def print_report(agent_stats: Dict, human_rates: Dict[str, float], agent: str = "PDDLAgent"):
    """Print a formatted Phy-Q report."""
    print("\n" + "=" * 70)
    print(f"PHY-Q SCORE REPORT - {agent}")
    print("=" * 70)
    
    print(f"\nOverall Phy-Q Score: {agent_stats['phyq_score']:.2f} / 100")
    print(f"Total Levels: {agent_stats['total_levels']}")
    print(f"Wins: {agent_stats['total_wins']} | Losses: {agent_stats['total_losses']}")
    if agent_stats['total_levels'] > 0:
        overall_rate = agent_stats['total_wins'] / agent_stats['total_levels'] * 100
        print(f"Overall Pass Rate: {overall_rate:.1f}%")
    
    print("\n" + "-" * 70)
    print("CATEGORY PERFORMANCE")
    print("-" * 70)
    for category in ["Force", "Motion", "Complex Reasoning"]:
        if category in agent_stats["category_pass_rates"]:
            rate = agent_stats["category_pass_rates"][category] * 100
            print(f"  {category}: {rate:.1f}%")
    
    print("\n" + "-" * 70)
    print(f"{'SCENARIO':<25} {'AGENT':>10} {'HUMAN':>10} {'DIFF':>10}")
    print("-" * 70)
    
    for scenario in SCENARIO_ORDER:
        agent_rate = agent_stats["scenario_pass_rates"].get(scenario)
        human_rate = human_rates.get(scenario)
        
        if agent_rate is not None:
            stats = agent_stats["scenario_stats"].get(scenario, {})
            wins = stats.get("wins", 0)
            total = wins + stats.get("losses", 0)
            agent_str = f"{agent_rate*100:5.1f}% ({wins}/{total})"
        else:
            agent_str = "N/A"
        
        if human_rate is not None:
            human_str = f"{human_rate*100:5.1f}%"
        else:
            human_str = "N/A"
        
        if agent_rate is not None and human_rate is not None:
            diff = (agent_rate - human_rate) * 100
            diff_str = f"{diff:+5.1f}%"
        else:
            diff_str = ""
        
        print(f"  {scenario:<23} {agent_str:>10} {human_str:>10} {diff_str:>10}")
    
    print("=" * 70 + "\n")


def get_all_agents(results: List[Dict]) -> List[str]:
    """Get list of all agents in the results (excluding Human)."""
    agents = set()
    for row in results:
        agent = row.get("agent", "")
        if agent and agent != "Human":
            agents.add(agent)
    return sorted(agents)


def print_comparison_table(results: List[Dict], agents: List[str], human_rates: Dict[str, float]):
    """Print a comparison table of all agents."""
    
    key_agents = ["PDDLAgent", "Human", "Datalab", "Bambirds", "Eagle's Wing"]
    display_agents = [a for a in key_agents if a in agents or a == "Human"]
    
    col_width = 12
    print("\n" + "=" * (20 + len(display_agents) * (col_width + 1)))
    print("AGENT COMPARISON TABLE")
    print("=" * (20 + len(display_agents) * (col_width + 1)))
    
    header = f"{'SCENARIO':<20}"
    for agent in display_agents:
        header += f" {agent:>{col_width}}"
    print(header)
    print("-" * (20 + len(display_agents) * (col_width + 1)))
    
    for scenario in SCENARIO_ORDER:
        row = f"{scenario:<20}"
        for agent in display_agents:
            if agent == "Human":
                human_rate = human_rates.get(scenario)
                if human_rate is not None:
                    row += f" {human_rate*100:>{col_width-1}.1f}%"
                else:
                    row += f" {'N/A':>{col_width}}"
            else:
                stats = compute_phyq_score(results, agent)
                rate = stats["scenario_pass_rates"].get(scenario)
                if rate is not None:
                    agent_stats = stats["scenario_stats"].get(scenario, {})
                    wins = agent_stats.get("wins", 0)
                    total = wins + agent_stats.get("losses", 0)
                    if total > 0:
                        row += f" {rate*100:>5.1f}%({total:>2})"
                    else:
                        row += f" {rate*100:>{col_width-1}.1f}%"
                else:
                    row += f" {'N/A':>{col_width}}"
        
        print(row)
    
    print("-" * (20 + len(display_agents) * (col_width + 1)))
    
    row = f"{'PHY-Q SCORE':<20}"
    for agent in display_agents:
        if agent == "Human":
            if human_rates:
                human_phyq = sum(human_rates.values()) / len(human_rates) * 100
                row += f" {human_phyq:>{col_width-1}.1f}%"
            else:
                row += f" {'N/A':>{col_width}}"
        else:
            stats = compute_phyq_score(results, agent)
            row += f" {stats['phyq_score']:>{col_width-1}.1f}%"
    print(row)
    print("=" * (20 + len(display_agents) * (col_width + 1)) + "\n")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    default_csv = os.path.join(project_root, "agent_comparison_results.csv")
    
    csv_path = sys.argv[1] if len(sys.argv) > 1 else default_csv
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found: {csv_path}")
        print("\nYou need to run the PDDL agent first to generate results.")
        print("Run: python main.py")
        print("\nOr specify a different CSV file:")
        print("  python agents/pddl/compute_phyq_from_csv.py path/to/results.csv")
        return {}
    
    print(f"Loading results from: {csv_path}")
    results = load_csv(csv_path)
    print(f"Loaded {len(results)} rows")
    
    agents = get_all_agents(results)
    human_rates = get_human_baseline(results)
    
    if "PDDLAgent" in agents:
        agent_stats = compute_phyq_score(results, "PDDLAgent")
        print_report(agent_stats, human_rates, "PDDLAgent")
    
    if len(agents) > 1:
        print_comparison_table(results, agents, human_rates)
    
    return compute_phyq_score(results, "PDDLAgent") if "PDDLAgent" in agents else {}


if __name__ == "__main__":
    main()
