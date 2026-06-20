"""
Test script for the AgentComparisonCSV module.

Run this script to verify CSV output format without needing the full game setup:
    python agents/pddl/test_comparison_csv.py
"""

import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))

sys.path.insert(0, script_dir)
os.chdir(project_root)

from comparison_csv import AgentComparisonCSV, create_comparison_csv


def test_comparison_csv():
    """Test the AgentComparisonCSV class with sample data."""
    
    test_csv_path = "test_agent_comparison.csv"
    human_baseline_path = "external/phy-q/playdata/broad_generalization_all_agents.csv"
    
    if os.path.exists(test_csv_path):
        os.remove(test_csv_path)
    
    print("=" * 60)
    print("TESTING AgentComparisonCSV")
    print("=" * 60)
    
    csv_writer = AgentComparisonCSV(
        output_path=test_csv_path,
        human_baseline_path=human_baseline_path
    )
    
    print(f"\nHuman baselines loaded: {len(csv_writer.human_baselines)}")
    for scenario, rate in list(csv_writer.human_baselines.items())[:5]:
        print(f"  {scenario}: {rate:.3f}")
    
    test_levels = [
        ("./Levels/phy_q/scenario_01_single_force/train/single_force_t01_00001.xml", True, "train"),
        ("./Levels/phy_q/scenario_01_single_force/train/single_force_t01_00002.xml", True, "train"),
        ("./Levels/phy_q/scenario_01_single_force/test/single_force_t02_00050.xml", False, "test"),
        ("./Levels/phy_q/scenario_03_rolling/train/rolling_t01_00041.xml", True, "train"),
        ("./Levels/phy_q/scenario_03_rolling/test/rolling_t02_00080.xml", False, "test"),
        ("./Levels/phy_q/scenario_06_bouncing/train/bouncing_t01_00010.xml", True, "train"),
        ("./Levels/phy_q/scenario_06_bouncing/test/bouncing_t03_00099.xml", False, "test"),
    ]
    
    print("\n" + "-" * 60)
    print("Writing test results...")
    print("-" * 60)
    
    for level_path, won, mode in test_levels:
        csv_writer.write_result(
            level_path=level_path,
            agent="PDDLAgent",
            won=won,
            mode=mode
        )
    
    csv_writer.print_summary()
    
    print("\n" + "-" * 60)
    print("Testing level path parsing...")
    print("-" * 60)
    
    test_path = "./Levels/phy_q/scenario_03_rolling/train/rolling_t01_00041.xml"
    print(f"Test path: {test_path}")
    print(f"  Level name: {csv_writer.extract_level_name(test_path)}")
    print(f"  Scenario: {csv_writer.extract_scenario(test_path)}")
    print(f"  Template: {csv_writer.extract_template(test_path)}")
    print(f"  Variation: {csv_writer.extract_variation(test_path)}")
    
    print("\n" + "-" * 60)
    print(f"CSV file created: {test_csv_path}")
    print("-" * 60)
    
    with open(test_csv_path, 'r') as f:
        print(f.read())
    
    os.remove(test_csv_path)
    print(f"\nTest CSV file cleaned up.")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    test_comparison_csv()
