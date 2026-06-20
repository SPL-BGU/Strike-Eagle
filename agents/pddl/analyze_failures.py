"""
Analyze PDDLAgent failures from the comparison CSV.
"""

import csv
import os
from collections import defaultdict

def analyze_failures():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    csv_path = os.path.join(project_root, "agent_comparison_results.csv")
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV not found at {csv_path}")
        return
    
    losses = []
    wins = []
    seen = set()
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['agent'] == 'PDDLAgent':
                level = row['level_name']
                if level not in seen:
                    seen.add(level)
                    if row['result'] == 'Lose':
                        losses.append(row)
                    else:
                        wins.append(row)
    
    print(f"Unique levels: {len(seen)}")
    print(f"Wins: {len(wins)}, Losses: {len(losses)}")
    if wins or losses:
        print(f"Win rate: {len(wins)/(len(wins)+len(losses))*100:.1f}%")
    
    # Group losses by template
    by_template = defaultdict(list)
    for row in losses:
        by_template[row['template']].append(row)
    
    # Group losses by mode
    by_mode = defaultdict(list)
    for row in losses:
        by_mode[row['mode']].append(row)
    
    print("\n" + "=" * 60)
    print("FAILURE ANALYSIS")
    print("=" * 60)
    
    print("\n--- Losses by Mode ---")
    for mode, rows in sorted(by_mode.items()):
        print(f"  {mode}: {len(rows)} losses")
    
    print("\n--- Losses by Template ---")
    for template, rows in sorted(by_template.items(), key=lambda x: -len(x[1])):
        print(f"  Template {template}: {len(rows)} losses")
        for row in rows:
            print(f"    - {row['level_name']} (variation {row['variation']}, {row['mode']})")
    
    # Find patterns in variation numbers
    print("\n--- Losses by Variation ---")
    by_variation = defaultdict(list)
    for row in losses:
        by_variation[row['variation']].append(row)
    
    for var, rows in sorted(by_variation.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0):
        print(f"  Variation {var}: {len(rows)} losses")
    
    # Compare with human performance on losing levels
    print("\n--- Human Performance on PDDLAgent's Losing Levels ---")
    print("(Note: Human rates are per-scenario, not per-level)")
    
    # Reload to get human rates
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        human_rates = {}
        for row in reader:
            if row['agent'] == 'Human':
                scenario = row['scenario']
                try:
                    human_rates[scenario] = float(row['result'])
                except ValueError:
                    pass
    
    for scenario, rate in human_rates.items():
        print(f"  {scenario}: Human pass rate = {rate*100:.1f}%")
    
    print("\n" + "=" * 60)
    print("DETAILED LOSS LIST")
    print("=" * 60)
    for i, row in enumerate(losses, 1):
        print(f"{i:2}. {row['level_name']}")
        print(f"    Template: {row['template']}, Variation: {row['variation']}, Mode: {row['mode']}")


if __name__ == "__main__":
    analyze_failures()
