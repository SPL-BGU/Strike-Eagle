#!/usr/bin/env python3
"""
Import Phy-Q benchmark levels into Strike-Eagle.

This script copies and organizes 7,500 levels from the Phy-Q benchmark
into the Strike-Eagle level structure, creating train/test splits.

Usage:
    python scripts/import_phyq_levels.py [--train-ratio 0.8] [--dry-run]
"""

import os
import shutil
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple


SCENARIO_MAP: Dict[Tuple[int, int], Tuple[int, str]] = {
    (1, 1): (1, "single_force"),
    (1, 2): (2, "multiple_forces"),
    (2, 1): (3, "rolling"),
    (2, 2): (4, "falling"),
    (2, 3): (5, "sliding"),
    (2, 4): (6, "bouncing"),
    (3, 1): (7, "relative_weight"),
    (3, 2): (8, "relative_height"),
    (3, 3): (9, "relative_width"),
    (3, 4): (10, "shape_difference"),
    (3, 5): (11, "non_greedy"),
    (3, 6): (12, "structural_analysis"),
    (3, 7): (13, "clearing_paths"),
    (3, 8): (14, "adequate_timing"),
    (3, 9): (15, "manoeuvring"),
}

SCENARIO_DESCRIPTIONS = {
    "single_force": "Some target objects can be destroyed with a single force",
    "multiple_forces": "Some target objects need multiple forces to destroy",
    "rolling": "Circular objects can be rolled along a surface to a target",
    "falling": "Objects can be fallen on to a target",
    "sliding": "Non-circular objects can be slid along a surface to a target",
    "bouncing": "Objects can be bounced off a surface to reach a target",
    "relative_weight": "Objects with correct weight need to be moved to reach a target",
    "relative_height": "Objects with correct height need to be moved to reach a target",
    "relative_width": "Objects with correct width or opening should be selected",
    "shape_difference": "Objects with correct shape need to be moved/destroyed",
    "non_greedy": "Actions need to be selected in the correct order based on physical consequences",
    "structural_analysis": "The correct target needs to be chosen to break the stability of a structure",
    "clearing_paths": "A path needs to be created before the target can be reached",
    "adequate_timing": "Correct actions need to be performed within time constraints",
    "manoeuvring": "Powers of objects need to be activated correctly to reach a target",
}


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def get_phyq_source_dir() -> Path:
    """Get the Phy-Q generated tasks source directory."""
    return get_project_root() / "external" / "phy-q" / "tasks" / "generated_tasks"


def get_levels_dest_dir() -> Path:
    """Get the destination directory for Phy-Q levels."""
    return get_project_root() / "ScienceBirds" / "win6.6" / "win" / "Levels" / "phy_q"


def discover_levels(source_dir: Path) -> Dict[Tuple[int, int, int], List[Path]]:
    """
    Discover all level files organized by (category, scenario, template).
    
    Returns:
        Dictionary mapping (category, scenario, template) to list of level file paths
    """
    levels = {}
    
    for category_dir in sorted(source_dir.iterdir()):
        if not category_dir.is_dir():
            continue
        try:
            category = int(category_dir.name)
        except ValueError:
            continue
            
        for scenario_dir in sorted(category_dir.iterdir()):
            if not scenario_dir.is_dir():
                continue
            try:
                scenario = int(scenario_dir.name)
            except ValueError:
                continue
                
            for template_dir in sorted(scenario_dir.iterdir()):
                if not template_dir.is_dir():
                    continue
                try:
                    template = int(template_dir.name)
                except ValueError:
                    continue
                
                xml_files = sorted(template_dir.glob("*.xml"))
                if xml_files:
                    levels[(category, scenario, template)] = xml_files
    
    return levels


def create_directory_structure(dest_dir: Path, dry_run: bool = False) -> None:
    """Create the destination directory structure."""
    for (cat, scen), (num, name) in SCENARIO_MAP.items():
        scenario_dir = dest_dir / f"scenario_{num:02d}_{name}"
        train_dir = scenario_dir / "train"
        test_dir = scenario_dir / "test"
        
        if not dry_run:
            train_dir.mkdir(parents=True, exist_ok=True)
            test_dir.mkdir(parents=True, exist_ok=True)
        else:
            print(f"Would create: {train_dir}")
            print(f"Would create: {test_dir}")


def copy_levels(
    levels: Dict[Tuple[int, int, int], List[Path]],
    dest_dir: Path,
    train_ratio: float = 0.8,
    dry_run: bool = False
) -> Dict:
    """
    Copy levels to destination with train/test split.
    
    Returns:
        Manifest dictionary with statistics
    """
    manifest = {
        "total_levels": 0,
        "train_levels": 0,
        "test_levels": 0,
        "scenarios": {},
        "templates": {}
    }
    
    for (category, scenario, template), file_list in sorted(levels.items()):
        key = (category, scenario)
        if key not in SCENARIO_MAP:
            print(f"Warning: Unknown scenario ({category}, {scenario}), skipping")
            continue
            
        num, name = SCENARIO_MAP[key]
        scenario_dir = dest_dir / f"scenario_{num:02d}_{name}"
        train_dir = scenario_dir / "train"
        test_dir = scenario_dir / "test"
        
        split_idx = int(len(file_list) * train_ratio)
        train_files = file_list[:split_idx]
        test_files = file_list[split_idx:]
        
        scenario_key = f"scenario_{num:02d}_{name}"
        template_key = f"{scenario_key}_template_{template:02d}"
        
        if scenario_key not in manifest["scenarios"]:
            manifest["scenarios"][scenario_key] = {
                "name": name,
                "description": SCENARIO_DESCRIPTIONS.get(name, ""),
                "category": category,
                "scenario_in_category": scenario,
                "train_count": 0,
                "test_count": 0,
                "templates": []
            }
        
        manifest["scenarios"][scenario_key]["templates"].append(template)
        manifest["scenarios"][scenario_key]["train_count"] += len(train_files)
        manifest["scenarios"][scenario_key]["test_count"] += len(test_files)
        
        manifest["templates"][template_key] = {
            "scenario": scenario_key,
            "template_num": template,
            "train_count": len(train_files),
            "test_count": len(test_files),
            "train_files": [],
            "test_files": []
        }
        
        for i, src_file in enumerate(train_files, 1):
            new_name = f"{name}_t{template:02d}_{i:05d}.xml"
            dest_file = train_dir / new_name
            
            if not dry_run:
                shutil.copy2(src_file, dest_file)
            else:
                print(f"Would copy: {src_file.name} -> {dest_file}")
            
            manifest["templates"][template_key]["train_files"].append(new_name)
            manifest["total_levels"] += 1
            manifest["train_levels"] += 1
        
        for i, src_file in enumerate(test_files, 1):
            new_name = f"{name}_t{template:02d}_{i:05d}.xml"
            dest_file = test_dir / new_name
            
            if not dry_run:
                shutil.copy2(src_file, dest_file)
            else:
                print(f"Would copy: {src_file.name} -> {dest_file}")
            
            manifest["templates"][template_key]["test_files"].append(new_name)
            manifest["total_levels"] += 1
            manifest["test_levels"] += 1
    
    return manifest


def save_manifest(manifest: Dict, dest_dir: Path, dry_run: bool = False) -> None:
    """Save the manifest file."""
    manifest_path = dest_dir / "manifest.json"
    
    manifest_summary = {
        "total_levels": manifest["total_levels"],
        "train_levels": manifest["train_levels"],
        "test_levels": manifest["test_levels"],
        "scenario_count": len(manifest["scenarios"]),
        "template_count": len(manifest["templates"]),
        "scenarios": {
            k: {
                "name": v["name"],
                "description": v["description"],
                "train_count": v["train_count"],
                "test_count": v["test_count"],
                "template_count": len(v["templates"])
            }
            for k, v in manifest["scenarios"].items()
        }
    }
    
    if not dry_run:
        with open(manifest_path, "w") as f:
            json.dump(manifest_summary, f, indent=2)
        print(f"Manifest saved to: {manifest_path}")
    else:
        print(f"Would save manifest to: {manifest_path}")
        print(json.dumps(manifest_summary, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Import Phy-Q benchmark levels into Strike-Eagle"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Ratio of levels to use for training (default: 0.8)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without copying files"
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Override source directory for Phy-Q levels"
    )
    parser.add_argument(
        "--dest",
        type=str,
        default=None,
        help="Override destination directory for levels"
    )
    
    args = parser.parse_args()
    
    source_dir = Path(args.source) if args.source else get_phyq_source_dir()
    dest_dir = Path(args.dest) if args.dest else get_levels_dest_dir()
    
    if not source_dir.exists():
        print(f"Error: Source directory does not exist: {source_dir}")
        print("Make sure to clone and extract the Phy-Q benchmark first:")
        print("  git clone https://github.com/phy-q/benchmark.git external/phy-q")
        print("  cd external/phy-q/tasks && unzip generated_tasks.zip")
        return 1
    
    print(f"Source: {source_dir}")
    print(f"Destination: {dest_dir}")
    print(f"Train ratio: {args.train_ratio}")
    print(f"Dry run: {args.dry_run}")
    print()
    
    print("Discovering levels...")
    levels = discover_levels(source_dir)
    
    total_files = sum(len(files) for files in levels.values())
    print(f"Found {len(levels)} templates with {total_files} total level files")
    print()
    
    print("Creating directory structure...")
    create_directory_structure(dest_dir, args.dry_run)
    print()
    
    print("Copying levels with train/test split...")
    manifest = copy_levels(levels, dest_dir, args.train_ratio, args.dry_run)
    print()
    
    print("Summary:")
    print(f"  Total levels: {manifest['total_levels']}")
    print(f"  Train levels: {manifest['train_levels']}")
    print(f"  Test levels: {manifest['test_levels']}")
    print(f"  Scenarios: {len(manifest['scenarios'])}")
    print(f"  Templates: {len(manifest['templates'])}")
    print()
    
    print("Saving manifest...")
    save_manifest(manifest, dest_dir, args.dry_run)
    
    print()
    print("Done!")
    return 0


if __name__ == "__main__":
    exit(main())
