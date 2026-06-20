#!/usr/bin/env python3
"""
Generate Science Birds config.xml files for Phy-Q benchmark levels.

This script creates different configuration files for training and evaluation:
- config_phyq_full.xml: All levels for comprehensive evaluation
- config_phyq_sample.xml: Subset of levels for quick experiments
- config_phyq_scenario_XX.xml: Per-scenario configs

Generalization Protocols (from Phy-Q paper):
- LOCAL: 80/20 split within each template (tests within-task generalization)
- BROAD: Train on some templates, test on others (tests cross-task generalization)

Usage:
    python scripts/generate_phyq_configs.py [--sample-size 10]
    python scripts/generate_phyq_configs.py --scenarios rolling bouncing falling
    python scripts/generate_phyq_configs.py --scenarios single_force --sample-size 20
    
    # Local generalization (80/20 within each template)
    python scripts/generate_phyq_configs.py --scenario single_force --generalization local
    
    # Local generalization with 30 levels per template (24 train, 6 test)
    python scripts/generate_phyq_configs.py --scenario single_force --generalization local --levels-per-template 30
    
    # Broad generalization (train on templates 1-4, test on 5-6)
    python scripts/generate_phyq_configs.py --scenario single_force --generalization broad --train-templates 1 2 3 4 --test-templates 5 6
    
    # Broad generalization with limited levels per template
    python scripts/generate_phyq_configs.py --scenario rolling --generalization broad --train-templates 1 2 3 --test-templates 4 5 --levels-per-template 20
"""

import os
import json
import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional
from xml.etree import ElementTree as ET
from xml.dom import minidom


SCENARIOS = [
    (1, "single_force"),
    (2, "multiple_forces"),
    (3, "rolling"),
    (4, "falling"),
    (5, "sliding"),
    (6, "bouncing"),
    (7, "relative_weight"),
    (8, "relative_height"),
    (9, "relative_width"),
    (10, "shape_difference"),
    (11, "non_greedy"),
    (12, "structural_analysis"),
    (13, "clearing_paths"),
    (14, "adequate_timing"),
    (15, "manoeuvring"),
]

# All valid scenario names for validation
VALID_SCENARIO_NAMES = [name for _, name in SCENARIOS]


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def get_levels_dir() -> Path:
    """Get the Phy-Q levels directory."""
    return get_project_root() / "ScienceBirds" / "win6.6" / "win" / "Levels" / "phy_q"


def get_config_dest_dir() -> Path:
    """Get the destination directory for config files."""
    return get_project_root() / "ScienceBirds" / "win6.6" / "win"


def discover_levels(levels_dir: Path) -> Dict[str, Dict[str, List[str]]]:
    """
    Discover all level files organized by scenario.
    
    Returns:
        Dictionary mapping scenario name to {"train": [...], "test": [...]}
    """
    levels = {}
    
    for scenario_dir in sorted(levels_dir.iterdir()):
        if not scenario_dir.is_dir() or not scenario_dir.name.startswith("scenario_"):
            continue
        
        scenario_name = scenario_dir.name
        levels[scenario_name] = {"train": [], "test": []}
        
        train_dir = scenario_dir / "train"
        test_dir = scenario_dir / "test"
        
        if train_dir.exists():
            for xml_file in sorted(train_dir.glob("*.xml")):
                rel_path = f"./Levels/phy_q/{scenario_name}/train/{xml_file.name}"
                levels[scenario_name]["train"].append(rel_path)
        
        if test_dir.exists():
            for xml_file in sorted(test_dir.glob("*.xml")):
                rel_path = f"./Levels/phy_q/{scenario_name}/test/{xml_file.name}"
                levels[scenario_name]["test"].append(rel_path)
    
    return levels


def prettify_xml(elem: ET.Element) -> str:
    """Return a pretty-printed XML string."""
    rough_string = ET.tostring(elem, encoding='unicode')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def create_config_xml(
    train_levels: List[str],
    test_levels: List[str],
    time_limit: int = 12000,
    attempt_limit: int = 1,
    interaction_limit: int = 100,
    checkpoint_time_limit: int = 200,
    checkpoint_interaction_limit: int = 200,
    interleaved: bool = True
) -> ET.Element:
    """
    Create a Science Birds config.xml structure.
    
    Args:
        train_levels: List of training level paths
        test_levels: List of test level paths
        time_limit: Time limit in seconds
        attempt_limit: Maximum attempts per level
        interaction_limit: Total interaction limit per level
        checkpoint_time_limit: Checkpoint time limit
        checkpoint_interaction_limit: Checkpoint interaction limit
        interleaved: If True, creates single trial with all levels in order
    
    Returns:
        ElementTree Element for the config
    """
    evaluation = ET.Element("evaluation")
    
    # Add novelty detection measurement element (matches original config)
    ET.SubElement(
        evaluation, 
        "novelty_detection_measurement",
        step="1",
        measure_in_training="True",
        measure_in_testing="True"
    )
    
    trials = ET.SubElement(evaluation, "trials")
    
    if interleaved:
        # Single trial with all levels (agent determines train/test from metadata)
        all_levels = train_levels + test_levels
        
        single_trial = ET.SubElement(
            trials, 
            "trial", 
            id="0",
            number_of_executions="1",
            checkpoint_time_limit=str(checkpoint_time_limit),
            checkpoint_interaction_limit=str(checkpoint_interaction_limit),
            notify_novelty="True"
        )
        level_set = ET.SubElement(
            single_trial, 
            "game_level_set",
            mode="training",  # Mode doesn't matter - agent uses metadata
            time_limit=str(time_limit),
            total_interaction_limit=str(interaction_limit * len(all_levels)),
            attempt_limit_per_level=str(attempt_limit),
            allow_level_selection="True"
        )
        for level_path in all_levels:
            ET.SubElement(level_set, "game_levels", level_path=level_path)
    else:
        # Original: separate trials for train and test
        if train_levels:
            train_trial = ET.SubElement(
                trials, 
                "trial", 
                id="0",
                number_of_executions="1",
                checkpoint_time_limit=str(checkpoint_time_limit),
                checkpoint_interaction_limit=str(checkpoint_interaction_limit),
                notify_novelty="True"
            )
            train_set = ET.SubElement(
                train_trial, 
                "game_level_set",
                mode="training",
                time_limit=str(time_limit),
                total_interaction_limit=str(interaction_limit * len(train_levels)),
                attempt_limit_per_level=str(attempt_limit),
                allow_level_selection="True"
            )
            for level_path in train_levels:
                ET.SubElement(train_set, "game_levels", level_path=level_path)
        
        if test_levels:
            test_trial = ET.SubElement(
                trials, 
                "trial", 
                id="1",
                number_of_executions="1",
                checkpoint_time_limit=str(checkpoint_time_limit),
                checkpoint_interaction_limit=str(checkpoint_interaction_limit),
                notify_novelty="True"
            )
            test_set = ET.SubElement(
                test_trial,
                "game_level_set",
                mode="testing",
                time_limit=str(time_limit),
                total_interaction_limit=str(interaction_limit * len(test_levels)),
                attempt_limit_per_level=str(attempt_limit),
                allow_level_selection="False"
            )
            for level_path in test_levels:
                ET.SubElement(test_set, "game_levels", level_path=level_path)
    
    return evaluation


def write_config(config: ET.Element, output_path: Path, dry_run: bool = False) -> None:
    """Write config XML to file."""
    xml_string = prettify_xml(config)
    xml_string = xml_string.replace('<?xml version="1.0" ?>\n', '<?xml version="1.0" encoding="utf-16"?>\n')
    
    if not dry_run:
        with open(output_path, 'w', encoding='utf-16') as f:
            f.write(xml_string)
        print(f"Written: {output_path}")
    else:
        print(f"Would write: {output_path}")


def write_config_metadata(
    output_path: Path,
    generalization_type: str,
    train_ratio: float = None,
    train_templates: List[int] = None,
    test_templates: List[int] = None,
    levels_per_template: int = None,
    seed: int = 42,
    scenario_filter: str = None,
    train_count: int = 0,
    test_count: int = 0,
    train_levels: List[str] = None,
    test_levels: List[str] = None,
    dry_run: bool = False
) -> None:
    """
    Write metadata JSON file alongside the config XML.
    
    This file can be read by the agent to automatically sync parameters.
    The metadata file will have the same name as the XML but with .meta.json extension.
    """
    metadata = {
        "config_xml": output_path.name,
        "generalization_type": generalization_type,
        "seed": seed,
        "train_count": train_count,
        "test_count": test_count,
    }
    
    if train_ratio is not None:
        metadata["train_ratio"] = train_ratio
    if train_templates is not None:
        metadata["train_templates"] = train_templates
    if test_templates is not None:
        metadata["test_templates"] = test_templates
    if levels_per_template is not None:
        metadata["levels_per_template"] = levels_per_template
    if scenario_filter is not None:
        metadata["scenario_filter"] = scenario_filter
    if train_levels is not None:
        metadata["train_levels"] = train_levels
    if test_levels is not None:
        metadata["test_levels"] = test_levels
    
    # Create metadata filename (replace .xml with .meta.json)
    meta_path = output_path.with_suffix('.meta.json')
    
    if not dry_run:
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        print(f"Written: {meta_path}")
    else:
        print(f"Would write: {meta_path}")


def generate_full_config(
    levels: Dict[str, Dict[str, List[str]]],
    output_dir: Path,
    dry_run: bool = False,
    name_suffix: Optional[str] = None,
    shuffle: bool = False
) -> None:
    """Generate config with all levels."""
    all_train = []
    all_test = []
    
    for scenario_name in sorted(levels.keys()):
        all_train.extend(levels[scenario_name]["train"])
        all_test.extend(levels[scenario_name]["test"])
    
    if shuffle:
        random.shuffle(all_train)
        random.shuffle(all_test)
    
    config = create_config_xml(all_train, all_test)
    filename = f"config_phyq_{name_suffix}_full.xml" if name_suffix else "config_phyq_full.xml"
    output_path = output_dir / filename
    write_config(config, output_path, dry_run)
    
    print(f"  Train levels: {len(all_train)}")
    print(f"  Test levels: {len(all_test)}")


def generate_sample_config(
    levels: Dict[str, Dict[str, List[str]]],
    output_dir: Path,
    sample_size: int = 10,
    dry_run: bool = False,
    name_suffix: Optional[str] = None,
    shuffle: bool = False
) -> None:
    """Generate config with a sample of levels from each template."""
    sample_train = []
    sample_test = []
    
    for scenario_name in sorted(levels.keys()):
        train_files = levels[scenario_name]["train"]
        test_files = levels[scenario_name]["test"]
        
        templates = {}
        for f in train_files:
            template = f.split("_t")[1].split("_")[0] if "_t" in f else "00"
            if template not in templates:
                templates[template] = {"train": [], "test": []}
            templates[template]["train"].append(f)
        
        for f in test_files:
            template = f.split("_t")[1].split("_")[0] if "_t" in f else "00"
            if template not in templates:
                templates[template] = {"train": [], "test": []}
            templates[template]["test"].append(f)
        
        for template, files in templates.items():
            sample_train.extend(files["train"][:sample_size])
            sample_test.extend(files["test"][:max(1, sample_size // 4)])
    
    if shuffle:
        random.shuffle(sample_train)
        random.shuffle(sample_test)
    
    config = create_config_xml(sample_train, sample_test)
    filename = f"config_phyq_{name_suffix}_sample.xml" if name_suffix else "config_phyq_sample.xml"
    output_path = output_dir / filename
    write_config(config, output_path, dry_run)
    
    print(f"  Train levels: {len(sample_train)}")
    print(f"  Test levels: {len(sample_test)}")


def generate_scenario_configs(
    levels: Dict[str, Dict[str, List[str]]],
    output_dir: Path,
    dry_run: bool = False,
    shuffle: bool = False
) -> None:
    """Generate per-scenario config files."""
    for scenario_name in sorted(levels.keys()):
        train_levels = list(levels[scenario_name]["train"])
        test_levels = list(levels[scenario_name]["test"])
        
        if not train_levels and not test_levels:
            continue
        
        if shuffle:
            random.shuffle(train_levels)
            random.shuffle(test_levels)
        
        config = create_config_xml(train_levels, test_levels)
        output_path = output_dir / f"config_phyq_{scenario_name}.xml"
        write_config(config, output_path, dry_run)


def generate_train_only_config(
    levels: Dict[str, Dict[str, List[str]]],
    output_dir: Path,
    dry_run: bool = False,
    name_suffix: Optional[str] = None,
    shuffle: bool = False
) -> None:
    """Generate config with only training levels (for development)."""
    all_train = []
    
    for scenario_name in sorted(levels.keys()):
        all_train.extend(levels[scenario_name]["train"])
    
    if shuffle:
        random.shuffle(all_train)
    
    config = create_config_xml(all_train, [])
    filename = f"config_phyq_{name_suffix}_train_only.xml" if name_suffix else "config_phyq_train_only.xml"
    output_path = output_dir / filename
    write_config(config, output_path, dry_run)
    
    print(f"  Train levels: {len(all_train)}")


def generate_test_only_config(
    levels: Dict[str, Dict[str, List[str]]],
    output_dir: Path,
    dry_run: bool = False,
    name_suffix: Optional[str] = None,
    shuffle: bool = False
) -> None:
    """Generate config with only test levels (for final evaluation)."""
    all_test = []
    
    for scenario_name in sorted(levels.keys()):
        all_test.extend(levels[scenario_name]["test"])
    
    if shuffle:
        random.shuffle(all_test)
    
    config = create_config_xml([], all_test)
    filename = f"config_phyq_{name_suffix}_test_only.xml" if name_suffix else "config_phyq_test_only.xml"
    output_path = output_dir / filename
    write_config(config, output_path, dry_run)
    
    print(f"  Test levels: {len(all_test)}")


def extract_template_from_path(path: str) -> int:
    """Extract template number from level path."""
    import re
    match = re.search(r'_t(\d+)_', path)
    return int(match.group(1)) if match else 0


def generate_local_generalization_config(
    levels: Dict[str, Dict[str, List[str]]],
    output_dir: Path,
    train_ratio: float = 0.8,
    dry_run: bool = False,
    name_suffix: Optional[str] = None,
    shuffle: bool = True,
    levels_per_template: Optional[int] = None,
    seed: int = 42,
    scenario_filter: Optional[str] = None
) -> None:
    """
    Generate config for LOCAL generalization: 80/20 split within each template.
    
    This tests within-task generalization - can the agent solve new instances
    of problem types it has seen during training?
    
    Order: Templates are kept as batches, but template order is shuffled.
    The agent will interleave train/test per template at runtime.
    
    Args:
        levels: Dictionary of levels by scenario
        output_dir: Directory to write config file
        train_ratio: Fraction of levels per template for training (default: 0.8)
        dry_run: If True, don't write file
        name_suffix: Suffix for output filename
        shuffle: If True, shuffle levels (default: True)
        levels_per_template: If specified, limit levels per template to this number
        seed: Random seed for reproducibility
        scenario_filter: Scenario name if filtering was applied
    """
    # Combine all levels from train and test folders
    all_levels_by_template: Dict[int, List[str]] = {}
    
    for scenario_name in sorted(levels.keys()):
        for level_path in levels[scenario_name]["train"] + levels[scenario_name]["test"]:
            template = extract_template_from_path(level_path)
            if template not in all_levels_by_template:
                all_levels_by_template[template] = []
            all_levels_by_template[template].append(level_path)
    
    # Collect train/test levels per template
    train_by_template: Dict[int, List[str]] = {}
    test_by_template: Dict[int, List[str]] = {}
    
    limit_str = f", max {levels_per_template} per template" if levels_per_template else ""
    print(f"  Local Generalization Split ({train_ratio:.0%} train / {1-train_ratio:.0%} test{limit_str}):")
    
    for template in sorted(all_levels_by_template.keys()):
        template_levels = all_levels_by_template[template].copy()
        
        # Shuffle within template to randomly select train/test
        if shuffle:
            random.shuffle(template_levels)
        
        # Apply levels_per_template limit if specified
        if levels_per_template and len(template_levels) > levels_per_template:
            template_levels = template_levels[:levels_per_template]
        
        split_idx = int(len(template_levels) * train_ratio)
        train_by_template[template] = template_levels[:split_idx]
        test_by_template[template] = template_levels[split_idx:]
        
        available = len(all_levels_by_template[template])
        used = len(template_levels)
        limit_info = f" (using {used}/{available})" if levels_per_template and used < available else ""
        print(f"    Template {template}: {len(train_by_template[template])} train, {len(test_by_template[template])} test{limit_info}")
    
    # Shuffle template order (but keep levels within each template together)
    template_order = list(all_levels_by_template.keys())
    if shuffle:
        random.shuffle(template_order)
    
    print(f"  Template order: {template_order}")
    
    # Build interleaved list: t1_train -> t1_test -> t2_train -> t2_test -> ...
    # This is the actual order levels will be played
    all_levels_interleaved = []
    train_levels = []  # For metadata
    test_levels = []   # For metadata
    
    for template in template_order:
        template_train = train_by_template[template]
        template_test = test_by_template[template]
        
        # Add train levels for this template
        all_levels_interleaved.extend(template_train)
        train_levels.extend(template_train)
        
        # Add test levels for this template
        all_levels_interleaved.extend(template_test)
        test_levels.extend(template_test)
    
    print(f"  Interleaved order: {len(all_levels_interleaved)} levels")
    print(f"    Example: {all_levels_interleaved[:3]}... -> ...{all_levels_interleaved[-3:]}")
    
    # Create config with single trial containing all levels in interleaved order
    config = create_config_xml(all_levels_interleaved, [], interleaved=True)
    filename = f"config_phyq_{name_suffix}_local.xml" if name_suffix else "config_phyq_local.xml"
    output_path = output_dir / filename
    write_config(config, output_path, dry_run)
    
    # Write metadata file for agent to read
    write_config_metadata(
        output_path=output_path,
        generalization_type="local",
        train_ratio=train_ratio,
        levels_per_template=levels_per_template,
        seed=seed,
        scenario_filter=scenario_filter,
        train_count=len(train_levels),
        test_count=len(test_levels),
        train_levels=train_levels,
        test_levels=test_levels,
        dry_run=dry_run
    )
    
    print(f"  Total train levels: {len(train_levels)}")
    print(f"  Total test levels: {len(test_levels)}")


def generate_broad_generalization_config(
    levels: Dict[str, Dict[str, List[str]]],
    output_dir: Path,
    train_templates: List[int],
    test_templates: List[int],
    dry_run: bool = False,
    name_suffix: Optional[str] = None,
    shuffle: bool = True,
    levels_per_template: Optional[int] = None,
    seed: int = 42,
    scenario_filter: Optional[str] = None
) -> None:
    """
    Generate config for BROAD generalization: train on some templates, test on others.
    
    This tests cross-task generalization - can the agent transfer the underlying
    physical reasoning rule to new problem structures it hasn't seen?
    
    Args:
        levels: Dictionary of levels by scenario
        output_dir: Directory to write config file
        train_templates: List of template numbers to use for training
        test_templates: List of template numbers to use for testing
        dry_run: If True, don't write file
        name_suffix: Suffix for output filename
        shuffle: If True, shuffle levels (default: True)
        levels_per_template: If specified, limit levels per template to this number
        seed: Random seed for reproducibility
        scenario_filter: Scenario name if filtering was applied
    """
    # Combine all levels from train and test folders
    all_levels_by_template: Dict[int, List[str]] = {}
    
    for scenario_name in sorted(levels.keys()):
        for level_path in levels[scenario_name]["train"] + levels[scenario_name]["test"]:
            template = extract_template_from_path(level_path)
            if template not in all_levels_by_template:
                all_levels_by_template[template] = []
            all_levels_by_template[template].append(level_path)
    
    # Validate templates
    available_templates = set(all_levels_by_template.keys())
    invalid_train = set(train_templates) - available_templates
    invalid_test = set(test_templates) - available_templates
    
    if invalid_train:
        print(f"  Warning: Train templates not found: {invalid_train}")
    if invalid_test:
        print(f"  Warning: Test templates not found: {invalid_test}")
    
    # Split by template
    train_levels = []
    test_levels = []
    
    limit_str = f" (max {levels_per_template} per template)" if levels_per_template else ""
    print(f"  Broad Generalization Split{limit_str}:")
    print(f"    Train templates: {train_templates}")
    print(f"    Test templates: {test_templates}")
    print()
    
    for template in sorted(all_levels_by_template.keys()):
        template_levels = all_levels_by_template[template].copy()
        available = len(template_levels)
        
        if shuffle:
            random.shuffle(template_levels)
        
        # Apply levels_per_template limit if specified
        if levels_per_template and len(template_levels) > levels_per_template:
            template_levels = template_levels[:levels_per_template]
        
        used = len(template_levels)
        limit_info = f" (using {used}/{available})" if levels_per_template and used < available else ""
        
        if template in train_templates:
            train_levels.extend(template_levels)
            print(f"    Template {template}: {len(template_levels)} levels -> TRAIN{limit_info}")
        elif template in test_templates:
            test_levels.extend(template_levels)
            print(f"    Template {template}: {len(template_levels)} levels -> TEST{limit_info}")
        else:
            print(f"    Template {template}: {available} levels -> EXCLUDED")
    
    # Final shuffle
    if shuffle:
        random.shuffle(train_levels)
        random.shuffle(test_levels)
    
    config = create_config_xml(train_levels, test_levels)
    filename = f"config_phyq_{name_suffix}_broad.xml" if name_suffix else "config_phyq_broad.xml"
    output_path = output_dir / filename
    write_config(config, output_path, dry_run)
    
    # Write metadata file for agent to read
    write_config_metadata(
        output_path=output_path,
        generalization_type="broad",
        train_templates=train_templates,
        test_templates=test_templates,
        levels_per_template=levels_per_template,
        seed=seed,
        scenario_filter=scenario_filter,
        train_count=len(train_levels),
        test_count=len(test_levels),
        dry_run=dry_run
    )
    
    print()
    print(f"  Total train levels: {len(train_levels)}")
    print(f"  Total test levels: {len(test_levels)}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Science Birds config files for Phy-Q levels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all standard configs for single_force
  python scripts/generate_phyq_configs.py --scenario single_force
  
  # Local generalization (80/20 within each template)
  python scripts/generate_phyq_configs.py --scenario single_force --generalization local
  
  # Local generalization with 70/30 split
  python scripts/generate_phyq_configs.py --scenario single_force --generalization local --train-ratio 0.7
  
  # Local generalization with 30 levels per template (24 train, 6 test per template)
  python scripts/generate_phyq_configs.py --scenario single_force --generalization local --levels-per-template 30
  
  # Broad generalization (train on templates 1-4, test on 5-6)
  python scripts/generate_phyq_configs.py --scenario single_force --generalization broad --train-templates 1 2 3 4 --test-templates 5 6
  
  # Broad generalization with limited levels per template
  python scripts/generate_phyq_configs.py --scenario rolling --generalization broad --train-templates 1 2 3 --test-templates 4 5 --levels-per-template 20
  
  # Only include specific templates (e.g., templates 1 and 2)
  python scripts/generate_phyq_configs.py --scenario single_force --generalization local --templates 1 2
  
  # Single template with limited levels
  python scripts/generate_phyq_configs.py --scenario single_force --generalization local --templates 1 --levels-per-template 10
"""
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10,
        help="Number of levels per template for sample config (default: 10)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without writing files"
    )
    parser.add_argument(
        "--levels-dir",
        type=str,
        default=None,
        help="Override levels directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory for config files"
    )
    parser.add_argument(
        "--scenario-configs",
        action="store_true",
        help="Also generate per-scenario config files"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        choices=VALID_SCENARIO_NAMES,
        help=f"Single scenario to include. Available: {', '.join(VALID_SCENARIO_NAMES)}"
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=None,
        metavar="SCENARIO",
        help=f"Multiple scenarios to include (space-separated). Available: {', '.join(VALID_SCENARIO_NAMES)}"
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Custom name suffix for output files (e.g., 'motion' creates config_phyq_motion_*.xml)"
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Randomize the order of levels in the generated configs"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible shuffling (default: 42)"
    )
    
    # Generalization protocol options
    parser.add_argument(
        "--generalization",
        type=str,
        choices=["local", "broad"],
        default=None,
        help="Generalization protocol: 'local' (80/20 within template) or 'broad' (train/test on different templates)"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="For local generalization: train/test split ratio (default: 0.8 = 80%% train)"
    )
    parser.add_argument(
        "--train-templates",
        type=int,
        nargs="+",
        default=None,
        help="For broad generalization: template numbers to train on (e.g., --train-templates 1 2 3 4)"
    )
    parser.add_argument(
        "--test-templates",
        type=int,
        nargs="+",
        default=None,
        help="For broad generalization: template numbers to test on (e.g., --test-templates 5 6)"
    )
    parser.add_argument(
        "--levels-per-template",
        type=int,
        default=None,
        help="Limit the number of levels per template (e.g., --levels-per-template 30). "
             "If not specified, uses all available levels."
    )
    parser.add_argument(
        "--templates",
        type=int,
        nargs="+",
        default=None,
        help="Only include specific template numbers (e.g., --templates 1 2 3). "
             "If not specified, uses all available templates."
    )
    
    args = parser.parse_args()
    
    # Handle --scenario as single item for --scenarios
    if args.scenario and not args.scenarios:
        args.scenarios = [args.scenario]
    
    # Validate scenario names if provided
    if args.scenarios:
        invalid = [s for s in args.scenarios if s not in VALID_SCENARIO_NAMES]
        if invalid:
            print(f"Error: Invalid scenario names: {invalid}")
            print(f"Valid scenarios: {', '.join(VALID_SCENARIO_NAMES)}")
            return 1
    
    # Validate broad generalization arguments
    if args.generalization == "broad":
        if not args.train_templates or not args.test_templates:
            print("Error: Broad generalization requires --train-templates and --test-templates")
            print("Example: --generalization broad --train-templates 1 2 3 4 --test-templates 5 6")
            return 1
    
    levels_dir = Path(args.levels_dir) if args.levels_dir else get_levels_dir()
    output_dir = Path(args.output_dir) if args.output_dir else get_config_dest_dir()
    
    if not levels_dir.exists():
        print(f"Error: Levels directory does not exist: {levels_dir}")
        print("Make sure to run the import script first:")
        print("  python scripts/import_phyq_levels.py")
        return 1
    
    print(f"Levels directory: {levels_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Dry run: {args.dry_run}")
    if args.scenarios:
        print(f"Filtering scenarios: {', '.join(args.scenarios)}")
    if args.output_name:
        print(f"Output name suffix: {args.output_name}")
    if args.templates:
        print(f"Filtering templates: {args.templates}")
    if args.generalization:
        print(f"Generalization mode: {args.generalization.upper()}")
        if args.generalization == "local":
            print(f"Train ratio: {args.train_ratio:.0%}")
        else:
            print(f"Train templates: {args.train_templates}")
            print(f"Test templates: {args.test_templates}")
        if args.levels_per_template:
            print(f"Levels per template: {args.levels_per_template}")
    
    # Always set seed for reproducibility
    seed = args.seed
    print(f"Random seed: {seed}")
    random.seed(seed)
    print()
    
    print("Discovering levels...")
    levels = discover_levels(levels_dir)
    
    # Filter scenarios if specified
    if args.scenarios:
        filtered_levels = {}
        for scenario_name, scenario_data in levels.items():
            # Extract the scenario name from "scenario_XX_name" format
            parts = scenario_name.split("_", 2)
            if len(parts) >= 3:
                name = parts[2]  # e.g., "rolling" from "scenario_03_rolling"
                if name in args.scenarios:
                    filtered_levels[scenario_name] = scenario_data
        levels = filtered_levels
        
        if not levels:
            print(f"Error: No levels found for scenarios: {args.scenarios}")
            return 1
    
    # Filter by template numbers if specified
    if args.templates:
        template_set = set(args.templates)
        for scenario_name in list(levels.keys()):
            levels[scenario_name]["train"] = [
                path for path in levels[scenario_name]["train"]
                if extract_template_from_path(path) in template_set
            ]
            levels[scenario_name]["test"] = [
                path for path in levels[scenario_name]["test"]
                if extract_template_from_path(path) in template_set
            ]
        
        # Remove scenarios with no levels left
        levels = {k: v for k, v in levels.items() if v["train"] or v["test"]}
        
        if not levels:
            print(f"Error: No levels found for templates: {args.templates}")
            return 1
    
    total_train = sum(len(v["train"]) for v in levels.values())
    total_test = sum(len(v["test"]) for v in levels.values())
    print(f"Found {len(levels)} scenarios")
    print(f"Total train levels: {total_train}")
    print(f"Total test levels: {total_test}")
    print()
    
    # Determine output file name suffix
    name_suffix = args.output_name if args.output_name else (
        "_".join(args.scenarios) if args.scenarios and len(args.scenarios) <= 3 
        else "custom" if args.scenarios 
        else None
    )
    
    # Determine scenario filter for metadata
    scenario_filter = args.scenarios[0] if args.scenarios and len(args.scenarios) == 1 else None
    
    # If generalization mode is specified, only generate that config
    if args.generalization:
        if args.generalization == "local":
            print("Generating LOCAL generalization config...")
            generate_local_generalization_config(
                levels, output_dir, 
                train_ratio=args.train_ratio,
                dry_run=args.dry_run, 
                name_suffix=name_suffix, 
                shuffle=True,
                levels_per_template=args.levels_per_template,
                seed=seed,
                scenario_filter=scenario_filter
            )
        else:  # broad
            print("Generating BROAD generalization config...")
            generate_broad_generalization_config(
                levels, output_dir,
                train_templates=args.train_templates,
                test_templates=args.test_templates,
                dry_run=args.dry_run,
                name_suffix=name_suffix,
                shuffle=True,
                levels_per_template=args.levels_per_template,
                seed=seed,
                scenario_filter=scenario_filter
            )
        print()
        print("Done!")
        return 0
    
    # Otherwise, generate all standard configs
    print("Generating full config...")
    generate_full_config(levels, output_dir, args.dry_run, name_suffix, args.shuffle)
    print()
    
    print("Generating sample config...")
    generate_sample_config(levels, output_dir, args.sample_size, args.dry_run, name_suffix, args.shuffle)
    print()
    
    print("Generating train-only config...")
    generate_train_only_config(levels, output_dir, args.dry_run, name_suffix, args.shuffle)
    print()
    
    print("Generating test-only config...")
    generate_test_only_config(levels, output_dir, args.dry_run, name_suffix, args.shuffle)
    print()
    
    if args.scenario_configs:
        print("Generating per-scenario configs...")
        generate_scenario_configs(levels, output_dir, args.dry_run, args.shuffle)
        print()
    
    print("Done!")
    return 0


if __name__ == "__main__":
    exit(main())
