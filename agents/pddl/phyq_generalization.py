"""
Phy-Q Generalization Protocol

Implements the two generalization evaluation protocols from the Phy-Q benchmark paper:

1. Local Generalization: 80/20 split within each template
   - Agents are trained on 80% of tasks within each template
   - Tested on remaining 20% of tasks from the same templates
   - Tests: Can the agent solve new instances of the same problem type?

2. Broad Generalization: Train on subset of templates, test on unseen templates  
   - Agents are trained on tasks from a subset of templates
   - Tested on tasks from different, unseen templates within the same scenario
   - Tests: Can the agent transfer the underlying physical rule to new structures?

Reference:
    Xue et al. "Phy-Q as a measure for physical reasoning intelligence"
    Nature Machine Intelligence, 2022
"""

import re
import json
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from pathlib import Path
from xml.etree import ElementTree as ET
from enum import Enum


class GeneralizationType(Enum):
    """Type of generalization evaluation."""
    LOCAL = "local"      # 80/20 split within each template
    BROAD = "broad"      # Train on some templates, test on other templates


@dataclass
class LevelInfo:
    """Information about a single level."""
    path: str
    scenario: str
    template: int
    level_id: int
    is_train: bool  # Original train/test from file path
    
    @classmethod
    def from_path(cls, path: str) -> Optional["LevelInfo"]:
        """Parse level info from path string."""
        path_str = str(path).replace("\\", "/")
        
        # Extract scenario from path (e.g., scenario_01_single_force -> single_force)
        scenario_match = re.search(r'scenario_\d+_(\w+)', path_str)
        scenario = scenario_match.group(1) if scenario_match else None
        
        if not scenario:
            # Try to extract from filename
            for scenario_name in ["single_force", "multiple_forces", "rolling", "falling", 
                                  "sliding", "bouncing", "relative_weight", "relative_height",
                                  "relative_width", "shape_difference", "non_greedy",
                                  "structural_analysis", "clearing_paths", "adequate_timing", "manoeuvring"]:
                if scenario_name in path_str:
                    scenario = scenario_name
                    break
        
        if not scenario:
            return None
            
        # Extract template number (e.g., _t01_ -> 1)
        template_match = re.search(r'_t(\d+)_', path_str)
        template = int(template_match.group(1)) if template_match else 0
        
        # Extract level ID (e.g., _00041.xml -> 41)
        level_id_match = re.search(r'_(\d{5})\.xml', path_str)
        level_id = int(level_id_match.group(1)) if level_id_match else 0
        
        # Determine if from train or test folder
        is_train = "/train/" in path_str
        
        return cls(
            path=path,
            scenario=scenario,
            template=template,
            level_id=level_id,
            is_train=is_train
        )


@dataclass
class GeneralizationSplit:
    """Result of a generalization split."""
    train_levels: List[str]
    test_levels: List[str]
    train_templates: Set[int]
    test_templates: Set[int]
    generalization_type: GeneralizationType
    split_ratio: float  # For local: train ratio; for broad: fraction of templates for training
    
    def __str__(self) -> str:
        lines = [
            f"Generalization Split ({self.generalization_type.value})",
            f"  Train levels: {len(self.train_levels)}",
            f"  Test levels: {len(self.test_levels)}",
            f"  Train templates: {sorted(self.train_templates)}",
            f"  Test templates: {sorted(self.test_templates)}",
        ]
        if self.generalization_type == GeneralizationType.LOCAL:
            lines.append(f"  Split ratio: {self.split_ratio:.0%} train / {1-self.split_ratio:.0%} test")
        else:
            lines.append(f"  Template split: {len(self.train_templates)} train / {len(self.test_templates)} test")
        return "\n".join(lines)


class PhyQGeneralizationProtocol:
    """
    Manages Phy-Q generalization evaluation protocols.
    
    Usage:
        # Local generalization (80/20 within templates)
        protocol = PhyQGeneralizationProtocol(
            config_path="config_phyq_single_force_full.xml",
            generalization_type="local",
            train_ratio=0.8,
            seed=42
        )
        
        # Broad generalization (train on templates 1-4, test on 5-6)
        protocol = PhyQGeneralizationProtocol(
            config_path="config_phyq_single_force_full.xml",
            generalization_type="broad",
            train_templates=[1, 2, 3, 4],
            test_templates=[5, 6],
            seed=42
        )
        
        # Get split
        split = protocol.get_split()
        print(f"Train levels: {len(split.train_levels)}")
        print(f"Test levels: {len(split.test_levels)}")
    """
    
    def __init__(
        self,
        config_path: str,
        generalization_type: str = "local",
        train_ratio: float = 0.8,
        train_templates: Optional[List[int]] = None,
        test_templates: Optional[List[int]] = None,
        seed: int = 42,
        scenario_filter: Optional[str] = None,
        levels_per_template: Optional[int] = None,
        train_levels_override: Optional[List[str]] = None,
        test_levels_override: Optional[List[str]] = None
    ):
        """
        Initialize the generalization protocol.
        
        Parameters:
        -----------
        config_path : str
            Path to Phy-Q config XML file
        generalization_type : str
            "local" for within-template split, "broad" for across-template split
        train_ratio : float
            For local generalization: fraction of levels per template to use for training (default: 0.8)
        train_templates : List[int], optional
            For broad generalization: list of template numbers to train on
        test_templates : List[int], optional
            For broad generalization: list of template numbers to test on
        seed : int
            Random seed for reproducible splits
        scenario_filter : str, optional
            Only include levels from this scenario (e.g., "single_force")
        levels_per_template : int, optional
            If specified, limit the number of levels per template
        train_levels_override : List[str], optional
            If provided, use these as train levels (from metadata)
        test_levels_override : List[str], optional
            If provided, use these as test levels (from metadata)
        """
        self.config_path = config_path
        self.generalization_type = GeneralizationType(generalization_type)
        self.train_ratio = train_ratio
        self.train_templates_config = train_templates
        self.test_templates_config = test_templates
        self.seed = seed
        self.scenario_filter = scenario_filter
        self.levels_per_template = levels_per_template
        self.train_levels_override = train_levels_override
        self.test_levels_override = test_levels_override
        
        self.rng = random.Random(seed)
        
        # Load and parse levels from config
        self.all_levels: List[LevelInfo] = []
        self.levels_by_template: Dict[int, List[LevelInfo]] = {}
        self.available_templates: Set[int] = set()
        
        self._load_levels_from_config()
        self._organize_by_template()
        
        # Compute split
        self.split: Optional[GeneralizationSplit] = None
        
    def _load_levels_from_config(self) -> None:
        """Load level paths from config XML file, preserving train/test assignment from XML."""
        try:
            tree = ET.parse(self.config_path)
            root = tree.getroot()
            
            # Track levels from config with their mode (training/testing)
            self._config_train_levels: List[str] = []
            self._config_test_levels: List[str] = []
            
            # Iterate through game_level_set elements to get mode
            for game_level_set in root.iter('game_level_set'):
                mode = game_level_set.get('mode', '').lower()
                is_train = mode == 'training'
                
                for game_level in game_level_set.iter('game_levels'):
                    level_path = game_level.get('level_path')
                    if level_path:
                        level_info = LevelInfo.from_path(level_path)
                        if level_info:
                            # Override is_train based on XML mode, not file path
                            level_info.is_train = is_train
                            
                            # Apply scenario filter if specified
                            if self.scenario_filter and level_info.scenario != self.scenario_filter:
                                continue
                            
                            self.all_levels.append(level_info)
                            
                            if is_train:
                                self._config_train_levels.append(level_path)
                            else:
                                self._config_test_levels.append(level_path)
            
            print(f"[PHY-Q GENERALIZATION] Loaded {len(self.all_levels)} levels from {self.config_path}")
            print(f"[PHY-Q GENERALIZATION] From config: {len(self._config_train_levels)} train, {len(self._config_test_levels)} test")
            
        except FileNotFoundError:
            print(f"[PHY-Q GENERALIZATION] Warning: Config file not found: {self.config_path}")
        except ET.ParseError as e:
            print(f"[PHY-Q GENERALIZATION] Warning: Failed to parse config: {e}")
    
    def _organize_by_template(self) -> None:
        """Organize levels by template number."""
        self.levels_by_template = {}
        
        for level in self.all_levels:
            if level.template not in self.levels_by_template:
                self.levels_by_template[level.template] = []
            self.levels_by_template[level.template].append(level)
        
        self.available_templates = set(self.levels_by_template.keys())
        
        print(f"[PHY-Q GENERALIZATION] Found {len(self.available_templates)} templates: {sorted(self.available_templates)}")
        for template, levels in sorted(self.levels_by_template.items()):
            train_count = sum(1 for l in levels if l.is_train)
            test_count = len(levels) - train_count
            print(f"  Template {template}: {len(levels)} levels ({train_count} train, {test_count} test)")
    
    def _compute_local_split(self) -> GeneralizationSplit:
        """
        Use the train/test split from metadata (if provided) or config XML.
        
        The config already has the correct split - we just need to organize by template
        and preserve the order.
        
        Order: Train template 1 -> Test template 1 -> Train template 2 -> Test template 2 -> ...
        """
        # Use overrides from metadata if provided (new single-trial format)
        if self.train_levels_override is not None and self.test_levels_override is not None:
            print(f"[PHY-Q GENERALIZATION] Using train/test split from METADATA")
            print(f"[PHY-Q GENERALIZATION] Train levels: {len(self.train_levels_override)}, Test levels: {len(self.test_levels_override)}")
            
            train_set = set(self.train_levels_override)
            test_set = set(self.test_levels_override)
            
            # Organize by template
            self._train_by_template = {}
            self._test_by_template = {}
            
            for level in self.all_levels:
                if level.path in train_set:
                    if level.template not in self._train_by_template:
                        self._train_by_template[level.template] = []
                    self._train_by_template[level.template].append(level.path)
                elif level.path in test_set:
                    if level.template not in self._test_by_template:
                        self._test_by_template[level.template] = []
                    self._test_by_template[level.template].append(level.path)
            
            # Determine template order from the overrides
            seen_templates = []
            for level_path in self.train_levels_override + self.test_levels_override:
                for level in self.all_levels:
                    if level.path == level_path and level.template not in seen_templates:
                        seen_templates.append(level.template)
                        break
            self._template_order = seen_templates
            
            print(f"[PHY-Q GENERALIZATION] Template order from metadata: {self._template_order}")
            
            # Print per-template breakdown
            for template in self._template_order:
                train_count = len(self._train_by_template.get(template, []))
                test_count = len(self._test_by_template.get(template, []))
                print(f"  Template {template}: {train_count} train, {test_count} test")
            
            return GeneralizationSplit(
                train_levels=self.train_levels_override,
                test_levels=self.test_levels_override,
                train_templates=self.available_templates.copy(),
                test_templates=self.available_templates.copy(),
                generalization_type=GeneralizationType.LOCAL,
                split_ratio=self.train_ratio
            )
        
        # Fallback: Use the split from config XML (already loaded in _load_levels_from_config)
        if hasattr(self, '_config_train_levels') and hasattr(self, '_config_test_levels'):
            print(f"[PHY-Q GENERALIZATION] Using train/test split from config XML")
            
            # Organize by template, preserving config order
            self._train_by_template = {}
            self._test_by_template = {}
            
            # Build a lookup for level -> template
            level_to_template = {}
            for level in self.all_levels:
                level_to_template[level.path] = level.template
            
            # Group train levels by template (preserving order from config)
            for level_path in self._config_train_levels:
                template = level_to_template.get(level_path)
                if template is not None:
                    if template not in self._train_by_template:
                        self._train_by_template[template] = []
                    self._train_by_template[template].append(level_path)
            
            # Group test levels by template (preserving order from config)
            for level_path in self._config_test_levels:
                template = level_to_template.get(level_path)
                if template is not None:
                    if template not in self._test_by_template:
                        self._test_by_template[template] = []
                    self._test_by_template[template].append(level_path)
            
            # Determine template order from the config (order of first appearance)
            seen_templates = []
            for level_path in self._config_train_levels + self._config_test_levels:
                template = level_to_template.get(level_path)
                if template is not None and template not in seen_templates:
                    seen_templates.append(template)
            self._template_order = seen_templates
            
            print(f"[PHY-Q GENERALIZATION] Template order from config: {self._template_order}")
            
            train_levels = self._config_train_levels.copy()
            test_levels = self._config_test_levels.copy()
        else:
            # Fallback: compute split (shouldn't happen if config is properly generated)
            print(f"[PHY-Q GENERALIZATION] WARNING: No config split found, computing split")
            self._train_by_template = {}
            self._test_by_template = {}
            
            for template, levels in self.levels_by_template.items():
                shuffled = levels.copy()
                self.rng.shuffle(shuffled)
                
                if self.levels_per_template and len(shuffled) > self.levels_per_template:
                    shuffled = shuffled[:self.levels_per_template]
                
                split_idx = int(len(shuffled) * self.train_ratio)
                self._train_by_template[template] = [l.path for l in shuffled[:split_idx]]
                self._test_by_template[template] = [l.path for l in shuffled[split_idx:]]
            
            self._template_order = list(self.levels_by_template.keys())
            self.rng.shuffle(self._template_order)
            
            train_levels = []
            test_levels = []
            for template in self._template_order:
                train_levels.extend(self._train_by_template[template])
                test_levels.extend(self._test_by_template[template])
        
        # Print per-template breakdown
        for template in self._template_order:
            train_count = len(self._train_by_template.get(template, []))
            test_count = len(self._test_by_template.get(template, []))
            print(f"  Template {template}: {train_count} train, {test_count} test")
        
        return GeneralizationSplit(
            train_levels=train_levels,
            test_levels=test_levels,
            train_templates=self.available_templates.copy(),
            test_templates=self.available_templates.copy(),
            generalization_type=GeneralizationType.LOCAL,
            split_ratio=self.train_ratio
        )
    
    def _compute_broad_split(self) -> GeneralizationSplit:
        """
        Compute broad generalization split: train on some templates, test on others.
        
        Uses configured train_templates and test_templates, or auto-computes a split
        if not specified.
        """
        train_templates = set(self.train_templates_config) if self.train_templates_config else set()
        test_templates = set(self.test_templates_config) if self.test_templates_config else set()
        
        # Auto-compute split if not fully specified
        if not train_templates or not test_templates:
            all_templates = sorted(self.available_templates)
            n_train = max(1, int(len(all_templates) * self.train_ratio))
            
            # Shuffle templates
            shuffled_templates = all_templates.copy()
            self.rng.shuffle(shuffled_templates)
            
            train_templates = set(shuffled_templates[:n_train])
            test_templates = set(shuffled_templates[n_train:])
            
            print(f"[PHY-Q GENERALIZATION] Auto-computed template split:")
            print(f"  Train templates: {sorted(train_templates)}")
            print(f"  Test templates: {sorted(test_templates)}")
        
        # Validate templates
        invalid_train = train_templates - self.available_templates
        invalid_test = test_templates - self.available_templates
        
        if invalid_train:
            print(f"[PHY-Q GENERALIZATION] Warning: Train templates not found: {invalid_train}")
            train_templates -= invalid_train
        
        if invalid_test:
            print(f"[PHY-Q GENERALIZATION] Warning: Test templates not found: {invalid_test}")
            test_templates -= invalid_test
        
        # Check for overlap
        overlap = train_templates & test_templates
        if overlap:
            print(f"[PHY-Q GENERALIZATION] Warning: Templates in both train and test: {overlap}")
            print("  Removing overlapping templates from test set.")
            test_templates -= overlap
        
        # Build level lists
        train_levels = []
        test_levels = []
        
        for template, levels in self.levels_by_template.items():
            # Shuffle levels within template before applying limit
            shuffled = levels.copy()
            self.rng.shuffle(shuffled)
            
            # Apply levels_per_template limit if specified
            if self.levels_per_template and len(shuffled) > self.levels_per_template:
                shuffled = shuffled[:self.levels_per_template]
            
            if template in train_templates:
                train_levels.extend([l.path for l in shuffled])
            elif template in test_templates:
                test_levels.extend([l.path for l in shuffled])
        
        # Shuffle final lists
        self.rng.shuffle(train_levels)
        self.rng.shuffle(test_levels)
        
        return GeneralizationSplit(
            train_levels=train_levels,
            test_levels=test_levels,
            train_templates=train_templates,
            test_templates=test_templates,
            generalization_type=GeneralizationType.BROAD,
            split_ratio=len(train_templates) / len(self.available_templates) if self.available_templates else 0
        )
    
    def get_split(self) -> GeneralizationSplit:
        """
        Get the train/test split according to the configured generalization type.
        
        Returns:
            GeneralizationSplit with train_levels and test_levels
        """
        if self.split is None:
            if self.generalization_type == GeneralizationType.LOCAL:
                self.split = self._compute_local_split()
            else:
                self.split = self._compute_broad_split()
            
            print(f"\n{self.split}\n")
        
        return self.split
    
    def get_train_levels(self) -> List[str]:
        """Get list of training level paths."""
        return self.get_split().train_levels
    
    def get_test_levels(self) -> List[str]:
        """Get list of test level paths."""
        return self.get_split().test_levels
    
    def get_all_levels_ordered(self) -> List[str]:
        """
        Get all levels in the order they should be played.
        
        For LOCAL generalization: interleaved by template
            t1 train -> t1 test -> t2 train -> t2 test -> ...
        
        For BROAD generalization: all train first, then all test
            all train levels -> all test levels
        """
        split = self.get_split()
        
        # For LOCAL generalization, use interleaved template order
        if self.generalization_type == GeneralizationType.LOCAL and hasattr(self, '_template_order'):
            levels = []
            for template in self._template_order:
                levels.extend(self._train_by_template.get(template, []))
                levels.extend(self._test_by_template.get(template, []))
            return levels
        
        # For BROAD or fallback: all train then all test
        return split.train_levels + split.test_levels
    
    def is_train_level(self, level_path: str) -> bool:
        """Check if a level path is in the training set."""
        return level_path in self.get_split().train_levels
    
    def is_test_level(self, level_path: str) -> bool:
        """Check if a level path is in the test set."""
        return level_path in self.get_split().test_levels
    
    def get_phase_for_level(self, level_idx: int) -> Tuple[str, bool]:
        """
        Get the phase (train/test) and should_learn flag for a level index.
        
        Parameters:
        -----------
        level_idx : int
            1-indexed level number
            
        Returns:
        --------
        Tuple[str, bool]
            ("train", True) for training levels
            ("test", False) for test levels
            ("complete", False) if past all levels
        """
        split = self.get_split()
        all_levels = self.get_all_levels_ordered()
        
        idx = level_idx - 1  # Convert to 0-indexed
        
        if idx >= len(all_levels):
            return ("complete", False)
        
        # Get the level path and check if it's train or test
        level_path = all_levels[idx]
        
        if level_path in split.train_levels:
            return ("train", True)
        else:
            return ("test", False)
    
    def get_level_path_for_index(self, level_idx: int) -> Optional[str]:
        """
        Get level path for a given 1-indexed level number.
        
        Parameters:
        -----------
        level_idx : int
            1-indexed level number
            
        Returns:
        --------
        str or None
            Level path, or None if index is out of range
        """
        all_levels = self.get_all_levels_ordered()
        idx = level_idx - 1  # Convert to 0-indexed
        
        if 0 <= idx < len(all_levels):
            return all_levels[idx]
        return None
    
    def print_summary(self) -> None:
        """Print detailed summary of the generalization protocol."""
        split = self.get_split()
        
        print("\n" + "=" * 70)
        print("PHY-Q GENERALIZATION PROTOCOL")
        print("=" * 70)
        
        print(f"\nConfiguration:")
        print(f"  Config file: {self.config_path}")
        print(f"  Generalization type: {self.generalization_type.value.upper()}")
        print(f"  Seed: {self.seed}")
        
        if self.generalization_type == GeneralizationType.LOCAL:
            print(f"  Train ratio: {self.train_ratio:.0%}")
        else:
            print(f"  Train templates: {sorted(split.train_templates)}")
            print(f"  Test templates: {sorted(split.test_templates)}")
        
        print(f"\nSplit Summary:")
        print(f"  Total levels: {len(split.train_levels) + len(split.test_levels)}")
        print(f"  Training levels: {len(split.train_levels)}")
        print(f"  Test levels: {len(split.test_levels)}")
        
        print(f"\nPer-Template Breakdown:")
        for template in sorted(self.available_templates):
            levels = self.levels_by_template[template]
            train_count = sum(1 for l in levels if l.path in split.train_levels)
            test_count = sum(1 for l in levels if l.path in split.test_levels)
            
            if template in split.train_templates and template in split.test_templates:
                role = "TRAIN+TEST"
            elif template in split.train_templates:
                role = "TRAIN only"
            elif template in split.test_templates:
                role = "TEST only"
            else:
                role = "excluded"
            
            print(f"  Template {template:2d}: {train_count:4d} train, {test_count:4d} test ({role})")
        
        print("\n" + "=" * 70)
    
    def __len__(self) -> int:
        """Return total number of levels."""
        return len(self.get_all_levels_ordered())


def create_local_generalization_protocol(
    config_path: str,
    train_ratio: float = 0.8,
    seed: int = 42,
    scenario_filter: Optional[str] = None,
    levels_per_template: Optional[int] = None,
    train_levels_override: Optional[List[str]] = None,
    test_levels_override: Optional[List[str]] = None
) -> PhyQGeneralizationProtocol:
    """
    Factory function for local generalization protocol.

    Local generalization tests if the agent can solve new instances of
    problem types it has seen during training.
    
    Args:
        levels_per_template: If specified, limit levels per template
        train_levels_override: If provided, use these as train levels (from metadata)
        test_levels_override: If provided, use these as test levels (from metadata)
    """
    return PhyQGeneralizationProtocol(
        config_path=config_path,
        generalization_type="local",
        train_ratio=train_ratio,
        seed=seed,
        scenario_filter=scenario_filter,
        levels_per_template=levels_per_template,
        train_levels_override=train_levels_override,
        test_levels_override=test_levels_override
    )


def create_broad_generalization_protocol(
    config_path: str,
    train_templates: Optional[List[int]] = None,
    test_templates: Optional[List[int]] = None,
    seed: int = 42,
    scenario_filter: Optional[str] = None,
    levels_per_template: Optional[int] = None
) -> PhyQGeneralizationProtocol:
    """
    Factory function for broad generalization protocol.
    
    Broad generalization tests if the agent can transfer the underlying
    physical reasoning rule to new problem structures it hasn't seen.
    
    If train_templates and test_templates are not specified, the protocol
    will auto-compute a ~80/20 template split.
    
    Args:
        levels_per_template: If specified, limit levels per template
    """
    return PhyQGeneralizationProtocol(
        config_path=config_path,
        generalization_type="broad",
        train_templates=train_templates,
        test_templates=test_templates,
        seed=seed,
        scenario_filter=scenario_filter,
        levels_per_template=levels_per_template
    )


def load_config_metadata(config_path: str) -> Optional[Dict]:
    """
    Load metadata JSON file for a config XML.
    
    The metadata file has the same name as the XML but with .meta.json extension.
    
    Args:
        config_path: Path to the config XML file
        
    Returns:
        Dictionary with metadata, or None if not found
    """
    config_path = Path(config_path)
    meta_path = config_path.with_suffix('.meta.json')
    
    print(f"[PHY-Q METADATA] Looking for metadata file: {meta_path}")
    
    if not meta_path.exists():
        print(f"[PHY-Q METADATA] No metadata file found at {meta_path}")
        print(f"[PHY-Q METADATA] Generate one using: python scripts/generate_phyq_configs.py --generalization <type> ...")
        return None
    
    try:
        with open(meta_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        print(f"[PHY-Q METADATA] Successfully loaded metadata from {meta_path}")
        print(f"[PHY-Q METADATA] Config settings:")
        print(f"  - Generalization type: {metadata.get('generalization_type', 'N/A')}")
        print(f"  - Train ratio: {metadata.get('train_ratio', 'N/A')}")
        print(f"  - Seed: {metadata.get('seed', 'N/A')}")
        print(f"  - Levels per template: {metadata.get('levels_per_template', 'unlimited')}")
        print(f"  - Scenario filter: {metadata.get('scenario_filter', 'none')}")
        if metadata.get('train_templates'):
            print(f"  - Train templates: {metadata.get('train_templates')}")
        if metadata.get('test_templates'):
            print(f"  - Test templates: {metadata.get('test_templates')}")
        print(f"  - Expected train count: {metadata.get('train_count', 'N/A')}")
        print(f"  - Expected test count: {metadata.get('test_count', 'N/A')}")
        
        return metadata
    except (json.JSONDecodeError, IOError) as e:
        print(f"[PHY-Q METADATA] Failed to load metadata: {e}")
        return None


def create_protocol_from_config(config_path: str) -> Optional[PhyQGeneralizationProtocol]:
    """
    Create a generalization protocol from a config file and its metadata.
    
    This function reads the .meta.json file alongside the config XML
    and creates the appropriate protocol with matching parameters.
    
    Args:
        config_path: Path to the config XML file
        
    Returns:
        PhyQGeneralizationProtocol configured from metadata, or None if metadata not found
    """
    print(f"\n[PHY-Q METADATA] Attempting to create protocol from config metadata...")
    print(f"[PHY-Q METADATA] Config path: {config_path}")
    
    metadata = load_config_metadata(config_path)
    
    if metadata is None:
        print(f"[PHY-Q METADATA] Cannot create protocol - metadata not available")
        return None
    
    gen_type = metadata.get("generalization_type", "local")
    print(f"\n[PHY-Q METADATA] Creating {gen_type.upper()} generalization protocol from metadata...")
    
    # Check if metadata has train/test level lists (new format)
    train_levels_from_meta = metadata.get("train_levels")
    test_levels_from_meta = metadata.get("test_levels")
    
    if gen_type == "local":
        protocol = create_local_generalization_protocol(
            config_path=config_path,
            train_ratio=metadata.get("train_ratio", 0.8),
            seed=metadata.get("seed", 42),
            scenario_filter=metadata.get("scenario_filter"),
            levels_per_template=metadata.get("levels_per_template"),
            train_levels_override=train_levels_from_meta,
            test_levels_override=test_levels_from_meta
        )
    else:  # broad
        protocol = create_broad_generalization_protocol(
            config_path=config_path,
            train_templates=metadata.get("train_templates"),
            test_templates=metadata.get("test_templates"),
            seed=metadata.get("seed", 42),
            scenario_filter=metadata.get("scenario_filter"),
            levels_per_template=metadata.get("levels_per_template")
        )
    
    print(f"[PHY-Q METADATA] Protocol created successfully from metadata!")
    return protocol


# Example template splits for common scenarios
TEMPLATE_SPLITS = {
    "single_force": {
        # 6 templates -> 4 train, 2 test
        "train": [1, 2, 3, 4],
        "test": [5, 6]
    },
    "rolling": {
        # Adjust based on actual template count
        "train": [1, 2, 3],
        "test": [4, 5]
    },
    # Add more scenarios as needed
}


def get_default_template_split(scenario: str) -> Tuple[List[int], List[int]]:
    """
    Get default template split for a scenario.
    
    Returns:
        Tuple of (train_templates, test_templates)
    """
    if scenario in TEMPLATE_SPLITS:
        return (TEMPLATE_SPLITS[scenario]["train"], TEMPLATE_SPLITS[scenario]["test"])
    else:
        # Default: no pre-defined split, protocol will auto-compute
        return (None, None)
