"""
Angle Training Protocol for PDDL Agent

Manages train/validation/test splits for angle selection during iterative learning.

Protocol:
- Every 5 training shots: Run a validation block of 5 shots
- Pattern: TTTTT VVVVV TTTTT VVVVV ... (5 train, 5 val, repeat)
- After 40 train shots: Run 10 test shots (no learning)
"""

import random
import numpy as np
from typing import Tuple, Optional, Dict, List, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from agents.pddl.phyq_metrics import PhyQMetrics


class AngleTrainingProtocol:
    """
    Manages angle selection with train/validation/test splits.
    
    Pool: 601 angles from 20.0° to 80.0° (0.1° steps)
    Default split: 70% train (~420), 15% val (~90), 15% test (~91)
    
    Protocol:
    - Every 5 training shots, run a validation block of 5 shots
    - Pattern: TTTTT VVVVV TTTTT VVVVV ... (5 train, 5 val, repeat)
    - After 40 train shots, run 10 test shots
    """
    
    def __init__(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
        val_every_n_levels: int = 5,
        val_shots_per_block: int = 5,
        test_after_n_trains: int = 40,
        test_shots: int = 10,
        angle_min: float = 20.0,
        angle_max: float = 80.0,
        angle_step: float = 0.1
    ):
        """
        Initialize the angle training protocol.
        
        Parameters:
        -----------
        train_ratio : float
            Fraction of angles for training (default: 0.70)
        val_ratio : float
            Fraction of angles for validation (default: 0.15)
        test_ratio : float
            Fraction of angles for testing (default: 0.15)
        seed : int
            Random seed for reproducible splits (default: 42)
        val_every_n_levels : int
            Run validation every N training levels (default: 5, so after levels 5, 10, 15, ...)
        val_shots_per_block : int
            Number of validation shots to run each validation block (default: 5)
        test_after_n_trains : int
            Enter test phase after N training shots (default: 40)
        test_shots : int
            Number of test shots to run (default: 10)
        angle_min : float
            Minimum angle in degrees (default: 20.0)
        angle_max : float
            Maximum angle in degrees (default: 80.0)
        angle_step : float
            Angle step size in degrees (default: 0.1)
        """
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.val_every_n_levels = val_every_n_levels
        self.val_shots_per_block = val_shots_per_block
        self.test_after_n_trains = test_after_n_trains
        self.test_shots = test_shots
        
        # Generate and split angle pool using STRATIFIED sampling
        # This ensures train/val/test all cover the full angle range evenly
        all_angles = np.round(np.arange(angle_min, angle_max + angle_step, angle_step), 1).tolist()
        
        rng = np.random.RandomState(seed)
        
        # Stratified split: assign each angle to train/val/test based on ratio
        # This ensures all sets cover the full 20°-80° range
        self.train_pool = []
        self.val_pool = []
        self.test_pool = []
        
        for angle in all_angles:
            r = rng.random()
            if r < train_ratio:
                self.train_pool.append(angle)
            elif r < train_ratio + val_ratio:
                self.val_pool.append(angle)
            else:
                self.test_pool.append(angle)
        
        # Shuffle each pool for random selection order
        rng.shuffle(self.train_pool)
        rng.shuffle(self.val_pool)
        rng.shuffle(self.test_pool)
        
        n = len(all_angles)
        
        # Counters
        self.level_count = 0
        self.train_count = 0
        self.val_count = 0
        self.test_count = 0
        
        # State
        self.phase = "training"  # "training", "validation_block", "testing", "complete"
        self.test_shots_done = 0
        self.val_block_remaining = 0  # Remaining validation shots in current block
        self._last_validated_train_count = 0  # Track which train_count triggered the last validation block
        
        # Results tracking
        self.train_results: List[Dict[str, Any]] = []
        self.val_results: List[Dict[str, Any]] = []
        self.test_results: List[Dict[str, Any]] = []
        
        # Collision sample storage for algorithm comparison
        # Each sample is a tuple: (pre_state: dict, post_state: dict)
        self.train_collision_samples: List[Tuple[Dict, Dict]] = []
        self.val_collision_samples: List[Tuple[Dict, Dict]] = []
        self.test_collision_samples: List[Tuple[Dict, Dict]] = []
        
        # High RMSE outlier tracking (RMSE > threshold)
        self.high_rmse_threshold = 40.0
        self.high_rmse_outliers: List[Dict[str, Any]] = []
        
        # Print initialization info
        print(f"\n{'='*60}")
        print("ANGLE TRAINING PROTOCOL INITIALIZED")
        print(f"{'='*60}")
        print(f"Total angles: {n} ({angle_min}° to {angle_max}°, step {angle_step}°)")
        
        # Show angle distribution for each pool (stratified split verification)
        def _pool_stats(pool):
            if not pool:
                return "empty"
            return f"range [{min(pool):.1f}° - {max(pool):.1f}°]"
        print(f"Train distribution: {_pool_stats(self.train_pool)}")
        print(f"Val distribution:   {_pool_stats(self.val_pool)}")
        print(f"Test distribution:  {_pool_stats(self.test_pool)}")
        print(f"Train pool:   {len(self.train_pool)} angles ({train_ratio*100:.0f}%)")
        print(f"Val pool:     {len(self.val_pool)} angles ({val_ratio*100:.0f}%)")
        print(f"Test pool:    {len(self.test_pool)} angles ({test_ratio*100:.0f}%)")
        print(f"Protocol:     Validate every {val_every_n_levels} train levels ({val_shots_per_block} val shots per block)")
        print(f"              Test after {test_after_n_trains} train shots ({test_shots} test shots)")
        print(f"Seed:         {seed}")
        print(f"{'='*60}\n")
    
    def get_next_shot(self) -> Tuple[Optional[float], str, bool]:
        """
        Get the next angle to shoot.
        
        Returns:
        --------
        Tuple[Optional[float], str, bool]
            - angle: The angle to shoot (None if complete)
            - phase: "train", "validation", "test", or "complete"
            - should_learn: True if model should update from this shot
        """
        # Check if we're in test phase
        if self.phase == "testing":
            if self.test_shots_done >= self.test_shots:
                self.phase = "complete"
                return None, "complete", False
            
            angle = random.choice(self.test_pool)
            self.test_shots_done += 1
            self.test_count += 1
            return angle, "test", False
        
        # Check if we're in a validation block
        if self.phase == "validation_block":
            if self.val_block_remaining > 0:
                angle = random.choice(self.val_pool)
                self.val_count += 1
                self.val_block_remaining -= 1
                self.level_count += 1
                
                # Check if validation block is complete
                if self.val_block_remaining == 0:
                    self.phase = "training"
                
                return angle, "validation", False
            else:
                # Block complete, return to training
                self.phase = "training"
        
        # Check if we should enter test phase
        if self.train_count >= self.test_after_n_trains and self.phase == "training":
            self.phase = "testing"
            print(f"\n{'='*60}")
            print(f"ENTERING TEST PHASE (after {self.train_count} train shots)")
            print(f"{'='*60}\n")
            return self.get_next_shot()  # Recurse to get test angle
        
        self.level_count += 1
        
        # Check if we should start a validation block (every N training levels)
        # Only trigger if validation is enabled (val_every_n_levels > 0) and we haven't already done a validation block for this train_count
        if (self.val_every_n_levels > 0 and
            self.train_count > 0 and 
            self.train_count % self.val_every_n_levels == 0 and
            self.train_count > self._last_validated_train_count):
            # Start validation block
            self.phase = "validation_block"
            self.val_block_remaining = self.val_shots_per_block - 1  # -1 because we're about to do one
            self._last_validated_train_count = self.train_count  # Remember we validated at this train_count
            
            angle = random.choice(self.val_pool)
            self.val_count += 1
            
            print(f"\n[VALIDATION BLOCK] Starting {self.val_shots_per_block} validation shots "
                  f"(after {self.train_count} train shots)")
            
            return angle, "validation", False
        else:
            # Training shot
            angle = random.choice(self.train_pool)
            self.train_count += 1
            return angle, "train", True
    
    def record_result(
        self,
        angle: float,
        phase: str,
        rmse: float,
        **extra
    ) -> None:
        """
        Record result for analysis.
        
        Parameters:
        -----------
        angle : float
            The angle used
        phase : str
            "train", "validation", or "test"
        rmse : float
            Root mean square error of trajectory prediction
        **extra : dict
            Additional metrics to record (e.g., gravity, velocity)
        """
        result = {"angle": angle, "rmse": rmse, "level": self.level_count, "phase": phase, **extra}
        
        if phase == "train":
            self.train_results.append(result)
        elif phase == "validation":
            self.val_results.append(result)
        elif phase == "test":
            self.test_results.append(result)
        
        # Track high RMSE outliers
        if rmse is not None and np.isfinite(rmse) and rmse > self.high_rmse_threshold:
            self.high_rmse_outliers.append(result.copy())
            print(f"\n[HIGH RMSE ALERT] RMSE={rmse:.2f} > {self.high_rmse_threshold} | "
                  f"Angle={angle}° | Phase={phase.upper()} | Level={self.level_count}")
    
    def record_collision(self, phase: str, pre_state: Dict, post_state: Dict) -> None:
        """
        Store collision sample for later algorithm evaluation.
        
        Parameters:
        -----------
        phase : str
            "train", "validation", or "test"
        pre_state : dict
            Pre-collision state with keys: x, y, v_x, v_y
        post_state : dict
            Post-collision state with keys: x, y, v_x, v_y
        """
        sample = (pre_state.copy(), post_state.copy())
        if phase == "train":
            self.train_collision_samples.append(sample)
        elif phase == "validation":
            self.val_collision_samples.append(sample)
        elif phase == "test":
            self.test_collision_samples.append(sample)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current protocol status."""
        return {
            "phase": self.phase,
            "level": self.level_count,
            "train_shots": self.train_count,
            "val_shots": self.val_count,
            "test_shots": self.test_count,
            "remaining_train": max(0, self.test_after_n_trains - self.train_count),
            "remaining_test": self.test_shots - self.test_shots_done if self.phase == "testing" else self.test_shots,
            "val_block_remaining": self.val_block_remaining,
            "train_pool_size": len(self.train_pool),
            "val_pool_size": len(self.val_pool),
            "test_pool_size": len(self.test_pool)
        }
    
    def print_status(self) -> None:
        """Print current protocol status."""
        s = self.get_status()
        
        if s["phase"] == "training":
            progress = f"Train: {s['train_shots']}/{self.test_after_n_trains}"
        elif s["phase"] == "validation_block":
            progress = f"Val Block: {self.val_block_remaining} remaining"
        elif s["phase"] == "testing":
            progress = f"Test: {s['test_shots']}/{self.test_shots}"
        else:
            progress = "Complete"
        
        print(f"[PROTOCOL] Phase: {s['phase'].upper():16} | Level: {s['level']:3} | "
              f"{progress} | Val: {s['val_shots']}")
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for all phases."""
        def calc_stats(results: List[Dict]) -> Dict[str, float]:
            if not results:
                return {"mean": 0, "std": 0, "min": 0, "max": 0, "count": 0}
            rmses = [r["rmse"] for r in results if r["rmse"] is not None and np.isfinite(r["rmse"])]
            if not rmses:
                return {"mean": 0, "std": 0, "min": 0, "max": 0, "count": 0}
            return {
                "mean": np.mean(rmses),
                "std": np.std(rmses),
                "min": np.min(rmses),
                "max": np.max(rmses),
                "count": len(rmses)
            }
        
        return {
            "train": calc_stats(self.train_results),
            "validation": calc_stats(self.val_results),
            "test": calc_stats(self.test_results)
        }
    
    def print_final_results(self, kb: Optional[Dict] = None, phyq_metrics: Optional["PhyQMetrics"] = None) -> None:
        """
        Print final results summary including trajectory RMSE and collision model comparison.
        
        Parameters:
        -----------
        kb : dict, optional
            Knowledge base containing trained collision models. If provided,
            collision model comparison will be printed.
        phyq_metrics : PhyQMetrics, optional
            Phy-Q benchmark metrics tracker. If provided, will print the 
            Phy-Q benchmark report for comparison with humans and other agents.
        """
        stats = self.get_summary_stats()
        
        print(f"\n{'='*70}")
        print("TRAINING PROTOCOL COMPLETE - FINAL RESULTS")
        print(f"{'='*70}")
        
        # Trajectory RMSE summary
        print("\n--- Trajectory Prediction RMSE ---")
        print(f"{'Phase':<12} {'Count':>6} {'Mean RMSE':>12} {'Std':>10} {'Min':>10} {'Max':>10}")
        print(f"{'-'*70}")
        
        for phase in ["train", "validation", "test"]:
            s = stats[phase]
            if s["count"] > 0:
                print(f"{phase.capitalize():<12} {s['count']:>6} {s['mean']:>12.2f} {s['std']:>10.2f} "
                      f"{s['min']:>10.2f} {s['max']:>10.2f}")
            else:
                print(f"{phase.capitalize():<12} {0:>6} {'N/A':>12} {'N/A':>10} {'N/A':>10} {'N/A':>10}")
        
        print(f"{'='*70}")
        
        # Analysis
        if stats["validation"]["count"] > 0 and stats["train"]["count"] > 0:
            train_val_gap = stats["validation"]["mean"] - stats["train"]["mean"]
            print(f"\nTrain-Val Gap: {train_val_gap:+.2f} RMSE", end="")
            if train_val_gap > stats["train"]["std"]:
                print(" (potential overfitting)")
            elif train_val_gap < 0:
                print(" (good generalization)")
            else:
                print(" (within expected range)")
        
        if stats["test"]["count"] > 0:
            print(f"Test RMSE:     {stats['test']['mean']:.2f} ± {stats['test']['std']:.2f}")
        
        print()
        
        # Print high RMSE outliers summary
        self.print_high_rmse_outliers()
        
        # Collision model comparison (if kb provided)
        if kb is not None:
            self.print_collision_comparison(kb)
        
        # Alpha hyperparameter validation
        self.validate_alpha()
        
        # Phy-Q benchmark report (if phyq_metrics provided)
        if phyq_metrics is not None:
            print(f"\n{'='*70}")
            print("PHY-Q BENCHMARK RESULTS")
            print(f"{'='*70}")
            phyq_metrics.print_report()
    
    def print_high_rmse_outliers(self) -> None:
        """Print summary of high RMSE outliers (RMSE > threshold)."""
        if not self.high_rmse_outliers:
            print(f"[OUTLIERS] No high RMSE outliers (threshold > {self.high_rmse_threshold})")
            return
        
        print(f"\n{'='*80}")
        print(f"HIGH RMSE OUTLIERS (RMSE > {self.high_rmse_threshold})")
        print(f"{'='*80}")
        print(f"{'Level':<8} {'Phase':<12} {'Angle':>8} {'RMSE':>10} {'Collisions':>12}")
        print(f"{'-'*80}")
        
        # Group by phase
        by_phase = {"train": [], "validation": [], "test": []}
        for outlier in self.high_rmse_outliers:
            phase = outlier.get("phase", "unknown")
            if phase in by_phase:
                by_phase[phase].append(outlier)
        
        for outlier in sorted(self.high_rmse_outliers, key=lambda x: x.get("level", 0)):
            level = outlier.get("level", "?")
            phase = outlier.get("phase", "?")
            angle = outlier.get("angle", "?")
            rmse = outlier.get("rmse", "?")
            n_collisions = outlier.get("n_collisions", "?")
            
            rmse_str = f"{rmse:.2f}" if isinstance(rmse, (int, float)) else str(rmse)
            angle_str = f"{angle:.1f}°" if isinstance(angle, (int, float)) else str(angle)
            
            print(f"{level:<8} {phase.upper():<12} {angle_str:>8} {rmse_str:>10} {n_collisions:>12}")
        
        print(f"{'-'*80}")
        print(f"Summary by Phase:")
        for phase, outliers in by_phase.items():
            if outliers:
                angles = [o.get("angle", 0) for o in outliers]
                rmses = [o.get("rmse", 0) for o in outliers]
                print(f"  {phase.upper():<12}: {len(outliers)} outliers | "
                      f"Angles: {min(angles):.1f}° - {max(angles):.1f}° | "
                      f"RMSE: {min(rmses):.1f} - {max(rmses):.1f}")
        
        # Check if outliers are concentrated in certain angle ranges
        all_angles = [o.get("angle", 0) for o in self.high_rmse_outliers]
        high_angles = [a for a in all_angles if a > 60]
        low_angles = [a for a in all_angles if a < 30]
        
        if len(high_angles) > len(self.high_rmse_outliers) * 0.5:
            print(f"\n[PATTERN] Most outliers ({len(high_angles)}/{len(all_angles)}) have HIGH angles (>60°)")
        elif len(low_angles) > len(self.high_rmse_outliers) * 0.5:
            print(f"\n[PATTERN] Most outliers ({len(low_angles)}/{len(all_angles)}) have LOW angles (<30°)")
        
        print(f"{'='*80}\n")
    
    def evaluate_collision_models(self, kb: Dict) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Evaluate trained collision models (General, CART, M5) on train/val/test samples.
        
        Uses models trained on train phase data to predict on all collected samples.
        Computes RMSE for each algorithm on each set.
        
        Parameters:
        -----------
        kb : dict
            Knowledge base containing trained collision models
        
        Returns:
        --------
        Dict with structure:
        {
            "train": {
                "general": {"v_x": rmse, "v_y": rmse, "y": rmse},
                "cart": {"v_x": rmse, "v_y": rmse, "y": rmse},
                "m5_l1": {"v_x": rmse, "v_y": rmse, "y": rmse},
                "m5_elasticnet": {"v_x": rmse, "v_y": rmse, "y": rmse}
            },
            "validation": {...},
            "test": {...}
        }
        """
        results = {
            "train": {},
            "validation": {},
            "test": {}
        }
        
        # Check if collision models exist
        if "collision" not in kb or "variables" not in kb["collision"]:
            print("[EVAL] No collision models found in KB")
            return results
        
        variables = kb["collision"]["variables"]
        
        # Algorithm names to evaluate
        algorithms = ["general", "cart", "m5_l1", "m5_elasticnet"]
        
        # Initialize result structure
        for phase in results:
            for algo in algorithms:
                results[phase][algo] = {}
        
        # Get samples for each phase
        phase_samples = {
            "train": self.train_collision_samples,
            "validation": self.val_collision_samples,
            "test": self.test_collision_samples
        }
        
        # Evaluate each variable (v_x, v_y, y)
        for var_name, var_data in variables.items():
            if var_name not in ["v_x", "v_y", "y"]:
                continue
                
            model_comparison = var_data.get("model_comparison")
            if model_comparison is None:
                continue
            
            # Get models
            general_model = model_comparison.get("general_model")
            cart_model = model_comparison.get("cart_model")
            m5_models = model_comparison.get("m5_models", {})
            
            # Evaluate on each phase
            for phase_name, samples in phase_samples.items():
                if not samples:
                    continue
                
                # Build feature matrix and target values from samples
                X = []
                y_true = []
                for pre_state, post_state in samples:
                    X.append([pre_state["x"], pre_state["y"], pre_state["v_x"], pre_state["v_y"]])
                    y_true.append(post_state[var_name])
                
                X = np.array(X)
                y_true = np.array(y_true)
                
                # Compute RMSE for each algorithm
                # General model
                if general_model is not None:
                    try:
                        X_poly = general_model.poly_features.transform(X)
                        y_pred = general_model.predict(X_poly)
                        rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
                        results[phase_name]["general"][var_name] = rmse
                    except Exception as e:
                        results[phase_name]["general"][var_name] = float('inf')
                
                # CART model
                if cart_model is not None:
                    try:
                        y_pred = cart_model.predict(X)
                        rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
                        results[phase_name]["cart"][var_name] = rmse
                    except Exception as e:
                        results[phase_name]["cart"][var_name] = float('inf')
                
                # M5 models
                for reg_type in ["l1", "elasticnet"]:
                    m5_model = m5_models.get(reg_type)
                    algo_name = f"m5_{reg_type}"
                    if m5_model is not None:
                        try:
                            y_pred = m5_model.predict(X)
                            rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
                            results[phase_name][algo_name][var_name] = rmse
                        except Exception as e:
                            results[phase_name][algo_name][var_name] = float('inf')
        
        # Store results for later access
        self.collision_eval_results = results
        return results
    
    def print_collision_comparison(self, kb: Dict) -> None:
        """
        Print collision model comparison table showing RMSE for each algorithm on each set.
        
        Parameters:
        -----------
        kb : dict
            Knowledge base containing trained collision models and LOO-CV stats
        """
        # Get evaluation results
        results = self.evaluate_collision_models(kb)
        
        # Get LOO-CV from training (stored in kb)
        loo_cv_results = {}
        if "collision" in kb and "variables" in kb["collision"]:
            for var_name, var_data in kb["collision"]["variables"].items():
                if var_name not in ["v_x", "v_y", "y"]:
                    continue
                model_comparison = var_data.get("model_comparison")
                if model_comparison:
                    loo_cv_results[var_name] = {
                        "general": model_comparison.get("general_stats", {}).get("loo_cv", float('inf')),
                        "cart": model_comparison.get("cart_stats", {}).get("loo_cv", float('inf')),
                        "m5_l1": model_comparison.get("m5_stats_by_reg", {}).get("l1", {}).get("loo_cv", float('inf')) if model_comparison.get("m5_stats_by_reg") else float('inf'),
                        "m5_elasticnet": model_comparison.get("m5_stats_by_reg", {}).get("elasticnet", {}).get("loo_cv", float('inf')) if model_comparison.get("m5_stats_by_reg") else float('inf')
                    }
        
        # Print header
        print(f"\n{'='*90}")
        print("COLLISION MODEL COMPARISON - RMSE by Algorithm and Set")
        print(f"{'='*90}")
        
        # Print sample counts
        print(f"\nCollision Samples: Train={len(self.train_collision_samples)}, "
              f"Val={len(self.val_collision_samples)}, Test={len(self.test_collision_samples)}")
        
        if not any(results[phase] for phase in results):
            print("\nNo collision data available for comparison.")
            return
        
        # Print table for each variable
        variables = ["v_x", "v_y", "y"]
        algorithms = ["general", "cart", "m5_l1", "m5_elasticnet"]
        algo_display = {"general": "General", "cart": "CART", "m5_l1": "M5-L1", "m5_elasticnet": "M5-ElasticNet"}
        
        for var_name in variables:
            print(f"\n--- {var_name} Prediction RMSE ---")
            print(f"{'Algorithm':<15} {'Train':>10} {'Val':>10} {'Test':>10} {'LOO-CV':>12}")
            print(f"{'-'*57}")
            
            for algo in algorithms:
                train_rmse = results["train"].get(algo, {}).get(var_name, float('inf'))
                val_rmse = results["validation"].get(algo, {}).get(var_name, float('inf'))
                test_rmse = results["test"].get(algo, {}).get(var_name, float('inf'))
                loo_cv = loo_cv_results.get(var_name, {}).get(algo, float('inf'))
                
                train_str = f"{train_rmse:>10.2f}" if np.isfinite(train_rmse) else f"{'N/A':>10}"
                val_str = f"{val_rmse:>10.2f}" if np.isfinite(val_rmse) else f"{'N/A':>10}"
                test_str = f"{test_rmse:>10.2f}" if np.isfinite(test_rmse) else f"{'N/A':>10}"
                loo_str = f"{loo_cv:>12.2f}" if np.isfinite(loo_cv) else f"{'N/A':>12}"
                
                print(f"{algo_display[algo]:<15} {train_str} {val_str} {test_str} {loo_str}")
        
        # Summary: Average RMSE across all variables
        print(f"\n--- Average RMSE (across v_x, v_y, y) ---")
        print(f"{'Algorithm':<15} {'Train':>10} {'Val':>10} {'Test':>10} {'LOO-CV':>12}")
        print(f"{'-'*57}")
        
        for algo in algorithms:
            train_rmses = [results["train"].get(algo, {}).get(v, float('inf')) for v in variables]
            val_rmses = [results["validation"].get(algo, {}).get(v, float('inf')) for v in variables]
            test_rmses = [results["test"].get(algo, {}).get(v, float('inf')) for v in variables]
            loo_cvs = [loo_cv_results.get(v, {}).get(algo, float('inf')) for v in variables]
            
            train_avg = np.mean([r for r in train_rmses if np.isfinite(r)]) if any(np.isfinite(r) for r in train_rmses) else float('inf')
            val_avg = np.mean([r for r in val_rmses if np.isfinite(r)]) if any(np.isfinite(r) for r in val_rmses) else float('inf')
            test_avg = np.mean([r for r in test_rmses if np.isfinite(r)]) if any(np.isfinite(r) for r in test_rmses) else float('inf')
            loo_avg = np.mean([r for r in loo_cvs if np.isfinite(r)]) if any(np.isfinite(r) for r in loo_cvs) else float('inf')
            
            train_str = f"{train_avg:>10.2f}" if np.isfinite(train_avg) else f"{'N/A':>10}"
            val_str = f"{val_avg:>10.2f}" if np.isfinite(val_avg) else f"{'N/A':>10}"
            test_str = f"{test_avg:>10.2f}" if np.isfinite(test_avg) else f"{'N/A':>10}"
            loo_str = f"{loo_avg:>12.2f}" if np.isfinite(loo_avg) else f"{'N/A':>12}"
            
            print(f"{algo_display[algo]:<15} {train_str} {val_str} {test_str} {loo_str}")
        
        print(f"\n{'='*90}")
    
    def is_complete(self) -> bool:
        """Check if protocol is complete."""
        return self.phase == "complete"
    
    def reset(self) -> None:
        """Reset the protocol to start fresh (keeps same angle pools)."""
        self.level_count = 0
        self.train_count = 0
        self.val_count = 0
        self.test_count = 0
        self.phase = "training"
        self.test_shots_done = 0
        self.val_block_remaining = 0
        self._last_validated_train_count = 0
        self.train_results = []
        self.val_results = []
        self.test_results = []
        self.train_collision_samples = []
        self.val_collision_samples = []
        self.test_collision_samples = []
        self.high_rmse_outliers = []
        print("[PROTOCOL] Reset complete. Ready for new training run.")
    
    def validate_alpha(self, alphas: List[float] = None, l1_ratio: float = 0.5) -> Dict[float, Dict[str, float]]:
        """
        Validate alpha hyperparameter by training models with different alpha values
        and evaluating on the validation set.
        
        Parameters:
        -----------
        alphas : List[float]
            List of alpha values to test (default: [0.1, 0.5, 1.0, 5.0, 10.0, 20.0, 50.0])
        l1_ratio : float
            L1/L2 ratio for ElasticNet (default: 0.5)
        
        Returns:
        --------
        Dict[float, Dict[str, float]]
            For each alpha: {"v_x": val_rmse, "v_y": val_rmse, "y": val_rmse, "avg": avg_rmse}
        """
        from sklearn.linear_model import Lasso, ElasticNet
        from sklearn.preprocessing import PolynomialFeatures
        
        if alphas is None:
            alphas = [0.01,0.05, 0.1,0.2, 0.5, 1.0, 5.0, 10.0]
        
        if not self.train_collision_samples:
            print("[ALPHA VALIDATION] No training samples available")
            return {}
        
        if not self.val_collision_samples:
            print("[ALPHA VALIDATION] No validation samples available")
            return {}
        
        # Build feature matrices
        X_train = np.array([[s[0]["x"], s[0]["y"], s[0]["v_x"], s[0]["v_y"]] 
                           for s in self.train_collision_samples])
        X_val = np.array([[s[0]["x"], s[0]["y"], s[0]["v_x"], s[0]["v_y"]] 
                         for s in self.val_collision_samples])
        
        # Build target arrays for each variable
        targets = {}
        for var in ["v_x", "v_y", "y"]:
            targets[var] = {
                "train": np.array([s[1][var] for s in self.train_collision_samples]),
                "val": np.array([s[1][var] for s in self.val_collision_samples])
            }
        
        # Polynomial features for General model
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_train_poly = poly.fit_transform(X_train)
        X_val_poly = poly.transform(X_val)
        
        results = {}
        
        print(f"\n{'='*90}")
        print("ALPHA HYPERPARAMETER VALIDATION")
        print(f"{'='*90}")
        print(f"Training samples: {len(self.train_collision_samples)}")
        print(f"Validation samples: {len(self.val_collision_samples)}")
        print(f"L1 ratio: {l1_ratio}")
        print(f"Testing alphas: {alphas}")
        print(f"\n{'-'*90}")
        
        # Header
        print(f"{'Alpha':>10} | {'v_x Val RMSE':>14} | {'v_y Val RMSE':>14} | {'y Val RMSE':>12} | {'Avg Val RMSE':>14}")
        print(f"{'-'*90}")
        
        best_alpha = None
        best_avg_rmse = float('inf')
        
        for alpha in alphas:
            results[alpha] = {}
            rmses = []
            
            for var in ["v_x", "v_y", "y"]:
                y_train = targets[var]["train"]
                y_val = targets[var]["val"]
                
                # Train ElasticNet with this alpha
                model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)
                model.fit(X_train_poly, y_train)
                
                # Predict on validation
                y_pred = model.predict(X_val_poly)
                rmse = np.sqrt(np.mean((y_pred - y_val) ** 2))
                results[alpha][var] = rmse
                rmses.append(rmse)
            
            avg_rmse = np.mean(rmses)
            results[alpha]["avg"] = avg_rmse
            
            # Track best
            if avg_rmse < best_avg_rmse:
                best_avg_rmse = avg_rmse
                best_alpha = alpha
            
            # Print row
            print(f"{alpha:>10.1f} | {results[alpha]['v_x']:>14.2f} | {results[alpha]['v_y']:>14.2f} | {results[alpha]['y']:>12.2f} | {avg_rmse:>14.2f}")
        
        print(f"{'-'*90}")
        print(f"\nBEST ALPHA: {best_alpha} (Avg Val RMSE: {best_avg_rmse:.2f})")
        print(f"{'='*90}\n")
        
        return results
