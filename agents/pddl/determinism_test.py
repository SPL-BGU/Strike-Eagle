"""
Determinism Test for ScienceBirds Simulation

Tests whether the simulation produces consistent trajectories for the same launch angle.
Shoots birds at 4 fixed angles (35, 50, 60, 75 degrees) and compares trajectories
across multiple shots to measure simulation consistency/variance.

Usage:
    from agents.pddl.determinism_test import DeterminismTester
    tester = DeterminismTester(agent)
    tester.run_test(shots_per_angle=3)
    tester.visualize_results()
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import time


class DeterminismTester:
    """
    Tests simulation determinism by shooting at fixed angles and comparing trajectories.
    
    Attributes:
        test_angles: List of angles to test (degrees)
        trajectories: Dict mapping angle -> list of trajectory arrays
        ground_collision_frames: Dict mapping angle -> list of collision frame indices
    """
    
    TEST_ANGLES = [35, 50, 60, 75]
    GROUND_LEVEL = 360  # y coordinate of ground in screen coords
    FRAME_RATE = 0.02   # 50 fps
    
    def __init__(self, agent):
        """
        Initialize the determinism tester.
        
        Parameters:
            agent: PDDLAgent instance with shoot_and_record capability
        """
        self.agent = agent
        self.trajectories: Dict[int, List[np.ndarray]] = defaultdict(list)
        self.ground_collision_frames: Dict[int, List[int]] = defaultdict(list)
        self.shot_count = 0
        self.current_angle_index = 0
        
    def get_next_angle(self) -> int:
        """Get the next angle to shoot at (cycles through TEST_ANGLES)."""
        angle = self.TEST_ANGLES[self.current_angle_index]
        self.current_angle_index = (self.current_angle_index + 1) % len(self.TEST_ANGLES)
        return angle
    
    def record_trajectory(self, angle: int, trajectory: np.ndarray, 
                          ground_collision_frame: Optional[int] = None):
        """
        Record a trajectory for a given angle.
        
        Parameters:
            angle: Launch angle in degrees
            trajectory: Numpy array of shape (N, 2) with x, y positions
            ground_collision_frame: Frame index where ground collision occurred (optional)
        """
        self.trajectories[angle].append(trajectory.copy())
        if ground_collision_frame is not None:
            self.ground_collision_frames[angle].append(ground_collision_frame)
        self.shot_count += 1
        
    def find_ground_collision_frame(self, trajectory: np.ndarray) -> Optional[int]:
        """
        Find the frame where the bird first hits the ground.
        
        Parameters:
            trajectory: Numpy array of shape (N, 2) with x, y positions
            
        Returns:
            Frame index of first ground collision, or None if no collision
        """
        # In natural coordinates, ground is at y = 280 (640 - 360)
        natural_ground = 640 - self.GROUND_LEVEL
        
        for i in range(1, len(trajectory)):
            prev_y = trajectory[i-1, 1]
            curr_y = trajectory[i, 1]
            
            # Check if bird crossed ground level (going down)
            if prev_y > natural_ground and curr_y <= natural_ground:
                return i
                
        return None
    
    def compute_trajectory_delta(self, traj1: np.ndarray, traj2: np.ndarray) -> np.ndarray:
        """
        Compute per-frame delta (error) between two trajectories.
        
        Parameters:
            traj1: First trajectory array (N, 2)
            traj2: Second trajectory array (M, 2)
            
        Returns:
            Array of euclidean distances at each frame (length = min(N, M))
        """
        min_len = min(len(traj1), len(traj2))
        deltas = np.sqrt(
            (traj1[:min_len, 0] - traj2[:min_len, 0])**2 + 
            (traj1[:min_len, 1] - traj2[:min_len, 1])**2
        )
        return deltas
    
    def compute_statistics(self, angle: int) -> Dict:
        """
        Compute trajectory statistics for a given angle.
        
        Returns dict with:
            - mean_trajectory: Average trajectory
            - std_trajectory: Standard deviation at each point
            - pairwise_deltas: All pairwise delta arrays
            - max_delta: Maximum delta observed
            - mean_delta: Mean delta across all pairs
        """
        trajs = self.trajectories[angle]
        if len(trajs) < 2:
            return None
            
        # Find minimum length
        min_len = min(len(t) for t in trajs)
        
        # Truncate all trajectories to same length
        truncated = np.array([t[:min_len] for t in trajs])
        
        # Compute mean and std
        mean_traj = np.mean(truncated, axis=0)
        std_traj = np.std(truncated, axis=0)
        
        # Compute pairwise deltas
        pairwise_deltas = []
        for i in range(len(trajs)):
            for j in range(i+1, len(trajs)):
                delta = self.compute_trajectory_delta(trajs[i], trajs[j])
                pairwise_deltas.append(delta)
        
        # Statistics
        all_deltas = np.concatenate(pairwise_deltas)
        
        return {
            'mean_trajectory': mean_traj,
            'std_trajectory': std_traj,
            'pairwise_deltas': pairwise_deltas,
            'max_delta': np.max(all_deltas),
            'mean_delta': np.mean(all_deltas),
            'std_delta': np.std(all_deltas),
            'num_shots': len(trajs)
        }
    
    def visualize_results(self, save_path: Optional[str] = None):
        """
        Visualize determinism test results.
        
        Creates a figure with subplots for each angle showing:
        - All trajectories overlaid
        - Delta/error analysis separated into:
          1. Full first segment (before ground collision)
          2. 20 frames before and after ground collision
        
        Parameters:
            save_path: Optional path to save the figure
        """
        # Check if we have any data
        angles_with_data = [a for a in self.TEST_ANGLES if len(self.trajectories[a]) >= 2]
        
        if not angles_with_data:
            print("Not enough data to visualize. Need at least 2 shots per angle.")
            return
            
        # Create figure with 3 rows per angle: trajectories, full deltas, collision deltas
        n_angles = len(angles_with_data)
        fig, axes = plt.subplots(n_angles, 3, figsize=(18, 5*n_angles))
        
        if n_angles == 1:
            axes = axes.reshape(1, -1)
            
        fig.suptitle('Simulation Determinism Test\n(Lower delta = more consistent)', 
                     fontsize=14, fontweight='bold')
        
        colors = plt.cm.Set2(np.linspace(0, 1, 10))
        
        for row, angle in enumerate(angles_with_data):
            trajs = self.trajectories[angle]
            stats = self.compute_statistics(angle)
            collision_frames = self.ground_collision_frames.get(angle, [])
            
            # === Column 1: Trajectory overlay ===
            ax1 = axes[row, 0]
            for i, traj in enumerate(trajs):
                ax1.plot(traj[:, 0], traj[:, 1], 
                        color=colors[i % len(colors)], 
                        alpha=0.7, linewidth=1.5,
                        label=f'Shot {i+1}')
                
                # Mark ground collision if available
                if i < len(collision_frames) and collision_frames[i] is not None:
                    cf = collision_frames[i]
                    if cf < len(traj):
                        ax1.scatter(traj[cf, 0], traj[cf, 1], 
                                   marker='x', s=100, c='red', zorder=5)
            
            # Draw ground line
            ax1.axhline(y=640-self.GROUND_LEVEL, color='brown', 
                       linestyle='--', alpha=0.5, label='Ground')
            
            ax1.set_title(f'Angle {angle}° - {len(trajs)} shots')
            ax1.set_xlabel('X position')
            ax1.set_ylabel('Y position')
            ax1.legend(loc='upper right', fontsize=8)
            ax1.set_aspect('equal')
            ax1.grid(True, alpha=0.3)
            
            # === Column 2: Full trajectory delta ===
            ax2 = axes[row, 1]
            
            for i, delta in enumerate(stats['pairwise_deltas']):
                frames = np.arange(len(delta))
                ax2.plot(frames, delta, alpha=0.6, linewidth=1)
            
            # Add mean delta line
            if stats['pairwise_deltas']:
                min_len = min(len(d) for d in stats['pairwise_deltas'])
                mean_delta = np.mean([d[:min_len] for d in stats['pairwise_deltas']], axis=0)
                ax2.plot(np.arange(min_len), mean_delta, 
                        color='red', linewidth=2, label='Mean delta')
            
            ax2.set_title(f'Full Trajectory Delta\nMax: {stats["max_delta"]:.2f}, Mean: {stats["mean_delta"]:.2f}')
            ax2.set_xlabel('Frame')
            ax2.set_ylabel('Delta (pixels)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # === Column 3: Around ground collision (±20 frames) ===
            ax3 = axes[row, 2]
            
            # Find valid collision frames
            valid_collisions = [cf for cf in collision_frames if cf is not None]
            
            if valid_collisions and len(trajs) >= 2:
                # Use first collision as reference point
                ref_collision = valid_collisions[0]
                window = 20
                
                for i, delta in enumerate(stats['pairwise_deltas']):
                    # Extract window around collision
                    start = max(0, ref_collision - window)
                    end = min(len(delta), ref_collision + window)
                    
                    if start < end:
                        # Relative frame numbers (centered at collision)
                        rel_frames = np.arange(start - ref_collision, end - ref_collision)
                        ax3.plot(rel_frames, delta[start:end], alpha=0.6, linewidth=1.5)
                
                # Add vertical line at collision point
                ax3.axvline(x=0, color='red', linestyle='--', 
                           label='Ground collision', linewidth=2)
                
                # Compute stats for collision window
                collision_deltas = []
                for delta in stats['pairwise_deltas']:
                    start = max(0, ref_collision - window)
                    end = min(len(delta), ref_collision + window)
                    if start < end:
                        collision_deltas.extend(delta[start:end])
                
                if collision_deltas:
                    collision_max = np.max(collision_deltas)
                    collision_mean = np.mean(collision_deltas)
                    ax3.set_title(f'±20 Frames Around Collision\n'
                                 f'Max: {collision_max:.2f}, Mean: {collision_mean:.2f}')
                else:
                    ax3.set_title('±20 Frames Around Collision\n(No data)')
            else:
                ax3.set_title('±20 Frames Around Collision\n(No collision detected)')
                ax3.text(0.5, 0.5, 'No ground collision\ndetected in trajectories',
                        transform=ax3.transAxes, ha='center', va='center',
                        fontsize=12, alpha=0.5)
            
            ax3.set_xlabel('Frame (relative to collision)')
            ax3.set_ylabel('Delta (pixels)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved determinism test results to {save_path}")
        
        plt.show()
        
    def visualize_single_update(self, angle: int, save_path: Optional[str] = None):
        """
        Visualize results for a single angle (called after each shot).
        
        Shows live updates of:
        - All trajectories for this angle
        - Delta analysis
        
        Parameters:
            angle: The angle to visualize
            save_path: Optional path to save the figure
        """
        trajs = self.trajectories[angle]
        
        if len(trajs) < 2:
            print(f"Need at least 2 shots at angle {angle}° to compare.")
            return
            
        stats = self.compute_statistics(angle)
        collision_frames = self.ground_collision_frames.get(angle, [])
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        fig.suptitle(f'Determinism Test - Angle {angle}° ({len(trajs)} shots)', 
                     fontsize=14, fontweight='bold')
        
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        
        # === Plot 1: Trajectory overlay ===
        ax1 = axes[0]
        for i, traj in enumerate(trajs):
            ax1.plot(traj[:, 0], traj[:, 1], 
                    color=colors[i % len(colors)], 
                    alpha=0.7, linewidth=1.5,
                    label=f'Shot {i+1}')
            
            # Mark ground collision
            if i < len(collision_frames) and collision_frames[i] is not None:
                cf = collision_frames[i]
                if cf < len(traj):
                    ax1.scatter(traj[cf, 0], traj[cf, 1], 
                               marker='x', s=100, c='red', zorder=5)
        
        ax1.axhline(y=640-self.GROUND_LEVEL, color='brown', 
                   linestyle='--', alpha=0.5, label='Ground')
        ax1.set_title('Trajectory Overlay')
        ax1.set_xlabel('X position')
        ax1.set_ylabel('Y position')
        ax1.legend(loc='upper right', fontsize=8)
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        
        # === Plot 2: Full trajectory delta ===
        ax2 = axes[1]
        
        pair_labels = []
        for i in range(len(trajs)):
            for j in range(i+1, len(trajs)):
                pair_labels.append(f'{i+1} vs {j+1}')
        
        for idx, delta in enumerate(stats['pairwise_deltas']):
            frames = np.arange(len(delta))
            ax2.plot(frames, delta, alpha=0.7, linewidth=1.5,
                    label=pair_labels[idx] if idx < 5 else None)
        
        ax2.set_title(f'Full Trajectory Delta\nMax: {stats["max_delta"]:.2f}px, '
                     f'Mean: {stats["mean_delta"]:.2f}px ± {stats["std_delta"]:.2f}')
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('Delta (pixels)')
        if len(pair_labels) <= 5:
            ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # === Plot 3: Around ground collision ===
        ax3 = axes[2]
        
        valid_collisions = [cf for cf in collision_frames if cf is not None]
        
        if valid_collisions and len(trajs) >= 2:
            ref_collision = valid_collisions[0]
            window = 20
            
            collision_deltas_all = []
            
            for idx, delta in enumerate(stats['pairwise_deltas']):
                start = max(0, ref_collision - window)
                end = min(len(delta), ref_collision + window)
                
                if start < end:
                    rel_frames = np.arange(start - ref_collision, end - ref_collision)
                    ax3.plot(rel_frames, delta[start:end], alpha=0.7, linewidth=1.5,
                            label=pair_labels[idx] if idx < 5 else None)
                    collision_deltas_all.extend(delta[start:end])
            
            ax3.axvline(x=0, color='red', linestyle='--', 
                       label='Collision', linewidth=2)
            
            if collision_deltas_all:
                coll_max = np.max(collision_deltas_all)
                coll_mean = np.mean(collision_deltas_all)
                ax3.set_title(f'±20 Frames Around Ground Collision\n'
                             f'Max: {coll_max:.2f}px, Mean: {coll_mean:.2f}px')
        else:
            ax3.set_title('±20 Frames Around Collision\n(No collision detected)')
            ax3.text(0.5, 0.5, 'No ground collision detected',
                    transform=ax3.transAxes, ha='center', va='center')
        
        ax3.set_xlabel('Frame (relative to collision)')
        ax3.set_ylabel('Delta (pixels)')
        if len(pair_labels) <= 5:
            ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
        
        plt.show()
        
    def print_summary(self):
        """Print a text summary of determinism test results."""
        print("\n" + "="*70)
        print("DETERMINISM TEST SUMMARY")
        print("="*70)
        
        for angle in self.TEST_ANGLES:
            trajs = self.trajectories[angle]
            print(f"\nAngle {angle}°: {len(trajs)} shots")
            
            if len(trajs) < 2:
                print("  (Need at least 2 shots to compute statistics)")
                continue
                
            stats = self.compute_statistics(angle)
            print(f"  Max delta: {stats['max_delta']:.2f} pixels")
            print(f"  Mean delta: {stats['mean_delta']:.2f} ± {stats['std_delta']:.2f} pixels")
            
            # Check collision data
            collisions = self.ground_collision_frames.get(angle, [])
            valid_collisions = [c for c in collisions if c is not None]
            if valid_collisions:
                print(f"  Ground collision frames: {valid_collisions}")
                if len(valid_collisions) > 1:
                    collision_std = np.std(valid_collisions)
                    print(f"  Collision frame variance: {collision_std:.2f} frames")
        
        print("\n" + "="*70)
        print(f"Total shots: {self.shot_count}")
        print("="*70 + "\n")


def integrate_with_pddl_agent(agent):
    """
    Integrate determinism testing with PDDLAgent.
    
    Modifies the agent to use fixed angles from the tester and
    record trajectories after each shot.
    
    Usage:
        tester = integrate_with_pddl_agent(agent)
        # Now agent.solve() will record trajectories
        # Call tester.visualize_single_update(angle) after each shot
        
    Parameters:
        agent: PDDLAgent instance
        
    Returns:
        DeterminismTester instance
    """
    tester = DeterminismTester(agent)
    
    # Store original solve method
    original_solve = agent.solve
    
    def determinism_solve():
        """Modified solve that uses fixed angles and records trajectories."""
        from agents.utility import GroundTruthType
        from agents.pddl.trajectory_parser import extract_real_trajectory
        from agents.pddl.pddl_files.segments import getSegmentsEvents
        
        # Get next test angle
        angle = tester.get_next_angle()
        print(f"\n[DETERMINISM TEST] Shooting at angle {angle}°")
        
        ground_truth_type = GroundTruthType.ground_truth_screenshot
        vision = agent._update_reader(ground_truth_type.value, agent.if_check_gt)
        
        sling = vision.find_slingshot_mbr()[0]
        sling.width, sling.height = sling.height, sling.width
        
        # Calculate release point for this angle
        release_point = agent.tp.find_release_point_random_power(sling, angle * np.pi / 180)
        
        # Shoot and record
        batch_gt = agent.ar.shoot_and_record_ground_truth(
            release_point.X, release_point.Y, 0, 0, 1, 0
        )
        
        # Extract trajectory
        groundtruth_trajectories, groundtruth_objects = extract_real_trajectory(
            batch_gt, angle, agent.model, agent.target_class
        )
        
        # Get bird trajectory
        if "redBird_0" in groundtruth_trajectories:
            bird_traj = np.array(groundtruth_trajectories["redBird_0"])
            
            # Detect events to find ground collision
            event_indexes_by_event, objects_features = getSegmentsEvents(
                groundtruth_trajectories, groundtruth_objects
            )
            
            # Get first ground collision frame
            ground_collisions = event_indexes_by_event.get("ground_collision", [])
            first_collision = ground_collisions[0] if ground_collisions else None
            
            # Record trajectory
            tester.record_trajectory(angle, bird_traj, first_collision)
            
            print(f"[DETERMINISM TEST] Recorded trajectory with {len(bird_traj)} frames")
            if first_collision:
                print(f"[DETERMINISM TEST] Ground collision at frame {first_collision}")
            
            # Show live visualization for this angle
            tester.visualize_single_update(angle)
        
        # Wait for next level
        time.sleep(3)
        
    # Replace solve method
    agent.solve = determinism_solve
    
    return tester
