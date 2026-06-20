import math

from agents.pddl.pddl_files.world_model.params import Params
from agents.pddl.pddl_files.world_model.world_model import WorldModel
from agents.pddl.pddl_files.pddl_parser import pddl_bird_position_before_pa_twang
from src.computer_vision.GroundTruthReader import GroundTruthReader
from src.computer_vision.game_object import GameObjectType


def get_birds(vision, sling, tp, agent_world_model: WorldModel, ref_angle_guess: float = 85.0):
    BIRD_TYPES = [GameObjectType.REDBIRD, GameObjectType.YELLOWBIRD, GameObjectType.BLACKBIRD,
                  GameObjectType.WHITEBIRD, GameObjectType.BLUEBIRD]
    bird_id = 0
    problem_data = dict()
    birds_types = vision.find_birds()
    ref = tp.get_reference_point(sling)
    
    # DEBUG: Log sling and reference point
    print(f"\n[BIRD DEBUG] Slingshot reference point:")
    print(f"  Screen coords: ref.X={ref.X}, ref.Y={ref.Y}")
    print(f"  PDDL Y = 640 - {ref.Y} = {640 - ref.Y}")
    
    for bird_type, birds in birds_types.items():
        for bird in birds:
            center_x = bird.X + bird.width / 2
            center_y_pddl = 640 - (bird.Y + bird.height / 2)
            # Match game release point (trajectory planner), then invert pa-twang for PDDL init
            release = tp.find_release_point_partial_power(
                sling, math.radians(ref_angle_guess), 1.0
            )
            launch_x = float(release.X)
            launch_y_pddl = 640 - float(release.Y)
            ref_x, ref_y = pddl_bird_position_before_pa_twang(
                launch_x, launch_y_pddl, ref_angle_guess
            )

            print(f"\n[BIRD DEBUG] bird_{bird_id}:")
            print(f"  Screen X: {bird.X}, Y: {bird.Y}")
            print(f"  Width: {bird.width}, Height: {bird.height}")
            print(f"  Center PDDL: ({center_x:.1f}, {center_y_pddl:.1f})")
            print(f"  Launch PDDL @ {ref_angle_guess:.1f}°: ({launch_x:.1f}, {launch_y_pddl:.1f})")
            print(f"  PDDL ref (before pa-twang): ({ref_x:.1f}, {ref_y:.1f})")
            print(f"  Sling ref (legacy): ({ref.X}, {640 - ref.Y})")

            problem_data[f"bird_{bird_id}"] = {
                "x_bird": ref_x,
                "y_bird": ref_y,
                "bird_id": bird_id,
                "bird_type": BIRD_TYPES.index(GameObjectType(bird_type)),
                "m_bird": bird.width * bird.height,  # check this because it is not mandatory
                "bird_radius": min(bird.width, bird.height) / 2,  # use min for tighter collision detection
                "v_bird": agent_world_model.hyperparams_values[Params.velocity],
                "bounce_count": 0,

                # "v_bird": 190.5
            }
            bird_id += 1
    return problem_data


def get_pigs(vision, sling, tp):
    pig_id = 0
    problem_data = dict()
    pigs = vision.find_pigs_mbr()
    for pig in pigs:
        temp_pt = pig.get_centre_point()

        # DEBUG: Log raw screen coordinates
        print(f"\n[PIG DEBUG] pig_{pig_id} raw screen coords:")
        print(f"  Screen X: {pig.X}, Y: {pig.Y}")
        print(f"  Width: {pig.width}, Height: {pig.height}")
        print(f"  Center point: ({pig.X + pig.width/2}, {pig.Y + pig.height/2})")
        
        center_x = pig.X + pig.width / 2
        center_y_pddl = 640 - (pig.Y + pig.height / 2)
        print(f"  PDDL center: ({center_x:.1f}, {center_y_pddl:.1f})")

        problem_data[f"pig_{pig_id}"] = {
            "x_pig": center_x,
            "y_pig": center_y_pddl,
            "m_pig": pig.width * pig.height,  # check this because it is not mandatory
            "pig_radius": min(pig.width, pig.height) / 2,  # use min for tighter collision detection
            "pig_life": 1  # check

        }
        pig_id += 1

    return problem_data


def get_blocks(vision, sling, tp):
    import numpy as np
    
    block_types = vision.find_blocks()
    x = 0
    blocks_data = {
        'wood': {
            'life': 0.75,
            'mass_coef': 0.375,
            'multi': 1
        },
        'ice': {
            'life': 0.75,
            'mass_coef': 0.375,
            'multi': 0.5
        },
        'TNT': {
            'life': 0.75,
            'mass_coef': 0.375,
            'multi': 0.5
        },

        'stone': {
            'life': 1.2,
            'mass_coef': 0.375,
            'multi': 2
        }
    }
    problem_data = dict()

    block_id = 0
    if not block_types:
        return {}
    for block_type, blocks in block_types.items():
        for block in blocks:
            # Get rotation angle if available
            angle = getattr(block, 'angle', 0) or 0
            
            # Calculate actual dimensions and center from vertices if available
            if hasattr(block, 'vertices') and block.vertices and len(block.vertices) >= 2:
                vertices = np.array(block.vertices)
                x_coords = vertices[:, 0]
                y_coords = vertices[:, 1]
                
                # Calculate actual bounding box from vertices
                actual_width = np.max(x_coords) - np.min(x_coords)
                actual_height = np.max(y_coords) - np.min(y_coords)
                center_x = (np.max(x_coords) + np.min(x_coords)) / 2
                center_y = (np.max(y_coords) + np.min(y_coords)) / 2
                
                # Use actual dimensions from vertices
                width = actual_width
                height = actual_height
                x_center = center_x
                y_center = center_y
            else:
                # Fallback to reported dimensions
                width = block.width
                height = block.height
                x_center = block.X + block.width / 2
                y_center = block.Y + block.height / 2
            
            # DEBUG: Log block info including rotation
            print(f"\n[BLOCK DEBUG] block_{block_id} ({block_type}):")
            print(f"  Screen X: {block.X}, Y: {block.Y}")
            print(f"  Reported Width: {block.width}, Height: {block.height}")
            print(f"  Angle: {angle}°")
            if hasattr(block, 'vertices') and block.vertices:
                print(f"  Vertices: {block.vertices}")
                print(f"  Actual from vertices: W={width:.0f}, H={height:.0f}, Center=({x_center:.0f}, {y_center:.0f})")
            
            problem_data[f"block_{block_id}"] = {
                "x_block": x_center,
                "y_block": 640 - y_center,
                "block_width": width,
                "block_height": height,
                "block_life": blocks_data[block_type]['life'] * blocks_data[block_type]['multi'],
                "block_mass": width * height * blocks_data[block_type]['mass_coef'],
                "block_stability": 1,
                # For visualization only (not written to PDDL):
                "_block_angle": angle,
                "_block_type": block_type,
            }
            block_id += 1

    return problem_data



def get_platforms(vision: GroundTruthReader, sling, tp):
    platform_id = 0
    problem_data = dict()
    
    # Collect platforms from multiple possible sources
    all_platforms = []
    
    # Try 'hill' key (common for ground truth)
    hills = vision.find_hill_mbr()
    if hills:
        print(f"[PLATFORM DEBUG] Found {len(hills)} objects via find_hill_mbr()")
        all_platforms.extend(hills)
    else:
        print("[PLATFORM DEBUG] find_hill_mbr() returned None/empty")
    
    # Also try 'Platform' key (alternative naming)
    try:
        platforms_alt = vision.find_platform_mbr()
        if platforms_alt:
            print(f"[PLATFORM DEBUG] Found {len(platforms_alt)} objects via find_platform_mbr()")
            all_platforms.extend(platforms_alt)
        else:
            print("[PLATFORM DEBUG] find_platform_mbr() returned None/empty")
    except Exception as e:
        print(f"[PLATFORM DEBUG] find_platform_mbr() error: {e}")
    
    # Debug: dump all object keys to see what's available
    if hasattr(vision, 'allObj') and vision.allObj:
        obj_keys = list(vision.allObj.keys()) if isinstance(vision.allObj, dict) else "not a dict"
        print(f"[PLATFORM DEBUG] Available object keys in vision.allObj: {obj_keys}")
    
    if not all_platforms:
        print("[PLATFORM DEBUG] No platforms found from any source!")
        return {}
    
    print(f"[PLATFORM DEBUG] Processing {len(all_platforms)} total platform(s)")
    
    for platform in all_platforms:
        # Get dimensions WITHOUT mutating the original object
        # The original has width/height swapped, so we swap them back for our use
        # Use local variables to avoid mutating the cached vision object
        width = platform.height  # Swap: use height as width
        height = platform.width  # Swap: use width as height
        
        # DEBUG: Log raw screen coordinates
        print(f"\n[PLATFORM DEBUG] platform_{platform_id} raw screen coords:")
        print(f"  Screen X: {platform.X}, Y: {platform.Y}")
        print(f"  Original W/H: {platform.width}/{platform.height} -> Using W/H: {width}/{height}")
        
        pddl_y = 640 - platform.Y - height / 2
        print(f"  PDDL Y = 640 - {platform.Y} - {height}/2 = {pddl_y}")
        
        problem_data[f"platform_{platform_id}"] = {
            "x_platform": platform.X + width / 2,
            "y_platform": pddl_y,
            "platform_width": width,
            "platform_height": height
        }
        platform_id += 1

    return problem_data
