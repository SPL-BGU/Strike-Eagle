from agents.pddl.pddl_files.pddl_parser import (
    block_bbox_from_game_object,
    pddl_bird_position_after_pa_twang,
    platform_bbox_from_game_object,
)
from agents.pddl.pddl_files.world_model.params import Params
from agents.pddl.pddl_files.world_model.world_model import WorldModel
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
            # Bird init position = sling reference point in PDDL coords.
            # pa-twang then kicks it by (-16*cos, -12*sin) to the actual launch position,
            # which matches the observed trajectory start (~sling center).
            ref_x = float(ref.X)
            ref_y = float(640 - ref.Y)

            print(f"\n[BIRD DEBUG] bird_{bird_id}:")
            print(f"  Screen X: {bird.X}, Y: {bird.Y}")
            print(f"  Width: {bird.width}, Height: {bird.height}")
            print(f"  Center PDDL: ({center_x:.1f}, {center_y_pddl:.1f})")
            after_x, after_y = pddl_bird_position_after_pa_twang(ref_x, ref_y, ref_angle_guess)
            print(f"  PDDL init (sling ref): ({ref_x:.1f}, {ref_y:.1f})")
            print(f"  After pa-twang @ dial {ref_angle_guess:.1f}° (bias-adjusted trig): "
                  f"({after_x:.1f}, {after_y:.1f})")

            problem_data[f"bird_{bird_id}"] = {
                "x_bird": ref_x,
                "y_bird": ref_y,
                "bird_id": bird_id,
                "bird_type": BIRD_TYPES.index(GameObjectType(bird_type)),
                "m_bird": bird.width * bird.height,  # check this because it is not mandatory
                "bird_radius": max(bird.width, bird.height) / 2,
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


# Block materials the PDDL planner should ignore entirely (skipped in get_blocks()).
# Ignored materials are not written to problem.pddl, so ENHSP treats their volume
# as empty and the Python forward simulator (which iterates problem_data for
# `block_*` keys) does not stop the bird on them either.
IGNORED_BLOCK_TYPES = frozenset({'ice'})


def get_blocks(vision, sling, tp, ignored_types=IGNORED_BLOCK_TYPES):
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
    ignored_types = frozenset(ignored_types or ())
    for block_type, blocks in block_types.items():
        if block_type in ignored_types:
            skipped = len(blocks) if blocks else 0
            if skipped:
                print(f"[BLOCK DEBUG] Skipping {skipped} '{block_type}' block(s) (ignored by planner)")
            continue
        for block in blocks:
            angle = getattr(block, "angle", 0) or 0
            bbox = block_bbox_from_game_object(block)
            width = bbox["block_width"]
            height = bbox["block_height"]

            print(f"\n[BLOCK DEBUG] block_{block_id} ({block_type}):")
            print(f"  Screen X: {block.X}, Y: {block.Y}")
            print(f"  Reported Width: {block.width}, Height: {block.height}")
            print(f"  Angle: {angle}°")
            if hasattr(block, "vertices") and block.vertices:
                print(f"  Vertices: {block.vertices}")
            print(
                f"  PDDL conservative AABB: W={width:.1f}, H={height:.1f}, "
                f"center=({bbox['x_block']:.1f}, {bbox['y_block']:.1f})"
            )

            problem_data[f"block_{block_id}"] = {
                **bbox,
                "block_life": blocks_data[block_type]["life"] * blocks_data[block_type]["multi"],
                "block_mass": width * height * blocks_data[block_type]["mass_coef"],
                "block_stability": 1,
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
        bbox = platform_bbox_from_game_object(platform)
        used_vertices = (
            hasattr(platform, "vertices")
            and platform.vertices
            and len(platform.vertices) >= 2
        )

        print(f"\n[PLATFORM DEBUG] platform_{platform_id} geometry:")
        print(f"  Source: {'vertices' if used_vertices else 'MBR fallback'}")
        print(
            f"  PDDL center=({bbox['x_platform']:.1f}, {bbox['y_platform']:.1f}), "
            f"size={bbox['platform_width']:.1f}x{bbox['platform_height']:.1f}"
        )
        top = bbox["y_platform"] + bbox["platform_height"] / 2
        print(f"  PDDL top={top:.1f}")

        problem_data[f"platform_{platform_id}"] = bbox
        platform_id += 1

    return problem_data
