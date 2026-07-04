import numpy as np
import math

_BLOCK_PREFIXES = ("wood_", "stone_", "ice_", "TNT_")

# Cache the active bird key to avoid repeated detection
_active_bird_cache = {}


def _find_active_bird(frames, i, groundtruth_objects=None):
    """
    Find the active bird in the current frame.
    
    For multi-bird levels, finds the bird that is:
    1. Present in the frame
    2. Has the most significant movement (largest velocity)
    
    Falls back to first available bird if no moving bird found.
    """
    bird_keys = [k for k in frames[i].keys() if k.startswith("redBird")]
    
    if not bird_keys:
        return None
    
    # If only one bird, return it
    if len(bird_keys) == 1:
        return bird_keys[0]
    
    # Find bird with most movement (highest velocity magnitude)
    best_bird = None
    max_velocity = -1
    
    for bird_key in bird_keys:
        bird = frames[i].get(bird_key, {})
        vx = abs(bird.get("v_x", 0))
        vy = abs(bird.get("v_y", 0))
        velocity = np.hypot(vx, vy)
        
        if velocity > max_velocity:
            max_velocity = velocity
            best_bird = bird_key
    
    # Fall back to first bird if none moving
    return best_bird if best_bird else bird_keys[0]


def _block_names_in_frame(frame):
    return [name for name in frame.keys() if name.startswith(_BLOCK_PREFIXES)]


def is_ground_collision(frames, groundtruth_objects, i):
    """
    Detect ground collision: bird hits the ground with significant downward velocity.
    
    Conditions for a true bounce:
    1. Previous frame: y > epsilon (bird was above ground)
    2. Current frame: y <= epsilon (bird hits ground)
    3. Bird had significant downward velocity (v_y < -threshold)
    
    Supports multi-bird scenarios by finding the active bird.
    """
    epsilon = 3
    velocity_threshold = -0.5  # Minimum downward velocity to count as collision (not just rolling)
    
    # Find active bird (supports multi-bird)
    bird_key = _find_active_bird(frames, i, groundtruth_objects)
    if bird_key is None or bird_key not in frames[i]:
        return False
    
    if i == 0:
        return False
    
    # Check if bird exists in previous frame
    if bird_key not in frames[i-1]:
        return False
    
    current_y = frames[i][bird_key]['y']
    prev_y = frames[i-1][bird_key]['y']
    prev_vy = frames[i-1][bird_key].get('v_y', 0)
    
    # Collision: bird was above ground, now at/below ground, AND was moving downward
    is_collision = (prev_y > epsilon and 
                    current_y <= epsilon and 
                    prev_vy < velocity_threshold)
    
    return is_collision


def is_hit(frames, groundtruth_objects, i):
    """
    Handles bird-pig collision and killing the pig if the conditions are met.

    Parameters:
    - frames: list of frame dicts containing object states
    - groundtruth_objects: dict of object dimensions
    - i: frame index
    
    Supports multi-bird scenarios by finding the active bird.
    Checks against all pigs in the scene.
    """
    # Find active bird (supports multi-bird)
    bird_key = _find_active_bird(frames, i, groundtruth_objects)
    if bird_key is None or bird_key not in frames[i]:
        return False
    
    # Find all pigs in the frame
    pig_keys = [k for k in frames[i].keys() if k.startswith("pig")]
    if not pig_keys:
        return False

    bird = frames[i][bird_key]
    # Unpack bird variables
    vx_bird = bird["v_x"]
    vy_bird = bird["v_y"]
    v_bird = np.hypot(vx_bird, vy_bird)

    x_bird, y_bird = bird["x"], bird["y"]
    
    # Get bird dimensions
    dim_bird = groundtruth_objects.get(bird_key, [14, 14])
    r_bird = min(dim_bird) / 2
    
    # Check collision against each pig
    for pig_key in pig_keys:
        pig = frames[i].get(pig_key)
        if not pig:
            continue
            
        x_pig, y_pig = pig["x"], pig["y"]
        
        dim_pig = groundtruth_objects.get(pig_key, [20, 20])
        r_pig = min(dim_pig) / 2

        dist_squared = (x_bird - x_pig) ** 2 + (y_bird - y_pig) ** 2
        radius_sum_squared = (r_bird + r_pig) ** 2

        if dist_squared <= radius_sum_squared:
            return True

    return False

def is_platform_collision(frames, groundtruth_properties, i, debug=False):
    """
    Detect collision between bird and platform (hill).
    
    Supports multi-bird scenarios by finding the active bird.
    """
    frame = frames[i]

    # Find active bird (supports multi-bird)
    bird_key = _find_active_bird(frames, i, groundtruth_properties)
    if bird_key is None:
        return False
    
    bird = frame.get(bird_key)
    if not bird:
        return False

    # Collect platforms
    platform_names = [name for name in frame.keys() if "hill" in name]
    if not platform_names:
        return False

    # Bird state
    x_b, y_b = bird["x"], bird["y"]
    vx_b, vy_b = bird.get("v_x", 0.0), bird.get("v_y", 0.0)
    v_bird = math.hypot(vx_b, vy_b)

    # Require motion to count collision
    if v_bird <= 0:
        if debug and i % 50 == 0:
            print(f"[PLATFORM DEBUG] Frame {i}: Bird not moving (v={v_bird})")
        return False

    # Get actual bird radius from groundtruth (like is_hit does) + epsilon for tolerance
    COLLISION_EPSILON = 6.0  # pixels - increased to catch near-miss passes
    bird_dims = groundtruth_properties.get(bird_key, [14, 14])  # default ~14px diameter
    if isinstance(bird_dims, (list, tuple)) and len(bird_dims) >= 2:
        r_bird = max(bird_dims) / 2 + COLLISION_EPSILON
    else:
        r_bird = 7.0 + COLLISION_EPSILON  # reasonable default

    for pname in platform_names:
        platform = frame[pname]
        props = groundtruth_properties.get(pname, None)
        
        # Skip if no valid dimensions
        if not props or not isinstance(props, (list, tuple)) or len(props) < 2:
            if debug and i == 0:
                print(f"[PLATFORM DEBUG] {pname} has invalid props: {props}")
            continue
        
        w, h = props[0], props[1]
        if w <= 0 or h <= 0:
            if debug and i == 0:
                print(f"[PLATFORM DEBUG] {pname} has zero dimensions: w={w}, h={h}")
            continue

        # Platform rect
        x_p, y_p = platform["x"], platform["y"]

        # Define rectangle bounds (assuming x,y is center)
        left   = x_p - w / 2
        right  = x_p + w / 2
        top    = y_p - h / 2
        bottom = y_p + h / 2

        # Closest point on rect to circle center
        closest_x = max(left, min(x_b, right))
        closest_y = max(top, min(y_b, bottom))

        # Distance from circle center to closest point
        dx = x_b - closest_x
        dy = y_b - closest_y
        dist_sq = dx * dx + dy * dy
        
        # Debug output for close frames
        if debug and dist_sq < 400:  # Within 20 pixels
            dist = math.sqrt(dist_sq)
            print(f"[PLATFORM DEBUG] Frame {i}: Bird({x_b:.1f},{y_b:.1f}) Platform({x_p:.1f},{y_p:.1f}) "
                  f"bounds[{left:.1f},{right:.1f}]x[{top:.1f},{bottom:.1f}] "
                  f"closest({closest_x:.1f},{closest_y:.1f}) dist={dist:.1f} r_bird={r_bird:.1f} "
                  f"{'COLLISION!' if dist <= r_bird else 'miss'}")

        if dist_sq <= r_bird * r_bird:
            return True

    return False


def is_block_collision(frames, groundtruth_properties, i):
    """
    Detect bird collision with blocks (wood/stone/ice/TNT).
    
    Supports multi-bird scenarios by finding the active bird.
    """
    frame = frames[i]
    
    # Find active bird (supports multi-bird)
    bird_key = _find_active_bird(frames, i, groundtruth_properties)
    if bird_key is None:
        return False
    
    bird = frame.get(bird_key)
    if not bird:
        return False

    block_names = _block_names_in_frame(frame)
    if not block_names:
        return False

    x_b, y_b = bird["x"], bird["y"]
    vx_b, vy_b = bird.get("v_x", 0.0), bird.get("v_y", 0.0)
    if math.hypot(vx_b, vy_b) <= 0:
        return False

    # Get actual bird radius from groundtruth + epsilon for tolerance
    COLLISION_EPSILON = 6.0  # pixels - increased to catch near-miss passes
    bird_dims = groundtruth_properties.get(bird_key, [14, 14])
    if isinstance(bird_dims, (list, tuple)) and len(bird_dims) >= 2:
        r_bird = max(bird_dims) / 2 + COLLISION_EPSILON
    else:
        r_bird = 7.0 + COLLISION_EPSILON
    
    for bname in block_names:
        block = frame[bname]
        props = groundtruth_properties.get(bname, {})
        if not props or len(props) < 2:
            continue

        x_bl, y_bl = block["x"], block["y"]
        w, h = props[0], props[1]
        left = x_bl - w / 2
        right = x_bl + w / 2
        top = y_bl - h / 2
        bottom = y_bl + h / 2

        closest_x = max(left, min(x_b, right))
        closest_y = max(top, min(y_b, bottom))
        dx = x_b - closest_x
        dy = y_b - closest_y
        if dx * dx + dy * dy <= r_bird * r_bird:
            return True

    return False