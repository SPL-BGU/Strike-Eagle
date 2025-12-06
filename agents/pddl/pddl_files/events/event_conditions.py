import numpy as np
import math

def is_ground_collision(frames,groundtruth_objects,i):
    epsilon=3
    if not "redBird_0" in frames[i]:
        return False
    else:
        return i > 0 and frames[i-1]["redBird_0"]['y'] <= epsilon and frames[i]["redBird_0"]['y'] > epsilon


def is_hit(frames,groundtruth_objects,i):
    """
    Handles bird-pig collision and killing the pig if the conditions are met.

    Parameters:
    - state: dict containing all required variable values
    - b: bird ID
    - p: pig ID
    """

    if not "redBird_0" in frames[i] or not "pig_0" in frames[i] :
        return False

    bird = frames[i]["redBird_0"]
    pig = frames[i]["pig_0"]
    # Unpack needed variables
    vx_bird = bird["v_x"]
    vy_bird = bird["v_y"]
    v_bird = np.hypot(vx_bird, vy_bird)

    x_bird, y_bird = bird["x"], bird["y"]
    x_pig, y_pig = pig["x"], pig["y"]

    r_bird = 3.5
    r_pig = 3.5


    dist_squared = (x_bird - x_pig) ** 2 + (y_bird - y_pig) ** 2
    radius_sum_squared = (r_bird + r_pig) ** 2

    if radius_sum_squared < dist_squared:
        return False

    return True

def is_platform_collision(frames, groundtruth_properties, i):
    frame = frames[i]

    # Bird check
    bird = frame.get("redBird_0")
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
        return False

    r_bird = 3.5  # fixed radius

    for pname in platform_names:
        platform = frame[pname]
        props = groundtruth_properties.get(pname, {})

        # Platform rect
        x_p, y_p = platform["x"], platform["y"]
        w = props[0]
        h = props[1]

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

        if dx * dx + dy * dy <= r_bird * r_bird:
            return True

    return False