import numpy as np
def is_ground_collision(frames,i):
    epsilon=3
    if not "redBird_0" in frames[i]:
        return False
    else:
        return i > 0 and frames[i-1]["redBird_0"]['y'] <= epsilon and frames[i]["redBird_0"]['y'] > epsilon


def is_hit(frames,i):
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

    if v_bird <= 0:
        return False

    dist_squared = (x_bird - x_pig) ** 2 + (y_bird - y_pig) ** 2
    radius_sum_squared = (r_bird + r_pig) ** 2

    if radius_sum_squared < dist_squared:
        return False

    return True


def is_platform_collision(frames,i):
    present_platforms = list(filter(lambda object_name: "hill" in object_name,frames[i]))
    if not "redBird_0" in frames[i] or present_platforms == []:
        return False
    for platform_name in present_platforms:
        bird = frames[i]["redBird_0"]
        platform = frames[i][platform_name]
        # Unpack needed variables
        vx_bird = bird["v_x"]
        vy_bird = bird["v_y"]
        v_bird = np.hypot(vx_bird, vy_bird)

        x_bird, y_bird = bird["x"], bird["y"]
        x_platform, y_platform = platform["x"], platform["y"]

        r_bird = 3.5
        r_platform = 3.5

        if v_bird <= 0:
            return False

        dist_squared = (x_bird - x_platform) ** 2 + (y_bird - y_platform) ** 2
        radius_sum_squared = (r_bird + r_platform) ** 2

        if radius_sum_squared < dist_squared:
            return False

        return True
