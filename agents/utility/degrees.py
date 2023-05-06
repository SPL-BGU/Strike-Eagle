import math
import numpy as np
def get_n_degrees_possibilities(deg_range: tuple, step: float) -> int:
    return int((deg_range[1]-deg_range[0]) / step)


def get_deg_from_index(min_deg: int, step: float, index: int) -> float:
    return min_deg + step * index


def get_random_deg(deg_range: tuple, step: float) -> float:
    """
        Get a random degree from range
    """
    n = get_n_degrees_possibilities(deg_range, step)
    i=np.random.randint(n)
    return get_deg_from_index(deg_range[0],step,i)
