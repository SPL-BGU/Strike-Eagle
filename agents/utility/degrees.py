import math
import numpy as np
def get_n_degrees_possibilities(min_deg: int, max_deg: int, step: float) -> int:
    return int((max_deg - min_deg) / step)


def get_deg_from_index(min_deg: int, step: float, index:int) -> float:
    return min_deg + step * index

def get_random_deg(deg_range: tuple, step: float) -> float:
    """
    Get a random degree from range
    """
    return