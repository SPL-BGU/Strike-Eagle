
def get_n_degrees_possibilities(min_deg: int, max_deg: int, step: float) -> int:
    return int((max_deg - min_deg) / step)


def get_deg_from_index(min_deg: int, step: float, index:int) -> float:
    return min_deg + step * index
