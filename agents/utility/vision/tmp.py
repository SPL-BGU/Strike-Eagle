from src.computer_vision.game_object import GameObject


def overlap(a: GameObject, b: GameObject, epsilon: float):
    """
    Check if a and a overlap
    :param a:
    :type a:
    :param b:
    :type b:
    :param epsilon:
    :type epsilon:
    :return:
    :rtype:
    """
    for vertex in a.vertices:
        if is_point_inside_object(vertex, b.vertices, epsilon):
            return True
    for vertex in b.vertices:
        if is_point_inside_object(vertex, a.vertices, epsilon):
            return True
    return False

def is_point_inside_object(point, obj: list, epsilon: float):
    intersections = 0
    for i in range(len(obj)):
        j = (i + 1) % len(obj)
        if (obj[i][1] > point[1] + epsilon) != (obj[j][1] > point[1] + epsilon) and point[
            0] + epsilon < (obj[j][0] - obj[i][0]) * (
                point[1] - obj[i][1]) + epsilon / (obj[j][1] - obj[i][1]) + obj[i][0] + epsilon:
            intersections += 1
    return intersections % 2 == 1


def above(a: GameObject, b: GameObject):
    """
    Define if a is above b
    :param a:
    :type a:
    :param b:
    :type b:
    :return:
    :rtype:
    """
    return a.bottom_right[1] < b.top_left[1]


def below(a: GameObject, b: GameObject):
    """
    Define if a is above b
    :param a:
    :type a:
    :param b:
    :type b:
    :return:
    :rtype:
    """
    return a.bottom_right[1] > b.top_left[1]


def right_to(a: GameObject, b: GameObject):
    """
    Define if a is right to b
    :param a:
    :type a:
    :param b:
    :type b:
    :return:
    :rtype:
    """
    return a.bottom_right[0] > b.top_left[0]


def left_to(a: GameObject, b: GameObject):
    """
    Define if a is left to b
    :param a:
    :type a:
    :param b:
    :type b:
    :return:
    :rtype:
    """
    return a.top_left[0] > b.bottom_right[0]


def close_top(a: GameObject, b: GameObject, epsilon: float):
    """
    Define if a is on b
    :param a:
    :type a:
    :param b:
    :type b:
    :param epsilon:
    :type epsilon
    :return:
    :rtype:
    """
    return overlap(a, b, epsilon) and abs(a.bottom_right[1] - b.top_left[1]) < epsilon


def close_below(a: GameObject, b: GameObject, epsilon: float):
    """
    Define if b is on a
    :param a:
    :type a:
    :param b:
    :type b:
    :param epsilon:
    :type epsilon
    :return:
    :rtype:
    """
    return overlap(a, b, epsilon) and abs(a.bottom_right[1] - b.top_left[1]) < epsilon


def close_right(a: GameObject, b: GameObject, epsilon: float):
    """
    Define if a is close to the right to b
    :param a:
    :type a:
    :param b:
    :type b:
    :param epsilon:
    :type epsilon
    :return:
    :rtype:
    """
    return overlap(a, b, epsilon) and abs(a.bottom_right[0] - b.top_left[0]) < epsilon


def close_left(a: GameObject, b: GameObject, epsilon: float):
    """
    Define if b is on a
    :param a:
    :type a:
    :param b:
    :type b:
    :param epsilon:
    :type epsilon
    :return:
    :rtype:
    """
    return overlap(a, b, epsilon) and abs(a.top_left[0] - b.bottom_right[0]) < epsilon
