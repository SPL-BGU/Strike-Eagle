from src.computer_vision.game_object import GameObject
import numpy as np


def center_of_mass(a: GameObject) -> np.ndarray:
    return np.mean(np.asarray(a.vertices), axis=0)


def relative_loaction(a: GameObject, b: GameObject) -> np.ndarray:
    return center_of_mass(a) - center_of_mass(b)
