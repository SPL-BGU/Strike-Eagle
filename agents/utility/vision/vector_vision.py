import copy

import numpy as np

from src.computer_vision.GroundTruthReader import GroundTruthReader
from src.computer_vision.game_object import GameObject
from agents.utility.vision.relations import *


class VectorVision(GroundTruthReader):
    def __init__(self, vision: GroundTruthReader):
        for key in vision.__dict__:
            setattr(self, key, vision.__dict__[key])
        self.epsilon = 4  # Number of pixels
        self.game_objects = None
        self.matrix = None

    def formulate_from_base_vision(self):
        # Flat game objets
        self.game_objects = [o for o_t in self.allObj.values() for o in o_t]

        # calculate matrix
        self.matrix = np.zeros((len(self.game_objects), len(self.game_objects), 2))

        for i, a in enumerate(self.game_objects):
            for j, b in enumerate(self.game_objects):
                self.matrix[i, j] = relative_loaction(a, b)
