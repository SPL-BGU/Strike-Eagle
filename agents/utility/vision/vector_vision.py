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
        self.game_objects = dict()
        self.matrix = None

    def formulate_from_base_vision(self):
        """
        Formulate VectorVision by updating and adding attributes to the GroundTruthReader
        :return:
        :rtype:
        """
        # Setup
        self.game_objects["center_of_mass"] = list()
        self.game_objects["object_types"] = list()

        # Flat game objets
        for o, o_t in self.allObj.items():
            self.game_objects["object_types"] += [(o, "Alive") for _ in o_t]
        self.game_objects["objects"] = [o for o_t in self.allObj.values() for o in o_t]

        # calculate matrix
        self.matrix = np.zeros((len(self.game_objects["objects"]), len(self.game_objects["objects"]), 2))

        for i, a in enumerate(self.game_objects["objects"]):
            self.game_objects["center_of_mass"].append(center_of_mass(a))
            for j, b in enumerate(self.game_objects["objects"]):
                self.matrix[i, j] = relative_loaction(a, b)

    def update_dead(self, base_vector_vision):
        """
        Update if objects are dead by seeing if they vanished from the base vision
        :param base_vector_vision:
        :type base_vector_vision:
        :return:
        :rtype:
        """
        i, j = 0, 0
        n_objects = max(len(self.game_objects["object_types"]), len(base_vector_vision.game_objects["object_types"]))
        for _ in range(n_objects):
            if self.game_objects["object_types"][i] != base_vector_vision.game_objects["object_types"][j]:
                # Update Dead
                self.game_objects["object_types"].insert(i, (base_vector_vision.game_objects["object_types"][j][0], "Dead"))
            j += 1
            i += 1

    def update_matrix(self):
        """
        Update matrix to have all objects from the game objects
        :param base_vector_vision:
        :type base_vector_vision:
        :return:
        :rtype:
        """
        for i, o_t in enumerate(self.game_objects["object_types"]):
            if o_t[1] == "Dead":
                self.matrix = np.insert(self.matrix, i, 0, axis=0)
                self.matrix = np.insert(self.matrix, i, 0, axis=1)
