from agents.utility.exceptions import GameSessionWonException,GameSessionLossException

from enum import Enum

class GroundTruthType(Enum):
    ground_truth_screenshot = "ground_truth_screenshot"
    ground_truth = "ground_truth"