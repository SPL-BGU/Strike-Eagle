from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np

from agents.pddl.pddl_files.world_model.process import Process
from agents.pddl.pddl_files.world_model.params import Params, HyperParams


class WorldModel:
    # should be 87.2

    param_identity = {
        Params.velocity_x: [Params.velocity, Params.angle],
        Params.velocity_y: [Params.velocity, Params.gravity, Params.angle],
        Params.x: [Params.velocity_x,
                   # Params.x_start
                   ],
        Params.y: [Params.velocity_y,
                   # Params.y_start
                   ],
    }

    process_identity = {
        Process.flight: [Params.x, Params.y]
    }

    hyperparams_values = {
        Params.gravity: 87.2,
        Params.velocity: 175.9259
    }

    def __init__(self,
                 hyperparams_values
                 ):
        self.hyperparams_values = self.hyperparams_values | hyperparams_values

    def get_all_hyper_parameters(self, process: Process) -> list:
        parameters = set()

        hyper_param_values = {param.value for param in HyperParams}

        def expand_param(param: Params):
            parameters.add(param)
            if param in self.param_identity:
                for dep in self.param_identity[param]:
                    expand_param(dep)

        for param in self.process_identity[process]:
            expand_param(param)

        return [param for param in parameters if param.value in hyper_param_values]

    def taylor_sin(self, x, n_terms=10):
        series_sum = 0
        for n in range(n_terms):
            # Calculate each term: (-1)^n * x^(2n+1) / (2n+1)!
            term = ((-1) ** n) * (x ** (2 * n + 1)) / np.math.factorial(2 * n + 1)
            series_sum += term
        return series_sum

    def taylor_cos(self, x, n_terms=10):
        series_sum = 0
        for n in range(n_terms):
            # Calculate each term: (-1)^n * x^(2n) / (2n)!
            term = ((-1) ** n) * (x ** (2 * n)) / np.math.factorial(2 * n)
            series_sum += term
        return series_sum
