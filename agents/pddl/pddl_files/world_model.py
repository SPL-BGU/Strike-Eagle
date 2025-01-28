from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
class WorldModel:


    # should be 87.2
    gravity = 87.2
    v_bird = 175.9259

    def __init__(self,
                 # should be 87.2
                 gravity=110,
                 v_bird = 60
                 ):
        self.gravity = gravity
        self.v_bird = v_bird

    def __str__(self):
        return f"gravity: {self.gravity},\tv_bird: {self.v_bird}"

    def gravity_values(self):
        return [self.gravity,self.v_bird]


    def taylor_sin(self,x, n_terms=10):
        series_sum = 0
        for n in range(n_terms):
            # Calculate each term: (-1)^n * x^(2n+1) / (2n+1)!
            term = ((-1) ** n) * (x ** (2 * n + 1)) / np.math.factorial(2 * n + 1)
            series_sum += term
        return series_sum

    def taylor_cos(self,x, n_terms=10):
        series_sum = 0
        for n in range(n_terms):
            # Calculate each term: (-1)^n * x^(2n) / (2n)!
            term = ((-1) ** n) * (x ** (2 * n)) / np.math.factorial(2 * n)
            series_sum += term
        return series_sum