from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
class WorldModel:


    # should be 87.2
    gravity =None
    v_bird = 175.9259

    def __init__(self,
                 # should be 87.2
                 gravity=87.2
                 ):
        self.gravity = gravity

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