from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, RegressorMixin
class WorldModel:


    # should be 87.2
    gravity =None
    v_bird = 175.9259

    def __init__(self,
                 # should be 87.2
                 gravity=90
                 ):
        self.gravity = gravity